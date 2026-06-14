#!/usr/bin/env python3
"""
common.py
=========

Shared helpers used across the pipeline so that names and behaviour stay
consistent between scripts 02-06:

* Paths to data / embeddings artifacts.
* The embedding-model loader (SPECTER2 with a graceful MiniLM fallback).
* L2 normalization.
* A tiny, fully-functional in-memory cosine index used as the local fallback
  for Pinecone (toggled by USE_LOCAL=1).

Nothing here talks to the network unless you actually load a model or call
Pinecone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("common")

# --------------------------------------------------------------------------- #
# Canonical paths (single source of truth for every script)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"

SUBSET_PARQUET = DATA_DIR / "arxiv_subset.parquet"
EMBEDDINGS_NPY = EMB_DIR / "embeddings.npy"
ID_MAP_JSON = EMB_DIR / "id_map.json"          # row index -> arxiv id
META_JSON = EMB_DIR / "metadata.json"          # arxiv id -> {title, authors, year, category}
MODEL_INFO_JSON = EMB_DIR / "model_info.json"  # which model / dim was used

# Preferred embedding model (matches the assignment) and its fallback.
SPECTER2_MODEL = "allenai/specter2_base"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization. Safe against zero vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingModel:
    """Thin wrapper so callers don't care which backend won."""

    name: str
    dim: int
    _encode_fn: Any

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        vecs = self._encode_fn(texts, batch_size, show_progress)
        vecs = np.asarray(vecs, dtype=np.float32)
        if normalize:
            vecs = l2_normalize(vecs)
        return vecs


def load_embedding_model() -> EmbeddingModel:
    """
    Load SPECTER2 (allenai/specter2_base) via sentence-transformers; fall back
    to all-MiniLM-L6-v2 if the SPECTER2 weights cannot be downloaded/loaded
    (e.g. offline).

    SPECTER2 is purpose-built for scientific documents: it is trained on the
    citation graph so that papers citing each other land close together. That
    makes it a much better default than a generic sentence encoder for this
    "search over scientific abstracts" task. Its model card recommends cosine
    similarity, which is why we L2-normalize and create the index with metric
    cosine.
    """
    from sentence_transformers import SentenceTransformer

    def _wrap(model: SentenceTransformer, name: str) -> EmbeddingModel:
        dim = int(model.get_sentence_embedding_dimension())

        def _encode(texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
            # We normalize ourselves in EmbeddingModel.encode for consistency.
            # sentence-transformers drives a tqdm progress bar internally when
            # show_progress_bar=True, which gives the batched progress the
            # assignment asks for.
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

        return EmbeddingModel(name=name, dim=dim, _encode_fn=_encode)

    try:
        log.info("Loading SPECTER2 model: %s", SPECTER2_MODEL)
        model = SentenceTransformer(SPECTER2_MODEL)
        return _wrap(model, SPECTER2_MODEL)
    except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
        log.warning(
            "Could not load SPECTER2 (%s). Falling back to %s. Reason: %s",
            SPECTER2_MODEL,
            FALLBACK_MODEL,
            exc,
        )
        model = SentenceTransformer(FALLBACK_MODEL)
        return _wrap(model, FALLBACK_MODEL)


# --------------------------------------------------------------------------- #
# Local fallback vector index (real, functional cosine search)
# --------------------------------------------------------------------------- #
@dataclass
class _Match:
    """Mimics the shape of a Pinecone query match so callers are uniform."""

    id: str
    score: float
    metadata: dict


class LocalCosineIndex:
    """
    Drop-in, in-memory replacement for a Pinecone index used when USE_LOCAL=1.

    Stores L2-normalized vectors, so a plain dot product equals cosine
    similarity. Supports metadata filtering on ``year`` (with $gte/$lte) and
    ``category`` (equality or $in), mirroring the subset of Pinecone's filter
    syntax that the search scripts use.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._meta: dict[str, dict] = {}

    # ---- write path ------------------------------------------------------ #
    def upsert(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self._ids.extend(ids)
        self._vectors = (
            vectors if self._vectors.size == 0 else np.vstack([self._vectors, vectors])
        )
        for _id, meta in zip(ids, metadatas):
            self._meta[_id] = meta

    # ---- filtering ------------------------------------------------------- #
    @staticmethod
    def _matches_filter(meta: dict, flt: dict | None) -> bool:
        if not flt:
            return True
        for field, condition in flt.items():
            value = meta.get(field)
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$gte" and not (value is not None and value >= target):
                        return False
                    if op == "$lte" and not (value is not None and value <= target):
                        return False
                    if op == "$eq" and value != target:
                        return False
                    if op == "$in" and value not in target:
                        return False
            else:  # bare equality
                if value != condition:
                    return False
        return True

    # ---- read path ------------------------------------------------------- #
    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        flt: dict | None = None,
        metric: str = "cosine",
    ) -> list[_Match]:
        if self._vectors.size == 0:
            return []

        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        if metric in ("cosine", "dotproduct"):
            # vectors already normalized; cosine == dot for normalized inputs.
            qn = l2_normalize(q) if metric == "cosine" else q
            scores = (self._vectors @ qn.ravel())
            higher_is_better = True
        elif metric == "euclidean":
            diffs = self._vectors - q
            scores = -np.linalg.norm(diffs, axis=1)  # negate so larger == closer
            higher_is_better = True
        else:
            raise ValueError(f"Unknown metric: {metric}")

        order = np.argsort(scores)[::-1] if higher_is_better else np.argsort(scores)

        results: list[_Match] = []
        for idx in order:
            _id = self._ids[idx]
            meta = self._meta.get(_id, {})
            if not self._matches_filter(meta, flt):
                continue
            results.append(_Match(id=_id, score=float(scores[idx]), metadata=meta))
            if len(results) >= top_k:
                break
        return results

    def __len__(self) -> int:
        return len(self._ids)


# --------------------------------------------------------------------------- #
# Artifact loaders shared by 03/04/05/06
# --------------------------------------------------------------------------- #
def load_embeddings() -> np.ndarray:
    if not EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_NPY} not found. Run scripts/02_embed.py first."
        )
    return np.load(EMBEDDINGS_NPY)


def load_id_map() -> list[str]:
    with ID_MAP_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_metadata() -> dict[str, dict]:
    with META_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_model_info() -> dict:
    with MODEL_INFO_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_local_index_from_artifacts() -> tuple[LocalCosineIndex, list[str], dict[str, dict]]:
    """Reconstruct an in-memory index from the saved 02_embed artifacts."""
    vectors = load_embeddings()
    id_map = load_id_map()
    meta = load_metadata()

    index = LocalCosineIndex(dim=vectors.shape[1])
    index.upsert(
        ids=list(id_map),
        vectors=vectors,
        metadatas=[meta.get(_id, {}) for _id in id_map],
    )
    return index, id_map, meta
