#!/usr/bin/env python3
"""
02_embed.py
===========

Step 2 of the pipeline.

Encode the prepared arXiv subset with the SPECTER2 scientific-document model,
L2-normalize the vectors, and persist them for the index-loading and search
steps.

Why SPECTER2?
-------------
SPECTER2 (``allenai/specter2``) is trained on the citation graph of scientific
papers, so semantically related papers end up close together even when their
abstracts share few words - exactly what we want for semantic search over
science. If the weights can't be downloaded (offline), we fall back to the
smaller, general-purpose ``all-MiniLM-L6-v2`` with a clear warning; everything
downstream still works, just with slightly weaker domain semantics.

What gets embedded?
-------------------
SPECTER2 expects ``title [SEP] abstract`` as input - the title carries a lot of
the topical signal, so we concatenate them. (For all-MiniLM we use the same
concatenation with a plain separator.)

Outputs (in embeddings/):
    embeddings.npy   float32 [N, D], L2-normalized
    id_map.json      list[str], row index -> arxiv id
    metadata.json    arxiv id -> {title, authors, year, category}
    model_info.json  {"model": ..., "dim": ..., "metric": "cosine"}

USAGE
-----
    python scripts/02_embed.py
    python scripts/02_embed.py --batch-size 64 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from common import (
    EMB_DIR,
    EMBEDDINGS_NPY,
    ID_MAP_JSON,
    META_JSON,
    MODEL_INFO_JSON,
    SUBSET_PARQUET,
    load_embedding_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("02_embed")

# SPECTER2 / sentence-transformers separator token.
SEP = " [SEP] "


def build_input_text(row: pd.Series) -> str:
    """Concatenate title and abstract the way SPECTER2 expects."""
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    return f"{title}{SEP}{abstract}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(SUBSET_PARQUET), help="Parquet from step 01.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoder batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on rows (for quick tests).")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit).copy()
    log.info("Loaded %d records from %s", len(df), args.input)

    model = load_embedding_model()
    log.info("Embedding with model=%s (dim=%d)", model.name, model.dim)

    texts = [build_input_text(row) for _, row in df.iterrows()]
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize=True,
        show_progress=True,
    )
    log.info("Encoded matrix shape=%s dtype=%s", vectors.shape, vectors.dtype)

    # ---- required diagnostics ------------------------------------------- #
    first_norm = float(np.linalg.norm(vectors[0])) if len(vectors) else 0.0
    print("\n========== EMBEDDING SUMMARY ==========")
    print(f"Total texts processed : {len(texts)}")
    print(f"Embedding dimension   : {vectors.shape[1]} (expect 768 for SPECTER2)")
    print(f"L2 norm of vector[0]  : {first_norm:.6f} (expect ~1.0 after normalization)")
    print(f"Model used            : {model.name}")
    print("=======================================\n")

    # ---- persist artifacts ---------------------------------------------- #
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, vectors)

    id_map = df["id"].astype(str).tolist()
    with ID_MAP_JSON.open("w", encoding="utf-8") as fh:
        json.dump(id_map, fh, ensure_ascii=False, indent=0)

    metadata = {
        str(row["id"]): {
            "title": str(row["title"]),
            "authors": str(row["authors"]),
            "year": int(row["year"]),
            "category": str(row["category"]),
        }
        for _, row in df.iterrows()
    }
    with META_JSON.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=0)

    model_info = {"model": model.name, "dim": model.dim, "metric": "cosine"}
    with MODEL_INFO_JSON.open("w", encoding="utf-8") as fh:
        json.dump(model_info, fh, ensure_ascii=False, indent=2)

    log.info("Saved embeddings -> %s", EMBEDDINGS_NPY)
    log.info("Saved id map     -> %s (%d ids)", ID_MAP_JSON, len(id_map))
    log.info("Saved metadata   -> %s", META_JSON)
    log.info("Saved model info -> %s : %s", MODEL_INFO_JSON, model_info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
