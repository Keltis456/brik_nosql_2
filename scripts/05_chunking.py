#!/usr/bin/env python3
"""
05_chunking.py
==============

Step 5 of the pipeline: chunking long abstracts and indexing the chunks.

Whole-abstract embeddings work well for short abstracts, but a long abstract
averaged into a single vector loses local detail: a query that matches one
sentence gets drowned out by the rest of the text. The fix is to split the text
into smaller chunks, embed each chunk separately, and search at the chunk level
so retrieval can hit the relevant passage directly.

This script:

  1. selects the 30 papers with the LONGEST abstracts from the prepared subset;
  2. splits each abstract with TWO strategies:
       * fixed       : fixed word-count windows with a small word overlap;
       * sentence    : whole sentences combined greedily up to a max word budget
                       (so sentences are never cut in the middle);
  3. builds a SEPARATE index per strategy. With USE_LOCAL=1 each strategy gets
     its own in-memory LocalCosineIndex namespace; with USE_LOCAL=0 each
     strategy gets its own Pinecone index (dimension 768, metric cosine);
  4. embeds every chunk with SPECTER2 and upserts it together with metadata
     (paper id, paper title, chunk text, chunk number, year, category) in
     batches with a tqdm progress bar;
  5. runs several test queries and prints the top-5 chunk results per strategy
     (article title + a snippet of the matching chunk).

USAGE
-----
    # Default: USE_LOCAL from .env, built-in test queries.
    python scripts/05_chunking.py

    # Custom queries and chunk sizes:
    python scripts/05_chunking.py \
        -q "graph neural networks for molecules" \
        -q "quantum error correction" \
        --fixed-size 60 --fixed-overlap 12 --sentence-max-words 60
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from common import (
    ROOT,
    SUBSET_PARQUET,
    LocalCosineIndex,
    load_embedding_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("05_chunking")

load_dotenv(ROOT / ".env")

SEP = " [SEP] "

# Built-in test queries (used when none are passed with -q).
DEFAULT_QUERIES = [
    "graph neural networks for molecules",
    "retrieval augmented generation for question answering",
    "quantum error correction surface code",
]


# --------------------------------------------------------------------------- #
# Chunkers (the two strategies required by the assignment)
# --------------------------------------------------------------------------- #
def chunk_fixed(text: str, size: int = 60, overlap: int = 12) -> list[str]:
    """
    Strategy 1: fixed word-count windows with a small overlap.

    Cheap and uniform, but it cuts mid-sentence, so a concept that straddles a
    boundary can be split across two vectors. The small overlap softens that by
    repeating a few boundary words in the next window.
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    chunks = [" ".join(words[i : i + size]) for i in range(0, len(words), step)]
    return [c for c in chunks if c.strip()]


def _split_sentences(text: str) -> list[str]:
    """NLTK sentence tokenizer if present; otherwise a decent regex fallback."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:  # noqa: BLE001
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]


def chunk_sentence(text: str, max_words: int = 60) -> list[str]:
    """
    Strategy 2: sentence-based chunks.

    Group whole sentences greedily until adding the next one would exceed
    max_words. Sentences are never cut in the middle, so each chunk is a clean
    semantic unit, which generally gives more meaningful embeddings.
    """
    sentences = _split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        n = len(s.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


# --------------------------------------------------------------------------- #
# Chunk record building
# --------------------------------------------------------------------------- #
def build_chunk_records(df: pd.DataFrame, strategy: str, args) -> list[dict]:
    """
    Turn each paper into a list of chunk records with full metadata.

    Each record: {id, text, metadata{paper_id, title, chunk_text, chunk_number,
    year, category}}.
    """
    records: list[dict] = []
    for _, row in df.iterrows():
        paper_id = str(row["id"])
        title = str(row["title"])
        abstract = str(row["abstract"])

        if strategy == "fixed":
            pieces = chunk_fixed(abstract, size=args.fixed_size, overlap=args.fixed_overlap)
        else:  # sentence
            pieces = chunk_sentence(abstract, max_words=args.sentence_max_words)

        for chunk_number, piece in enumerate(pieces):
            records.append(
                {
                    "id": f"{paper_id}#{strategy}#{chunk_number}",
                    "text": f"{title}{SEP}{piece}",
                    "metadata": {
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_text": piece,
                        "chunk_number": chunk_number,
                        "year": int(row["year"]),
                        "category": str(row["category"]),
                    },
                }
            )
    return records


# --------------------------------------------------------------------------- #
# Index backends (separate index / namespace per strategy)
# --------------------------------------------------------------------------- #
def index_local(strategy: str, records: list[dict], model, batch_size: int) -> LocalCosineIndex:
    """USE_LOCAL path: one in-memory index per strategy (a separate namespace)."""
    index = LocalCosineIndex(dim=model.dim)
    log.info("[%s] embedding + upserting %d chunks into local namespace ...", strategy, len(records))

    for start in tqdm(
        range(0, len(records), batch_size),
        desc=f"local upsert [{strategy}]",
        unit="batch",
    ):
        batch = records[start : start + batch_size]
        vecs = model.encode([r["text"] for r in batch], batch_size=batch_size, normalize=True)
        index.upsert(
            ids=[r["id"] for r in batch],
            vectors=vecs,
            metadatas=[r["metadata"] for r in batch],
        )
    log.info("[%s] local index ready (%d vectors).", strategy, len(index))
    return index


def query_local(index: LocalCosineIndex, query_vec: np.ndarray, top_k: int):
    return index.query(query_vec, top_k=top_k, metric="cosine")


def index_pinecone(strategy: str, records: list[dict], model, batch_size: int, recreate: bool = False):
    """USE_LOCAL=0 path: one Pinecone index per strategy (dim 768, cosine)."""
    import time

    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    base = os.getenv("PINECONE_INDEX", "arxiv-semantic-search")
    index_name = f"{base}-chunks-{strategy}"
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    pc = Pinecone(api_key=api_key)
    existing = [ix["name"] for ix in pc.list_indexes()]
    if recreate and index_name in existing:
        log.info("[%s] recreate requested: deleting index '%s' for a clean rebuild...", strategy, index_name)
        pc.delete_index(index_name)
        existing = [ix["name"] for ix in pc.list_indexes()]

    if index_name not in existing:
        log.info("[%s] creating Pinecone index '%s' (dim=%d, cosine)...", strategy, index_name, model.dim)
        pc.create_index(
            name=index_name,
            dimension=model.dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        for _ in range(30):
            if pc.describe_index(index_name)["status"]["ready"]:
                break
            time.sleep(1)
    else:
        log.info("[%s] reusing Pinecone index '%s' (pass --recreate for a clean rebuild).", strategy, index_name)

    index = pc.Index(index_name)

    for start in tqdm(
        range(0, len(records), batch_size),
        desc=f"pinecone upsert [{strategy}]",
        unit="batch",
    ):
        batch = records[start : start + batch_size]
        vecs = model.encode([r["text"] for r in batch], batch_size=batch_size, normalize=True)
        index.upsert(
            vectors=[
                {"id": r["id"], "values": vecs[i].tolist(), "metadata": r["metadata"]}
                for i, r in enumerate(batch)
            ]
        )
    return index


def query_pinecone(index, query_vec: np.ndarray, top_k: int):
    from types import SimpleNamespace

    res = index.query(vector=query_vec.tolist(), top_k=top_k, include_metadata=True)
    return [
        SimpleNamespace(id=m["id"], score=m["score"], metadata=m.get("metadata", {}))
        for m in res["matches"]
    ]


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def print_chunk_results(strategy: str, query: str, matches) -> None:
    print(f"\n--- strategy={strategy} | query={query!r} (top {len(matches)}) ---")
    if not matches:
        print("  (no results)")
        return
    for rank, m in enumerate(matches, start=1):
        md = m.metadata or {}
        snippet = str(md.get("chunk_text", ""))
        snippet = (snippet[:160] + " ...") if len(snippet) > 160 else snippet
        print(f"  {rank}. score={m.score:+.4f}  {md.get('title', '?')}")
        print(f"       chunk #{md.get('chunk_number', '?')}: \"{snippet}\"")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--query", action="append", default=None, help="Test query (repeatable).")
    parser.add_argument("--input", default=str(SUBSET_PARQUET))
    parser.add_argument("--num-papers", type=int, default=30, help="How many longest-abstract papers to use.")
    parser.add_argument("--fixed-size", type=int, default=60, help="Words per fixed chunk.")
    parser.add_argument("--fixed-overlap", type=int, default=12, help="Word overlap for fixed chunks.")
    parser.add_argument("--sentence-max-words", type=int, default=60, help="Max words per sentence chunk.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embed/upsert batch size.")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query per strategy.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the per-strategy chunk indexes for a clean rebuild.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    # Select the papers with the LONGEST abstracts (by word count).
    df = df.assign(_abs_words=df["abstract"].astype(str).str.split().str.len())
    df = df.sort_values("_abs_words", ascending=False).head(args.num_papers).reset_index(drop=True)
    log.info(
        "Selected %d longest-abstract papers (abstract words: max=%d, min=%d).",
        len(df),
        int(df["_abs_words"].max()),
        int(df["_abs_words"].min()),
    )

    # Test queries. If none are passed with -q, derive them from the titles of a
    # few of the SELECTED papers. The chunk demo corpus is ONLY these 30
    # long-abstract papers, which (with a representative random sample) are
    # arbitrary topics. Fixed generic queries would rarely match them, so we
    # query with the selected papers' own titles: this makes the chunk search
    # meaningful and lets us compare how fixed vs sentence chunking retrieves
    # the relevant passages.
    if args.query:
        queries = args.query
    else:
        titles = df["title"].astype(str).tolist()
        idxs = [0, len(titles) // 2, len(titles) - 1] if len(titles) >= 3 else list(range(len(titles)))
        queries = [titles[i] for i in idxs]
        log.info("No -q given; using titles of %d selected papers as demo queries.", len(queries))

    model = load_embedding_model()
    log.info("Model: %s (dim=%d)", model.name, model.dim)

    use_local = os.getenv("USE_LOCAL", "1") == "1"
    log.info("Backend: %s", "local in-memory (one namespace per strategy)" if use_local else "Pinecone (one index per strategy)")

    strategies = ["fixed", "sentence"]
    backends: dict[str, object] = {}

    for strategy in strategies:
        records = build_chunk_records(df, strategy, args)
        log.info(
            "[%s] %d chunks from %d papers (avg %.1f chunks/paper).",
            strategy,
            len(records),
            len(df),
            len(records) / max(1, len(df)),
        )
        if use_local:
            backends[strategy] = index_local(strategy, records, model, args.batch_size)
        else:
            backends[strategy] = index_pinecone(strategy, records, model, args.batch_size, recreate=args.recreate)

    # ---- run the test queries against each strategy --------------------- #
    print("\n========== CHUNK SEARCH RESULTS ==========")
    for query in queries:
        query_vec = model.encode([query], normalize=True)[0]
        for strategy in strategies:
            if use_local:
                matches = query_local(backends[strategy], query_vec, args.top_k)
            else:
                matches = query_pinecone(backends[strategy], query_vec, args.top_k)
            print_chunk_results(strategy, query, matches)
    print("\n==========================================")

    # ----------------------------------------------------------------- #
    # DISCUSSION (which strategy gives more meaningful chunks) - see README.
    # ----------------------------------------------------------------- #
    # * sentence : chunks are whole sentences, so each chunk is a self-contained
    #              semantic unit. Embeddings are cleaner because no sentence is
    #              cut in the middle, which is why this usually gives the more
    #              meaningful chunks for short factual queries.
    # * fixed    : simplest and most uniform, but it cuts mid-sentence; the small
    #              overlap helps a boundary-straddling concept survive in at least
    #              one window, at the cost of some redundant vectors.
    # Chunk size is the key knob: too small (10-15 words) and a chunk lacks the
    # context the model needs, so embeddings get noisy; too large (hundreds of
    # words) and you are back to averaging away local detail. The sweet spot is
    # task-dependent and should match the model's context and the query style.
    return 0


if __name__ == "__main__":
    sys.exit(main())
