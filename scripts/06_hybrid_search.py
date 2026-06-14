#!/usr/bin/env python3
"""
06_hybrid_search.py
===================

Step 6 of the pipeline: hybrid lexical + semantic search fused with
Reciprocal Rank Fusion (RRF).

Two retrievers make COMPLEMENTARY errors:

  * BM25 (lexical / sparse): scores by exact token overlap. Great at rare,
    precise terms (acronyms, identifiers, surnames), blind to synonyms and
    paraphrase.
  * Vector search (dense / semantic): scores by embedding similarity. Captures
    meaning and paraphrase, but can miss a query that hinges on one rare exact
    token the embedding smears over.

RRF fuses their two ranked lists WITHOUT needing comparable scores:

        score(d) = sum over retrievers  1 / (k + rank_r(d)),   k = 60

Only RANKS matter, so we never have to reconcile BM25's unbounded scores with
cosine's [-1,1]. A document ranked high by EITHER retriever gets a solid fused
score; a document ranked high by BOTH wins. That is why hybrid typically beats
each component alone - it recovers the documents each method individually
misses.

Backend for the vector side: real Pinecone (USE_LOCAL=0) or the local in-memory
cosine index (USE_LOCAL=1). BM25 always runs locally over the parquet text.

USAGE
-----
    python scripts/06_hybrid_search.py -q "RRF for dense passage retrieval"
    python scripts/06_hybrid_search.py -q "transformer attention molecules" --top-k 5
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
    build_local_index_from_artifacts,
    load_embedding_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("06_hybrid_search")

load_dotenv(ROOT / ".env")

RRF_K = 60  # standard fusion constant from Cormack et al. (2009)

# Three default test queries (the assignment asks for 3), exercised through the
# three methods (BM25, vector, hybrid). Override with one or more -q flags.
DEFAULT_QUERIES = [
    "reciprocal rank fusion for dense passage retrieval",
    "graph neural networks for molecular property prediction",
    "quantum error correction surface code",
]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------- #
# Retrievers
# --------------------------------------------------------------------------- #
def bm25_rank(corpus_tokens, ids, query: str, top_k: int) -> list[tuple[str, float]]:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in order]


class VectorRetriever:
    """Builds the dense backend once, then answers many queries."""

    def __init__(self):
        self.model = load_embedding_model()
        self.use_local = os.getenv("USE_LOCAL", "1") == "1"
        if self.use_local:
            self.index, _, _ = build_local_index_from_artifacts()
            self._pc_index = None
        else:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX", "arxiv-semantic-search")
            self._pc_index = Pinecone(api_key=api_key).Index(index_name)

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vec = self.model.encode([query], normalize=True)[0]
        if self.use_local:
            matches = self.index.query(query_vec, top_k=top_k, metric="cosine")
            return [(m.id, m.score) for m in matches]
        res = self._pc_index.query(
            vector=query_vec.tolist(), top_k=top_k, include_metadata=False
        )
        return [(m["id"], m["score"]) for m in res["matches"]]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked ID lists. Each list is ordered best-first.
    Returns (id, fused_score) sorted best-first.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):  # rank is 1-based
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def run_one_query(query, df, ids, titles, corpus_tokens, retriever, top_k, pool):
    """Run a single query through BM25, vector, and hybrid, then print all three."""
    bm25_list = bm25_rank(corpus_tokens, ids, query, pool)
    vec_list = retriever.rank(query, pool)

    bm25_ids = [i for i, _ in bm25_list]
    vec_ids = [i for i, _ in vec_list]
    fused = reciprocal_rank_fusion([bm25_ids, vec_ids], k=RRF_K)

    print(f"\n########## QUERY: {query!r} ##########")

    def show(name, ranked):
        print(f"\n--- {name} (top {top_k}) ---")
        for rank, item in enumerate(ranked[:top_k], 1):
            doc_id = item[0] if isinstance(item, tuple) else item
            print(f"  {rank}. {titles.get(doc_id, doc_id)}")

    show("BM25 (lexical only)", bm25_list)
    show("Vector (semantic only)", vec_list)
    show("Hybrid (RRF k=60)", fused)

    # ---- Evidence that hybrid recovers what each method misses ---------- #
    top_bm25 = set(bm25_ids[:top_k])
    top_vec = set(vec_ids[:top_k])
    top_hybrid = {doc_id for doc_id, _ in fused[:top_k]}

    only_bm25 = top_bm25 - top_vec
    only_vec = top_vec - top_bm25
    print("\n--- Complementarity analysis ---")
    print(f"  In BM25 top-{top_k} but NOT vector: {len(only_bm25)} docs")
    print(f"  In vector top-{top_k} but NOT BM25: {len(only_vec)} docs")
    recovered = top_hybrid & (only_bm25 | only_vec)
    print(
        f"  Hybrid top-{top_k} includes {len(recovered)} doc(s) that ONE method "
        f"alone would have missed -> this is the RRF win."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--query", action="append", default=None, help="Test query (repeatable; defaults to 3 built-in queries).")
    parser.add_argument("--input", default=str(SUBSET_PARQUET))
    parser.add_argument("--top-k", type=int, default=5, help="Results to display.")
    parser.add_argument("--pool", type=int, default=50, help="Candidates pulled from each retriever before fusion.")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES

    df = pd.read_parquet(args.input)
    ids = df["id"].astype(str).tolist()
    titles = dict(zip(ids, df["title"].astype(str)))

    # BM25 corpus = title + abstract.
    corpus_tokens = [tokenize(f"{t} {a}") for t, a in zip(df["title"], df["abstract"])]

    # Build the dense retriever once and reuse it for every query.
    retriever = VectorRetriever()

    for query in tqdm(queries, desc="Queries", unit="query"):
        run_one_query(query, df, ids, titles, corpus_tokens, retriever, args.top_k, args.pool)

    return 0


if __name__ == "__main__":
    sys.exit(main())
