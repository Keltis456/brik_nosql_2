#!/usr/bin/env python3
"""
04_search.py
============

Step 4 of the pipeline: query-time semantic search.

Three modes (select with --mode):

  (a) semantic      Pure semantic search: embed the query, return nearest
                    neighbours by cosine similarity.

  (b) filtered      Semantic search + metadata filters: restrict to a year
                    range (--year-min/--year-max) and/or a category
                    (--category). This is the classic "vector search with a
                    structured pre-filter" pattern.

  (c) metric        Run the SAME query under three similarity metrics
                    (cosine, dotproduct, euclidean) and print the rankings side
                    by side. See the discussion block at the bottom for why the
                    rankings agree for L2-normalized SPECTER2 vectors.

Backend: real Pinecone when USE_LOCAL=0, otherwise the in-memory cosine index
(common.LocalCosineIndex). The query vector is embedded with the SAME model
saved by 02_embed (read from model_info.json), so dimensions always match.

USAGE
-----
    python scripts/04_search.py --mode semantic -q "graph neural networks for molecules"
    python scripts/04_search.py --mode filtered -q "retrieval augmented generation" \
        --year-min 2024 --category cs.CL
    python scripts/04_search.py --mode metric   -q "quantum error correction"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv

from common import (
    ROOT,
    build_local_index_from_artifacts,
    load_embedding_model,
    load_model_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("04_search")

load_dotenv(ROOT / ".env")


# --------------------------------------------------------------------------- #
# Filter construction (Pinecone-style; LocalCosineIndex understands the same)
# --------------------------------------------------------------------------- #
def build_filter(year_min: int | None, year_max: int | None, category: str | None) -> dict | None:
    flt: dict = {}
    year_cond: dict = {}
    if year_min is not None:
        year_cond["$gte"] = year_min
    if year_max is not None:
        year_cond["$lte"] = year_max
    if year_cond:
        flt["year"] = year_cond
    if category:
        flt["category"] = {"$eq": category}
    return flt or None


# --------------------------------------------------------------------------- #
# Backend abstraction
# --------------------------------------------------------------------------- #
class Backend:
    """Uniform .search(query_vec, top_k, flt, metric) over Pinecone / local."""

    def __init__(self):
        self.use_local = os.getenv("USE_LOCAL", "1") == "1"
        if self.use_local:
            self.index, _, _ = build_local_index_from_artifacts()
            log.info("Backend: local in-memory index (%d vectors).", len(self.index))
        else:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX", "arxiv-semantic-search")
            self.index = Pinecone(api_key=api_key).Index(index_name)
            log.info("Backend: Pinecone index '%s'.", index_name)

    def search(self, query_vec: np.ndarray, top_k: int, flt: dict | None, metric: str = "cosine"):
        if self.use_local:
            return self.index.query(query_vec, top_k=top_k, flt=flt, metric=metric)
        # Pinecone: metric is fixed at index-creation time (cosine), so the
        # `metric` arg only affects the local index. We still pass the filter.
        res = self.index.query(
            vector=query_vec.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=flt,
        )
        # Normalize to the same shape as LocalCosineIndex matches.
        from types import SimpleNamespace

        return [
            SimpleNamespace(id=m["id"], score=m["score"], metadata=m.get("metadata", {}))
            for m in res["matches"]
        ]


def print_results(title: str, matches) -> None:
    print(f"\n=== {title} ===")
    if not matches:
        print("  (no results)")
        return
    for rank, m in enumerate(matches, start=1):
        md = m.metadata or {}
        print(
            f"  {rank:>2}. score={m.score:+.4f}  [{md.get('category','?')}, {md.get('year','?')}]  "
            f"{md.get('title','?')}"
        )


# Built-in queries used when the script is run with no -q (pipeline demo).
DEFAULT_DEMO_QUERIES = [
    "graph neural networks for molecular property prediction",
    "retrieval augmented generation for question answering",
    "quantum error correction surface code",
]


def _run_semantic(backend, query_vec, query, top_k):
    matches = backend.search(query_vec, top_k=top_k, flt=None, metric="cosine")
    print_results(f"Semantic search: {query!r}", matches)


def _run_filtered(backend, query_vec, query, top_k, flt):
    log.info("Applying metadata filter: %s", flt)
    matches = backend.search(query_vec, top_k=top_k, flt=flt, metric="cosine")
    print_results(f"Semantic + filter ({flt}): {query!r}", matches)


def _run_metric(backend, query_vec, query, top_k):
    # Metric comparison fully applies to the LOCAL index, where we re-score with
    # each metric. For Pinecone the metric is fixed at index creation, so we note
    # that and only run cosine there.
    for metric in ("cosine", "dotproduct", "euclidean"):
        if not backend.use_local and metric != "cosine":
            print(f"\n[skipped {metric}: Pinecone index metric is fixed at creation]")
            continue
        matches = backend.search(query_vec, top_k=top_k, flt=None, metric=metric)
        print_results(f"metric={metric}: {query!r}", matches)


def _print_metric_discussion() -> None:
    print(
        "\n--- Metric discussion (cosine vs dot vs euclidean) ---\n"
        "All vectors are L2-normalized in 02_embed, so they lie on the unit\n"
        "hypersphere and the three metrics induce the SAME ranking:\n"
        "  cosine(a,b)      = a.b           (because |a|=|b|=1)\n"
        "  dotproduct(a,b)  = a.b           (identical to cosine here)\n"
        "  euclidean(a,b)^2 = 2 - 2*(a.b)   (smaller distance <=> larger dot)\n"
        "Scores differ, order is identical. SPECTER2 is trained with a cosine\n"
        "objective, which is why we normalize and use cosine. Without\n"
        "normalization, dotproduct would reward longer vectors and could\n"
        "disagree with cosine."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-q", "--query", default=None,
        help="Natural-language query. If omitted, runs a built-in demo across all three modes.",
    )
    parser.add_argument("--mode", choices=["semantic", "filtered", "metric", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    # Embed queries with the SAME model used at indexing time.
    model_info = load_model_info()
    model = load_embedding_model()
    if model.dim != int(model_info["dim"]):
        log.warning(
            "Loaded model dim=%d but index was built with dim=%d. "
            "Re-run 02_embed if you changed models.",
            model.dim,
            model_info["dim"],
        )

    backend = Backend()

    def embed(q):
        return model.encode([q], normalize=True)[0]

    # ---- demo mode: no query supplied, exercise all three modes ---------- #
    if args.query is None:
        log.info("No -q provided; running built-in demo across all three modes.")
        for q in DEFAULT_DEMO_QUERIES:
            _run_semantic(backend, embed(q), q, args.top_k)
        q_filt = DEFAULT_DEMO_QUERIES[0]
        _run_filtered(backend, embed(q_filt), q_filt, args.top_k, build_filter(2024, None, None))
        q_metric = DEFAULT_DEMO_QUERIES[2]
        _run_metric(backend, embed(q_metric), q_metric, args.top_k)
        _print_metric_discussion()
        return 0

    # ---- single-query mode ----------------------------------------------- #
    query_vec = embed(args.query)
    if args.mode in ("semantic", "all"):
        _run_semantic(backend, query_vec, args.query, args.top_k)
    if args.mode in ("filtered", "all"):
        flt = build_filter(args.year_min, args.year_max, args.category)
        _run_filtered(backend, query_vec, args.query, args.top_k, flt)
    if args.mode in ("metric", "all"):
        _run_metric(backend, query_vec, args.query, args.top_k)
        _print_metric_discussion()

    return 0


if __name__ == "__main__":
    sys.exit(main())
