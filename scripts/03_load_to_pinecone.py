#!/usr/bin/env python3
"""
03_load_to_pinecone.py
======================

Step 3 of the pipeline.

Create a Pinecone serverless index with the correct dimension and the cosine
metric, then upsert all embeddings together with their metadata
(title, authors, year, category) in batches.

LOCAL FALLBACK
--------------
If ``USE_LOCAL=1`` (read from .env), this script does NOT touch Pinecone at all.
Instead it verifies that the saved embeddings + metadata can be loaded into the
in-memory ``LocalCosineIndex`` (see common.py). That same index is what
04/06 use when USE_LOCAL=1, so this step becomes a quick consistency check and
lets a student run the entire pipeline with no Pinecone account.

Set USE_LOCAL=0 and provide PINECONE_API_KEY to use the real service.

USAGE
-----
    python scripts/03_load_to_pinecone.py
    python scripts/03_load_to_pinecone.py --batch-size 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

from common import (
    ROOT,
    build_local_index_from_artifacts,
    load_embeddings,
    load_id_map,
    load_metadata,
    load_model_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("03_load_to_pinecone")

load_dotenv(ROOT / ".env")


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_local() -> int:
    """USE_LOCAL path: build & sanity-check the in-memory index."""
    log.info("USE_LOCAL=1 -> using in-memory cosine index (no Pinecone).")
    index, id_map, meta = build_local_index_from_artifacts()
    log.info("Local index built with %d vectors (dim=%d).", len(index), index.dim)

    # Smoke query: search with the first stored vector; it should retrieve itself.
    import numpy as np

    vectors = load_embeddings()
    probe = vectors[0]
    results = index.query(probe, top_k=3, metric="cosine")
    log.info("Sanity query top-3 ids: %s", [r.id for r in results])
    if results and results[0].id == id_map[0]:
        log.info("OK: nearest neighbour of vector[0] is itself (score=%.4f).", results[0].score)
    return 0


def load_pinecone(batch_size: int) -> int:
    """Real Pinecone path."""
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "your-pinecone-api-key-here":
        log.error("PINECONE_API_KEY missing/placeholder. Set it in .env or use USE_LOCAL=1.")
        return 1

    index_name = os.getenv("PINECONE_INDEX", "arxiv-semantic-search")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    model_info = load_model_info()
    dim = int(model_info["dim"])
    metric = model_info.get("metric", "cosine")

    pc = Pinecone(api_key=api_key)

    existing = [ix["name"] for ix in pc.list_indexes()]
    if index_name not in existing:
        log.info("Creating index '%s' (dim=%d, metric=%s)...", index_name, dim, metric)
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        log.info("Index '%s' already exists; reusing.", index_name)

    index = pc.Index(index_name)

    vectors = load_embeddings()
    id_map = load_id_map()
    meta = load_metadata()
    log.info("Upserting %d vectors in batches of %d ...", len(id_map), batch_size)

    items = []
    for i, _id in enumerate(id_map):
        m = meta.get(_id, {})
        items.append(
            {
                "id": _id,
                "values": vectors[i].tolist(),
                "metadata": {
                    "title": m.get("title", ""),
                    "authors": m.get("authors", ""),
                    "year": int(m.get("year", 0)),
                    "category": m.get("category", ""),
                },
            }
        )

    total = 0
    n_batches = (len(items) + batch_size - 1) // batch_size
    for batch in tqdm(
        _chunks(items, batch_size),
        total=n_batches,
        desc="Upserting to Pinecone",
        unit="batch",
    ):
        index.upsert(vectors=batch)
        total += len(batch)

    stats = index.describe_index_stats()
    log.info("Done. Index stats: %s", stats)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=200, help="Upsert batch size (Pinecone).")
    args = parser.parse_args()

    use_local = os.getenv("USE_LOCAL", "1") == "1"
    if use_local:
        return load_local()
    return load_pinecone(args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
