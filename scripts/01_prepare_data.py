#!/usr/bin/env python3
"""
01_prepare_data.py
==================

Step 1 of the semantic-search pipeline.

Read the arXiv metadata dataset (JSONL), clean it, keep a 5k-10k subset, and
save a tidy Parquet file with a fixed schema:

    id, title, abstract, authors, year, category

DATASET
-------
The full dataset is the "arXiv Dataset" on Kaggle:

    https://www.kaggle.com/datasets/Cornell-University/arxiv

Download it (it is a single large JSONL file, usually named
``arxiv-metadata-oai-snapshot.json``, ~4 GB) and point this script at it with
``--input``. Because that file is huge and cannot ship inside a homework repo,
this script FALLS BACK to a small synthetic sample (``data/sample.jsonl``,
~40 fake-but-realistic records) so the whole pipeline can be smoke-tested
offline. The synthetic records share the exact same JSON schema as the real
Kaggle dump, so nothing downstream changes.

USAGE
-----
    # Offline smoke test (uses data/sample.jsonl automatically):
    python scripts/01_prepare_data.py

    # Real run against the Kaggle dump:
    python scripts/01_prepare_data.py \
        --input data/arxiv-metadata-oai-snapshot.json \
        --max-records 8000

Output: data/arxiv_subset.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths & logging
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_REAL_INPUT = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
SAMPLE_INPUT = DATA_DIR / "sample.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "arxiv_subset.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("01_prepare_data")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean_text(text: str | None) -> str:
    """Collapse whitespace/newlines and strip - arXiv abstracts are wrapped."""
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _parse_year(record: dict) -> int | None:
    """
    Extract a publication year.

    The synthetic sample carries an explicit ``year`` field. The real Kaggle
    dump does not; it has ``versions`` (with timestamps) and ``update_date``.
    We try, in order: explicit year -> first version timestamp -> update_date.
    """
    if record.get("year"):
        try:
            return int(record["year"])
        except (TypeError, ValueError):
            pass

    versions = record.get("versions") or []
    if versions:
        # e.g. "Mon, 2 Apr 2007 19:18:42 GMT"
        created = versions[0].get("created", "")
        for token in created.replace(",", "").split():
            if token.isdigit() and len(token) == 4:
                return int(token)

    update_date = record.get("update_date", "")  # e.g. "2008-11-26"
    if update_date[:4].isdigit():
        return int(update_date[:4])

    return None


def _normalize_authors(record: dict) -> str:
    """
    Return a single human-readable authors string.

    The real dump may give ``authors`` (a string) or ``authors_parsed`` (a list
    of [last, first, suffix]). The sample uses a plain ``authors`` string.
    """
    if record.get("authors"):
        return _clean_text(record["authors"])

    parsed = record.get("authors_parsed") or []
    names = []
    for parts in parsed:
        last = parts[0] if len(parts) > 0 else ""
        first = parts[1] if len(parts) > 1 else ""
        names.append(_clean_text(f"{first} {last}"))
    return ", ".join(n for n in names if n)


def _primary_category(record: dict) -> str:
    """arXiv 'categories' is a space-separated list; keep the first one."""
    cats = record.get("category") or record.get("categories") or ""
    cats = _clean_text(cats)
    return cats.split()[0] if cats else ""


def iter_records(path: Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file, skipping malformed lines."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON on line %d", line_no)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def build_subset(
    input_path: Path,
    max_records: int,
    min_abstract_chars: int,
    categories: list[str] | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    seen_ids: set[str] = set()

    # tqdm gives a live progress bar; total is unknown for a streamed JSONL, so
    # we drive it by the number of accepted rows toward max_records.
    progress = tqdm(total=max_records, desc="Selecting records", unit="rec")

    for record in iter_records(input_path):
        arxiv_id = _clean_text(record.get("id"))
        title = _clean_text(record.get("title"))
        abstract = _clean_text(record.get("abstract"))
        category = _primary_category(record)

        # ---- cleaning / filtering rules ---------------------------------- #
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        if not title or len(abstract) < min_abstract_chars:
            continue
        if categories and category not in categories:
            continue

        rows.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": _normalize_authors(record),
                "year": _parse_year(record),
                "category": category,
            }
        )
        seen_ids.add(arxiv_id)
        progress.update(1)

        if len(rows) >= max_records:
            log.info("Reached max_records=%d, stopping early.", max_records)
            break

    progress.close()

    df = pd.DataFrame(rows, columns=["id", "title", "abstract", "authors", "year", "category"])
    # Year may be missing for a few records; fill with a sentinel and cast.
    df["year"] = df["year"].fillna(0).astype(int)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to arXiv JSONL. Defaults to the real Kaggle dump if present, "
        "otherwise falls back to data/sample.jsonl.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path.")
    parser.add_argument("--max-records", type=int, default=8000, help="Subset size (target 5k-10k).")
    parser.add_argument("--min-abstract-chars", type=int, default=80, help="Drop too-short abstracts.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional whitelist of primary categories (e.g. cs.LG cs.IR hep-th).",
    )
    args = parser.parse_args()

    # ---- resolve input with graceful fallback ---------------------------- #
    if args.input is not None:
        input_path = args.input
    elif DEFAULT_REAL_INPUT.exists():
        input_path = DEFAULT_REAL_INPUT
        log.info("Using real Kaggle dump: %s", input_path)
    else:
        input_path = SAMPLE_INPUT
        log.warning(
            "Real arXiv dump not found. Falling back to synthetic sample: %s\n"
            "    To use real data, download from "
            "https://www.kaggle.com/datasets/Cornell-University/arxiv "
            "and pass --input <path>.",
            input_path,
        )

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        return 1

    log.info("Reading from %s ...", input_path)
    df = build_subset(
        input_path=input_path,
        max_records=args.max_records,
        min_abstract_chars=args.min_abstract_chars,
        categories=args.categories,
    )

    if df.empty:
        log.error("No records survived cleaning. Check filters / input file.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    log.info("Wrote %d records -> %s", len(df), args.output)

    # ---- required summary output ---------------------------------------- #
    print("\n========== DATASET SUMMARY ==========")
    print(f"Total records kept: {len(df)}")

    print("\nTop-10 category distribution:")
    for cat, count in df["category"].value_counts().head(10).items():
        print(f"  {cat:<24} {count:>6}")

    print("\nYear distribution:")
    year_counts = df.loc[df["year"] > 0, "year"].value_counts().sort_index()
    if year_counts.empty:
        print("  (no parseable years)")
    else:
        for year, count in year_counts.items():
            print(f"  {int(year):<6} {count:>6}")
        print(
            f"  Year range: {int(year_counts.index.min())} - "
            f"{int(year_counts.index.max())}"
        )

    print("\nSample record:")
    sample = df.iloc[0].to_dict()
    for key, value in sample.items():
        shown = str(value)
        if len(shown) > 200:
            shown = shown[:200] + " ..."
        print(f"  {key:<10}: {shown}")
    print("=====================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
