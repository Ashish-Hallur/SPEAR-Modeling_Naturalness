#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from llm_outputs_common import COMMON_FIELDNAMES, DEFAULT_DATA_ROOT, build_candidate_rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output_csv", type=Path, default=Path("tmp/llm_candidate_manifest.csv"))
    ap.add_argument("--status_csv", type=Path, default=Path("tmp/llm_candidate_status.csv"))
    return ap.parse_args()


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_counts(rows: list[dict], label: str):
    print(f"{label}: {len(rows):,}")
    for key, value in sorted(Counter((r.get("model_name", ""), r.get("split", ""), r.get("source_style", "")) for r in rows).items()):
        model, split, style = key
        print(f"  {model}/{split}/{style}: {value:,}")


def main():
    args = parse_args()
    rows, skipped = build_candidate_rows(args.data_root)

    pair_ids = [row["pair_id"] for row in rows]
    assert len(pair_ids) == len(set(pair_ids)), "Usable candidate pair_id values are not unique"

    write_rows(args.output_csv, rows, COMMON_FIELDNAMES)
    status_fields = [
        "pair_id",
        "model_name",
        "split",
        "source_style",
        "prompt_stem",
        "metadata_csv",
        "audio_path",
        "answer_audio_path",
        "status_pair",
    ]
    write_rows(args.status_csv, skipped, status_fields)

    print_counts(rows, "usable candidates")
    print_counts(skipped, "skipped candidates")
    print(f"Wrote manifest: {args.output_csv}")
    print(f"Wrote status manifest: {args.status_csv}")


if __name__ == "__main__":
    main()
