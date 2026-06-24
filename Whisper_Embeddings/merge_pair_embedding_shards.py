#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    shard_files = sorted(args.shards_dir.glob("*.csv"))
    assert shard_files, f"No shard CSVs found in {args.shards_dir}"

    with open(shard_files[0], "r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    assert fieldnames is not None, "Could not read header from first shard"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for shard in shard_files:
            with open(shard, "r", newline="", encoding="utf-8") as fin:
                for row in csv.DictReader(fin):
                    pair_id = row["pair_id"]
                    if pair_id in seen:
                        raise ValueError(f"Duplicate pair_id across shards: {pair_id}")
                    seen.add(pair_id)
                    writer.writerow(row)

    print(f"Wrote merged embedding CSV: {args.output_csv}")
    print(f"Rows: {len(seen):,}")


if __name__ == "__main__":
    main()