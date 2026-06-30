#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--expected_shards", type=int, default=0)
    ap.add_argument("--allow_partial", action="store_true")
    return ap.parse_args()


def shard_index(path: Path) -> int:
    matches = re.findall(r"shard_(\d+)\.csv$", path.name)
    assert matches, f"Could not parse shard index from {path.name}"
    return int(matches[-1])


def main():
    args = parse_args()
    shard_files = sorted(args.shards_dir.glob("*.csv"), key=shard_index)
    assert shard_files, f"No shard CSVs found in {args.shards_dir}"

    if args.expected_shards and not args.allow_partial:
        found = {shard_index(path) for path in shard_files}
        expected = set(range(args.expected_shards))
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        assert not missing, f"Missing {len(missing)} shard CSVs in {args.shards_dir}: {missing[:20]}"
        assert not extra, f"Found shard indices outside expected range in {args.shards_dir}: {extra[:20]}"

    with open(shard_files[0], "r", newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames
    assert fieldnames is not None, "Could not read header from first shard"
    assert "merge_key" in fieldnames, "Expected merge_key column in embedding shards"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for shard in shard_files:
            with open(shard, "r", newline="", encoding="utf-8") as fin:
                for row in csv.DictReader(fin):
                    merge_key = row["merge_key"]
                    assert merge_key, f"Empty merge_key in {shard}"
                    if merge_key in seen:
                        raise ValueError(f"Duplicate merge_key across shards: {merge_key}")
                    seen.add(merge_key)
                    writer.writerow(row)

    print(f"Wrote merged embedding CSV: {args.output_csv}")
    print(f"Rows: {len(seen):,}")


if __name__ == "__main__":
    main()
