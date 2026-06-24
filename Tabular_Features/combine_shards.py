#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    return ap.parse_args()


def shard_sort_key(path: Path):
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return stem


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(input_dir.glob(args.pattern), key=shard_sort_key)
    assert shards, f"No shard files matched {input_dir / args.pattern}"

    expected_header = None
    rows_written = 0

    with open(output_csv, "w", newline="") as out_f:
        writer = None
        for shard in shards:
            with open(shard, "r", newline="") as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise AssertionError(f"Empty shard file: {shard}")

                if expected_header is None:
                    expected_header = header
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                else:
                    assert header == expected_header, f"Header mismatch in {shard}"

                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    print(f"Combined {len(shards):,} shards into {output_csv}")
    print(f"Rows written: {rows_written:,}")


if __name__ == "__main__":
    main()
