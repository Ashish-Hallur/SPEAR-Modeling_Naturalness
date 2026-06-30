#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--expected_shards", type=int, default=0)
    ap.add_argument("--allow_partial", action="store_true")
    return ap.parse_args()


def shard_index(path: Path):
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def shard_sort_key(path: Path):
    idx = shard_index(path)
    if idx is not None:
        return idx
    return path.stem


def check_expected_shards(shards, expected_shards, input_dir, pattern):
    if expected_shards <= 0:
        return

    found = {}
    unindexed = []
    for shard in shards:
        idx = shard_index(shard)
        if idx is None:
            unindexed.append(shard.name)
        else:
            found.setdefault(idx, []).append(shard.name)

    assert not unindexed, f"Could not parse shard index from files: {unindexed[:20]}"

    duplicate = {idx: names for idx, names in found.items() if len(names) > 1}
    assert not duplicate, f"Duplicate shard indices under {input_dir}: {duplicate}"

    missing = [idx for idx in range(expected_shards) if idx not in found]
    extra = sorted(idx for idx in found if idx < 0 or idx >= expected_shards)
    assert not missing, (
        f"Expected {expected_shards} shards matching {input_dir / pattern}, "
        f"but found {len(found)}. Missing {len(missing)} shard(s): {missing[:40]}"
    )
    assert not extra, f"Found shard indices outside 0..{expected_shards - 1}: {extra[:40]}"


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(input_dir.glob(args.pattern), key=shard_sort_key)
    assert shards, f"No shard files matched {input_dir / args.pattern}"
    if args.expected_shards > 0 and not args.allow_partial:
        check_expected_shards(shards, args.expected_shards, input_dir, args.pattern)

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
