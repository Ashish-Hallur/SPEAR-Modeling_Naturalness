#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from llm_outputs_common import COMMON_FIELDNAMES, build_merge_key


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=Path, required=True)
    ap.add_argument("--vox_csv", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    return ap.parse_args()


def load_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["merge_key"] = build_merge_key(row)
            rows.append(row)
    assert rows, f"No rows loaded from {path}"
    return rows, reader.fieldnames or []


def index_by_merge_key(rows, name):
    out = {}
    duplicates = []
    for row in rows:
        key = row["merge_key"]
        if key in out:
            duplicates.append(key)
        out[key] = row
    assert not duplicates, f"{name} merge_key is not unique; examples: {duplicates[:10]}"
    return out


def main():
    args = parse_args()
    metrics_rows, metrics_header = load_rows(args.metrics_csv)
    vox_rows, vox_header = load_rows(args.vox_csv)

    metrics_by_key = index_by_merge_key(metrics_rows, "metrics")
    vox_by_key = index_by_merge_key(vox_rows, "vox")

    metrics_keys = set(metrics_by_key)
    vox_keys = set(vox_by_key)
    shared_keys = sorted(metrics_keys & vox_keys)
    assert shared_keys, "No intersecting merge keys between metrics and vox"

    print(f"Loaded metrics rows: {len(metrics_rows):,}")
    print(f"Loaded vox rows:     {len(vox_rows):,}")
    print(f"Rows only in metrics: {len(metrics_keys - vox_keys):,}")
    print(f"Rows only in vox:     {len(vox_keys - metrics_keys):,}")
    print(f"Intersected rows:     {len(shared_keys):,}")

    shared_header_cols = set(metrics_header) & set(vox_header)
    for col in sorted(set(COMMON_FIELDNAMES) & shared_header_cols):
        for key in shared_keys:
            left = str(metrics_by_key[key].get(col, ""))
            right = str(vox_by_key[key].get(col, ""))
            assert left == right, f"Mismatch in shared column {col!r} for merge_key {key!r}: {left!r} != {right!r}"

    vox_only_cols = [col for col in vox_header if col not in metrics_header]
    output_header = list(metrics_header) + ["merge_key"] + vox_only_cols

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_header)
        writer.writeheader()
        for key in shared_keys:
            metrics_row = metrics_by_key[key]
            vox_row = vox_by_key[key]
            out = {col: metrics_row.get(col, "") for col in metrics_header}
            out["merge_key"] = key
            for col in vox_only_cols:
                out[col] = vox_row.get(col, "")
            writer.writerow(out)

    print(f"Wrote merged dataset: {args.output_csv}")
    print(f"Final shape: ({len(shared_keys)}, {len(output_header)})")


if __name__ == "__main__":
    main()
