#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from emotivoice_pair_common import build_merge_key


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=Path, required=True)
    ap.add_argument("--vox_csv", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    return ap.parse_args()


def load_rows(path: Path):
    with open(path, "r", newline="", encoding="utf-8") as f:
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

    expected_same = {
        "pair_id",
        "session_id",
        "interaction_id",
        "dyad_id",
        "prompt_participant_id",
        "response_participant_id",
        "turn_index",
        "source_wav_path_prompt",
        "source_json_path_prompt",
        "source_wav_path_response",
        "source_json_path_response",
        "prompt_start_s",
        "prompt_end_s",
        "prompt_duration_s",
        "response_start_s",
        "response_end_s",
        "response_duration_s",
        "latency_s",
        "overlap_s",
        "prompt_transcript",
        "response_transcript",
        "prompt_word_count",
        "response_word_count",
        "relationship",
        "relationship_detail",
        "status_relationship",
        "prompt_tts_model",
        "response_tts_model",
        "prompt_is_emotivoice",
        "response_is_emotivoice",
        "is_human_prompt_emotivoice_response",
        "emotivoice_row_dir",
        "prompt_track",
        "response_track",
        "status_pair",
        "source_dataset_split",
        "split",
        "split_seed",
        "split_unit",
    }
    shared_header_cols = set(metrics_header) & set(vox_header)
    for col in sorted(expected_same & shared_header_cols):
        for key in shared_keys:
            left = str(metrics_by_key[key].get(col, ""))
            right = str(vox_by_key[key].get(col, ""))
            assert left == right, f"Mismatch in shared column {col!r} for merge_key {key!r}: {left!r} != {right!r}"

    vox_only_cols = [col for col in vox_header if col not in metrics_header]
    output_header = ["pair_id"]
    for col in metrics_header:
        if col != "pair_id":
            output_header.append(col)
    output_header.append("merge_key")
    output_header.extend(vox_only_cols)
    output_header.append("pair_id_old")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_header)
        writer.writeheader()
        for key in shared_keys:
            metrics_row = metrics_by_key[key]
            vox_row = vox_by_key[key]
            out = {}
            out["pair_id"] = key
            for col in metrics_header:
                if col != "pair_id":
                    out[col] = metrics_row.get(col, "")
            out["merge_key"] = key
            for col in vox_only_cols:
                out[col] = vox_row.get(col, "")
            out["pair_id_old"] = metrics_row["pair_id"]
            writer.writerow(out)

    print(f"Wrote merged dataset: {args.output_csv}")
    print(f"Final shape: ({len(shared_keys)}, {len(output_header)})")


if __name__ == "__main__":
    main()
