#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--vox_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    return ap.parse_args()


def build_merge_key(df: pd.DataFrame) -> pd.Series:
    required = [
        "session_id",
        "interaction_id",
        "dyad_id",
        "prompt_participant_id",
        "response_participant_id",
        "turn_index",
        "prompt_start_s",
        "prompt_end_s",
        "response_start_s",
        "response_end_s",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing key columns: {missing}"

    # normalize numeric formatting so string keys line up
    tmp = df.copy()
    for col in ["prompt_start_s", "prompt_end_s", "response_start_s", "response_end_s"]:
        tmp[col] = tmp[col].astype(float).round(6)

    key = (
        tmp["session_id"].astype(str)
        + "|"
        + tmp["interaction_id"].astype(str)
        + "|"
        + tmp["dyad_id"].astype(str)
        + "|"
        + tmp["prompt_participant_id"].astype(str)
        + "|"
        + tmp["response_participant_id"].astype(str)
        + "|"
        + tmp["turn_index"].astype(str)
        + "|"
        + tmp["prompt_start_s"].astype(str)
        + "|"
        + tmp["prompt_end_s"].astype(str)
        + "|"
        + tmp["response_start_s"].astype(str)
        + "|"
        + tmp["response_end_s"].astype(str)
    )
    return key


def report_duplicates(df: pd.DataFrame, name: str, key_col: str):
    dup_mask = df[key_col].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = int(df.loc[dup_mask, key_col].nunique())
    print(f"{name}: duplicate rows on {key_col} = {n_dup_rows:,}, duplicate keys = {n_dup_keys:,}")
    if n_dup_rows > 0:
        print(df.loc[dup_mask, [key_col]].head(10))


def main():
    args = parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    vox = pd.read_csv(args.vox_csv)

    print(f"Loaded metrics: {metrics.shape}")
    print(f"Loaded vox:     {vox.shape}")

    metrics["merge_key"] = build_merge_key(metrics)
    vox["merge_key"] = build_merge_key(vox)

    report_duplicates(metrics, "metrics", "pair_id")
    report_duplicates(vox, "vox", "pair_id")

    assert metrics["merge_key"].is_unique, "metrics merge_key is not unique"
    assert vox["merge_key"].is_unique, "vox merge_key is not unique"

    metrics_keys = set(metrics["merge_key"])
    vox_keys = set(vox["merge_key"])

    only_metrics = metrics_keys - vox_keys
    only_vox = vox_keys - metrics_keys

    print(f"Rows only in metrics: {len(only_metrics):,}")
    print(f"Rows only in vox:     {len(only_vox):,}")

    # keep intersection only
    metrics_i = metrics[metrics["merge_key"].isin(vox_keys)].copy()
    vox_i = vox[vox["merge_key"].isin(metrics_keys)].copy()

    print(f"Intersected metrics: {metrics_i.shape}")
    print(f"Intersected vox:     {vox_i.shape}")

    overlap_cols = set(metrics_i.columns).intersection(set(vox_i.columns))
    overlap_cols.remove("merge_key")

    # columns that should match exactly if present in both
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
        "status_pair",
        "source_dataset_split",
        "split",
        "split_seed",
        "split_unit",
    }

    shared_to_check = sorted(expected_same.intersection(overlap_cols))

    # align by merge_key before checking equality
    metrics_i = metrics_i.sort_values("merge_key").reset_index(drop=True)
    vox_i = vox_i.sort_values("merge_key").reset_index(drop=True)

    for col in shared_to_check:
        left = metrics_i[col].fillna("__NA__").astype(str)
        right = vox_i[col].fillna("__NA__").astype(str)
        same = left.equals(right)
        assert same, f"Mismatch in shared column: {col}"

    vox_only = [c for c in vox_i.columns if c not in metrics_i.columns and c != "merge_key"]

    final_df = pd.merge(
        metrics_i,
        vox_i[["merge_key"] + vox_only],
        on="merge_key",
        how="inner",
        validate="one_to_one",
    )

    # replace weak pair_id with a globally unique one
    final_df["pair_id_old"] = final_df["pair_id"]
    final_df["pair_id"] = final_df["merge_key"]

    # put pair_id first
    cols = final_df.columns.tolist()
    cols.remove("pair_id")
    final_df = final_df[["pair_id"] + cols]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print(f"Wrote merged dataset: {args.output_csv}")
    print(f"Final shape: {final_df.shape}")


if __name__ == "__main__":
    main()