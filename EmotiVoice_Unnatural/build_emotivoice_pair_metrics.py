#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import parselmouth
from tqdm import tqdm

from emotivoice_pair_common import (
    COMMON_FIELDNAMES,
    DEFAULT_INPUT_ROOT,
    MIN_SEGMENT_SEC,
    MAX_SEGMENT_SEC,
    PAUSE_THRESHOLD_SEC,
    build_pair_candidates_for_row_dir,
    public_row,
    selected_row_dirs,
)


F0_FLOOR = 75
F0_CEILING = 500
MIN_VOICED_RATIO = 0.05
EPS = 1e-6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def compute_local_temporal(words, start_s, end_s):
    if len(words) < 2:
        return {
            "speech_rate_local": np.nan,
            "articulation_rate_local": np.nan,
            "pause_ratio_local": np.nan,
            "mean_pause_duration_local": np.nan,
            "status_temporal": "TOO_FEW_WORDS_FOR_TEMPORAL",
        }

    total_dur = end_s - start_s
    assert (MIN_SEGMENT_SEC - EPS) <= total_dur <= (MAX_SEGMENT_SEC + EPS), total_dur

    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= PAUSE_THRESHOLD_SEC:
            pauses.append(gap)

    pause_total = float(np.sum(pauses)) if pauses else 0.0
    pause_mean = float(np.mean(pauses)) if pauses else 0.0
    speech_active = max(total_dur - pause_total, 0.0)

    word_count = len(words)
    speech_rate_wpm = 60.0 * word_count / total_dur
    articulation_rate_wpm = 60.0 * word_count / speech_active if speech_active > 0 else np.nan
    pause_ratio = pause_total / total_dur

    return {
        "speech_rate_local": speech_rate_wpm,
        "articulation_rate_local": articulation_rate_wpm,
        "pause_ratio_local": pause_ratio,
        "mean_pause_duration_local": pause_mean,
        "status_temporal": "OK",
    }


def robust_trimmed_stats(voiced, low_p=10, high_p=90):
    lo = np.percentile(voiced, low_p)
    hi = np.percentile(voiced, high_p)
    trimmed = voiced[(voiced >= lo) & (voiced <= hi)]
    assert len(trimmed) > 0
    return {
        "f0_mean": float(np.mean(trimmed)),
        "f0_sd": float(np.std(trimmed)),
        "f0_range": float(hi - lo),
    }


def compute_segment_f0(sound_cache, wav_path, start_s, end_s):
    wav_path = str(wav_path)
    if wav_path not in sound_cache:
        sound_cache[wav_path] = parselmouth.Sound(wav_path)

    snd = sound_cache[wav_path]
    seg = snd.extract_part(from_time=start_s, to_time=end_s)
    pitch = seg.to_pitch_ac(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEILING)

    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    total_frames = len(f0)
    voiced_frames = len(voiced)

    if total_frames == 0:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": 0.0,
            "status_prosody": "NO_PITCH_FRAMES",
        }

    voiced_ratio = voiced_frames / total_frames

    if voiced_frames == 0:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": voiced_ratio,
            "status_prosody": "NO_VOICED_FRAMES",
        }

    stats = robust_trimmed_stats(voiced, 10, 90)
    stats["voiced_ratio"] = voiced_ratio
    stats["status_prosody"] = "LOW_VOICED_RATIO" if voiced_ratio < MIN_VOICED_RATIO else "OK"
    return stats


def add_metric_features(row, sound_cache):
    prompt_f0 = compute_segment_f0(
        sound_cache,
        row["source_wav_path_prompt"],
        row["prompt_start_s"],
        row["prompt_end_s"],
    )
    response_f0 = compute_segment_f0(
        sound_cache,
        row["source_wav_path_response"],
        row["response_start_s"],
        row["response_end_s"],
    )
    prompt_temp = compute_local_temporal(row["_prompt_words"], row["prompt_start_s"], row["prompt_end_s"])
    response_temp = compute_local_temporal(row["_response_words"], row["response_start_s"], row["response_end_s"])

    out = public_row(row)
    out.update(
        {
            "prompt_f0_mean": prompt_f0["f0_mean"],
            "prompt_f0_sd": prompt_f0["f0_sd"],
            "prompt_f0_range": prompt_f0["f0_range"],
            "prompt_voiced_ratio": prompt_f0["voiced_ratio"],
            "response_f0_mean": response_f0["f0_mean"],
            "response_f0_sd": response_f0["f0_sd"],
            "response_f0_range": response_f0["f0_range"],
            "response_voiced_ratio": response_f0["voiced_ratio"],
            "prompt_speech_rate_local": prompt_temp["speech_rate_local"],
            "prompt_articulation_rate_local": prompt_temp["articulation_rate_local"],
            "prompt_pause_ratio_local": prompt_temp["pause_ratio_local"],
            "prompt_mean_pause_duration_local": prompt_temp["mean_pause_duration_local"],
            "response_speech_rate_local": response_temp["speech_rate_local"],
            "response_articulation_rate_local": response_temp["articulation_rate_local"],
            "response_pause_ratio_local": response_temp["pause_ratio_local"],
            "response_mean_pause_duration_local": response_temp["mean_pause_duration_local"],
            "status_prompt_prosody": prompt_f0["status_prosody"],
            "status_response_prosody": response_f0["status_prosody"],
            "status_prompt_temporal": prompt_temp["status_temporal"],
            "status_response_temporal": response_temp["status_temporal"],
        }
    )
    return out


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = COMMON_FIELDNAMES + [
        "prompt_f0_mean", "prompt_f0_sd", "prompt_f0_range", "prompt_voiced_ratio",
        "response_f0_mean", "response_f0_sd", "response_f0_range", "response_voiced_ratio",
        "prompt_speech_rate_local", "prompt_articulation_rate_local",
        "prompt_pause_ratio_local", "prompt_mean_pause_duration_local",
        "response_speech_rate_local", "response_articulation_rate_local",
        "response_pause_ratio_local", "response_mean_pause_duration_local",
        "status_prompt_prosody", "status_response_prosody",
        "status_prompt_temporal", "status_response_temporal",
    ]

    row_dirs = selected_row_dirs(input_root, args.shard_idx, args.num_shards, args.limit)
    sound_cache = {}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_dir in tqdm(row_dirs, desc=f"emotivoice metrics shard {args.shard_idx}/{args.num_shards}"):
            for row in build_pair_candidates_for_row_dir(row_dir):
                writer.writerow(add_metric_features(row, sound_cache))


if __name__ == "__main__":
    main()
