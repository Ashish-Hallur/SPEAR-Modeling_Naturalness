#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import parselmouth
from tqdm import tqdm

from llm_outputs_common import COMMON_FIELDNAMES, PAUSE_THRESHOLD_SEC, public_row, selected_rows_from_csv


F0_FLOOR = 75
F0_CEILING = 500
MIN_VOICED_RATIO = 0.05
FRAME_SEC = 0.025
HOP_SEC = 0.010
EPS = 1e-6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=Path, default=Path("tmp/llm_candidate_manifest.csv"))
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def robust_trimmed_stats(voiced, low_p=10, high_p=90):
    lo = np.percentile(voiced, low_p)
    hi = np.percentile(voiced, high_p)
    trimmed = voiced[(voiced >= lo) & (voiced <= hi)]
    if len(trimmed) == 0:
        return None
    return {
        "f0_mean": float(np.mean(trimmed)),
        "f0_sd": float(np.std(trimmed)),
        "f0_range": float(hi - lo),
    }


def sound_segment(sound_cache, wav_path, start_s, end_s):
    wav_path = str(wav_path)
    if wav_path not in sound_cache:
        sound_cache[wav_path] = parselmouth.Sound(wav_path)
    assert end_s > start_s, f"Invalid segment bounds: {wav_path} {start_s}-{end_s}"
    return sound_cache[wav_path].extract_part(from_time=start_s, to_time=end_s)


def compute_segment_f0(sound_cache, wav_path, start_s, end_s):
    try:
        seg = sound_segment(sound_cache, wav_path, start_s, end_s)
    except (AssertionError, parselmouth.PraatError):
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": np.nan,
            "status_prosody": "SEGMENT_INVALID_FOR_PITCH",
        }

    if seg.get_total_duration() < 0.05:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": np.nan,
            "status_prosody": "SEGMENT_TOO_SHORT_FOR_PITCH",
        }

    try:
        pitch = seg.to_pitch_ac(pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEILING)
    except parselmouth.PraatError:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": np.nan,
            "status_prosody": "PITCH_ANALYSIS_FAILED",
        }
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    total_frames = len(f0)

    if total_frames == 0:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": 0.0,
            "status_prosody": "NO_PITCH_FRAMES",
        }

    voiced_ratio = len(voiced) / total_frames
    if len(voiced) == 0:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": voiced_ratio,
            "status_prosody": "NO_VOICED_FRAMES",
        }

    out = robust_trimmed_stats(voiced, 10, 90)
    if out is None:
        return {
            "f0_mean": np.nan,
            "f0_sd": np.nan,
            "f0_range": np.nan,
            "voiced_ratio": voiced_ratio,
            "status_prosody": "EMPTY_TRIMMED_F0",
        }
    out["voiced_ratio"] = voiced_ratio
    out["status_prosody"] = "LOW_VOICED_RATIO" if voiced_ratio < MIN_VOICED_RATIO else "OK"
    return out


def waveform_from_segment(seg):
    values = np.asarray(seg.values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=0)
    return values.astype(np.float32, copy=False), int(round(seg.sampling_frequency))


def speech_segments_from_energy(y, sr):
    if y.size == 0:
        return []
    frame = max(1, int(round(FRAME_SEC * sr)))
    hop = max(1, int(round(HOP_SEC * sr)))
    if y.size < frame:
        rms = np.sqrt(np.mean(y ** 2) + EPS)
        return [(0.0, y.size / sr)] if rms > EPS else []

    rms = []
    centers = []
    for start in range(0, y.size - frame + 1, hop):
        chunk = y[start:start + frame]
        rms.append(float(np.sqrt(np.mean(chunk ** 2) + EPS)))
        centers.append((start + frame / 2) / sr)

    rms = np.asarray(rms, dtype=np.float32)
    threshold = max(float(np.percentile(rms, 65)) * 0.35, float(np.max(rms)) * 0.05, 1e-4)
    speech = rms >= threshold
    segments = []
    open_start = None
    for is_speech, center in zip(speech, centers):
        if is_speech and open_start is None:
            open_start = max(0.0, center - FRAME_SEC / 2)
        elif not is_speech and open_start is not None:
            segments.append((open_start, min(y.size / sr, center + FRAME_SEC / 2)))
            open_start = None
    if open_start is not None:
        segments.append((open_start, y.size / sr))
    return merge_short_gaps(segments, PAUSE_THRESHOLD_SEC)


def merge_short_gaps(segments, gap_s):
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def compute_energy_temporal(sound_cache, wav_path, start_s, end_s, word_count):
    total_dur = end_s - start_s
    assert total_dur > 0, total_dur
    seg = sound_segment(sound_cache, wav_path, start_s, end_s)
    y, sr = waveform_from_segment(seg)
    speech_segments = speech_segments_from_energy(y, sr)
    speech_sec = float(sum(end - start for start, end in speech_segments))

    if not speech_segments:
        return {
            "speech_rate_local": 60.0 * word_count / total_dur if word_count > 0 else np.nan,
            "articulation_rate_local": np.nan,
            "pause_ratio_local": np.nan,
            "mean_pause_duration_local": np.nan,
            "vad_speech_sec": 0.0,
            "vad_pause_sec": np.nan,
            "status_temporal": "NO_SPEECH_FRAMES",
        }

    pauses = []
    for i in range(1, len(speech_segments)):
        gap = speech_segments[i][0] - speech_segments[i - 1][1]
        if gap >= PAUSE_THRESHOLD_SEC:
            pauses.append(gap)
    pause_total = float(np.sum(pauses)) if pauses else 0.0
    pause_mean = float(np.mean(pauses)) if pauses else 0.0

    status = "OK" if word_count >= 2 else "TOO_FEW_WORDS_FOR_TEMPORAL"
    return {
        "speech_rate_local": 60.0 * word_count / total_dur if word_count > 0 else np.nan,
        "articulation_rate_local": 60.0 * word_count / speech_sec if word_count > 0 and speech_sec > 0 else np.nan,
        "pause_ratio_local": pause_total / total_dur,
        "mean_pause_duration_local": pause_mean,
        "vad_speech_sec": speech_sec,
        "vad_pause_sec": pause_total,
        "status_temporal": status,
    }


def add_metric_features(row, sound_cache):
    prompt_f0 = compute_segment_f0(
        sound_cache,
        row["source_wav_path_prompt"],
        float(row["prompt_start_s"]),
        float(row["prompt_end_s"]),
    )
    response_f0 = compute_segment_f0(
        sound_cache,
        row["source_wav_path_response"],
        float(row["response_start_s"]),
        float(row["response_end_s"]),
    )
    prompt_temp = compute_energy_temporal(
        sound_cache,
        row["source_wav_path_prompt"],
        float(row["prompt_start_s"]),
        float(row["prompt_end_s"]),
        int(row["prompt_word_count"] or 0),
    )
    response_temp = compute_energy_temporal(
        sound_cache,
        row["source_wav_path_response"],
        float(row["response_start_s"]),
        float(row["response_end_s"]),
        int(row["response_word_count"] or 0),
    )

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
            "prompt_vad_speech_sec": prompt_temp["vad_speech_sec"],
            "prompt_vad_pause_sec": prompt_temp["vad_pause_sec"],
            "response_speech_rate_local": response_temp["speech_rate_local"],
            "response_articulation_rate_local": response_temp["articulation_rate_local"],
            "response_pause_ratio_local": response_temp["pause_ratio_local"],
            "response_mean_pause_duration_local": response_temp["mean_pause_duration_local"],
            "response_vad_speech_sec": response_temp["vad_speech_sec"],
            "response_vad_pause_sec": response_temp["vad_pause_sec"],
            "status_prompt_prosody": prompt_f0["status_prosody"],
            "status_response_prosody": response_f0["status_prosody"],
            "status_prompt_temporal": prompt_temp["status_temporal"],
            "status_response_temporal": response_temp["status_temporal"],
        }
    )
    return out


def main():
    args = parse_args()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = COMMON_FIELDNAMES + [
        "prompt_f0_mean", "prompt_f0_sd", "prompt_f0_range", "prompt_voiced_ratio",
        "response_f0_mean", "response_f0_sd", "response_f0_range", "response_voiced_ratio",
        "prompt_speech_rate_local", "prompt_articulation_rate_local",
        "prompt_pause_ratio_local", "prompt_mean_pause_duration_local",
        "prompt_vad_speech_sec", "prompt_vad_pause_sec",
        "response_speech_rate_local", "response_articulation_rate_local",
        "response_pause_ratio_local", "response_mean_pause_duration_local",
        "response_vad_speech_sec", "response_vad_pause_sec",
        "status_prompt_prosody", "status_response_prosody",
        "status_prompt_temporal", "status_response_temporal",
    ]

    rows = selected_rows_from_csv(args.input_csv, args.shard_idx, args.num_shards, args.limit)
    sound_cache = {}

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"llm metrics shard {args.shard_idx}/{args.num_shards}"):
            writer.writerow(add_metric_features(row, sound_cache))


if __name__ == "__main__":
    main()
