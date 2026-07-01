#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from llm_outputs_common import COMMON_FIELDNAMES, MAX_CLIP_SEC, MIN_VOX_SEC, public_row, selected_rows_from_csv


TARGET_SR = 16000
DEFAULT_VOX_RELEASE_DIR = Path(__file__).resolve().parents[1] / "vox-profile-release"
EPS = 1e-6


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=Path, default=Path("tmp/llm_candidate_manifest.csv"))
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--vox_release_dir", type=Path, default=DEFAULT_VOX_RELEASE_DIR)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def load_audio_cache(audio_cache, wav_path):
    wav_path = str(wav_path)
    if wav_path not in audio_cache:
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        audio_cache[wav_path] = (wav.squeeze(0), sr)
    return audio_cache[wav_path]


def slice_audio(audio_cache, wav_path, start_s, end_s):
    wav, sr = load_audio_cache(audio_cache, wav_path)
    if wav.numel() == 0:
        return wav.clone(), sr
    start = int(round(float(start_s) * sr))
    end = int(round(float(end_s) * sr))
    assert end > start, f"Invalid slice bounds for {wav_path}: {start_s}, {end_s}"
    assert end <= wav.numel(), f"Slice end out of bounds for {wav_path}: {end} > {wav.numel()}"
    return wav[start:end].clone(), sr


def prepare_audio_for_vox(wav, sr):
    if wav.numel() == 0:
        return wav.float()
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, TARGET_SR).squeeze(0)
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-8)
    wav = wav / max(rms.item(), 1e-4)
    return wav


def speech_only_chunks_from_waveform(wav_16k, vad_model, get_speech_timestamps):
    wav_cpu = wav_16k.detach().cpu()
    speech_ts = get_speech_timestamps(wav_cpu, vad_model, sampling_rate=TARGET_SR)
    if not speech_ts:
        return [], 0.0

    speech_audio = torch.cat([wav_cpu[t["start"]: t["end"]] for t in speech_ts])
    speech_sec = speech_audio.numel() / TARGET_SR

    max_len = int(MAX_CLIP_SEC * TARGET_SR)
    min_len = int(MIN_VOX_SEC * TARGET_SR)
    chunks = []
    for i in range(0, speech_audio.numel(), max_len):
        chunk = speech_audio[i:i + max_len]
        if chunk.numel() >= min_len:
            chunks.append(chunk)
    return chunks, speech_sec


def empty_speaker(status):
    return {
        "age": np.nan,
        "gender": "",
        "gender_confidence": np.nan,
        "num_chunks": 0,
        "speech_sec": 0.0,
        "status": status,
    }


def empty_avd(status):
    return {
        "arousal": np.nan,
        "valence": np.nan,
        "dominance": np.nan,
        "num_chunks": 0,
        "speech_sec": 0.0,
        "status": status,
    }


def check_vox_duration(start_s, end_s):
    duration = float(end_s) - float(start_s)
    if duration < MIN_VOX_SEC - EPS:
        return "TOO_SHORT_FOR_VOX"
    assert duration <= MAX_CLIP_SEC + EPS, duration
    return "OK"


def aggregate_age_gender_for_segment(
    audio_cache,
    wav_path,
    start_s,
    end_s,
    age_sex_model,
    device,
    vad_model,
    get_speech_timestamps,
):
    duration_status = check_vox_duration(start_s, end_s)
    if duration_status != "OK":
        return empty_speaker(duration_status)

    try:
        wav, sr = slice_audio(audio_cache, wav_path, start_s, end_s)
    except AssertionError:
        return empty_speaker("SEGMENT_OUT_OF_BOUNDS")
    except RuntimeError as exc:
        if "Failed to decode audio" not in str(exc):
            raise
        return empty_speaker("AUDIO_DECODE_FAILED")

    wav_16k = prepare_audio_for_vox(wav, sr)
    if wav_16k.numel() == 0:
        return empty_speaker("EMPTY_AUDIO")

    chunks, speech_sec = speech_only_chunks_from_waveform(wav_16k, vad_model, get_speech_timestamps)
    if not chunks:
        out = empty_speaker("NO_SPEECH_AFTER_VAD")
        out["speech_sec"] = speech_sec
        return out

    age_vals = []
    sex_probs = []
    for chunk in chunks:
        x = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            age_out, sex_out = age_sex_model(x)
        age_vals.append(age_out.squeeze().item() * 100.0)
        sex_probs.append(F.softmax(sex_out, dim=1).squeeze(0).cpu().numpy())

    mean_age = float(np.mean(age_vals))
    mean_sex = np.mean(np.stack(sex_probs, axis=0), axis=0)
    gender_idx = int(np.argmax(mean_sex))
    return {
        "age": mean_age,
        "gender": "female" if gender_idx == 0 else "male",
        "gender_confidence": float(np.max(mean_sex)),
        "num_chunks": len(chunks),
        "speech_sec": speech_sec,
        "status": "OK",
    }


def aggregate_avd_for_segment(
    audio_cache,
    wav_path,
    start_s,
    end_s,
    emotion_model,
    device,
    vad_model,
    get_speech_timestamps,
):
    duration_status = check_vox_duration(start_s, end_s)
    if duration_status != "OK":
        return empty_avd(duration_status)

    try:
        wav, sr = slice_audio(audio_cache, wav_path, start_s, end_s)
    except AssertionError:
        return empty_avd("SEGMENT_OUT_OF_BOUNDS")
    except RuntimeError as exc:
        if "Failed to decode audio" not in str(exc):
            raise
        return empty_avd("AUDIO_DECODE_FAILED")

    wav_16k = prepare_audio_for_vox(wav, sr)
    if wav_16k.numel() == 0:
        return empty_avd("EMPTY_AUDIO")

    chunks, speech_sec = speech_only_chunks_from_waveform(wav_16k, vad_model, get_speech_timestamps)
    if not chunks:
        out = empty_avd("NO_SPEECH_AFTER_VAD")
        out["speech_sec"] = speech_sec
        return out

    a_vals, v_vals, d_vals = [], [], []
    for chunk in chunks:
        x = chunk.unsqueeze(0).to(device)
        with torch.no_grad():
            arousal_out, valence_out, dominance_out = emotion_model(x)
        a_vals.append(arousal_out.squeeze().item())
        v_vals.append(valence_out.squeeze().item())
        d_vals.append(dominance_out.squeeze().item())

    return {
        "arousal": float(np.mean(a_vals)),
        "valence": float(np.mean(v_vals)),
        "dominance": float(np.mean(d_vals)),
        "num_chunks": len(chunks),
        "speech_sec": speech_sec,
        "status": "OK",
    }


def speaker_cache_key(row, role):
    return (
        row[f"source_wav_path_{role}"],
        row[f"{role}_start_s"],
        row[f"{role}_end_s"],
    )


def add_vox_features(
    row,
    audio_cache,
    speaker_cache,
    age_sex_model,
    emotion_model,
    device,
    vad_model,
    get_speech_timestamps,
):
    for role in ["prompt", "response"]:
        key = speaker_cache_key(row, role)
        if key not in speaker_cache:
            speaker_cache[key] = aggregate_age_gender_for_segment(
                audio_cache,
                row[f"source_wav_path_{role}"],
                float(row[f"{role}_start_s"]),
                float(row[f"{role}_end_s"]),
                age_sex_model,
                device,
                vad_model,
                get_speech_timestamps,
            )

    prompt_avd = aggregate_avd_for_segment(
        audio_cache,
        row["source_wav_path_prompt"],
        float(row["prompt_start_s"]),
        float(row["prompt_end_s"]),
        emotion_model,
        device,
        vad_model,
        get_speech_timestamps,
    )
    response_avd = aggregate_avd_for_segment(
        audio_cache,
        row["source_wav_path_response"],
        float(row["response_start_s"]),
        float(row["response_end_s"]),
        emotion_model,
        device,
        vad_model,
        get_speech_timestamps,
    )

    prompt_spk = speaker_cache[speaker_cache_key(row, "prompt")]
    response_spk = speaker_cache[speaker_cache_key(row, "response")]

    out = public_row(row)
    out.update(
        {
            "prompt_arousal": prompt_avd["arousal"],
            "prompt_valence": prompt_avd["valence"],
            "prompt_dominance": prompt_avd["dominance"],
            "prompt_age": prompt_spk["age"],
            "prompt_gender": prompt_spk["gender"],
            "prompt_gender_confidence": prompt_spk["gender_confidence"],
            "prompt_vox_num_chunks": prompt_avd["num_chunks"],
            "prompt_vox_support_speech_sec": prompt_avd["speech_sec"],
            "response_arousal": response_avd["arousal"],
            "response_valence": response_avd["valence"],
            "response_dominance": response_avd["dominance"],
            "response_age": response_spk["age"],
            "response_gender": response_spk["gender"],
            "response_gender_confidence": response_spk["gender_confidence"],
            "response_vox_num_chunks": response_avd["num_chunks"],
            "response_vox_support_speech_sec": response_avd["speech_sec"],
            "status_prompt_vox": prompt_avd["status"],
            "status_response_vox": response_avd["status"],
            "status_prompt_speaker_profile": prompt_spk["status"],
            "status_response_speaker_profile": response_spk["status"],
        }
    )
    return out


def main():
    args = parse_args()
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.vox_release_dir.resolve()))
    from src.model.age_sex.wavlm_demographics import WavLMWrapper as AgeSexModel
    from src.model.emotion.wavlm_emotion_dim import WavLMWrapper as EmotionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    age_sex_model = AgeSexModel.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device).eval()
    emotion_model = EmotionModel.from_pretrained("tiantiaf/wavlm-large-msp-podcast-emotion-dim").to(device).eval()

    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = vad_utils[0]
    vad_model = vad_model.cpu().eval()

    fieldnames = COMMON_FIELDNAMES + [
        "prompt_arousal", "prompt_valence", "prompt_dominance",
        "prompt_age", "prompt_gender", "prompt_gender_confidence",
        "prompt_vox_num_chunks", "prompt_vox_support_speech_sec",
        "response_arousal", "response_valence", "response_dominance",
        "response_age", "response_gender", "response_gender_confidence",
        "response_vox_num_chunks", "response_vox_support_speech_sec",
        "status_prompt_vox", "status_response_vox",
        "status_prompt_speaker_profile", "status_response_speaker_profile",
    ]

    rows = selected_rows_from_csv(args.input_csv, args.shard_idx, args.num_shards, args.limit)
    audio_cache = {}
    speaker_cache = {}

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tqdm(rows, desc=f"llm vox shard {args.shard_idx}/{args.num_shards}"):
            writer.writerow(
                add_vox_features(
                    row,
                    audio_cache,
                    speaker_cache,
                    age_sex_model,
                    emotion_model,
                    device,
                    vad_model,
                    get_speech_timestamps,
                )
            )


if __name__ == "__main__":
    main()
