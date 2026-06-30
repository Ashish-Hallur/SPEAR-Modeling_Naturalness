#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 15.0
MAX_LATENCY_SEC = 1.0
MAX_OVERLAP_SEC = 0.25
MERGE_GAP_SEC = 1.0
VAD_TOLERANCE_SEC = 0.5

TRAIN_FRAC = 0.60
VAL_FRAC = 0.10
TEST_FRAC = 0.30

TARGET_SR = 16000
VOX_MIN_SEC = 3.0
VOX_MAX_SEC = 15.0

BACKCHANNEL_SET = {
    "oh", "oh.", "oh!", "oh?", "yeah", "yeah.", "yeah!", "yes", "yes.", "yep",
    "yep.", "no", "no.", "nah", "nah.", "mm-hmm", "mm-hmm.", "mmhmm", "uh-huh",
    "uh-huh.", "uh huh", "right", "right.", "okay", "okay.", "ok", "ok.",
    "cool", "cool.", "sure", "sure.", "alright", "alright.", "hmm", "hmm.",
    "hm", "hm.", "mhm", "mhm.", "wow", "wow."
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyad_lookup_csv", type=str, required=True)
    ap.add_argument("--relationships_csv", type=str, required=True)
    ap.add_argument("--output_train_csv", type=str, required=True)
    ap.add_argument("--output_test_csv", type=str, required=True)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


VOX_RELEASE_DIR = Path(__file__).resolve().parents[1] / "vox-profile-release"
sys.path.insert(0, str(VOX_RELEASE_DIR))

from src.model.age_sex.wavlm_demographics import WavLMWrapper as AgeSexModel
from src.model.emotion.wavlm_emotion_dim import WavLMWrapper as EmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

age_sex_model = AgeSexModel.from_pretrained("tiantiaf/wavlm-large-age-sex").to(DEVICE).eval()
emotion_model = EmotionModel.from_pretrained("tiantiaf/wavlm-large-msp-podcast-emotion-dim").to(DEVICE).eval()

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True,
)

(
    get_speech_timestamps,
    save_audio,
    read_audio,
    VADIterator,
    collect_chunks,
) = vad_utils

vad_model = vad_model.cpu().eval()


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_participant_id(raw: str) -> str:
    raw = str(raw).strip()
    if raw.startswith("P"):
        raw = raw[1:]
    return raw


def canonical_participant_id(raw: str) -> str:
    return f"P{normalize_participant_id(raw)}"


def words_to_text(words):
    return " ".join(w["word"] for w in words).strip()


def parse_stem(stem: str):
    parts = stem.split("_")
    assert len(parts) == 4, f"Unexpected filename stem format: {stem}"
    vendor_id = parts[0]
    session_id = parts[1][1:]
    interaction_id = parts[2][1:]
    participant_id = parts[3][1:]
    return vendor_id, session_id, interaction_id, participant_id


def resolve_relpath(base_dir: Path, relpath: str) -> Path:
    return (base_dir / relpath).resolve()


def parse_source_dataset_split(relpath: str) -> str:
    parts = Path(relpath).parts
    hits = [part for part in parts if part in {"train", "dev", "test"}]
    assert len(hits) == 1, f"Expected exactly one train/dev/test path component in {relpath}"
    return hits[0]


def output_split_for_source(source_dataset_split: str) -> str:
    assert source_dataset_split in {"train", "dev", "test"}, source_dataset_split
    return "test" if source_dataset_split == "test" else "train"


def normalize_relationship_detail(rel, rel_detail):
    rel = str(rel).strip()
    rel_detail = str(rel_detail).strip()
    if rel_detail == "":
        rel_detail = "UNKNOWN"
    if rel == "stranger":
        return "UNKNOWN"
    if rel_detail == "stranger":
        return "UNKNOWN"
    return rel_detail


def load_relationships(path: Path):
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["vendor_id"], row["session_id"])
            relationship = row["relationship"].strip()
            relationship_detail = normalize_relationship_detail(
                relationship, row["relationship_detail"]
            )
            out[key] = {
                "relationship": relationship,
                "relationship_detail": relationship_detail,
            }
    assert out, "relationships.csv loaded empty"
    return out


def load_dyads(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            source1 = parse_source_dataset_split(row["participant1_relpath"])
            source2 = parse_source_dataset_split(row["participant2_relpath"])
            assert source1 == source2, (
                f"Mixed source dataset splits in dyad row {i}: {source1} vs {source2}"
            )
            rows.append(
                {
                    "row_index": i,
                    "participant1_id": canonical_participant_id(row["participant1_id"]),
                    "participant1_relpath": row["participant1_relpath"],
                    "participant2_id": canonical_participant_id(row["participant2_id"]),
                    "participant2_relpath": row["participant2_relpath"],
                    "source_dataset_split": source1,
                    "output_split": output_split_for_source(source1),
                }
            )
    assert rows, "dyad_lookup.csv is empty"
    return rows


def build_speaker_component_split(dyad_rows, seed):
    graph = defaultdict(set)
    dyad_row_counts = defaultdict(int)

    for row in dyad_rows:
        a = row["participant1_id"]
        b = row["participant2_id"]
        graph[a].add(b)
        graph[b].add(a)
        dyad_id = "|".join(sorted([a, b]))
        dyad_row_counts[dyad_id] += 1

    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        q = deque([node])
        visited.add(node)
        comp_nodes = []

        while q:
            cur = q.popleft()
            comp_nodes.append(cur)
            for nbr in graph[cur]:
                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)

        comp_nodes = sorted(comp_nodes)
        comp_node_set = set(comp_nodes)

        comp_weight = 0
        comp_dyads = set()
        for row in dyad_rows:
            a = row["participant1_id"]
            b = row["participant2_id"]
            if a in comp_node_set and b in comp_node_set:
                dyad_id = "|".join(sorted([a, b]))
                if dyad_id not in comp_dyads:
                    comp_dyads.add(dyad_id)
                    comp_weight += dyad_row_counts[dyad_id]

        components.append({"nodes": comp_nodes, "weight": comp_weight})

    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=lambda x: x["weight"], reverse=True)

    total_weight = sum(c["weight"] for c in components)
    train_target = TRAIN_FRAC * total_weight
    val_target = VAL_FRAC * total_weight
    test_target = TEST_FRAC * total_weight

    buckets = {
        "train": {"weight": 0, "components": []},
        "val": {"weight": 0, "components": []},
        "test": {"weight": 0, "components": []},
    }

    def score_bucket(name, add_weight):
        target = {"train": train_target, "val": val_target, "test": test_target}[name]
        return abs((buckets[name]["weight"] + add_weight) - target)

    for comp in components:
        best_bucket = min(
            ["train", "val", "test"],
            key=lambda name: score_bucket(name, comp["weight"]),
        )
        buckets[best_bucket]["components"].append(comp)
        buckets[best_bucket]["weight"] += comp["weight"]

    speaker_to_split = {}
    for split_name, bucket in buckets.items():
        for comp in bucket["components"]:
            for spk in comp["nodes"]:
                speaker_to_split[spk] = split_name

    return speaker_to_split


def extract_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_valid_words(transcript_seg):
    words = []
    for w in transcript_seg.get("words", []):
        word = str(w.get("word", "")).strip()
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        if not word:
            continue
        words.append({"word": word, "start": start, "end": end})
    words.sort(key=lambda x: x["start"])
    return words


def extract_transcript_events(json_data, participant_id, wav_path, json_path):
    events = []
    segs = json_data.get("metadata:transcript", [])
    for seg in segs:
        words = get_valid_words(seg)
        if not words:
            continue
        start = float(words[0]["start"])
        end = float(words[-1]["end"])
        transcript = seg.get("transcript", "").strip()
        if not transcript:
            transcript = words_to_text(words)
        events.append(
            {
                "participant_id": participant_id,
                "wav_path": str(wav_path),
                "json_path": str(json_path),
                "start": start,
                "end": end,
                "transcript": transcript,
                "words": words,
            }
        )
    events.sort(key=lambda x: x["start"])
    return events


def merge_local_same_speaker_events(events, gap_s=MERGE_GAP_SEC):
    if not events:
        return []
    merged = [events[0].copy()]
    for ev in events[1:]:
        cur = merged[-1]
        assert ev["participant_id"] == cur["participant_id"]
        gap = ev["start"] - cur["end"]
        if gap <= gap_s:
            cur["end"] = max(cur["end"], ev["end"])
            cur["words"].extend(ev["words"])
            cur["words"].sort(key=lambda x: x["start"])
            cur["transcript"] = words_to_text(cur["words"])
        else:
            merged.append(ev.copy())
    return merged


def extract_vad_segments(json_data):
    vad = json_data.get("metadata:vad", [])
    out = []
    for seg in vad:
        s = seg.get("start")
        e = seg.get("end")
        assert s is not None and e is not None, f"Invalid VAD segment: {seg}"
        s = float(s)
        e = float(e)
        if e == s:
            continue
        assert e > s, f"Invalid VAD segment with end<=start: {seg}"
        out.append((s, e))
    out.sort(key=lambda x: x[0])
    return out


def merge_vad_segments(vad_segments, gap_s=MERGE_GAP_SEC):
    if not vad_segments:
        return []
    merged = [vad_segments[0]]
    for s, e in vad_segments[1:]:
        cs, ce = merged[-1]
        if s - ce <= gap_s:
            merged[-1] = (cs, max(ce, e))
        else:
            merged.append((s, e))
    return merged


def refine_bounds_with_vad(start_s, end_s, merged_vad, tolerance_s=VAD_TOLERANCE_SEC):
    direct = []
    for s, e in merged_vad:
        if e >= start_s and s <= end_s:
            direct.append((s, e))

    if direct:
        refined_start = min(s for s, _ in direct)
        refined_end = max(e for _, e in direct)
        assert refined_end > refined_start
        return refined_start, refined_end

    near = []
    for s, e in merged_vad:
        left_gap = max(0.0, start_s - e)
        right_gap = max(0.0, s - end_s)
        gap = max(left_gap, right_gap)
        if gap <= tolerance_s:
            near.append((gap, s, e))

    if near:
        near.sort(key=lambda x: x[0])
        _, s, e = near[0]
        return s, e

    return None


def clip_bounds(start_s, end_s, mode):
    dur = end_s - start_s
    assert dur > 0
    if dur <= MAX_SEGMENT_SEC:
        return start_s, end_s
    if mode == "prompt":
        return end_s - MAX_SEGMENT_SEC, end_s
    if mode == "response":
        return start_s, start_s + MAX_SEGMENT_SEC
    raise ValueError(f"Unknown mode: {mode}")


def select_words_in_bounds(words, start_s, end_s):
    selected = [w for w in words if w["start"] >= start_s and w["end"] <= end_s]
    selected.sort(key=lambda x: x["start"])
    return selected


def is_backchannel(transcript, words):
    txt = normalize_text(transcript)
    if txt in BACKCHANNEL_SET:
        return True
    if len(words) <= 3:
        toks = [normalize_text(w["word"]) for w in words]
        toks = [t for t in toks if t]
        if toks and all(t in BACKCHANNEL_SET for t in toks):
            return True
    return False


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
    start = int(round(start_s * sr))
    end = int(round(end_s * sr))
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


def speech_only_chunks_from_waveform(wav_16k):
    wav_cpu = wav_16k.detach().cpu()
    speech_ts = get_speech_timestamps(wav_cpu, vad_model, sampling_rate=TARGET_SR)

    if not speech_ts:
        return [], 0.0

    speech_audio = torch.cat([wav_cpu[t["start"]: t["end"]] for t in speech_ts])
    speech_sec = speech_audio.numel() / TARGET_SR

    max_len = int(VOX_MAX_SEC * TARGET_SR)
    min_len = int(VOX_MIN_SEC * TARGET_SR)

    chunks = []
    for i in range(0, speech_audio.numel(), max_len):
        chunk = speech_audio[i:i + max_len]
        if chunk.numel() >= min_len:
            chunks.append(chunk)

    return chunks, speech_sec


def aggregate_age_gender_for_speaker(audio_cache, wav_path):
    try:
        wav, sr = load_audio_cache(audio_cache, wav_path)
    except RuntimeError as exc:
        if "Failed to decode audio" not in str(exc):
            raise
        return {
            "age": np.nan,
            "gender": "",
            "gender_confidence": np.nan,
            "num_chunks": 0,
            "speech_sec": 0.0,
            "status": "AUDIO_DECODE_FAILED",
        }

    wav_16k = prepare_audio_for_vox(wav, sr)
    if wav_16k.numel() == 0:
        return {
            "age": np.nan,
            "gender": "",
            "gender_confidence": np.nan,
            "num_chunks": 0,
            "speech_sec": 0.0,
            "status": "EMPTY_AUDIO",
        }

    chunks, speech_sec = speech_only_chunks_from_waveform(wav_16k)

    if not chunks:
        return {
            "age": np.nan,
            "gender": "",
            "gender_confidence": np.nan,
            "num_chunks": 0,
            "speech_sec": speech_sec,
            "status": "NO_SPEECH_AFTER_VAD",
        }

    age_vals = []
    sex_probs = []
    for chunk in chunks:
        x = chunk.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            age_out, sex_out = age_sex_model(x)
        age_vals.append(age_out.squeeze().item() * 100.0)
        sex_probs.append(F.softmax(sex_out, dim=1).squeeze(0).cpu().numpy())

    mean_age = float(np.mean(age_vals))
    mean_sex = np.mean(np.stack(sex_probs, axis=0), axis=0)
    gender_idx = int(np.argmax(mean_sex))
    gender = "female" if gender_idx == 0 else "male"
    gender_conf = float(np.max(mean_sex))

    return {
        "age": mean_age,
        "gender": gender,
        "gender_confidence": gender_conf,
        "num_chunks": len(chunks),
        "speech_sec": speech_sec,
        "status": "OK",
    }


def aggregate_avd_for_segment(audio_cache, wav_path, start_s, end_s):
    try:
        wav, sr = slice_audio(audio_cache, wav_path, start_s, end_s)
    except RuntimeError as exc:
        if "Failed to decode audio" not in str(exc):
            raise
        return {
            "arousal": np.nan,
            "valence": np.nan,
            "dominance": np.nan,
            "num_chunks": 0,
            "speech_sec": 0.0,
            "status": "AUDIO_DECODE_FAILED",
        }

    wav_16k = prepare_audio_for_vox(wav, sr)
    if wav_16k.numel() == 0:
        return {
            "arousal": np.nan,
            "valence": np.nan,
            "dominance": np.nan,
            "num_chunks": 0,
            "speech_sec": 0.0,
            "status": "EMPTY_AUDIO",
        }

    chunks, speech_sec = speech_only_chunks_from_waveform(wav_16k)

    if not chunks:
        return {
            "arousal": np.nan,
            "valence": np.nan,
            "dominance": np.nan,
            "num_chunks": 0,
            "speech_sec": speech_sec,
            "status": "NO_SPEECH_AFTER_VAD",
        }

    a_vals, v_vals, d_vals = [], [], []
    for chunk in chunks:
        x = chunk.unsqueeze(0).to(DEVICE)
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


def build_pair_rows_for_dyad(
    dyad_row,
    relationships_map,
    split_name,
    source_dataset_split,
    dyad_lookup_parent,
    audio_cache,
    speaker_cache,
):
    wav1 = resolve_relpath(dyad_lookup_parent, dyad_row["participant1_relpath"])
    wav2 = resolve_relpath(dyad_lookup_parent, dyad_row["participant2_relpath"])
    json1 = wav1.with_suffix(".json")
    json2 = wav2.with_suffix(".json")

    assert wav1.exists(), f"Missing wav: {wav1}"
    assert wav2.exists(), f"Missing wav: {wav2}"
    assert json1.exists(), f"Missing json: {json1}"
    assert json2.exists(), f"Missing json: {json2}"

    vendor1, session1, interaction1, p1_raw = parse_stem(wav1.stem)
    vendor2, session2, interaction2, p2_raw = parse_stem(wav2.stem)

    p1 = canonical_participant_id(p1_raw)
    p2 = canonical_participant_id(p2_raw)

    assert p1 == dyad_row["participant1_id"], f"Participant mismatch: {p1} vs {dyad_row['participant1_id']}"
    assert p2 == dyad_row["participant2_id"], f"Participant mismatch: {p2} vs {dyad_row['participant2_id']}"
    assert vendor1 == vendor2, f"Vendor mismatch: {wav1} vs {wav2}"
    assert session1 == session2, f"Session mismatch: {wav1} vs {wav2}"
    assert interaction1 == interaction2, f"Interaction mismatch: {wav1} vs {wav2}"

    dyad_id = "|".join(sorted([p1, p2]))
    session_id = session1
    interaction_id = interaction1

    rel_key = (vendor1, session1)
    rel_info = relationships_map.get(rel_key)
    if rel_info is None:
        relationship = "UNKNOWN"
        relationship_detail = "UNKNOWN"
        status_relationship = "MISSING_RELATIONSHIP_METADATA"
    else:
        relationship = rel_info["relationship"]
        relationship_detail = rel_info["relationship_detail"]
        status_relationship = "OK"

    j1 = extract_json(json1)
    j2 = extract_json(json2)

    ev1 = merge_local_same_speaker_events(
        extract_transcript_events(j1, p1, wav1, json1),
        gap_s=MERGE_GAP_SEC,
    )
    ev2 = merge_local_same_speaker_events(
        extract_transcript_events(j2, p2, wav2, json2),
        gap_s=MERGE_GAP_SEC,
    )

    vad1 = merge_vad_segments(extract_vad_segments(j1), gap_s=MERGE_GAP_SEC)
    vad2 = merge_vad_segments(extract_vad_segments(j2), gap_s=MERGE_GAP_SEC)

    if str(wav1) not in speaker_cache:
        speaker_cache[str(wav1)] = aggregate_age_gender_for_speaker(audio_cache, wav1)
    if str(wav2) not in speaker_cache:
        speaker_cache[str(wav2)] = aggregate_age_gender_for_speaker(audio_cache, wav2)

    all_events = ev1 + ev2
    all_events.sort(key=lambda x: (x["start"], x["end"], x["participant_id"]))

    out_rows = []
    turn_index = 0

    for i in range(1, len(all_events)):
        prev_ev = all_events[i - 1]
        cur_ev = all_events[i]

        if prev_ev["participant_id"] == cur_ev["participant_id"]:
            continue
        if is_backchannel(prev_ev["transcript"], prev_ev["words"]):
            continue
        if is_backchannel(cur_ev["transcript"], cur_ev["words"]):
            continue

        prompt_vad = vad1 if prev_ev["participant_id"] == p1 else vad2
        response_vad = vad1 if cur_ev["participant_id"] == p1 else vad2

        prompt_refined = refine_bounds_with_vad(prev_ev["start"], prev_ev["end"], prompt_vad)
        response_refined = refine_bounds_with_vad(cur_ev["start"], cur_ev["end"], response_vad)

        if prompt_refined is None or response_refined is None:
            continue

        prompt_start, prompt_end = prompt_refined
        response_start, response_end = response_refined

        prompt_start, prompt_end = clip_bounds(prompt_start, prompt_end, mode="prompt")
        response_start, response_end = clip_bounds(response_start, response_end, mode="response")

        prompt_duration = prompt_end - prompt_start
        response_duration = response_end - response_start

        if prompt_duration < MIN_SEGMENT_SEC:
            continue
        if response_duration < MIN_SEGMENT_SEC:
            continue

        overlap_s = max(0.0, prompt_end - response_start)
        latency_s = max(0.0, response_start - prompt_end)

        if overlap_s > MAX_OVERLAP_SEC:
            continue
        if latency_s > MAX_LATENCY_SEC:
            continue

        prompt_words = select_words_in_bounds(prev_ev["words"], prompt_start, prompt_end)
        response_words = select_words_in_bounds(cur_ev["words"], response_start, response_end)

        prompt_transcript = words_to_text(prompt_words)
        response_transcript = words_to_text(response_words)

        turn_index += 1
        pair_id = f"{interaction_id}|{prev_ev['participant_id']}|{cur_ev['participant_id']}|{turn_index:04d}"

        prompt_avd = aggregate_avd_for_segment(audio_cache, prev_ev["wav_path"], prompt_start, prompt_end)
        response_avd = aggregate_avd_for_segment(audio_cache, cur_ev["wav_path"], response_start, response_end)

        prompt_spk = speaker_cache[prev_ev["wav_path"]]
        response_spk = speaker_cache[cur_ev["wav_path"]]

        out_rows.append(
            {
                "pair_id": pair_id,
                "session_id": session_id,
                "interaction_id": interaction_id,
                "dyad_id": dyad_id,
                "prompt_participant_id": prev_ev["participant_id"],
                "response_participant_id": cur_ev["participant_id"],
                "turn_index": turn_index,

                "source_wav_path_prompt": prev_ev["wav_path"],
                "source_json_path_prompt": prev_ev["json_path"],
                "source_wav_path_response": cur_ev["wav_path"],
                "source_json_path_response": cur_ev["json_path"],

                "prompt_start_s": prompt_start,
                "prompt_end_s": prompt_end,
                "prompt_duration_s": prompt_duration,

                "response_start_s": response_start,
                "response_end_s": response_end,
                "response_duration_s": response_duration,

                "latency_s": latency_s,
                "overlap_s": overlap_s,

                "prompt_transcript": prompt_transcript,
                "response_transcript": response_transcript,
                "prompt_word_count": len(prompt_words),
                "response_word_count": len(response_words),

                "relationship": relationship,
                "relationship_detail": relationship_detail,
                "status_relationship": status_relationship,

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

                "status_pair": "OK",
                "status_prompt_vox": prompt_avd["status"],
                "status_response_vox": response_avd["status"],
                "status_prompt_speaker": prompt_spk["status"],
                "status_response_speaker": response_spk["status"],

                "source_dataset_split": source_dataset_split,
                "split": split_name,
                "split_seed": None,
                "split_unit": "seamless_dataset_path",
            }
        )

    return out_rows


def main():
    args = parse_args()

    dyad_lookup_csv = Path(args.dyad_lookup_csv).resolve()
    relationships_csv = Path(args.relationships_csv).resolve()
    output_train_csv = Path(args.output_train_csv).resolve()
    output_test_csv = Path(args.output_test_csv).resolve()
    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    output_test_csv.parent.mkdir(parents=True, exist_ok=True)

    dyad_rows = load_dyads(dyad_lookup_csv)

    shard_rows = [row for row in dyad_rows if (row["row_index"] % args.num_shards) == args.shard_idx]
    if args.limit > 0:
        shard_rows = shard_rows[: args.limit]

    relationships_map = load_relationships(relationships_csv)
    audio_cache = {}
    speaker_cache = {}

    fieldnames = [
        "pair_id", "session_id", "interaction_id", "dyad_id",
        "prompt_participant_id", "response_participant_id", "turn_index",
        "source_wav_path_prompt", "source_json_path_prompt",
        "source_wav_path_response", "source_json_path_response",
        "prompt_start_s", "prompt_end_s", "prompt_duration_s",
        "response_start_s", "response_end_s", "response_duration_s",
        "latency_s", "overlap_s",
        "prompt_transcript", "response_transcript",
        "prompt_word_count", "response_word_count",
        "relationship", "relationship_detail", "status_relationship",
        "prompt_arousal", "prompt_valence", "prompt_dominance",
        "prompt_age", "prompt_gender", "prompt_gender_confidence",
        "prompt_vox_num_chunks", "prompt_vox_support_speech_sec",
        "response_arousal", "response_valence", "response_dominance",
        "response_age", "response_gender", "response_gender_confidence",
        "response_vox_num_chunks", "response_vox_support_speech_sec",
        "status_pair", "status_prompt_vox", "status_response_vox",
        "status_prompt_speaker", "status_response_speaker",
        "source_dataset_split", "split", "split_seed", "split_unit",
    ]

    with open(output_train_csv, "w", newline="") as train_f, open(output_test_csv, "w", newline="") as test_f:
        writers = {
            "train": csv.DictWriter(train_f, fieldnames=fieldnames),
            "test": csv.DictWriter(test_f, fieldnames=fieldnames),
        }
        for writer in writers.values():
            writer.writeheader()

        for row in tqdm(shard_rows, desc=f"vox shard {args.shard_idx}/{args.num_shards}"):
            pair_rows = build_pair_rows_for_dyad(
                row,
                relationships_map=relationships_map,
                split_name=row["output_split"],
                source_dataset_split=row["source_dataset_split"],
                dyad_lookup_parent=dyad_lookup_csv.parent,
                audio_cache=audio_cache,
                speaker_cache=speaker_cache,
            )

            writer = writers[row["output_split"]]
            for pr in pair_rows:
                writer.writerow(pr)


if __name__ == "__main__":
    main()