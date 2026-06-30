#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/home/ahallur1/spear/NIPS_Experiments/emotivoice_full_v2")

MIN_SEGMENT_SEC = 3.0
MAX_SEGMENT_SEC = 15.0
MAX_LATENCY_SEC = 1.0
MAX_OVERLAP_SEC = 0.25
MERGE_GAP_SEC = 1.0
VAD_TOLERANCE_SEC = 0.5
PAUSE_THRESHOLD_SEC = 0.2

SOURCE_DATASET_SPLIT = "emotivoice_full_v2"
EXTERNAL_SPLIT = "external"
SPLIT_UNIT = "emotivoice_row_dir"
SESSION_ID = "emotivoice_full_v2"

TRACK_TO_SPEAKER = {
    "speaker1": "A",
    "speaker2": "B",
}

TRACK_TO_FILENAME = {
    "speaker1": "speaker1_track",
    "speaker2": "speaker2_track",
}

REQUIRED_ROW_FILES = [
    "metadata.csv",
    "speaker1_track.wav",
    "speaker1_track.json",
    "speaker2_track.wav",
    "speaker2_track.json",
]

BACKCHANNEL_SET = {
    "oh", "oh.", "oh!", "oh?", "yeah", "yeah.", "yeah!", "yes", "yes.", "yep",
    "yep.", "no", "no.", "nah", "nah.", "mm-hmm", "mm-hmm.", "mmhmm", "uh-huh",
    "uh-huh.", "uh huh", "right", "right.", "okay", "okay.", "ok", "ok.",
    "cool", "cool.", "sure", "sure.", "alright", "alright.", "hmm", "hmm.",
    "hm", "hm.", "mhm", "mhm.", "wow", "wow.",
}

COMMON_FIELDNAMES = [
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
    "prompt_tts_model", "response_tts_model",
    "prompt_is_emotivoice", "response_is_emotivoice",
    "is_human_prompt_emotivoice_response",
    "emotivoice_row_dir", "prompt_track", "response_track",
    "status_pair",
    "source_dataset_split", "split", "split_seed", "split_unit",
]


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def canonical_participant_id(raw: str) -> str:
    raw = str(raw).strip()
    if raw.startswith("P"):
        return raw
    return f"P{raw}"


def parse_row_dir_name(row_dir_name: str) -> dict:
    match = re.match(r"^row_(?P<row_index>.+)_p1_(?P<p1>P[^_]+)_p2_(?P<p2>P[^_]+)$", row_dir_name)
    assert match, f"Unexpected EmotiVoice row directory name: {row_dir_name}"
    return {
        "row_index_label": match.group("row_index"),
        "p1": canonical_participant_id(match.group("p1")),
        "p2": canonical_participant_id(match.group("p2")),
    }


def participant_for_track(row_info: dict, track: str) -> str:
    assert track in TRACK_TO_SPEAKER, f"Unknown track: {track}"
    return row_info["p1"] if track == "speaker1" else row_info["p2"]


def words_to_text(words):
    return " ".join(w["word"] for w in words).strip()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
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


def extract_transcript_events(json_data, row_dir: Path, row_info: dict, track: str):
    stem = TRACK_TO_FILENAME[track]
    wav_path = row_dir / f"{stem}.wav"
    json_path = row_dir / f"{stem}.json"
    participant_id = participant_for_track(row_info, track)

    events = []
    for seg in json_data.get("metadata:transcript", []):
        words = get_valid_words(seg)
        if not words:
            continue
        transcript = seg.get("transcript", "").strip()
        if not transcript:
            transcript = words_to_text(words)
        events.append(
            {
                "track": track,
                "metadata_speaker": TRACK_TO_SPEAKER[track],
                "participant_id": participant_id,
                "wav_path": str(wav_path.resolve()),
                "json_path": str(json_path.resolve()),
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
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
    merged[0]["words"] = list(merged[0]["words"])
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
            new_ev = ev.copy()
            new_ev["words"] = list(ev["words"])
            merged.append(new_ev)
    return merged


def extract_vad_segments(json_data):
    out = []
    for seg in json_data.get("metadata:vad", []):
        start = seg.get("start")
        end = seg.get("end")
        assert start is not None and end is not None, f"Invalid VAD segment: {seg}"
        start = float(start)
        end = float(end)
        if end == start:
            continue
        assert end > start, f"Invalid VAD segment with end<=start: {seg}"
        out.append((start, end))
    out.sort(key=lambda x: x[0])
    return out


def merge_vad_segments(vad_segments, gap_s=MERGE_GAP_SEC):
    if not vad_segments:
        return []
    merged = [vad_segments[0]]
    for start, end in vad_segments[1:]:
        cur_start, cur_end = merged[-1]
        if start - cur_end <= gap_s:
            merged[-1] = (cur_start, max(cur_end, end))
        else:
            merged.append((start, end))
    return merged


def refine_bounds_with_vad(start_s, end_s, merged_vad, tolerance_s=VAD_TOLERANCE_SEC):
    direct = [(s, e) for s, e in merged_vad if e >= start_s and s <= end_s]
    if direct:
        refined_start = min(s for s, _ in direct)
        refined_end = max(e for _, e in direct)
        assert refined_end > refined_start
        return refined_start, refined_end

    near = []
    for start, end in merged_vad:
        left_gap = max(0.0, start_s - end)
        right_gap = max(0.0, start - end_s)
        gap = max(left_gap, right_gap)
        if gap <= tolerance_s:
            near.append((gap, start, end))
    if near:
        near.sort(key=lambda x: x[0])
        _, start, end = near[0]
        return start, end
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


def load_metadata_models(row_dir: Path):
    metadata_path = row_dir / "metadata.csv"
    assert metadata_path.exists(), f"Missing metadata.csv: {metadata_path}"
    counts = defaultdict(Counter)
    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = str(row.get("spk", "")).strip()
            if not speaker:
                continue
            model = str(row.get("tts_model", "")).strip()
            if not model:
                model = "orig"
            counts[speaker][model] += 1
    assert counts, f"No speaker metadata loaded from {metadata_path}"

    out = {}
    for speaker, counter in counts.items():
        out[speaker] = counter.most_common(1)[0][0]
    return out


def tts_model_for_event(event, metadata_models):
    return metadata_models.get(event["metadata_speaker"], "UNKNOWN")


def base_row_dirs(input_root: Path):
    input_root = input_root.resolve()
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def missing_required_row_files(row_dir: Path):
    return [name for name in REQUIRED_ROW_FILES if not (row_dir / name).exists()]


def selected_row_dirs(input_root: Path, shard_idx: int, num_shards: int, limit: int = 0):
    assert num_shards > 0, f"num_shards must be positive, got {num_shards}"
    assert 0 <= shard_idx < num_shards, f"shard_idx {shard_idx} outside 0..{num_shards - 1}"
    assigned_rows = [path for i, path in enumerate(base_row_dirs(input_root)) if (i % num_shards) == shard_idx]
    rows = []
    skipped = []
    for row_dir in assigned_rows:
        missing = missing_required_row_files(row_dir)
        if missing:
            skipped.append((row_dir, missing))
            continue
        rows.append(row_dir)

    if skipped:
        skipped_names = ", ".join(f"{path.name} missing {missing}" for path, missing in skipped)
        print(
            f"Skipping {len(skipped)} incomplete EmotiVoice row dirs in shard "
            f"{shard_idx}/{num_shards}: {skipped_names}",
            file=sys.stderr,
        )

    if limit > 0:
        rows = rows[:limit]
    return rows


def build_merge_key(row: dict) -> str:
    required = [
        "session_id", "interaction_id", "dyad_id",
        "prompt_participant_id", "response_participant_id", "turn_index",
        "prompt_start_s", "prompt_end_s", "response_start_s", "response_end_s",
    ]
    missing = [col for col in required if col not in row]
    assert not missing, f"Missing merge key columns: {missing}"
    return (
        str(row["session_id"])
        + "|"
        + str(row["interaction_id"])
        + "|"
        + str(row["dyad_id"])
        + "|"
        + str(row["prompt_participant_id"])
        + "|"
        + str(row["response_participant_id"])
        + "|"
        + str(row["turn_index"])
        + "|"
        + str(round(float(row["prompt_start_s"]), 6))
        + "|"
        + str(round(float(row["prompt_end_s"]), 6))
        + "|"
        + str(round(float(row["response_start_s"]), 6))
        + "|"
        + str(round(float(row["response_end_s"]), 6))
    )


def build_pair_candidates_for_row_dir(row_dir: Path):
    row_info = parse_row_dir_name(row_dir.name)
    metadata_models = load_metadata_models(row_dir)

    json_by_track = {}
    events_by_track = {}
    vad_by_track = {}
    for track, stem in TRACK_TO_FILENAME.items():
        wav_path = row_dir / f"{stem}.wav"
        json_path = row_dir / f"{stem}.json"
        assert wav_path.exists(), f"Missing wav: {wav_path}"
        assert json_path.exists(), f"Missing json: {json_path}"

        json_data = load_json(json_path)
        json_by_track[track] = json_data
        events_by_track[track] = merge_local_same_speaker_events(
            extract_transcript_events(json_data, row_dir, row_info, track),
            gap_s=MERGE_GAP_SEC,
        )
        vad_by_track[track] = merge_vad_segments(
            extract_vad_segments(json_data),
            gap_s=MERGE_GAP_SEC,
        )

    all_events = []
    for track_events in events_by_track.values():
        all_events.extend(track_events)
    all_events.sort(key=lambda x: (x["start"], x["end"], x["participant_id"]))

    dyad_id = "|".join(sorted([row_info["p1"], row_info["p2"]]))
    interaction_id = row_dir.name
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

        prompt_refined = refine_bounds_with_vad(
            prev_ev["start"],
            prev_ev["end"],
            vad_by_track[prev_ev["track"]],
        )
        response_refined = refine_bounds_with_vad(
            cur_ev["start"],
            cur_ev["end"],
            vad_by_track[cur_ev["track"]],
        )
        if prompt_refined is None or response_refined is None:
            continue

        prompt_start, prompt_end = clip_bounds(*prompt_refined, mode="prompt")
        response_start, response_end = clip_bounds(*response_refined, mode="response")
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

        turn_index += 1
        prompt_tts = tts_model_for_event(prev_ev, metadata_models)
        response_tts = tts_model_for_event(cur_ev, metadata_models)
        prompt_is_emotivoice = prompt_tts == "emotivoice"
        response_is_emotivoice = response_tts == "emotivoice"

        row = {
            "pair_id": f"{interaction_id}|{prev_ev['participant_id']}|{cur_ev['participant_id']}|{turn_index:04d}",
            "session_id": SESSION_ID,
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
            "prompt_transcript": words_to_text(prompt_words),
            "response_transcript": words_to_text(response_words),
            "prompt_word_count": len(prompt_words),
            "response_word_count": len(response_words),
            "relationship": "UNKNOWN",
            "relationship_detail": "UNKNOWN",
            "status_relationship": "MISSING_RELATIONSHIP_METADATA",
            "prompt_tts_model": prompt_tts,
            "response_tts_model": response_tts,
            "prompt_is_emotivoice": prompt_is_emotivoice,
            "response_is_emotivoice": response_is_emotivoice,
            "is_human_prompt_emotivoice_response": (not prompt_is_emotivoice) and response_is_emotivoice,
            "emotivoice_row_dir": str(row_dir.resolve()),
            "prompt_track": prev_ev["track"],
            "response_track": cur_ev["track"],
            "status_pair": "OK",
            "source_dataset_split": SOURCE_DATASET_SPLIT,
            "split": EXTERNAL_SPLIT,
            "split_seed": None,
            "split_unit": SPLIT_UNIT,
            "_prompt_words": prompt_words,
            "_response_words": response_words,
        }
        out_rows.append(row)

    return out_rows


def public_row(row: dict):
    return {key: value for key, value in row.items() if not key.startswith("_")}
