#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import wave
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None


DEFAULT_DATA_ROOT = Path("/home/tthebau1/SPEAR/SPEARBench/benchmark/data/seamless_2t_2s_questions")
SESSION_ID = "spearbench_seamless_2t_2s_questions"
SOURCE_DATASET_SPLIT = "spearbench_seamless_2t_2s_questions"
SPLIT_UNIT = "spearbench_question"

MIN_VOX_SEC = 3.0
MAX_CLIP_SEC = 15.0
PAUSE_THRESHOLD_SEC = 0.2

COMMON_FIELDNAMES = [
    "pair_id",
    "candidate_id",
    "session_id",
    "interaction_id",
    "dyad_id",
    "prompt_participant_id",
    "response_participant_id",
    "turn_index",
    "model_name",
    "is_original_reference",
    "split",
    "source_style",
    "source_dataset_split",
    "split_seed",
    "split_unit",
    "conversation_id",
    "speakers",
    "prompt_stem",
    "response_stem",
    "finish_reason",
    "answer_start_time",
    "latency_s",
    "overlap_s",
    "prompt_audio_path",
    "response_audio_path",
    "source_wav_path_prompt",
    "source_wav_path_response",
    "source_json_path_prompt",
    "source_json_path_response",
    "prompt_start_s",
    "prompt_end_s",
    "prompt_duration_s",
    "response_start_s",
    "response_end_s",
    "response_duration_s",
    "prompt_audio_duration_s",
    "response_audio_duration_s",
    "prompt_clip_start_s",
    "prompt_clip_end_s",
    "response_clip_start_s",
    "response_clip_end_s",
    "prompt_was_clipped",
    "response_was_clipped",
    "prompt_vox_compatible",
    "response_vox_compatible",
    "prompt_transcript",
    "response_transcript",
    "response_metadata_transcript",
    "response_asr_transcript",
    "prompt_word_count",
    "response_word_count",
    "relationship",
    "relationship_detail",
    "status_relationship",
    "status_pair",
    "status_prompt_audio",
    "status_response_audio",
    "status_prompt_source",
    "status_response_source",
]


def benchmark_root(data_root: Path) -> Path:
    data_root = data_root.resolve()
    assert data_root.name == "seamless_2t_2s_questions", data_root
    return data_root.parents[1]


def resolve_benchmark_path(data_root: Path, raw_path: str) -> Path:
    raw = str(raw_path).strip()
    assert raw, "Cannot resolve empty benchmark path"
    path = Path(raw)
    if path.is_absolute():
        return path
    return (benchmark_root(data_root) / path).resolve()


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audio_duration_s(path: Path) -> float:
    if sf is not None:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def positive_float(value) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    out = float(text)
    if out <= 0:
        return None
    return out


def prompt_duration_from_metadata(prompt_row: dict, prompt_path: Path) -> float:
    return audio_duration_s(prompt_path)


def response_duration_from_metadata(output_row: dict, response_path: Path) -> float:
    return audio_duration_s(response_path)


def prompt_clip_bounds(duration_s: float):
    assert duration_s > 0, duration_s
    if duration_s <= MAX_CLIP_SEC:
        return 0.0, duration_s, False
    return duration_s - MAX_CLIP_SEC, duration_s, True


def response_clip_bounds(duration_s: float):
    assert duration_s > 0, duration_s
    if duration_s <= MAX_CLIP_SEC:
        return 0.0, duration_s, False
    return 0.0, MAX_CLIP_SEC, True


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", str(text)))


def prompt_stem_from_row(row: dict) -> str:
    return Path(row["audio_path"]).stem


def parse_turn_index(prompt_stem: str) -> int:
    suffix = prompt_stem.rsplit("_", 1)[-1]
    assert suffix.isdigit(), f"Could not parse turn index from prompt stem: {prompt_stem}"
    return int(suffix)


def parse_last_speaker(transcript: str) -> str:
    matches = re.findall(r"\b(P[0-9A-Z]+):", str(transcript))
    return matches[-1] if matches else ""


def parse_first_speaker(transcript: str) -> str:
    match = re.search(r"\b(P[0-9A-Z]+):", str(transcript))
    return match.group(1) if match else ""


def participant_fallback(speakers: str, index: int) -> str:
    parts = [p for p in str(speakers).split("|") if p]
    if index < len(parts):
        return parts[index]
    return ""


def normalize_model_name_for_id(model_name: str) -> str:
    return str(model_name).replace("|", "_").replace("/", "_")


def build_pair_id(model_name: str, split: str, source_style: str, prompt_stem: str, response_stem: str) -> str:
    return "|".join([normalize_model_name_for_id(model_name), split, source_style, prompt_stem, response_stem])


def build_merge_key(row: dict) -> str:
    return (
        str(row["session_id"])
        + "|"
        + str(row["pair_id"])
        + "|"
        + str(round(float(row["prompt_start_s"]), 6))
        + "|"
        + str(round(float(row["prompt_end_s"]), 6))
        + "|"
        + str(round(float(row["response_start_s"]), 6))
        + "|"
        + str(round(float(row["response_end_s"]), 6))
    )


def load_input_index(data_root: Path):
    out = {}
    for metadata_path in sorted((data_root / "inputs").glob("*/*/metadata.csv")):
        split = metadata_path.parents[1].name
        source_style = metadata_path.parent.name
        for row in read_csv_rows(metadata_path):
            stem = prompt_stem_from_row(row)
            out[(split, source_style, stem)] = row
    return out


def load_asr_transcript_index(output_dir: Path):
    path = output_dir / "whisper-large-v3_transcripts.csv"
    if not path.exists():
        return {}
    out = {}
    for row in read_csv_rows(path):
        answer_path = row.get("answer_audio_path", "")
        if answer_path:
            out[answer_path] = row
            out[Path(answer_path).name] = row
    return out


def candidate_metadata_paths(data_root: Path):
    return sorted((data_root / "outputs").glob("*/*/*/metadata.csv"))


def build_candidate_rows(data_root: Path):
    data_root = data_root.resolve()
    input_index = load_input_index(data_root)
    rows = []
    skipped = []

    for metadata_path in candidate_metadata_paths(data_root):
        source_style = metadata_path.parent.name
        split = metadata_path.parents[1].name
        model_name = metadata_path.parents[2].name
        asr_index = load_asr_transcript_index(metadata_path.parent)

        for output_row in read_csv_rows(metadata_path):
            prompt_stem = prompt_stem_from_row(output_row)
            input_row = input_index.get((split, source_style, prompt_stem))
            prompt_row = input_row if input_row is not None else output_row
            prompt_status = "INPUT_METADATA" if input_row is not None else "OUTPUT_METADATA_FALLBACK"

            prompt_path = resolve_benchmark_path(data_root, prompt_row["audio_path"])
            response_raw = output_row.get("answer_audio_path", "")
            if not response_raw:
                skipped.append(make_skipped_row(data_root, metadata_path, output_row, "MISSING_RESPONSE_AUDIO_PATH"))
                continue
            response_path = resolve_benchmark_path(data_root, response_raw)
            if not prompt_path.exists():
                skipped.append(make_skipped_row(data_root, metadata_path, output_row, "MISSING_PROMPT_AUDIO_FILE"))
                continue
            if not response_path.exists():
                skipped.append(make_skipped_row(data_root, metadata_path, output_row, "MISSING_RESPONSE_AUDIO_FILE"))
                continue

            prompt_duration = prompt_duration_from_metadata(prompt_row, prompt_path)
            response_duration = response_duration_from_metadata(output_row, response_path)
            p_start, p_end, p_clipped = prompt_clip_bounds(prompt_duration)
            r_start, r_end, r_clipped = response_clip_bounds(response_duration)

            asr_row = asr_index.get(response_raw) or asr_index.get(Path(response_raw).name) or {}
            response_asr = str(asr_row.get("ASR_transcript_answer", "")).strip()
            response_metadata = str(output_row.get("transcript_answer", "")).strip()
            response_transcript = response_asr or response_metadata
            prompt_transcript = str(prompt_row.get("transcript_question", output_row.get("transcript_question", ""))).strip()
            speakers = str(output_row.get("speakers", prompt_row.get("speakers", ""))).strip()
            prompt_participant = parse_last_speaker(prompt_transcript) or participant_fallback(speakers, 0)
            if model_name == "original":
                response_participant = parse_first_speaker(response_metadata) or participant_fallback(speakers, 1)
            else:
                response_participant = model_name

            answer_start_time = float(output_row.get("answer_start_time") or 0.0)
            response_stem = response_path.stem
            pair_id = build_pair_id(model_name, split, source_style, prompt_stem, response_stem)
            row = {
                "pair_id": pair_id,
                "candidate_id": pair_id,
                "session_id": SESSION_ID,
                "interaction_id": str(output_row.get("conversation_id", prompt_row.get("conversation_id", ""))),
                "dyad_id": speakers,
                "prompt_participant_id": prompt_participant,
                "response_participant_id": response_participant,
                "turn_index": parse_turn_index(prompt_stem),
                "model_name": model_name,
                "is_original_reference": model_name == "original",
                "split": split,
                "source_style": source_style,
                "source_dataset_split": SOURCE_DATASET_SPLIT,
                "split_seed": "",
                "split_unit": SPLIT_UNIT,
                "conversation_id": str(output_row.get("conversation_id", prompt_row.get("conversation_id", ""))),
                "speakers": speakers,
                "prompt_stem": prompt_stem,
                "response_stem": response_stem,
                "finish_reason": str(output_row.get("finish_reason", "")),
                "answer_start_time": answer_start_time,
                "latency_s": max(0.0, answer_start_time),
                "overlap_s": max(0.0, -answer_start_time),
                "prompt_audio_path": str(prompt_path),
                "response_audio_path": str(response_path),
                "source_wav_path_prompt": str(prompt_path),
                "source_wav_path_response": str(response_path),
                "source_json_path_prompt": "",
                "source_json_path_response": "",
                "prompt_start_s": p_start,
                "prompt_end_s": p_end,
                "prompt_duration_s": p_end - p_start,
                "response_start_s": r_start,
                "response_end_s": r_end,
                "response_duration_s": r_end - r_start,
                "prompt_audio_duration_s": prompt_duration,
                "response_audio_duration_s": response_duration,
                "prompt_clip_start_s": p_start,
                "prompt_clip_end_s": p_end,
                "response_clip_start_s": r_start,
                "response_clip_end_s": r_end,
                "prompt_was_clipped": p_clipped,
                "response_was_clipped": r_clipped,
                "prompt_vox_compatible": (p_end - p_start) >= MIN_VOX_SEC,
                "response_vox_compatible": (r_end - r_start) >= MIN_VOX_SEC,
                "prompt_transcript": prompt_transcript,
                "response_transcript": response_transcript,
                "response_metadata_transcript": response_metadata,
                "response_asr_transcript": response_asr,
                "prompt_word_count": word_count(prompt_transcript),
                "response_word_count": word_count(response_transcript),
                "relationship": "UNKNOWN",
                "relationship_detail": "UNKNOWN",
                "status_relationship": "MISSING_RELATIONSHIP_METADATA",
                "status_pair": "OK",
                "status_prompt_audio": "OK",
                "status_response_audio": "OK",
                "status_prompt_source": prompt_status,
                "status_response_source": "OUTPUT_METADATA",
            }
            rows.append(row)

    return rows, skipped


def make_skipped_row(data_root: Path, metadata_path: Path, output_row: dict, status: str):
    split = metadata_path.parents[1].name
    source_style = metadata_path.parent.name
    model_name = metadata_path.parents[2].name
    prompt_stem = prompt_stem_from_row(output_row)
    return {
        "pair_id": build_pair_id(model_name, split, source_style, prompt_stem, Path(output_row.get("answer_audio_path", "")).stem),
        "model_name": model_name,
        "split": split,
        "source_style": source_style,
        "prompt_stem": prompt_stem,
        "metadata_csv": str(metadata_path),
        "audio_path": output_row.get("audio_path", ""),
        "answer_audio_path": output_row.get("answer_audio_path", ""),
        "status_pair": status,
    }


def selected_rows_from_csv(input_csv: Path, shard_idx: int, num_shards: int, limit: int = 0):
    assert num_shards > 0, f"num_shards must be positive, got {num_shards}"
    assert 0 <= shard_idx < num_shards, f"shard_idx {shard_idx} outside 0..{num_shards - 1}"
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = [row for i, row in enumerate(csv.DictReader(f)) if (i % num_shards) == shard_idx]
    if limit > 0:
        rows = rows[:limit]
    return rows


def public_row(row: dict):
    return {key: value for key, value in row.items() if not key.startswith("_")}
