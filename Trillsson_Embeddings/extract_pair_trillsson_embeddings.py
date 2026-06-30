#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy import signal
from scipy.io import wavfile
from tqdm.auto import tqdm


TARGET_SR = 16000
DEFAULT_MODEL_HANDLE = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1"
DEFAULT_OUTPUT_KEY = "auto"


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract Trillsson embeddings for prompt/response segments from the final pair dataset."
    )
    ap.add_argument("--input_train_csv", type=Path, required=True, help="Train final merged pair dataset CSV.")
    ap.add_argument("--input_test_csv", type=Path, required=True, help="Test final merged pair dataset CSV.")
    ap.add_argument("--out_train_dir", type=Path, required=True, help="Train output root dir.")
    ap.add_argument("--out_test_dir", type=Path, required=True, help="Test output root dir.")
    ap.add_argument("--output_train_csv", type=Path, required=True, help="Train shard output CSV path.")
    ap.add_argument("--output_test_csv", type=Path, required=True, help="Test shard output CSV path.")

    ap.add_argument("--model_handle", type=str, default=DEFAULT_MODEL_HANDLE)
    ap.add_argument("--trillsson_variant", type=str, default="trillsson4")
    ap.add_argument(
        "--embedding_output_key",
        type=str,
        default=DEFAULT_OUTPUT_KEY,
        help="SavedModel pooled embedding key. Use auto to choose embedding/output/embeddings/default or the only rank-2 output.",
    )
    ap.add_argument(
        "--sequence_output_key",
        type=str,
        default="",
        help="Optional SavedModel output key for [B,T,D] representations.",
    )
    ap.add_argument(
        "--save_full_seq",
        action="store_true",
        help="Save a sequence output if the SavedModel exposes one.",
    )

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)

    return ap.parse_args()


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def safe_pair_id_filename(pair_id: str, role: str, merge_key: str) -> str:
    safe = pair_id.replace("|", "__").replace("/", "_").replace("\\", "_")
    return f"{safe}__{short_hash(merge_key)}__{role}"


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    sr, y = wavfile.read(str(path))
    if np.issubdtype(y.dtype, np.integer):
        info = np.iinfo(y.dtype)
        if info.min == 0:
            midpoint = (info.max + 1) / 2.0
            y = (y.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), info.max))
            y = y.astype(np.float32) / scale
    else:
        y = y.astype(np.float32, copy=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr


def empty_embedding_info(status: str) -> Dict[str, str]:
    return {
        "vec_path": "",
        "vec_shape": "",
        "seq_path": "",
        "seq_shape": "",
        "status": status,
    }


def prepare_segment_audio(path: Path, start_s: float, end_s: float) -> Tuple[Optional[np.ndarray], str]:
    try:
        y, sr = load_audio(path)
        seg = slice_audio(y, sr, start_s, end_s)
        seg = resample_to_16k(seg, sr)
    except AssertionError as exc:
        if "Empty segment" in str(exc):
            return None, "EMPTY_AUDIO"
        return None, "AUDIO_SEGMENT_INVALID"
    except (RuntimeError, OSError, ValueError) as exc:
        if "Failed to decode audio" in str(exc):
            return None, "AUDIO_DECODE_FAILED"
        return None, "AUDIO_LOAD_FAILED"

    if seg.size == 0:
        return None, "EMPTY_AUDIO"
    return seg, "OK"


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = int(round(start_s * sr))
    end = int(round(end_s * sr))
    assert end > start, f"Invalid slice: start={start_s}, end={end_s}"
    assert start >= 0, f"Negative start index for slice: {start_s}"
    assert end <= len(y), f"Slice end exceeds audio length: {end} > {len(y)}"
    seg = y[start:end]
    assert seg.size > 0, f"Empty segment for slice: {start_s}-{end_s}"
    return seg.astype(np.float32, copy=False)


def resample_to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return y.astype(np.float32, copy=False)
    gcd = np.gcd(sr, TARGET_SR)
    up = TARGET_SR // gcd
    down = sr // gcd
    return signal.resample_poly(y, up, down).astype(np.float32, copy=False)


def pad_audio_batch(batch_audio_np: List[np.ndarray]) -> np.ndarray:
    assert batch_audio_np, "Cannot pad empty batch"
    lengths = [int(a.shape[0]) for a in batch_audio_np]
    assert min(lengths) > 0, f"Found empty audio in batch: {lengths}"
    max_len = max(lengths)
    out = np.zeros((len(batch_audio_np), max_len), dtype=np.float32)
    for i, audio in enumerate(batch_audio_np):
        out[i, : audio.shape[0]] = audio.astype(np.float32, copy=False)
    return out


@dataclass
class SegmentMeta:
    pair_id: str
    merge_key: str
    role: str
    dyad_id: str
    src_wav: str
    start_s: float
    end_s: float


def build_embedding_paths(out_dir: Path, dyad_id: str, pair_id: str, merge_key: str, role: str, save_full_seq: bool):
    stem = safe_pair_id_filename(pair_id, role, merge_key)
    vec_path = out_dir / "embeds_vec" / role / dyad_id / f"{stem}.npy"
    seq_path = out_dir / "embeds_seq" / role / dyad_id / f"{stem}.npy" if save_full_seq else None
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    if seq_path is not None:
        seq_path.parent.mkdir(parents=True, exist_ok=True)
    return vec_path, seq_path


def write_npy_if_needed(path: Path, arr: np.ndarray):
    if not path.exists():
        np.save(path, arr)


def tensor_to_numpy(x: tf.Tensor) -> np.ndarray:
    return x.numpy().astype(np.float32, copy=False)



def choose_pooled_output(outputs: Dict[str, tf.Tensor], embedding_output_key: str) -> Tuple[str, tf.Tensor]:
    if embedding_output_key != "auto":
        assert embedding_output_key in outputs, (
            f"embedding_output_key={embedding_output_key!r} not found. "
            f"Available outputs: {sorted(outputs.keys())}"
        )
        return embedding_output_key, outputs[embedding_output_key]

    for key in ["embedding", "output", "embeddings", "default"]:
        if key in outputs and len(outputs[key].shape) == 2:
            return key, outputs[key]

    rank2 = [(k, v) for k, v in outputs.items() if len(v.shape) == 2]
    assert len(rank2) == 1, (
        "Could not auto-select pooled Trillsson output; pass --embedding_output_key explicitly. "
        f"Rank-2 candidates: {[k for k, _ in rank2]}; all outputs: {sorted(outputs.keys())}"
    )
    return rank2[0]

def choose_sequence_output(outputs: Dict[str, tf.Tensor], sequence_output_key: str) -> Optional[tf.Tensor]:
    if sequence_output_key:
        assert sequence_output_key in outputs, (
            f"sequence_output_key={sequence_output_key!r} not found. "
            f"Available outputs: {sorted(outputs.keys())}"
        )
        seq = outputs[sequence_output_key]
        assert len(seq.shape) == 3, (
            f"Expected sequence output {sequence_output_key!r} to be rank 3 [B,T,D], got shape {seq.shape}"
        )
        return seq

    candidates = [(k, v) for k, v in outputs.items() if len(v.shape) == 3]
    if not candidates:
        return None
    assert len(candidates) == 1, (
        "Multiple rank-3 sequence outputs found; pass --sequence_output_key explicitly: "
        f"{[k for k, _ in candidates]}"
    )
    return candidates[0][1]


def extract_batch_trillsson(
    model: tf.keras.layers.Layer,
    batch_audio_np: List[np.ndarray],
    embedding_output_key: str,
    save_full_seq: bool,
    sequence_output_key: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    batch = pad_audio_batch(batch_audio_np)
    outputs = model(tf.convert_to_tensor(batch, dtype=tf.float32))

    if isinstance(outputs, dict):
        resolved_key, pooled_tensor = choose_pooled_output(outputs, embedding_output_key)
        seq_tensor = choose_sequence_output(outputs, sequence_output_key) if save_full_seq else None
    else:
        assert embedding_output_key in ("", "auto", "default", "embedding", "output", "embeddings"), (
            "Model returned a tensor rather than a dict; use --embedding_output_key auto/default/embedding/output/embeddings or blank."
        )
        pooled_tensor = outputs
        seq_tensor = None

    assert len(pooled_tensor.shape) == 2, f"Expected pooled embedding [B,D], got shape {pooled_tensor.shape}"
    assert int(pooled_tensor.shape[0]) == len(batch_audio_np), (
        f"Batch size mismatch: output {pooled_tensor.shape[0]} vs input {len(batch_audio_np)}"
    )

    pooled = tensor_to_numpy(pooled_tensor)
    full_seq = tensor_to_numpy(seq_tensor) if seq_tensor is not None else None
    if save_full_seq:
        assert full_seq is not None, "Requested --save_full_seq, but the Trillsson SavedModel exposed no rank-3 output"
        assert full_seq.shape[0] == len(batch_audio_np), (
            f"Sequence batch size mismatch: output {full_seq.shape[0]} vs input {len(batch_audio_np)}"
        )
    return pooled, full_seq


def leading_output_fields() -> List[str]:
    return [
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
        "source_dataset_split",
        "split",
        "split_seed",
        "split_unit",
        "merge_key",
        "pair_id_old",
    ]


def output_fieldnames() -> List[str]:
    return leading_output_fields() + [
        "prompt_trillsson_vec_path",
        "prompt_trillsson_vec_shape",
        "prompt_trillsson_seq_path",
        "prompt_trillsson_seq_shape",
        "status_prompt_trillsson",
        "response_trillsson_vec_path",
        "response_trillsson_vec_shape",
        "response_trillsson_seq_path",
        "response_trillsson_seq_shape",
        "status_response_trillsson",
        "embedding_model_name",
        "trillsson_variant",
        "embedding_output_key",
        "sequence_output_key",
        "target_sr",
    ]


def make_output_row(row: Dict[str, str], prompt_info: Dict[str, str], response_info: Dict[str, str], args) -> Dict[str, Any]:
    out = {k: row.get(k, "") for k in leading_output_fields()}
    out.update(
        {
            "prompt_trillsson_vec_path": prompt_info["vec_path"],
            "prompt_trillsson_vec_shape": prompt_info["vec_shape"],
            "prompt_trillsson_seq_path": prompt_info["seq_path"],
            "prompt_trillsson_seq_shape": prompt_info["seq_shape"],
            "status_prompt_trillsson": prompt_info["status"],
            "response_trillsson_vec_path": response_info["vec_path"],
            "response_trillsson_vec_shape": response_info["vec_shape"],
            "response_trillsson_seq_path": response_info["seq_path"],
            "response_trillsson_seq_shape": response_info["seq_shape"],
            "status_response_trillsson": response_info["status"],
            "embedding_model_name": args.model_handle,
            "trillsson_variant": args.trillsson_variant,
            "embedding_output_key": args.embedding_output_key,
            "sequence_output_key": args.sequence_output_key,
            "target_sr": TARGET_SR,
        }
    )
    return out


def flush_batch(
    model: tf.keras.layers.Layer,
    batch_audio: List[np.ndarray],
    batch_meta: List[SegmentMeta],
    args,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    pooled, full_seq = extract_batch_trillsson(
        model=model,
        batch_audio_np=batch_audio,
        embedding_output_key=args.embedding_output_key,
        save_full_seq=args.save_full_seq,
        sequence_output_key=args.sequence_output_key,
    )

    saved: Dict[Tuple[str, str], Dict[str, str]] = {}

    for i, meta in enumerate(batch_meta):
        vec_path, seq_path = build_embedding_paths(
            out_dir=args.out_dir,
            dyad_id=meta.dyad_id,
            pair_id=meta.pair_id,
            merge_key=meta.merge_key,
            role=meta.role,
            save_full_seq=args.save_full_seq,
        )

        write_npy_if_needed(vec_path, pooled[i])
        if seq_path is not None:
            assert full_seq is not None
            write_npy_if_needed(seq_path, full_seq[i])

        saved[(meta.merge_key, meta.role)] = {
            "vec_path": str(vec_path),
            "vec_shape": str(tuple(pooled[i].shape)),
            "seq_path": str(seq_path) if seq_path is not None else "",
            "seq_shape": str(tuple(full_seq[i].shape)) if seq_path is not None and full_seq is not None else "",
            "status": "OK",
        }

    return saved


def process_table(
    args,
    model: tf.keras.layers.Layer,
    input_csv: Path,
    out_dir: Path,
    output_csv: Path,
    split_name: str,
):
    args.out_dir = out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for i, row in enumerate(reader) if (i % args.num_shards) == args.shard_idx]

    if args.limit > 0:
        rows = rows[: args.limit]

    batch_audio: List[np.ndarray] = []
    batch_meta: List[SegmentMeta] = []
    pending_rows: List[Dict[str, str]] = []

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=output_fieldnames())
        writer.writeheader()

        if not rows:
            print(f"No {split_name} rows selected for shard {args.shard_idx}/{args.num_shards}")
            return

        for row in tqdm(rows, desc=f"trillsson {split_name} shard {args.shard_idx}/{args.num_shards}", unit="row"):
            pair_id = row["pair_id"]
            dyad_id = row["dyad_id"]

            prompt_wav = Path(row["source_wav_path_prompt"]).resolve()
            response_wav = Path(row["source_wav_path_response"]).resolve()

            assert prompt_wav.exists(), f"Missing prompt wav: {prompt_wav}"
            assert response_wav.exists(), f"Missing response wav: {response_wav}"

            prompt_start = float(row["prompt_start_s"])
            prompt_end = float(row["prompt_end_s"])
            response_start = float(row["response_start_s"])
            response_end = float(row["response_end_s"])

            seg_p, prompt_status = prepare_segment_audio(prompt_wav, prompt_start, prompt_end)
            seg_r, response_status = prepare_segment_audio(response_wav, response_start, response_end)

            if seg_p is None or seg_r is None:
                prompt_info = empty_embedding_info(prompt_status if seg_p is None else "SKIPPED_PAIR_FAILURE")
                response_info = empty_embedding_info(response_status if seg_r is None else "SKIPPED_PAIR_FAILURE")
                writer.writerow(make_output_row(row, prompt_info, response_info, args))
                continue

            batch_audio.append(seg_p)
            batch_meta.append(
                SegmentMeta(
                    pair_id=pair_id,
                    merge_key=row["merge_key"],
                    role="prompt",
                    dyad_id=dyad_id,
                    src_wav=str(prompt_wav),
                    start_s=prompt_start,
                    end_s=prompt_end,
                )
            )

            batch_audio.append(seg_r)
            batch_meta.append(
                SegmentMeta(
                    pair_id=pair_id,
                    merge_key=row["merge_key"],
                    role="response",
                    dyad_id=dyad_id,
                    src_wav=str(response_wav),
                    start_s=response_start,
                    end_s=response_end,
                )
            )

            pending_rows.append(row)

            if len(batch_audio) >= args.batch_size:
                saved = flush_batch(model, batch_audio, batch_meta, args)

                for prow in pending_rows:
                    prompt_info = saved[(prow["merge_key"], "prompt")]
                    response_info = saved[(prow["merge_key"], "response")]
                    writer.writerow(make_output_row(prow, prompt_info, response_info, args))

                batch_audio.clear()
                batch_meta.clear()
                pending_rows.clear()

        if batch_audio:
            saved = flush_batch(model, batch_audio, batch_meta, args)
            for prow in pending_rows:
                prompt_info = saved[(prow["merge_key"], "prompt")]
                response_info = saved[(prow["merge_key"], "response")]
                writer.writerow(make_output_row(prow, prompt_info, response_info, args))


def main():
    args = parse_args()

    assert args.batch_size > 0, f"batch_size must be positive, got {args.batch_size}"
    model = hub.KerasLayer(args.model_handle, trainable=False)

    process_table(
        args=args,
        model=model,
        input_csv=args.input_train_csv,
        out_dir=args.out_train_dir,
        output_csv=args.output_train_csv,
        split_name="train",
    )
    process_table(
        args=args,
        model=model,
        input_csv=args.input_test_csv,
        out_dir=args.out_test_dir,
        output_csv=args.output_test_csv,
        split_name="test",
    )


if __name__ == "__main__":
    main()
