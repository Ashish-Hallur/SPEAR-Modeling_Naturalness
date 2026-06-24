#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import torchaudio
except Exception as e:
    raise RuntimeError("torchaudio is required for resampling") from e


TARGET_SR = 16000


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract Whisper encoder embeddings for prompt/response segments from the final pair dataset."
    )
    ap.add_argument("--input_train_csv", type=Path, required=True, help="Train final merged pair dataset CSV.")
    ap.add_argument("--input_test_csv", type=Path, required=True, help="Test final merged pair dataset CSV.")
    ap.add_argument("--out_train_dir", type=Path, required=True, help="Train output root dir.")
    ap.add_argument("--out_test_dir", type=Path, required=True, help="Test output root dir.")
    ap.add_argument("--output_train_csv", type=Path, required=True, help="Train shard output CSV path.")
    ap.add_argument("--output_test_csv", type=Path, required=True, help="Test shard output CSV path.")

    ap.add_argument("--vox_release_dir", type=Path, required=True,
                    help="Path to vox-profile-release root so WhisperWrapper can be imported.")
    ap.add_argument("--model_name", type=str, default="tiantiaf/whisper-large-v3-msp-podcast-emotion-dim")
    ap.add_argument("--hidden_state_index", type=int, default=10,
                    help="Hidden-state index to extract. Default=10.")
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--save_full_seq", action="store_true",
                    help="Also save full [T,D] sequence for the selected hidden state.")
    ap.add_argument("--amp", action="store_true", help="Use autocast on CUDA.")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)

    return ap.parse_args()


def short_hash(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def safe_pair_id_filename(pair_id: str, role: str) -> str:
    safe = pair_id.replace("|", "__").replace("/", "_").replace("\\", "_")
    return f"{safe}__{role}"


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(str(path), "r") as f:
        sr = f.samplerate
        y = f.read(dtype="float32", always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    return y, sr


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
    x = torch.from_numpy(y).float().unsqueeze(0)
    x = torchaudio.functional.resample(x, sr, TARGET_SR)
    return x.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def pool_sequence(x_btd: torch.Tensor, how: str) -> torch.Tensor:
    if how == "mean":
        return x_btd.mean(dim=1)
    if how == "max":
        return x_btd.max(dim=1).values
    raise ValueError(f"Unsupported pool: {how}")


def ensure_whisper_expected_mel_len(model: torch.nn.Module, input_features: torch.Tensor) -> torch.Tensor:
    max_src = getattr(getattr(model, "backbone_model", model).config, "max_source_positions", None)
    if max_src is None:
        return input_features
    expected = int(max_src) * 2
    cur = int(input_features.shape[-1])
    if cur == expected:
        return input_features
    if cur < expected:
        return F.pad(input_features, (0, expected - cur))
    return input_features[..., :expected]


@dataclass
class SegmentMeta:
    pair_id: str
    role: str
    dyad_id: str
    src_wav: str
    start_s: float
    end_s: float


def build_embedding_paths(out_dir: Path, dyad_id: str, pair_id: str, role: str, save_full_seq: bool):
    stem = safe_pair_id_filename(pair_id, role)
    vec_path = out_dir / "embeds_vec" / role / dyad_id / f"{stem}.npy"
    seq_path = out_dir / "embeds_seq" / role / dyad_id / f"{stem}.npy" if save_full_seq else None
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    if seq_path is not None:
        seq_path.parent.mkdir(parents=True, exist_ok=True)
    return vec_path, seq_path


def write_npy_if_needed(path: Path, arr: np.ndarray):
    if not path.exists():
        np.save(path, arr)


def extract_batch_hidden(
    model: torch.nn.Module,
    device: torch.device,
    batch_audio_np: List[np.ndarray],
    hidden_state_index: int,
    pool: str,
    amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = model.feature_extractor(
        [a.astype(np.float32, copy=False) for a in batch_audio_np],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )
    input_features = feats.input_features.to(device)
    input_features = ensure_whisper_expected_mel_len(model, input_features)

    if hasattr(model, "_embed_positions_750") and not getattr(model, "_did_set_embedpos", False):
        model.backbone_model.encoder.embed_positions = torch.nn.Embedding.from_pretrained(
            model._embed_positions_750.to(device), freeze=True
        )
        model._did_set_embedpos = True

    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                enc = model.backbone_model.encoder(input_features, output_hidden_states=True)
        else:
            enc = model.backbone_model.encoder(input_features, output_hidden_states=True)

    hs = enc.hidden_states
    assert hs is not None, "Model did not return hidden states"
    assert -len(hs) <= hidden_state_index < len(hs), (
        f"hidden_state_index={hidden_state_index} out of range for {len(hs)} hidden states"
    )

    chosen = hs[hidden_state_index].detach().float()
    pooled = pool_sequence(chosen, pool).cpu().numpy().astype(np.float32)
    full_seq = chosen.cpu().numpy().astype(np.float32)
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
        "prompt_emb_vec_path",
        "prompt_emb_vec_shape",
        "prompt_emb_seq_path",
        "prompt_emb_seq_shape",
        "response_emb_vec_path",
        "response_emb_vec_shape",
        "response_emb_seq_path",
        "response_emb_seq_shape",
        "embedding_model_name",
        "whisper_hidden_state_index",
        "embedding_pool",
        "target_sr",
    ]


def make_output_row(row: Dict[str, str], prompt_info: Dict[str, str], response_info: Dict[str, str], args) -> Dict[str, Any]:
    out = {k: row.get(k, "") for k in leading_output_fields()}
    out.update(
        {
            "prompt_emb_vec_path": prompt_info["vec_path"],
            "prompt_emb_vec_shape": prompt_info["vec_shape"],
            "prompt_emb_seq_path": prompt_info["seq_path"],
            "prompt_emb_seq_shape": prompt_info["seq_shape"],
            "response_emb_vec_path": response_info["vec_path"],
            "response_emb_vec_shape": response_info["vec_shape"],
            "response_emb_seq_path": response_info["seq_path"],
            "response_emb_seq_shape": response_info["seq_shape"],
            "embedding_model_name": args.model_name,
            "whisper_hidden_state_index": args.hidden_state_index,
            "embedding_pool": args.pool,
            "target_sr": TARGET_SR,
        }
    )
    return out


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_audio: List[np.ndarray],
    batch_meta: List[SegmentMeta],
    args,
) -> Dict[Tuple[str, str], Dict[str, str]]:
    pooled, full_seq = extract_batch_hidden(
        model=model,
        device=device,
        batch_audio_np=batch_audio,
        hidden_state_index=args.hidden_state_index,
        pool=args.pool,
        amp=args.amp,
    )

    saved: Dict[Tuple[str, str], Dict[str, str]] = {}

    for i, meta in enumerate(batch_meta):
        vec_path, seq_path = build_embedding_paths(
            out_dir=args.out_dir,
            dyad_id=meta.dyad_id,
            pair_id=meta.pair_id,
            role=meta.role,
            save_full_seq=args.save_full_seq,
        )

        write_npy_if_needed(vec_path, pooled[i])
        if seq_path is not None:
            write_npy_if_needed(seq_path, full_seq[i])

        saved[(meta.pair_id, meta.role)] = {
            "vec_path": str(vec_path),
            "vec_shape": str(tuple(pooled[i].shape)),
            "seq_path": str(seq_path) if seq_path is not None else "",
            "seq_shape": str(tuple(full_seq[i].shape)) if seq_path is not None else "",
        }

    return saved


def process_table(
    args,
    model: torch.nn.Module,
    device: torch.device,
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

        for row in tqdm(rows, desc=f"whisper {split_name} shard {args.shard_idx}/{args.num_shards}", unit="row"):
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

            y_p, sr_p = load_audio(prompt_wav)
            y_r, sr_r = load_audio(response_wav)

            seg_p = slice_audio(y_p, sr_p, prompt_start, prompt_end)
            seg_r = slice_audio(y_r, sr_r, response_start, response_end)

            seg_p = resample_to_16k(seg_p, sr_p)
            seg_r = resample_to_16k(seg_r, sr_r)

            batch_audio.append(seg_p)
            batch_meta.append(
                SegmentMeta(
                    pair_id=pair_id,
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
                    role="response",
                    dyad_id=dyad_id,
                    src_wav=str(response_wav),
                    start_s=response_start,
                    end_s=response_end,
                )
            )

            pending_rows.append(row)

            if len(batch_audio) >= args.batch_size:
                saved = flush_batch(model, device, batch_audio, batch_meta, args)

                for prow in pending_rows:
                    prompt_info = saved[(prow["pair_id"], "prompt")]
                    response_info = saved[(prow["pair_id"], "response")]
                    writer.writerow(make_output_row(prow, prompt_info, response_info, args))

                batch_audio.clear()
                batch_meta.clear()
                pending_rows.clear()

        if batch_audio:
            saved = flush_batch(model, device, batch_audio, batch_meta, args)
            for prow in pending_rows:
                prompt_info = saved[(prow["pair_id"], "prompt")]
                response_info = saved[(prow["pair_id"], "response")]
                writer.writerow(make_output_row(prow, prompt_info, response_info, args))


def main():
    args = parse_args()

    sys.path.append(str(args.vox_release_dir))
    from src.model.emotion.whisper_emotion_dim import WhisperWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperWrapper.from_pretrained(args.model_name).to(device)
    model.eval()

    process_table(
        args=args,
        model=model,
        device=device,
        input_csv=args.input_train_csv,
        out_dir=args.out_train_dir,
        output_csv=args.output_train_csv,
        split_name="train",
    )
    process_table(
        args=args,
        model=model,
        device=device,
        input_csv=args.input_test_csv,
        out_dir=args.out_test_dir,
        output_csv=args.output_test_csv,
        split_name="test",
    )


if __name__ == "__main__":
    main()