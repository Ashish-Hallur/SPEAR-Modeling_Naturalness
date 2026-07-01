#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_VOX_RELEASE_DIR = Path(__file__).resolve().parents[1] / "vox-profile-release"
WHISPER_DIR = Path(__file__).resolve().parents[1] / "Whisper_Embeddings"
LLM_LEADING_FIELDS = [
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
    "latency_s",
    "overlap_s",
    "finish_reason",
    "answer_start_time",
    "prompt_transcript",
    "response_transcript",
    "response_metadata_transcript",
    "response_asr_transcript",
    "prompt_word_count",
    "response_word_count",
    "relationship",
    "relationship_detail",
    "source_dataset_split",
    "split",
    "source_style",
    "split_seed",
    "split_unit",
    "conversation_id",
    "speakers",
    "prompt_stem",
    "response_stem",
    "status_pair",
    "status_prompt_audio",
    "status_response_audio",
    "status_prompt_source",
    "status_response_source",
    "merge_key",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Extract Whisper encoder embeddings for SPEARBench LLM candidates.")
    ap.add_argument("--input_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--vox_release_dir", type=Path, default=DEFAULT_VOX_RELEASE_DIR)
    ap.add_argument("--model_name", type=str, default="tiantiaf/whisper-large-v3-msp-podcast-emotion-dim")
    ap.add_argument("--hidden_state_index", type=int, default=10)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument("--save_full_seq", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(WHISPER_DIR))
    import torch
    import extract_pair_whisper10_embeddings as base

    base.leading_output_fields = lambda: LLM_LEADING_FIELDS

    sys.path.append(str(args.vox_release_dir.resolve()))
    from src.model.emotion.whisper_emotion_dim import WhisperWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperWrapper.from_pretrained(args.model_name).to(device)
    model.eval()

    base.process_table(
        args=args,
        model=model,
        device=device,
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        output_csv=args.output_csv,
        split_name="llm_outputs",
    )


if __name__ == "__main__":
    main()
