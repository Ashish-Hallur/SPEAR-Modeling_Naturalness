#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRILLSSON_DIR = Path(__file__).resolve().parents[1] / "Trillsson_Embeddings"
DEFAULT_MODEL_HANDLE = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1"
DEFAULT_OUTPUT_KEY = "auto"
EMOTIVOICE_LEADING_FIELDS = [
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
    "prompt_tts_model",
    "response_tts_model",
    "prompt_is_emotivoice",
    "response_is_emotivoice",
    "is_human_prompt_emotivoice_response",
    "emotivoice_row_dir",
    "prompt_track",
    "response_track",
    "source_dataset_split",
    "split",
    "split_seed",
    "split_unit",
    "merge_key",
    "pair_id_old",
]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract Trillsson embeddings for EmotiVoice external prompt/response pairs."
    )
    ap.add_argument("--input_csv", type=Path, required=True, help="Final merged EmotiVoice pair dataset CSV.")
    ap.add_argument("--out_dir", type=Path, required=True, help="Embedding output root dir.")
    ap.add_argument("--output_csv", type=Path, required=True, help="Shard output CSV path.")
    ap.add_argument("--model_handle", type=str, default=DEFAULT_MODEL_HANDLE)
    ap.add_argument("--trillsson_variant", type=str, default="trillsson4")
    ap.add_argument("--embedding_output_key", type=str, default=DEFAULT_OUTPUT_KEY)
    ap.add_argument("--sequence_output_key", type=str, default="")
    ap.add_argument("--save_full_seq", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_idx", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    assert args.batch_size > 0, f"batch_size must be positive, got {args.batch_size}"

    sys.path.insert(0, str(TRILLSSON_DIR))
    import tensorflow_hub as hub
    import extract_pair_trillsson_embeddings as base

    base.leading_output_fields = lambda: EMOTIVOICE_LEADING_FIELDS

    model = hub.KerasLayer(args.model_handle, trainable=False)
    base.process_table(
        args=args,
        model=model,
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        output_csv=args.output_csv,
        split_name="external",
    )


if __name__ == "__main__":
    main()
