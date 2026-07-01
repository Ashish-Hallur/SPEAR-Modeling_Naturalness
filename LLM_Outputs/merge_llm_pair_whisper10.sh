#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs
mkdir -p ./tmp

python ../Whisper_Embeddings/merge_pair_embedding_shards.py \
  --shards_dir ./tmp/llm_pair_whisper10/shards \
  --output_csv ./tmp/llm_pair_whisper10.csv \
  --expected_shards 431
