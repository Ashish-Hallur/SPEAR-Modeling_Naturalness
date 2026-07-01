#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs
mkdir -p ./tmp

python ../Whisper_Embeddings/merge_pair_embedding_shards.py \
  --shards_dir ./tmp/llm_pair_trillsson/shards \
  --output_csv ./tmp/llm_pair_trillsson.csv \
  --expected_shards 431
