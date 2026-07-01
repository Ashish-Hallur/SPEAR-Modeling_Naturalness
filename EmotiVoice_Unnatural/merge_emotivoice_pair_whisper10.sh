#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural
mkdir -p ./tmp

python ../Whisper_Embeddings/merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_whisper10/shards \
  --output_csv ./tmp/pair_whisper10.csv \
  --expected_shards 431
