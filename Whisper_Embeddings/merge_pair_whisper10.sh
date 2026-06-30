#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Whisper_Embeddings
mkdir -p ./tmp

python merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_whisper10/train/shards \
  --output_csv ./tmp/pair_whisper10_train.csv \
  --expected_shards 431

python merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_whisper10/test/shards \
  --output_csv ./tmp/pair_whisper10_test.csv \
  --expected_shards 431
