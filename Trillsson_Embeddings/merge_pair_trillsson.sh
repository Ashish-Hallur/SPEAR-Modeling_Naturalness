#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Trillsson_Embeddings
mkdir -p ./tmp

python merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_trillsson/train/shards \
  --output_csv ./tmp/pair_trillsson_train.csv \
  --expected_shards 431

python merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_trillsson/test/shards \
  --output_csv ./tmp/pair_trillsson_test.csv \
  --expected_shards 431
