#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural
mkdir -p ./tmp

python ../Trillsson_Embeddings/merge_pair_embedding_shards.py \
  --shards_dir ./tmp/pair_trillsson/shards \
  --output_csv ./tmp/pair_trillsson.csv \
  --expected_shards 431
