#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural
mkdir -p ./tmp

python ../Tabular_Features/combine_shards.py \
  --input_dir ./tmp/metrics_shards \
  --pattern 'pair_metrics_shard_*.csv' \
  --output_csv ./tmp/pair_metrics.csv \
  --expected_shards 432

python ../Tabular_Features/combine_shards.py \
  --input_dir ./tmp/vox_shards \
  --pattern 'pair_vox_shard_*.csv' \
  --output_csv ./tmp/pair_vox.csv \
  --expected_shards 432

python merge_emotivoice_pair_datasets.py \
  --metrics_csv ./tmp/pair_metrics.csv \
  --vox_csv ./tmp/pair_vox.csv \
  --output_csv ./tmp/pair_final_dataset.csv
