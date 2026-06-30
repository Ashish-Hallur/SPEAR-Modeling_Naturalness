#!/bin/bash
set -euo pipefail

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features
mkdir -p ./tmp

python combine_shards.py \
  --input_dir ./tmp/metrics_shards/train \
  --pattern 'pair_metrics_train_shard_*.csv' \
  --output_csv ./tmp/pair_metrics_train.csv \
  --expected_shards 432

python combine_shards.py \
  --input_dir ./tmp/metrics_shards/test \
  --pattern 'pair_metrics_test_shard_*.csv' \
  --output_csv ./tmp/pair_metrics_test.csv \
  --expected_shards 432

python combine_shards.py \
  --input_dir ./tmp/vox_shards/train \
  --pattern 'pair_vox_train_shard_*.csv' \
  --output_csv ./tmp/pair_vox_train.csv \
  --expected_shards 432

python combine_shards.py \
  --input_dir ./tmp/vox_shards/test \
  --pattern 'pair_vox_test_shard_*.csv' \
  --output_csv ./tmp/pair_vox_test.csv \
  --expected_shards 432

python merge_pair_datasets.py \
  --metrics_csv ./tmp/pair_metrics_train.csv \
  --vox_csv ./tmp/pair_vox_train.csv \
  --output_csv ./tmp/pair_final_dataset_train.csv

python merge_pair_datasets.py \
  --metrics_csv ./tmp/pair_metrics_test.csv \
  --vox_csv ./tmp/pair_vox_test.csv \
  --output_csv ./tmp/pair_final_dataset_test.csv
