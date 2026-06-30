#!/bin/bash
#SBATCH --job-name=spear_pair_metrics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-431
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/logs/metrics_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/logs/metrics_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate /home/ahallur1/.conda/envs/py310

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features
mkdir -p ./tmp/metrics_shards/train ./tmp/metrics_shards/test ./logs

which python
python --version

python build_pair_metrics.py \
  --dyad_lookup_csv /export/fs06/corpora8/seamless_interaction/datasets/assets/dyad_lookup.csv \
  --relationships_csv /export/fs06/corpora8/seamless_interaction/datasets/assets/relationships.csv \
  --output_train_csv ./tmp/metrics_shards/train/pair_metrics_train_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --output_test_csv ./tmp/metrics_shards/test/pair_metrics_test_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 432
