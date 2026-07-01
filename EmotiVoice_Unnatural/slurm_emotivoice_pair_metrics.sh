#!/bin/bash
#SBATCH --job-name=emotivoice_pair_metrics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-431
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural/logs/metrics_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural/logs/metrics_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate /home/ahallur1/.conda/envs/py310

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural
mkdir -p ./tmp/metrics_shards ./logs

which python
python --version

python build_emotivoice_pair_metrics.py \
  --input_root /home/ahallur1/spear/NIPS_Experiments/emotivoice_full_v2 \
  --output_csv ./tmp/metrics_shards/pair_metrics_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 432
