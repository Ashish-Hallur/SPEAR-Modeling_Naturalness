#!/bin/bash
#SBATCH --job-name=llm_pair_vox
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-431
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/vox_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/vox_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate vox

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs
mkdir -p ./tmp/vox_shards ./logs

which python
python --version

python build_llm_pair_vox.py \
  --input_csv ./tmp/llm_candidate_manifest.csv \
  --output_csv ./tmp/vox_shards/llm_pair_vox_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --vox_release_dir /home/ahallur1/spear/SPEAR-Modeling_Naturalness/vox-profile-release \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 432
