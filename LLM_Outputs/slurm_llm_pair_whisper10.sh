#!/bin/bash
#SBATCH --job-name=llm_pair_whisper10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-430
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/pair_whisper10_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/pair_whisper10_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate vox

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs
mkdir -p ./tmp/llm_pair_whisper10/shards ./tmp/llm_pair_whisper10 ./logs

which python
python --version

python extract_llm_whisper10_embeddings.py \
  --input_csv ./tmp/llm_pair_final_dataset.csv \
  --out_dir ./tmp/llm_pair_whisper10 \
  --output_csv ./tmp/llm_pair_whisper10/shards/llm_pair_whisper10_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --vox_release_dir /home/ahallur1/spear/SPEAR-Modeling_Naturalness/vox-profile-release \
  --hidden_state_index 10 \
  --pool mean \
  --batch_size 8 \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 431 \
  --save_full_seq
