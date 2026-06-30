#!/bin/bash
#SBATCH --job-name=emotivoice_pair_trillsson
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-430
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural/logs/pair_trillsson_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural/logs/pair_trillsson_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate trillsson

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/EmotiVoice_Unnatural
mkdir -p ./tmp/pair_trillsson/shards ./tmp/pair_trillsson ./logs

which python
python --version

python extract_emotivoice_trillsson_embeddings.py \
  --input_csv ./tmp/pair_final_dataset.csv \
  --out_dir ./tmp/pair_trillsson \
  --output_csv ./tmp/pair_trillsson/shards/pair_trillsson_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --model_handle https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1 \
  --trillsson_variant trillsson4 \
  --embedding_output_key auto \
  --batch_size 16 \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 431
