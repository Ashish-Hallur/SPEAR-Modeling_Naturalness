#!/bin/bash
#SBATCH --job-name=spear_pair_trillsson
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-430
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Trillsson_Embeddings/logs/pair_trillsson_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Trillsson_Embeddings/logs/pair_trillsson_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

module purge
module load conda
conda deactivate
conda activate /home/ahallur1/miniconda3/envs/trillsson

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Trillsson_Embeddings
mkdir -p ./tmp/pair_trillsson/train/shards ./tmp/pair_trillsson/test/shards ./logs

python extract_pair_trillsson_embeddings.py \
  --input_train_csv /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/tmp/pair_final_dataset_train.csv \
  --input_test_csv /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/tmp/pair_final_dataset_test.csv \
  --out_train_dir ./tmp/pair_trillsson/train \
  --out_test_dir ./tmp/pair_trillsson/test \
  --output_train_csv ./tmp/pair_trillsson/train/shards/pair_trillsson_train_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --output_test_csv ./tmp/pair_trillsson/test/shards/pair_trillsson_test_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --model_handle https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson4/1 \
  --trillsson_variant trillsson4 \
  --embedding_output_key auto \
  --batch_size 16 \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 431
