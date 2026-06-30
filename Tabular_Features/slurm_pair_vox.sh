#!/bin/bash
#SBATCH --job-name=spear_pair_vox
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-431
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/logs/vox_%A_%a.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features/logs/vox_%A_%a.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

module purge
module load conda
conda deactivate
conda activate /home/ahallur1/miniconda3/envs/vox

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/Tabular_Features
mkdir -p ./tmp/vox_shards/train ./tmp/vox_shards/test ./logs

python build_pair_vox.py \
  --dyad_lookup_csv /export/fs06/corpora8/seamless_interaction/datasets/assets/dyad_lookup.csv \
  --relationships_csv /export/fs06/corpora8/seamless_interaction/datasets/assets/relationships.csv \
  --output_train_csv ./tmp/vox_shards/train/pair_vox_train_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --output_test_csv ./tmp/vox_shards/test/pair_vox_test_shard_${SLURM_ARRAY_TASK_ID}.csv \
  --shard_idx ${SLURM_ARRAY_TASK_ID} \
  --num_shards 432
