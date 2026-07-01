#!/bin/bash
#SBATCH --job-name=llm_candidate_manifest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/manifest_%j.out
#SBATCH --error=/home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs/logs/manifest_%j.err
#SBATCH --mail-user="ahallur1@jh.edu"
#SBATCH --mail-type=ALL

set -euo pipefail

source /home/ahallur1/miniconda3/etc/profile.d/conda.sh
conda activate /home/ahallur1/.conda/envs/py310

cd /home/ahallur1/spear/SPEAR-Modeling_Naturalness/LLM_Outputs
mkdir -p ./tmp ./logs

which python
python --version

python build_llm_candidate_manifest.py \
  --data_root /home/tthebau1/SPEAR/SPEARBench/benchmark/data/seamless_2t_2s_questions \
  --output_csv ./tmp/llm_candidate_manifest.csv \
  --status_csv ./tmp/llm_candidate_status.csv
