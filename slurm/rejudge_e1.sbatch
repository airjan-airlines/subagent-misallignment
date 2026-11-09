#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rejudge_%j.out
# Re-score E1 with the v2 judge on one A6000. Submit from ~/ccsteer: sbatch slurm/rejudge_e1.sbatch
set -e
cd ~/ccsteer
. .venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
python scripts/rejudge_e1.py --e1 results/e1_qwen3-1.7b --labels data/judge_validation_e1.jsonl
