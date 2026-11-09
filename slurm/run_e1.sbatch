#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/e1_%j.out
# E1 (free-form steering) on one A6000. Submit from ~/ccsteer: sbatch slurm/run_e1.sbatch [model] [outdir]
set -e
cd ~/ccsteer
. venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
python scripts/run_e1.py --model "${1:-Qwen/Qwen3-1.7B}" --e0 results/e0_qwen3-1.7b --out "${2:-results/e1_qwen3-1.7b}"
