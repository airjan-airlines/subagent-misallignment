#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/e0_%j.out
# Initial experiment E0 on one A6000. Submit from ~/ccsteer: sbatch slurm/run_initial.sbatch [model] [outdir]
set -e
cd ~/ccsteer
. venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
python scripts/run_initial.py --model "${1:-Qwen/Qwen3-1.7B}" --out "${2:-results/e0_qwen3-1.7b}"
