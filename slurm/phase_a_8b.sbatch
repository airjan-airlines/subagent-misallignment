#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase_a_%j.out
# Phase A on Qwen3-8B: E0-style A/B direction, then T1 + T3 (free-form direction + steering sweep), then T2.
# Submit from ~/ccsteer: sbatch slurm/phase_a_8b.sbatch
set -e
cd ~/ccsteer
. venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
LAYERS="16 18 20 22 24 26"
python scripts/run_initial.py --model Qwen/Qwen3-8B --out results/e0_qwen3-8b
python scripts/run_e1.py --model Qwen/Qwen3-8B --e0 results/e0_qwen3-8b --out results/t1_qwen3-8b \
    --layers $LAYERS --coefs -1 -0.5 -0.25 0.25 0.5 0.75
python scripts/run_content_control.py --model Qwen/Qwen3-8B --ff results/t1_qwen3-8b/ff_vectors.pt \
    --ab results/e0_qwen3-8b/vectors.pt --layers $LAYERS --out results/t2_qwen3-8b
