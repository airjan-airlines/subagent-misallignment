#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase_b_%j.out
# T2 (content control) and T6 (two-agent pilot) on Qwen3-1.7B. Submit from ~/ccsteer: sbatch slurm/phase_b_pilot.sbatch
set -e
cd ~/ccsteer
. venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
python scripts/run_content_control.py --model Qwen/Qwen3-1.7B --ff results/e1_qwen3-1.7b/ff_vectors.pt \
    --ab results/e0_qwen3-1.7b/vectors.pt --layers 14 16 18 20 --out results/t2_qwen3-1.7b
python scripts/run_two_agent.py --model Qwen/Qwen3-1.7B --ff results/e1_qwen3-1.7b/ff_vectors.pt \
    --ab results/e0_qwen3-1.7b/vectors.pt --steer-layer 16 --steer-coef 0.5 --read-layers 14 16 18 \
    --out results/t6_qwen3-1.7b
