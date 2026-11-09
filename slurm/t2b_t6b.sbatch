#!/bin/bash
#SBATCH --partition=ocf-hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/t2b_t6b_%j.out
# T6b (conductor decision check) and T2b (intent direction) on Qwen3-1.7B and Qwen3-8B.
# Submit from ~/ccsteer: sbatch slurm/t2b_t6b.sbatch
set -e
cd ~/ccsteer
. venv/bin/activate
export PYTHONPATH=src HF_HOME=~/hf_cache
python scripts/run_decision_check.py --model Qwen/Qwen3-8B --out results/t6b_qwen3-8b
python scripts/run_decision_check.py --model Qwen/Qwen3-1.7B --out results/t6b_qwen3-1.7b
python scripts/run_intent_direction.py --model Qwen/Qwen3-1.7B --t1 results/e1_qwen3-1.7b \
    --ab results/e0_qwen3-1.7b/vectors.pt --layers 14 16 18 20 --steer-layers 14 16 --out results/t2b_qwen3-1.7b
python scripts/run_intent_direction.py --model Qwen/Qwen3-8B --t1 results/t1_qwen3-8b \
    --ab results/e0_qwen3-8b/vectors.pt --layers 16 18 20 22 24 --steer-layers 16 18 --out results/t2b_qwen3-8b
