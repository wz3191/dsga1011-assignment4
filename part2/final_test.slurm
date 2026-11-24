#!/bin/bash
#SBATCH --job-name=a4-final-test
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/wz3191/nyu-dsga1011-a4/part2
#SBATCH --output=logs/final-test-%j.out
#SBATCH --error=logs/final-test-%j.err

set -eo pipefail
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate dsga1011-hw4-part2

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export SLURM_EXPORT_ENV=ALL

date
hostname
nvidia-smi -L

# Generate test SQL using the selected checkpoint and decoding settings.
"$CONDA_PREFIX/bin/python" -u final_test_generate.py generate \
    --checkpoint runs/lr1e4_36295/checkpoints/ft_experiments/lr1e4/best \
    --beams 4 \
    --batch_size 8 \
    --max_new_tokens 512 \
    --output results/t5_ft_experiment_test.sql

date
