#!/bin/bash
#SBATCH --job-name=a4-baseline-beam4
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/wz3191/nyu-dsga1011-a4/part2
#SBATCH --output=logs/beam4-%j.out
#SBATCH --error=logs/beam4-%j.err

set -eo pipefail
source /share/apps/pyenv/py3.9/etc/profile.d/conda.sh
conda activate dsga1011-hw4-part2
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export SLURM_EXPORT_ENV=ALL
date
hostname
nvidia-smi -L
# Generate SQL only; score it separately on CPU after this job completes.
"$CONDA_PREFIX/bin/python" -u beam_eval.py generate
date
