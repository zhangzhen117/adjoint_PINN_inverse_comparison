#!/bin/bash
# Generic single-script SLURM launcher, for validations and one-off analyses.
#
#   sbatch slurm/run_python.sh path/to/script.py [args...]
#   sbatch -p gpu -A gk-condo slurm/run_python.sh ...     # override the partition
#
# Sweeps should use slurm/run_sweep.sh instead, which arrays over configurations.
#
# Pinned to l40s: every reported wall-clock number in the paper is L40S, so timing
# runs must not land on a 3090 or A6000. See run_sweep.sh for the full rationale.

#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -t 01:00:00
#SBATCH --output slurm/logs/%x_%j.log
#SBATCH -e slurm/logs/%x_%j.err
#SBATCH -J pyrun

set -euo pipefail

SCRIPT="${1:?usage: sbatch slurm/run_python.sh <script.py> [args...]}"
shift || true
REPO=/oscar/data/gk/zzhan536/playing_center/adjoint_test/adjoint_PINN_inverse_comparison

echo "=== $SCRIPT on $(hostname) at $(date -Is)"

# The login profile activates a .venv that shadows conda on PATH; drop it first.
unset VIRTUAL_ENV || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opt_2nd
PY="${CONDA_PREFIX}/bin/python"

cd "${REPO}"
mkdir -p slurm/logs results
export PYTHONPATH="${REPO}:${REPO}/Darcy_New:${PYTHONPATH:-}"
# Fixed BLAS threads so adjoint timings stay comparable (see run_sweep.sh).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

"${PY}" -u "${SCRIPT}" "$@"

echo "=== done $(date -Is)"
