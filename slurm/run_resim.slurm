#!/bin/bash
#SBATCH --job-name=cylresim
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.log

REPO=/oscar/data/gk/zzhan536/playing_center/adjoint_test/adjoint_PINN_inverse_comparison
GRID="$REPO/cases/case4_cylinder/gridstudy"

# gmsh needs libGLU, which is present on the GPU nodes but not on batch.
module load mesa-glu/9.0.2-ikgf
# The repo venv is auto-activated by the login profile and shadows
# `conda activate`, so call the conda interpreter by path. scipy lives in
# opt_2nd; the venv has neither scipy nor a working matplotlib.
PY_EXE=~/miniconda3/envs/opt_2nd/bin/python
cd "$GRID"
echo "=== $(date) on $(hostname) | $($PY_EXE -c 'import sys;print(sys.executable)') ==="
$PY_EXE resim_eps_u.py
echo "=== done $(date) ==="
