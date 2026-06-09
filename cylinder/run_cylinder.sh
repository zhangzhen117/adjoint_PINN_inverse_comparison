#!/bin/bash
# Consolidated production run of the whole cylinder pipeline: executes
# run_cylinder.ipynb headless (forward solve -> obs -> gradient check ->
# adjoint inversion -> forward PINN -> inverse PINN + all history plots).
# Solver is numpy/scipy/gmsh, PINN is torch -- both live in env opt_2nd, so one
# kernel runs everything. GPU node.
#   sbatch run_cylinder.sh

#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH -t 04:00:00
#SBATCH --output run_cylinder_%j.log
#SBATCH -e run_cylinder_%j.err
#SBATCH -J cyl_run

set -e
echo "Job started on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opt_2nd
which python; python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /oscar/data/gk/zzhan536/playing_center/adjoint_test/cylinder
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 run_cylinder.ipynb
echo "Job finished at $(date)"
