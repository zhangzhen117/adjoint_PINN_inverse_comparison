#!/bin/bash
# SLURM job: run the inverse-PINN noise sweep with per-run field comparisons.
# Example:
#   sbatch run_pinn_inv_noise_sweep.sh

#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH -t 12:00:00
#SBATCH --output pinn_inv_noise_sweep_%j.log
#SBATCH -e pinn_inv_noise_sweep_%j.err
#SBATCH -J cyl_inv_noise

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opt_2nd
which python
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /oscar/data/gk/zzhan536/playing_center/adjoint_test/cylinder
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

N_RUNS=${N_RUNS:-10}
SEED0=${SEED0:-42}
NOISE=${NOISE:-0.01}

python -u pinn_inv_noise_sweep.py \
  --n-runs "${N_RUNS}" \
  --seed0 "${SEED0}" \
  --noise "${NOISE}"

echo "Job finished at $(date)"