#!/bin/bash
#SBATCH --job-name=ofdata
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.log

REPO=/oscar/data/gk/zzhan536/playing_center/adjoint_test/adjoint_PINN_inverse_comparison
GRID="$REPO/cases/case4_cylinder/gridstudy"

# No `set -e`: the module command returns non-zero benignly and killed
# an earlier attempt after stage 1. Each stage is checked explicitly.
cd "$GRID"
PY_EXE=~/miniconda3/envs/opt_2nd/bin/python

echo "=== stage 1: probe locations ==="
$PY_EXE make_of_dataset.py --stage points || { echo "STAGE1 FAILED"; exit 1; }

module purge
module load gcc/11.5.0-wfyo openmpi/4.1.8-iqkf
source /oscar/data/gk/zzhan536/OpenFOAM/OpenFOAM-v2512/etc/bashrc

cd cases/L5_trans
echo "=== stage 2: reconstructPar t=80,85 ==="
reconstructPar -time '80,85' 2>&1 | tail -4
[ -d 80 ] || { echo "STAGE2 FAILED: no reconstructed t=80"; exit 1; }
echo "=== stage 3: sample U at the FEM nodes ==="
postProcess -func icProbes -time '80,85' 2>&1 | tail -6
[ -d postProcessing/icProbes ] || { echo "STAGE3 FAILED"; exit 1; }
cd "$GRID"

module purge
echo "=== stage 4: assemble npz ==="
$PY_EXE make_of_dataset.py --stage assemble
echo "=== done $(date) ==="
