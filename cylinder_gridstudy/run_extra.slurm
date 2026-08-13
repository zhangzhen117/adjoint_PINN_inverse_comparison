#!/bin/bash
#SBATCH --job-name=cylgrid2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.log

set -o pipefail
module purge
module load gcc/11.5.0-wfyo openmpi/4.1.8-iqkf
source /oscar/data/gk/zzhan536/OpenFOAM/OpenFOAM-v2512/etc/bashrc || exit 1

CASES=(L5_trans L3dt0.00125_trans)
declare -A RANKS=( [L5_trans]=32 [L3dt0.00125_trans]=16 )
NAME="${CASES[$SLURM_ARRAY_TASK_ID]}"
CASE="$SLURM_SUBMIT_DIR/cases/$NAME"
cd "$CASE" || exit 1
NP=${RANKS[$NAME]:-16}
cat > system/decomposeParDict <<EOD
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains $NP;
method          scotch;
EOD

echo "=== $NAME | pimpleFoam | np=$NP | $(hostname) | $(date) ==="
rm -rf processor* postProcessing constant/polyMesh
blockMesh 2>&1 | tee log.blockMesh | tail -2
checkMesh 2>&1 | tee log.checkMesh | grep -E 'cells:|non-orthogonality Max|Max skewness|Mesh OK'
decomposePar -force 2>&1 | tee log.decomposePar | tail -1
mpirun -np "$NP" pimpleFoam -parallel 2>&1 | tee log.pimpleFoam | tail -3
reconstructPar -latestTime 2>&1 | tee log.reconstructPar | tail -1
echo "=== done $NAME $(date) ==="
