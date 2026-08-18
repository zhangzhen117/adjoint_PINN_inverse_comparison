#!/bin/bash
# SLURM array driver for the ablation sweeps.
#
# One array task runs exactly one (configuration x seed) row of one bundle, so the
# peak-memory counters in common/instrument.py measure that run alone rather than a
# high-water mark shared with other configurations in the same process.
#
# Usage:
#   sbatch --array=0-19 slurm/run_sweep.sh A          # bundle A, 20 rows
#   PARTITION=3090 sbatch --array=0-74 slurm/run_sweep.sh E
#
# The row count must match `--count` on the same module; submitting a wider
# array is harmless (extra tasks exit 0 with "row out of range"), submitting a
# narrower one silently drops rows, so always read the count first.
#
# HARDWARE IS PINNED TO L40S AND MUST STAY THAT WAY. Every wall-clock number in the
# paper was measured on "a single compute node equipped with an NVIDIA L40S GPU and a
# dual-socket AMD EPYC 9554 CPU", and referee R3.2 is specifically about normalizing
# cost -- so a run that lands on a 3090 or an A6000 would silently corrupt the cost
# table. common.instrument.require_l40s() aborts the run if the GPU is not an L40S.
# The l40s-gcondo partition has 15 nodes x 8 GPUs, so pinning costs no throughput.
#
# CPU contention matters here as much as the GPU: the adjoint inversions are
# CPU-bound (dense BLAS in Burgers, sparse FEM in Darcy and cylinder), and l40s nodes
# are shared. Requesting 16 cores per task makes SLURM's cgroup pin each task to its
# own cores, so eight concurrent array tasks on one node do not contend. Set
# EXCLUSIVE=1 to take whole nodes instead, for the headline timing runs.

#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96g
#SBATCH -t 08:00:00
#SBATCH --output slurm/logs/%x_%A_%a.log
#SBATCH -e slurm/logs/%x_%A_%a.err
#SBATCH -J ablate

set -euo pipefail

BUNDLE="${1:?usage: sbatch [--array=0-N] slurm/run_sweep.sh <bundle A|B|C|D|E>}"
REPO=/oscar/data/gk/zzhan536/playing_center/adjoint_test/adjoint_PINN_inverse_comparison

echo "=== bundle=${BUNDLE} task=${SLURM_ARRAY_TASK_ID:-0} host=$(hostname) start=$(date -Is)"

# The login profile activates a .venv that shadows conda on PATH; drop it before
# activating opt_2nd, otherwise `python` resolves to a 3.9 interpreter with no torch.
unset VIRTUAL_ENV || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opt_2nd
PY="${CONDA_PREFIX}/bin/python"

"${PY}" -V
"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# Refuse to produce timings on anything but an L40S (see the header note).
PYTHONPATH="${REPO}" "${PY}" -c "
from common.instrument import require_l40s
info = require_l40s()
print('hardware OK:', info['gpu_name'], '|', info['cpu_model'], '|', info['node'])
"
# SSBroyden needs the patched scipy (see README); fail loudly rather than silently
# degrading every run in the sweep to plain BFGS.
"${PY}" - <<'EOF'
import inspect, sys, scipy.optimize._optimize as o
if "method_bfgs" not in inspect.getsource(o):
    sys.exit("FATAL: scipy is not the patched build; SSBroyden2 would fall back to BFGS")
print("patched scipy: OK")
EOF

cd "${REPO}"
mkdir -p slurm/logs results
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
# Hold the BLAS thread count fixed at 8 regardless of the cgroup allocation, so that
# adjoint timings are comparable across bundles and against the numbers already in
# the paper. Do not tie this to SLURM_CPUS_PER_TASK.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# The two Appendix-B ablations live under ablations/, the rest under sweeps/.
case "${BUNDLE}" in
  B|B1) MODULE="ablations.b1_optimizer.bundle_${BUNDLE}" ;;
  F)    MODULE="ablations.b2_architecture.bundle_${BUNDLE}" ;;
  *)    MODULE="sweeps.bundle_${BUNDLE}" ;;
esac

"${PY}" -u -m "${MODULE}" --row "${SLURM_ARRAY_TASK_ID:-0}"

echo "=== done $(date -Is)"
