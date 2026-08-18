#!/bin/bash
#SBATCH -A gk-l40s-gcondo
#SBATCH -p l40s-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH --mem=48g
#SBATCH -t 02:00:00
#SBATCH -J darcyfield
#SBATCH --output=slurm/logs/%x_%j.log

# The login profile activates a .venv that shadows conda on PATH; drop it first,
# otherwise `python` resolves to a 3.9 interpreter with neither scipy nor torch.
unset VIRTUAL_ENV || true
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opt_2nd
PY="${CONDA_PREFIX}/bin/python"
cd "$SLURM_SUBMIT_DIR"
echo "=== $(date) $(hostname) | $("${PY}" -V 2>&1) ==="
"${PY}" -c "import scipy, torch; print('scipy', scipy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
PYTHONPATH="$SLURM_SUBMIT_DIR" "${PY}" analysis/darcy_field_runs.py
echo "=== done $(date) ==="
