"""Ablation sweep drivers, one module per bundle.

Each module exposes ``rows()`` -- the full list of configurations, in a fixed order
-- and a ``--row N`` entry point that runs exactly one of them and appends a single
record to ``results/<bundle>.jsonl``. One SLURM array task per row, so peak-memory
counters measure a single run and a crashed configuration cannot take the sweep
with it.

    python -m sweeps.bundle_A --count      # how many rows (for --array=0-N)
    python -m sweeps.bundle_A --list       # show them
    python -m sweeps.bundle_A --row 3      # run one
"""
