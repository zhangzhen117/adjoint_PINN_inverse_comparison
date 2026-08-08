"""Uniform result schema for the ablation sweeps.

Every run of every benchmark appends one JSON line to ``results/<bundle>.jsonl``.
All revision tables and figures are then generated from a single dataframe, which
is what makes it possible to assert -- in the verification step -- that each number
printed in the manuscript traces back to a specific recorded run. The Test 2 numbers
in the submitted version could not be reproduced from the committed history files,
and this schema is the mechanism that stops that recurring.

One line per run, never a summary: means and standard deviations across the five
seeds are computed at read time, so the raw spread is always recoverable.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, Iterable

# Canonical column order. Anything not listed here still round-trips (it is written
# verbatim), but these are the fields every row is expected to carry.
RESULT_FIELDS = (
    # --- identity of the run -------------------------------------------------
    "bundle",          # ablation bundle: A | B | C | D | E
    "benchmark",       # burgers | darcy | allencahn | cylinder
    "method",          # adjoint | pinn | eki
    "representation",  # nn | grid | kl | scalar
    "optimizer",       # ssbroyden2 | bfgs | lbfgsb | adam | soap
    "arch",            # e.g. "w32d3-tanh" or "w64d4-fourier"
    "seed",
    # --- the experimental axes ----------------------------------------------
    "noise",           # observation noise level (relative)
    "gamma",           # H1 / smoothness regularization weight
    "beta",            # Tikhonov weight
    # --- outcomes -------------------------------------------------------------
    "eps_f",           # relative L2 error of the recovered parameter field
    "eps_u",           # relative L2 error of the re-simulated state
    "converged",       # False when the run hit the wall-clock cap
    "stop_reason",     # "tolerance" | "maxiter" | "walltime_cap"
    # --- cost (referee R3.2) --------------------------------------------------
    "runtime_s",
    "forward_solves",
    "adjoint_sweeps",
    "pde_residual_evals",
    "n_fev",
    "n_iter",
    "n_params",
    "peak_host_bytes",
    "peak_device_bytes",
    "traj_bytes_analytic",
    "hessian_bytes_analytic",
    # --- provenance -------------------------------------------------------------
    "hostname",
    "node",
    "gpu_name",
    "cpu_model",
    "slurm_job_id",
    "git_commit",
    "cfg_snapshot",
)


def _provenance() -> dict[str, Any]:
    import socket
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip() or None
    except Exception:
        commit = None

    from common.instrument import hardware_info

    return {
        **hardware_info(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": commit,
    }


def make_record(**kwargs: Any) -> dict[str, Any]:
    """Build one result row, filling unknown canonical fields with ``None``.

    Provenance (hostname, SLURM job id, git commit) is attached automatically so a
    row can always be traced back to the job that produced it.
    """
    row: dict[str, Any] = {k: None for k in RESULT_FIELDS}
    row.update(_provenance())
    row.update(kwargs)
    return row


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays and dataclasses into JSON-serializable form."""
    import dataclasses

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy scalars and arrays, torch tensors
    for attr in ("item", "tolist"):
        if hasattr(obj, attr):
            try:
                return _jsonable(getattr(obj, attr)())
            except Exception:
                pass
    return str(obj)


def append_record(path: str, record: dict[str, Any]) -> None:
    """Append one record as a JSON line, creating parent directories as needed.

    Opened in append mode with a single ``write`` of one line, so concurrent SLURM
    array tasks writing to the same file do not interleave: on Linux, an ``O_APPEND``
    write below the pipe buffer size is atomic.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    line = json.dumps(_jsonable(record), sort_keys=False)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def load_records(pattern: str = "results/*.jsonl") -> "Any":
    """Load every matching ``*.jsonl`` into one pandas DataFrame.

    Rows with a malformed JSON line are skipped with a warning rather than aborting
    the whole aggregation, so one crashed array task cannot block the analysis.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for path in sorted(glob(pattern)):
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"warning: skipping {path}:{lineno}: {exc}")
    if not rows:
        return pd.DataFrame(columns=list(RESULT_FIELDS))

    df = pd.DataFrame(rows)
    ordered = [c for c in RESULT_FIELDS if c in df.columns]
    extra = [c for c in df.columns if c not in RESULT_FIELDS]
    return df[ordered + extra]


def summarize(df: "Any", by: Iterable[str], metrics: Iterable[str] = ("eps_f", "eps_u", "runtime_s")) -> "Any":
    """Group by the given columns and report mean, std and count for each metric.

    ``count`` is included deliberately: it makes an incomplete seed set visible in
    the table instead of silently averaging over three runs where five were claimed.
    """
    by = list(by)
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(by, dropna=False)[metrics].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
