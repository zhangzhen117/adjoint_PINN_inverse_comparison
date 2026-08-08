"""Shared machinery for the ablation sweeps.

The four benchmarks each keep their own solver, adjoint, PINN and config, but the
revision requires them to emit *comparable* records: same seed handling, same cost
counters, same result schema. That common layer lives here.

- :mod:`common.seeding`     -- one ``set_seed`` used by every entry point.
- :mod:`common.instrument`  -- cost and memory counters (referee R3.2).
- :mod:`common.sweep`       -- the ``results/*.jsonl`` record schema and loader.
"""

from common.seeding import set_seed
from common.instrument import (Counters, RunTimer, hardware_info, require_l40s,
                               ssbroyden_hessian_bytes, trajectory_bytes)
from common.sweep import RESULT_FIELDS, append_record, load_records, make_record

__all__ = [
    "set_seed",
    "Counters",
    "RunTimer",
    "hardware_info",
    "require_l40s",
    "ssbroyden_hessian_bytes",
    "trajectory_bytes",
    "RESULT_FIELDS",
    "append_record",
    "load_records",
    "make_record",
]
