"""Shared argument handling for the bundle drivers."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Sequence


def main(bundle: str, rows: Sequence[dict], run_one: Callable[[dict, int], dict],
         results_path: str | None = None) -> int:
    """Standard ``--count`` / ``--list`` / ``--row N`` entry point.

    ``run_one(row, index)`` performs the run and returns the record to append.
    Rows are addressed positionally, so ``rows()`` must be deterministic -- an array
    task and the analysis stage have to agree on what row 7 was.
    """
    from common.sweep import append_record

    results_path = results_path or f"results/{bundle}.jsonl"

    ap = argparse.ArgumentParser(prog=f"sweeps.bundle_{bundle}")
    ap.add_argument("--row", type=int, help="index of the configuration to run")
    ap.add_argument("--count", action="store_true", help="print the number of rows")
    ap.add_argument("--list", action="store_true", help="print every row as JSON")
    ap.add_argument("--out", default=results_path)
    args = ap.parse_args()

    if args.count:
        print(len(rows))
        return 0

    if args.list:
        for i, r in enumerate(rows):
            print(f"{i:4d}  {json.dumps(r, sort_keys=True)}")
        return 0

    if args.row is None:
        ap.error("one of --row, --count or --list is required")

    if not (0 <= args.row < len(rows)):
        # A wider --array than the row count is harmless; say so and exit clean so
        # the surplus tasks do not show up as failures.
        print(f"row {args.row} out of range (bundle {bundle} has {len(rows)} rows); "
              f"nothing to do")
        return 0

    row = dict(rows[args.row])
    print(f"=== bundle {bundle} row {args.row}/{len(rows) - 1}: "
          f"{json.dumps(row, sort_keys=True)}", flush=True)

    record = run_one(row, args.row)
    append_record(args.out, record)
    print(f"=== appended to {args.out}", flush=True)

    interesting = ("eps_f", "eps_u", "runtime_s", "converged", "stop_reason",
                   "n_fev", "n_iter", "n_params")
    print("=== " + "  ".join(f"{k}={record.get(k)}" for k in interesting if k in record))
    return 0


def cli(bundle: str, rows_fn: Callable[[], Sequence[dict]],
        run_one: Callable[[dict, int], dict]) -> None:
    sys.exit(main(bundle, list(rows_fn()), run_one))
