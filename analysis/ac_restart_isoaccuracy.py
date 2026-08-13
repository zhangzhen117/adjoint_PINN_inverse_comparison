"""Iso-accuracy cost of the Allen-Cahn PINN-warm-started restart.

Run to the same 200-iteration rule as the cold baseline, the restart costs about
the same wall-clock and ends far more accurate. That is one true statement, but it
is not the one the hybrid claim wants: a practitioner who only needs the accuracy
the cold start delivers can stop as soon as the restart reaches it.

So this measures the other axis. For each restart, find the first objective
evaluation whose reaction error is at or below the cold start's *final* mean, and
report the wall-clock spent up to that point. The result is the cost of reaching
cold-start accuracy, against the cold start's own cost of reaching it.

One caveat has to travel with the number. Stopping there requires knowing eps_f,
which requires the true reaction, so it is not a rule that could be applied during
an actual inversion -- the same objection that applies to the cylinder's best
iterate. It measures how much faster the warm start converges, not a stopping
criterion anyone could implement. Reported as such.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def traces(pattern, fname):
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "results", pattern, fname))):
        z = np.load(f, allow_pickle=True)
        ev = z["evals"]
        out.append((np.array([e["rel_l2_f"] for e in ev], float),
                    float(z["runtime_sec"])))
    return out


def main():
    cold = traces(os.path.join("D_runs", "ac_adj_s*"), "ac3d_adj.npz")
    warm = traces(os.path.join("H_runs", "ac_restart_s*"), "ac3d_adj_restart.npz")
    assert cold and warm, "missing runs"

    cold_final = np.array([e[-1] for e, _ in cold])
    target = cold_final.mean()
    t_cold = np.array([t for _, t in cold])
    print(f"cold start: eps_f {cold_final.mean():.4e} +- {cold_final.std(ddof=1):.2e}"
          f"   t {t_cold.mean():.0f} +- {t_cold.std(ddof=1):.0f} s")
    print(f"target for the iso-accuracy comparison: {target:.4e}\n")

    rows = []
    for i, (e, t) in enumerate(warm):
        per_eval = t / len(e)
        hit = np.where(e <= target)[0]
        if hit.size == 0:
            print(f"  restart s{i}: never reaches the target (best {e.min():.3e})")
            continue
        k = int(hit[0]) + 1
        rows.append((k, k * per_eval, e[-1], t, len(e)))
        print(f"  restart s{i}: reaches {target:.3e} at eval {k:3d}/{len(e)} "
              f"= {k*per_eval:6.0f} s   (final {e[-1]:.3e} at {t:.0f} s)")

    a = np.array(rows, float)
    k, ti, fin, tfull = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    print(f"\n  iso-accuracy cost : {ti.mean():.0f} +- {ti.std(ddof=1):.0f} s"
          f"   = {100*ti.mean()/t_cold.mean():.0f}% of the cold start")
    print(f"  evaluations       : {k.mean():.0f} +- {k.std(ddof=1):.0f}"
          f"   of {a[:,4].mean():.0f} run")
    print(f"  run to completion : {tfull.mean():.0f} +- {tfull.std(ddof=1):.0f} s"
          f"   = {100*tfull.mean()/t_cold.mean():.0f}% of the cold start,"
          f"  eps_f {fin.mean():.3e} +- {fin.std(ddof=1):.1e}")

    # including the PINN that supplied the warm start
    D = [json.loads(l) for l in open(os.path.join(REPO, "results", "D.jsonl"))
         if l.strip().startswith("{")]
    t_pinn = np.mean([r["runtime_s"] for r in D
                      if r["benchmark"] == "allencahn" and r["method"] == "pinn"])
    print(f"\n  PINN warm start   : {t_pinn:.0f} s")
    print(f"  hybrid, iso-acc   : {t_pinn + ti.mean():.0f} s"
          f"   = {100*(t_pinn+ti.mean())/t_cold.mean():.0f}% of the cold start")
    print(f"  hybrid, complete  : {t_pinn + tfull.mean():.0f} s"
          f"   = {100*(t_pinn+tfull.mean())/t_cold.mean():.0f}% of the cold start")
    np.savez(os.path.join(REPO, "cylinder_gridstudy", "ac_restart_isoaccuracy.npz"),
             target=target, k=k, t_iso=ti, t_full=tfull, eps_final=fin,
             t_cold=t_cold, t_pinn=t_pinn)


if __name__ == "__main__":
    main()
