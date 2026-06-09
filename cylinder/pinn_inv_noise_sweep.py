"""Run the cylinder inverse PINN across multiple observation-noise samples.

For each run this script:
1. Regenerates the probe observation with a distinct noise seed.
2. Trains the inverse PINN from the same deterministic optimization seed.
3. Uses the recovered viscosity in the FEM solver to march from the saved
   saturated IC to the terminal time.
4. Measures terminal-field errors for u, v, and the full velocity vector.
5. Saves per-run terminal comparison figures and rich result files.

Aggregate mean/std summaries are written in addition to the full per-run data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
"""Run the cylinder inverse PINN across multiple observation-noise samples.

For each run this script:
1. Regenerates the probe observation with a distinct noise seed.
2. Trains the inverse PINN from the same deterministic optimization seed.
3. Uses the recovered viscosity in the FEM solver to march from the saved
   saturated IC to the terminal time.
4. Measures terminal-field errors for u, v, and the full velocity vector.
5. Saves per-run terminal comparison figures and rich result files.

Aggregate mean/std summaries are written in addition to the full per-run data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
os.chdir(THIS_DIR)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cylinder_api import CylinderAPI
from cylinder_config import CylinderRunConfig
from cylinder_solver import create_solver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of inverse-PINN runs with distinct noise samples.")
    parser.add_argument("--seed0", type=int, default=42,
                        help="First observation-noise seed; run k uses seed0 + k.")
    parser.add_argument("--noise", type=float, default=None,
                        help="Observation-noise standard deviation. Defaults to CylinderRunConfig().obs_noise.")
    parser.add_argument("--output-dir", type=str, default="history/pinn_inv_noise_sweep",
                        help="Directory for per-run artifacts and summary tables.")
    parser.add_argument("--fig-dir", type=str, default="figures/pinn_inv_noise_sweep",
                        help="Directory for per-run figures.")
    return parser


def load_reference_terminal(cfg: CylinderRunConfig):
    mesh, _, solver = create_solver(cfg.solver_cfg("prod"))
    with np.load(cfg.saturated_path, mmap_mode="r") as z:
        times = np.array(z["period_times"] if "period_times" in z.files else z["times"])
        snaps = np.array(z["period_snaps"] if "period_snaps" in z.files else z["snaps"])
        u0 = np.array(z["u0"])

    idx_T = int(np.argmin(np.abs(times - cfg.T_obs)))
    t_ref = float(times[idx_T])
    u_ref = np.array(snaps[idx_T])
    tri = mtri.Triangulation(mesh.coords_p1[:, 0], mesh.coords_p1[:, 1], mesh.elem_p1)
    return mesh, solver, tri, u0, u_ref, t_ref


def mass_rel_l2(mass_matrix, approx: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    diff = approx - ref
    abs_sq = float(diff @ (mass_matrix @ diff))
    ref_sq = float(ref @ (mass_matrix @ ref))
    abs_l2 = float(np.sqrt(max(abs_sq, 0.0)))
    rel_l2 = float(abs_l2 / np.sqrt(max(ref_sq, 1e-300)))
    return abs_l2, rel_l2


def terminal_field_metrics(solver, u0: np.ndarray, u_ref: np.ndarray, nu: float,
                           T: float, dt: float) -> dict[str, object]:
    t0 = time.time()
    out = solver.solve_forward(T=T, nu=float(nu), u0=u0, dt=dt, ramp_T=0.0,
                               adaptive=False, save_dt=T, verbose=False)
    runtime = time.time() - t0
    u_T = np.array(out["u_final"])
    nps = solver.mesh.n_p2

    ux_ref = u_ref[:nps]
    uy_ref = u_ref[nps:2*nps]
    ux_T = u_T[:nps]
    uy_T = u_T[nps:2*nps]

    ux_abs, ux_rel = mass_rel_l2(solver.M, ux_T, ux_ref)
    uy_abs, uy_rel = mass_rel_l2(solver.M, uy_T, uy_ref)
    uv_abs, uv_rel = mass_rel_l2(solver.M_u, u_T, u_ref)

    return {
        "u_final": u_T,
        "solve_runtime_sec": runtime,
        "n_steps": int(out["n_steps"]),
        "u_abs_l2": ux_abs,
        "u_rel_l2": ux_rel,
        "v_abs_l2": uy_abs,
        "v_rel_l2": uy_rel,
        "uv_abs_l2": uv_abs,
        "uv_rel_l2": uv_rel,
    }


def p1_field(mesh, snap: np.ndarray, component: str) -> np.ndarray:
    nps = mesh.n_p2
    if component == "u":
        return snap[mesh._p1_to_p2]
    if component == "v":
        return snap[nps + mesh._p1_to_p2]
    raise ValueError("component must be 'u' or 'v'")


def save_terminal_comparison_plots(run_dir: Path, mesh, tri, cfg: CylinderRunConfig,
                                   u_ref: np.ndarray, u_pred: np.ndarray, probe_xy: np.ndarray,
                                   nu_rec: float, nu_true: float, metrics: dict[str, object]) -> list[str]:
    plots_dir = run_dir / "terminal_fields"
    plots_dir.mkdir(parents=True, exist_ok=True)
    th = np.linspace(0, 2*np.pi, 200)
    saved = []

    for component, pretty, rel_key in (("u", "$u$", "u_rel_l2"), ("v", "$v$", "v_rel_l2")):
        ref_p1 = p1_field(mesh, u_ref, component)
        pred_p1 = p1_field(mesh, u_pred, component)
        err_p1 = pred_p1 - ref_p1
        vmax = float(max(np.max(np.abs(ref_p1)), np.max(np.abs(pred_p1)), 1e-12))
        emax = float(max(np.max(np.abs(err_p1)), 1e-12))

        fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
        panels = [
            (ref_p1, f"FEM terminal {pretty}", "RdBu_r", -vmax, vmax),
            (pred_p1, f"Recovered-nu terminal {pretty}", "RdBu_r", -vmax, vmax),
            (np.abs(err_p1), f"|error|, rel L2={metrics[rel_key]:.3e}", "magma", 0.0, emax),
        ]
        for axi, (fld, ttl, cmap, vmin, vmax_use) in zip(ax, panels):
            tc = axi.tripcolor(tri, fld, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax_use)
            axi.fill(0.5*np.cos(th), 0.5*np.sin(th), color="k")
            axi.set_aspect("equal")
            axi.set_xlim(cfg.x0, cfg.x1)
            axi.set_ylim(cfg.y0, cfg.y1)
            axi.set_title(ttl)
            axi.grid(alpha=0.2)
            fig.colorbar(tc, ax=axi, shrink=0.8)
        ax[0].plot(probe_xy[:, 0], probe_xy[:, 1], "gP", ms=5, mec="k", mew=0.4)
        fig.suptitle(
            "Terminal field comparison (%s) | nu_rec=%.6f, nu_true=%.6f"
            % (component, nu_rec, nu_true)
        )
        fig.tight_layout()
        out_path = plots_dir / f"terminal_{component}_compare.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        saved.append(str(out_path))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    abs_u = p1_field(mesh, u_pred - u_ref, "u")
    abs_v = p1_field(mesh, u_pred - u_ref, "v")
    mag = np.sqrt(abs_u**2 + abs_v**2)
    ref_mag = np.sqrt(p1_field(mesh, u_ref, "u")**2 + p1_field(mesh, u_ref, "v")**2)
    for axi, fld, ttl in zip(ax, (mag, ref_mag), ("|velocity error|", "|reference velocity|")):
        tc = axi.tripcolor(tri, fld, shading="gouraud", cmap="viridis")
        axi.fill(0.5*np.cos(th), 0.5*np.sin(th), color="k")
        axi.set_aspect("equal")
        axi.set_xlim(cfg.x0, cfg.x1)
        axi.set_ylim(cfg.y0, cfg.y1)
        axi.set_title(ttl)
        axi.grid(alpha=0.2)
        fig.colorbar(tc, ax=axi, shrink=0.8)
    fig.suptitle("Terminal vector-field summary | rel L2=%.3e" % metrics["uv_rel_l2"])
    fig.tight_layout()
    out_path = plots_dir / "terminal_vector_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    saved.append(str(out_path))
    return saved


def summarize(rows: list[dict[str, object]]) -> dict[str, float]:
    keys = [
        "runtime_sec",
        "solve_runtime_sec",
        "total_runtime_sec",
        "abs_nu_err",
        "rel_nu_err",
        "u_abs_l2",
        "u_rel_l2",
        "v_abs_l2",
        "v_rel_l2",
        "uv_abs_l2",
        "uv_rel_l2",
    ]
    summary = {}
    for key in keys:
        arr = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=0))
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run",
        "obs_seed",
        "noise",
        "nu_true",
        "nu_rec",
        "abs_nu_err",
        "rel_nu_err",
        "runtime_sec",
        "solve_runtime_sec",
        "total_runtime_sec",
        "u_abs_l2",
        "u_rel_l2",
        "v_abs_l2",
        "v_rel_l2",
        "uv_abs_l2",
        "uv_rel_l2",
        "n_steps",
        "obs_path",
        "pinn_inv_npz",
        "pinn_inv_model",
        "run_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> None:
    args = build_parser().parse_args()

    base_cfg = CylinderRunConfig()
    if args.noise is not None:
        base_cfg.obs_noise = args.noise

    out_dir = (THIS_DIR / args.output_dir).resolve()
    fig_root = (THIS_DIR / args.fig_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    mesh, solver, tri, u0, u_ref, t_ref = load_reference_terminal(base_cfg)
    print("reference terminal snapshot loaded at t=%.3f for T_obs=%.3f"
          % (t_ref, base_cfg.T_obs), flush=True)
    print("noise std = %.4f | runs = %d | inverse seed fixed at %d"
          % (base_cfg.obs_noise, args.n_runs, base_cfg.seed_inv), flush=True)

    rows: list[dict[str, object]] = []
    for run in range(args.n_runs):
        cfg = copy.deepcopy(base_cfg)
        cfg.obs_seed = args.seed0 + run
        run_dir = out_dir / f"run_{run:02d}"
        fig_dir = fig_root / f"run_{run:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        cfg.hist_dir = str(run_dir)
        cfg.fig_dir = str(fig_dir)
        cfg.obs_path = str(run_dir / "probe_obs.npz")
        cfg.pinn_inv_npz = str(run_dir / "pinn_inv.npz")
        cfg.pinn_inv_model = str(run_dir / "pinn_inv_model.pt")

        api = CylinderAPI(cfg)
        api.build_observation(verbose=False)

        t0 = time.time()
        result = api.train_pinn_inverse()
        train_runtime = time.time() - t0
        field = terminal_field_metrics(solver, u0, u_ref, result["nu_rec"], cfg.T_obs, cfg.dt)
        total_runtime = train_runtime + field["solve_runtime_sec"]

        plot_paths = save_terminal_comparison_plots(
            fig_dir, mesh, tri, cfg, u_ref, field["u_final"], result["probe_xy"],
            result["nu_rec"], result["nu_true"], field
        )

        row = {
            "run": run,
            "obs_seed": cfg.obs_seed,
            "noise": cfg.obs_noise,
            "nu_true": result["nu_true"],
            "nu_rec": result["nu_rec"],
            "abs_nu_err": abs(result["nu_rec"] - result["nu_true"]),
            "rel_nu_err": result["rel_err"],
            "runtime_sec": result["runtime_sec"],
            "solve_runtime_sec": field["solve_runtime_sec"],
            "total_runtime_sec": total_runtime,
            "u_abs_l2": field["u_abs_l2"],
            "u_rel_l2": field["u_rel_l2"],
            "v_abs_l2": field["v_abs_l2"],
            "v_rel_l2": field["v_rel_l2"],
            "uv_abs_l2": field["uv_abs_l2"],
            "uv_rel_l2": field["uv_rel_l2"],
            "n_steps": field["n_steps"],
            "obs_path": cfg.obs_path,
            "pinn_inv_npz": cfg.pinn_inv_npz,
            "pinn_inv_model": cfg.pinn_inv_model,
            "run_dir": str(run_dir),
            "fig_dir": str(fig_dir),
            "plot_paths": plot_paths,
        }
        rows.append(row)

        np.savez(
            run_dir / "terminal_fields.npz",
            u_ref=u_ref,
            u_pred=field["u_final"],
            nu_true=result["nu_true"],
            nu_rec=result["nu_rec"],
            obs_seed=cfg.obs_seed,
            noise=cfg.obs_noise,
            runtime_sec=result["runtime_sec"],
            solve_runtime_sec=field["solve_runtime_sec"],
            total_runtime_sec=total_runtime,
            u_abs_l2=field["u_abs_l2"],
            u_rel_l2=field["u_rel_l2"],
            v_abs_l2=field["v_abs_l2"],
            v_rel_l2=field["v_rel_l2"],
            uv_abs_l2=field["uv_abs_l2"],
            uv_rel_l2=field["uv_rel_l2"],
            cfg_snapshot=str(cfg.snapshot()),
        )
        (run_dir / "run_summary.json").write_text(json.dumps(row, indent=2))

        print(
            "run %02d | seed=%d | nu_rec=%.6f | rel_nu=%.4e | relL2(u)=%.4e | relL2(v)=%.4e | relL2(uv)=%.4e | train=%.1fs | solve=%.1fs"
            % (
                run,
                cfg.obs_seed,
                row["nu_rec"],
                row["rel_nu_err"],
                row["u_rel_l2"],
                row["v_rel_l2"],
                row["uv_rel_l2"],
                row["runtime_sec"],
                row["solve_runtime_sec"],
            ),
            flush=True,
        )

    summary = summarize(rows)
    summary.update({
        "n_runs": args.n_runs,
        "noise": base_cfg.obs_noise,
        "seed0": args.seed0,
        "cfg_snapshot": base_cfg.snapshot(),
    })

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    npz_path = out_dir / "summary.npz"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    np.savez(npz_path, runs=np.array(rows, dtype=object), summary=np.array(summary, dtype=object))

    print("\nSaved:")
    print("  %s" % csv_path)
    print("  %s" % json_path)
    print("  %s" % npz_path)
    print("\nStatistics over %d runs:" % args.n_runs)
    for key in (
        "runtime_sec",
        "solve_runtime_sec",
        "total_runtime_sec",
        "rel_nu_err",
        "u_rel_l2",
        "v_rel_l2",
        "uv_rel_l2",
    ):
        print("  %-18s mean=%.4e  std=%.4e" % (key, summary[f"{key}_mean"], summary[f"{key}_std"]))


if __name__ == "__main__":
    main()
    base_cfg = CylinderRunConfig()
    if args.noise is not None:
        base_cfg.obs_noise = args.noise

    out_dir = (THIS_DIR / args.output_dir).resolve()
    fig_root = (THIS_DIR / args.fig_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    mesh, solver, tri, u0, u_ref, t_ref = load_reference_terminal(base_cfg)
    print("reference terminal snapshot loaded at t=%.3f for T_obs=%.3f"
          % (t_ref, base_cfg.T_obs), flush=True)
    print("noise std = %.4f | runs = %d | inverse seed fixed at %d"
          % (base_cfg.obs_noise, args.n_runs, base_cfg.seed_inv), flush=True)

    rows: list[dict[str, object]] = []
    for run in range(args.n_runs):
        cfg = copy.deepcopy(base_cfg)
        cfg.obs_seed = args.seed0 + run
        run_dir = out_dir / f"run_{run:02d}"
        fig_dir = fig_root / f"run_{run:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        cfg.hist_dir = str(run_dir)
        cfg.fig_dir = str(fig_dir)
        cfg.obs_path = str(run_dir / "probe_obs.npz")
        cfg.pinn_inv_npz = str(run_dir / "pinn_inv.npz")
        cfg.pinn_inv_model = str(run_dir / "pinn_inv_model.pt")

        api = CylinderAPI(cfg)
        api.build_observation(verbose=False)

        t0 = time.time()
        result = api.train_pinn_inverse()
        train_runtime = time.time() - t0
        field = terminal_field_metrics(solver, u0, u_ref, result["nu_rec"], cfg.T_obs, cfg.dt)
        total_runtime = train_runtime + field["solve_runtime_sec"]

        plot_paths = save_terminal_comparison_plots(
            fig_dir, mesh, tri, cfg, u_ref, field["u_final"], result["probe_xy"],
            result["nu_rec"], result["nu_true"], field
        )

        row = {
            "run": run,
            "obs_seed": cfg.obs_seed,
            "noise": cfg.obs_noise,
            "nu_true": result["nu_true"],
            "nu_rec": result["nu_rec"],
            "abs_nu_err": abs(result["nu_rec"] - result["nu_true"]),
            "rel_nu_err": result["rel_err"],
            "runtime_sec": result["runtime_sec"],
            "solve_runtime_sec": field["solve_runtime_sec"],
            "total_runtime_sec": total_runtime,
            "u_abs_l2": field["u_abs_l2"],
            "u_rel_l2": field["u_rel_l2"],
            "v_abs_l2": field["v_abs_l2"],
            "v_rel_l2": field["v_rel_l2"],
            "uv_abs_l2": field["uv_abs_l2"],
            "uv_rel_l2": field["uv_rel_l2"],
            "n_steps": field["n_steps"],
            "obs_path": cfg.obs_path,
            "pinn_inv_npz": cfg.pinn_inv_npz,
            "pinn_inv_model": cfg.pinn_inv_model,
            "run_dir": str(run_dir),
            "fig_dir": str(fig_dir),
            "plot_paths": plot_paths,
        }
        rows.append(row)

        np.savez(
            run_dir / "terminal_fields.npz",
            u_ref=u_ref,
            u_pred=field["u_final"],
            nu_true=result["nu_true"],
            nu_rec=result["nu_rec"],
            obs_seed=cfg.obs_seed,
            noise=cfg.obs_noise,
            runtime_sec=result["runtime_sec"],
            solve_runtime_sec=field["solve_runtime_sec"],
            total_runtime_sec=total_runtime,
            u_abs_l2=field["u_abs_l2"],
            u_rel_l2=field["u_rel_l2"],
            v_abs_l2=field["v_abs_l2"],
            v_rel_l2=field["v_rel_l2"],
            uv_abs_l2=field["uv_abs_l2"],
            uv_rel_l2=field["uv_rel_l2"],
            cfg_snapshot=str(cfg.snapshot()),
        )
        (run_dir / "run_summary.json").write_text(json.dumps(row, indent=2))

        print(
            "run %02d | seed=%d | nu_rec=%.6f | rel_nu=%.4e | relL2(u)=%.4e | relL2(v)=%.4e | relL2(uv)=%.4e | train=%.1fs | solve=%.1fs"
            % (
                run,
                cfg.obs_seed,
                row["nu_rec"],
                row["rel_nu_err"],
                row["u_rel_l2"],
                row["v_rel_l2"],
                row["uv_rel_l2"],
                row["runtime_sec"],
                row["solve_runtime_sec"],
            ),
            flush=True,
        )

    summary = summarize(rows)
    summary.update({
        "n_runs": args.n_runs,
        "noise": base_cfg.obs_noise,
        "seed0": args.seed0,
        "cfg_snapshot": base_cfg.snapshot(),
    })

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    npz_path = out_dir / "summary.npz"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"runs": rows, "summary": summary}, indent=2))
    np.savez(npz_path, runs=np.array(rows, dtype=object), summary=np.array(summary, dtype=object))

    print("\nSaved:")
    print("  %s" % csv_path)
    print("  %s" % json_path)
    print("  %s" % npz_path)
    print("\nStatistics over %d runs:" % args.n_runs)
    for key in (
        "runtime_sec",
        "solve_runtime_sec",
        "total_runtime_sec",
        "rel_nu_err",
        "u_rel_l2",
        "v_rel_l2",
        "uv_rel_l2",
    ):
        print("  %-18s mean=%.4e  std=%.4e" % (key, summary[f"{key}_mean"], summary[f"{key}_std"]))


if __name__ == "__main__":
    main()