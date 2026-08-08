"""Cost and memory counters for the hardware-independent cost comparison.

Referee R3.2 objects that wall-clock time is hardware-specific and that the adjoint
and PINN implementations may not be equally optimized, and asks instead for the
number of forward solves, the number of PDE residual evaluations, the memory
consumption, and the peak storage requirement. Referee R3.3 makes the related point
that the trajectory-storage argument is asserted rather than measured.

This module supplies those numbers. Some already existed piecemeal -- ``nfev``/``nit``
in the cylinder history files, and ``len(eval_loss)`` versus ``len(iter_loss)`` in
Burgers and Allen-Cahn -- but they were neither uniform nor complete. Peak memory was
not recorded anywhere.

Two of the quantities are reported both analytically and as measured, because they
are the crux of the paper's scalability argument and a referee should be able to
check one against the other:

- **trajectory storage**: the adjoint must retain the whole discrete state history
  for the backward sweep, which is :func:`trajectory_bytes` = ``n_x * (n_t+1) * 8``.
- **quasi-Newton curvature storage**: SSBroyden carries a *dense* inverse Hessian,
  which is :func:`ssbroyden_hessian_bytes` = ``8 * n_params**2``. This is what makes
  the method infeasible past ~1e4 parameters and is the direct answer to R3.7.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import asdict, dataclass, field


def trajectory_bytes(n_x: int, n_t: int, itemsize: int = 8) -> int:
    """Bytes needed to cache a full discrete state trajectory for the adjoint sweep.

    ``n_t`` is the number of steps, so ``n_t + 1`` states are stored including the
    initial condition. ``itemsize`` is 8 because every benchmark runs in float64.
    """
    return int(n_x) * (int(n_t) + 1) * int(itemsize)


def ssbroyden_hessian_bytes(n_params: int, itemsize: int = 8) -> int:
    """Bytes held by a dense ``n_params x n_params`` inverse-Hessian approximation.

    Grows quadratically: 1e3 parameters -> 8 MB, 1e4 -> 800 MB, 1e5 -> 80 GB. This is
    the quantitative form of the answer to R3.7 ("could SSBroyden be prohibitively
    expensive for large-scale coupled problems?").
    """
    return int(itemsize) * int(n_params) ** 2


def hardware_info() -> dict:
    """Identify the GPU, CPU and node a run executed on.

    Every reported wall-clock number must come from the same hardware as the rest
    of the paper -- an NVIDIA L40S with a dual-socket AMD EPYC CPU -- otherwise the
    cost table R3.2 asks for would be comparing timings across machines. Recording
    the hardware in each row makes a stray run detectable after the fact;
    :func:`require_l40s` makes it detectable before the run starts.
    """
    import platform
    import socket

    info = {
        "node": socket.gethostname(),
        "cpu_model": platform.processor() or None,
        "gpu_name": None,
        "n_cpus_visible": os.cpu_count(),
    }

    # platform.processor() is often empty on Linux; read the real model name.
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return info


def require_l40s(strict: bool = True) -> dict:
    """Fail fast unless this run is on an L40S GPU.

    Timings from a 3090 or an A6000 are not comparable with the numbers already in
    the manuscript, and a silently mis-scheduled array task would corrupt the cost
    table rather than announce itself. Set ``strict=False`` (or the environment
    variable ``ALLOW_NON_L40S=1``) for accuracy-only runs whose wall-clock is not
    reported.
    """
    info = hardware_info()
    name = (info.get("gpu_name") or "").lower()
    if "l40s" in name:
        return info
    if not strict or os.environ.get("ALLOW_NON_L40S") == "1":
        print(f"warning: running on {info.get('gpu_name')!r}, not an L40S; "
              f"timings from this run must not be reported")
        return info
    raise RuntimeError(
        f"refusing to run: expected an L40S GPU, found {info.get('gpu_name')!r} on "
        f"{info.get('node')}. Submit to -p l40s-gcondo -A gk-l40s-gcondo, or set "
        f"ALLOW_NON_L40S=1 if this run's wall-clock will not be reported."
    )


def peak_host_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is reported in kilobytes on Linux. This is a high-water mark for
    the whole process, so it is only meaningful when one run owns the process --
    which is why the sweep runner launches one configuration per SLURM array task
    rather than looping in-process.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def peak_device_bytes() -> int:
    """Peak CUDA allocation in bytes, or 0 when running on CPU."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def reset_peak_device() -> None:
    """Reset the CUDA high-water mark so a run measures only its own allocation."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class Counters:
    """Work counters accumulated over one inversion.

    ``forward_solves`` and ``adjoint_sweeps`` count full PDE solves, which is the
    hardware-independent unit R3.2 asks for. ``pde_residual_evals`` is the PINN's
    analogue: collocation points times residual evaluations. ``n_fev``/``n_iter``
    separate function evaluations from optimizer iterations -- the manuscript
    currently conflates the two when it says the cylinder adjoint "converges in 8
    iterations" (the saved run has ``nit=3, nfev=8``).
    """

    forward_solves: int = 0
    adjoint_sweeps: int = 0
    pde_residual_evals: int = 0
    n_fev: int = 0
    n_iter: int = 0
    # Static properties of the configuration, filled in by the caller.
    n_params: int = 0
    traj_bytes_analytic: int = 0
    hessian_bytes_analytic: int = 0

    def forward(self, k: int = 1) -> None:
        self.forward_solves += k

    def adjoint(self, k: int = 1) -> None:
        self.adjoint_sweeps += k

    def residual(self, n_points: int) -> None:
        self.pde_residual_evals += int(n_points)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunTimer:
    """Wall-clock plus peak-memory capture for one inversion.

    Used as a context manager::

        with RunTimer() as t:
            ...
        record = t.as_dict()          # runtime_s, peak_host_bytes, peak_device_bytes
    """

    runtime_s: float = 0.0
    peak_host_bytes: int = 0
    peak_device_bytes: int = 0
    _t0: float = field(default=0.0, repr=False)

    def __enter__(self) -> "RunTimer":
        reset_peak_device()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.runtime_s = time.perf_counter() - self._t0
        self.peak_host_bytes = peak_host_bytes()
        self.peak_device_bytes = peak_device_bytes()
        return False  # never suppress exceptions

    def as_dict(self) -> dict:
        return {
            "runtime_s": self.runtime_s,
            "peak_host_bytes": self.peak_host_bytes,
            "peak_device_bytes": self.peak_device_bytes,
        }
