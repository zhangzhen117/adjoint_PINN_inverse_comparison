"""Output format for the manuscript figures.

The plot scripts write PNG into paper_overleaf/ for the working build. The
publisher wants separate high-resolution files in TIFF, EPS or JPEG, so the same
scripts double as the export path: set FIG_DPI and FIG_FMT (and optionally
FIG_OUTDIR) and they re-render the identical figure at submission resolution
instead of the screen one.

EPS is vector, so a dpi is meaningless for it and the requirement is satisfied by
construction; it is the right choice for the line plots. The field plots are
filled contours, where a raster at the stated dpi is both smaller and faster to
open than the equivalent vector file.
"""

from __future__ import annotations

import os


def dpi(default: int) -> int:
    """Rendering resolution: FIG_DPI if set, else the script's screen default."""
    return int(os.environ.get("FIG_DPI", default))


def save_kwargs() -> dict:
    """Extra savefig arguments for the chosen format.

    An uncompressed TIFF of a full-page figure at 650 dpi is ~147 MB, which no
    submission system wants; LZW is lossless and takes it to a few MB, since
    filled contours are large flat regions.
    """
    fmt = (os.environ.get("FIG_FMT") or "").lstrip(".").lower()
    if fmt in ("tif", "tiff"):
        return {"pil_kwargs": {"compression": "tiff_lzw"}}
    if fmt in ("jpg", "jpeg"):
        return {"pil_kwargs": {"quality": 95}}
    return {}


def target(path: str) -> str:
    """Redirect ``path`` to FIG_OUTDIR and FIG_FMT when those are set."""
    fmt = os.environ.get("FIG_FMT")
    outdir = os.environ.get("FIG_OUTDIR")
    if fmt:
        path = os.path.splitext(path)[0] + "." + fmt.lstrip(".")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        # "_new" distinguished the replacement figures from the submitted ones
        # while both were in flight; the exported file should carry the name the
        # manuscript uses.
        base = os.path.basename(path).replace("_new.", ".")
        path = os.path.join(outdir, base)
    return path
