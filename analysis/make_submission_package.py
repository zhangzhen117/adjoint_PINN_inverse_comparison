"""Build a flat, self-contained submission archive.

The publisher wants one archive with no subfolders, figures in TIFF, EPS or JPEG
named in manuscript order, and tables as editable text (which LaTeX gives for
free). This assembles that: it re-exports every script-backed figure at
submission settings, converts the rest, renames them to their manuscript numbers,
rewrites the \\includegraphics keys in a copy of main.tex, and compiles the copy
in place so the archive is known to build before it is uploaded.

Format. Everything goes out as JPEG at quality 95, rendered at 750 dpi where a
script can regenerate it. JPEG is on the publisher's allowed list and pdflatex
reads it natively; TIFF does not compile under pdflatex, so an archive using it
could not be built from its own contents, and EPS, while vector and therefore
free of any dpi question, is not what this journal asked for.

Five figures come from notebook cells that depend on in-memory training state
rather than from a script, so they can only be carried across at the resolution
they were saved at. Their effective dpi at a 190 mm column is printed in the
report so the shortfall is visible rather than implied.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper_overleaf")
OUT = os.path.join(REPO, "submission_package")

# manuscript figure number -> source basename. Figure 2 is a two-panel figure, so
# its subfigures carry the 2a/2b suffixes the guidelines use.
FIGURES = [
    ("Figure1",  "B_training_history",                  "jpg"),
    ("Figure2a", "B_learned_force",                     "jpg"),
    ("Figure2b", "B_terminal_solution",                 "jpg"),
    ("Figure3",  "D_training_history",                  "jpg"),
    ("Figure4",  "D_three_method_comparison",           "jpg"),
    ("Figure5",  "D_three_method_solution_comparison",  "jpg"),
    ("Figure6",  "AC_training_history",                 "jpg"),
    ("Figure7",  "AC_force_recovery",                   "jpg"),
    ("Figure8",  "AC_terminal_solution_slice",          "jpg"),
    ("Figure9",  "C_optimization_history_4panel",       "jpg"),
    ("Figure10", "C_pinn_inv_compare_3times",           "jpg"),
]

# figures a script can regenerate: script -> the basenames it writes
SCRIPTED = {
    "plot_burgers_history.py":  ["B_training_history"],
    "plot_darcy_history.py":    ["D_training_history"],
    "plot_ac_history.py":       ["AC_training_history"],
    "plot_cylinder_history.py": ["C_optimization_history_4panel"],
    "plot_darcy_fields.py":     ["D_three_method_comparison",
                                 "D_three_method_solution_comparison"],
}
DPI = 750
COLUMN_IN = 190 / 25.4          # full text width of the journal page


def run(cmd, **kw):
    env = dict(os.environ, PYTHONPATH=REPO, **kw.pop("env", {}))
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)


def regenerate(stage):
    """Re-export the script-backed figures into `stage` at submission settings."""
    for script, names in SCRIPTED.items():
        fmt = next(f for n, b, f in FIGURES if b == names[0])
        r = run([sys.executable, os.path.join("analysis", script)],
                env={"FIG_FMT": fmt, "FIG_DPI": str(DPI), "FIG_OUTDIR": stage})
        if r.returncode:
            raise SystemExit(f"{script} failed:\n{r.stderr[-2000:]}")
        print(f"  regenerated {', '.join(names)} as .{fmt}")


def to_jpeg(src, dst):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    dpi = im.info.get("dpi", (DPI, DPI))
    im.convert("RGB").save(dst, "JPEG", quality=95, dpi=dpi)
    return im.size, dpi[0]


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    stage = os.path.join(OUT, "_stage")
    os.makedirs(stage)

    print("Re-exporting script-backed figures")
    regenerate(stage)

    print("\nPlacing figures under their manuscript numbers")
    report = []
    for num, base, fmt in FIGURES:
        staged = os.path.join(stage, f"{base}.{fmt}")
        dst = os.path.join(OUT, f"{num}.{fmt}")
        if os.path.exists(staged):                       # freshly exported
            shutil.move(staged, dst)
            src_note = "regenerated"
        else:                                            # notebook figure: convert
            png = os.path.join(PAPER, f"{base}.png")
            if not os.path.exists(png):
                raise SystemExit(f"missing source for {num}: {png}")
            to_jpeg(png, dst)
            src_note = "converted from the saved render"
        size = os.path.getsize(dst) / 1048576
        if fmt == "jpg":
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            w, _ = Image.open(dst).size
            eff = w / COLUMN_IN
            report.append((num, base, fmt, f"{w} px", f"{eff:.0f} dpi at 190 mm",
                           f"{size:.1f} MB", src_note))
        else:
            report.append((num, base, fmt, "vector", "resolution-independent",
                           f"{size:.1f} MB", src_note))
    shutil.rmtree(stage)

    print("\nCopying the manuscript sources")
    for f in ("references.bib", "elsarticle-num.bst"):
        shutil.copy2(os.path.join(PAPER, f), OUT)
        print(f"  {f}")
    # elsarticle ships as .dtx/.ins; unpack the class so the archive stands alone
    cls = subprocess.run(["kpsewhich", "elsarticle.cls"], capture_output=True,
                         text=True).stdout.strip()
    if cls and os.path.exists(cls):
        shutil.copy2(cls, OUT)
        print("  elsarticle.cls")

    print("\nRewriting the figure keys in main.tex")
    keys = {base: num for num, base, _ in FIGURES}
    pat = re.compile(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}")

    def swap(m):
        opts, name = m.group(1) or "", os.path.splitext(m.group(2))[0]
        if name not in keys:
            raise SystemExit(f"unmapped figure in main.tex: {name}")
        return f"\\includegraphics{opts}{{{keys[name]}}}"

    # Both \revzz and \revaa draw on zzrev, and the revision tables set it
    # directly, so one definition takes the whole submission copy to black. The
    # Overleaf source keeps its colour for the authors' own reading.
    black = 0
    out_lines, n = [], 0
    for line in open(os.path.join(PAPER, "main.tex")):
        if line.startswith("\\definecolor{zzrev}"):
            line, black = "\\definecolor{zzrev}{RGB}{0,0,0}\n", 1
        # a commented-out \includegraphics is not a figure; the preamble carries
        # one from the class template
        code = line.split("%", 1)[0] if not line.lstrip().startswith("%") else ""
        if pat.search(code):
            line, k = pat.subn(swap, line)
            n += k
        out_lines.append(line)
    print(f"  {n} \\includegraphics keys rewritten to Figure<N>")
    if not black:
        raise SystemExit("did not find the zzrev definition to set to black")
    print("  revision colour set to black")
    open(os.path.join(OUT, "main.tex"), "w").write("".join(out_lines))

    print("\nCompiling the archive from its own contents")
    for step in (["pdflatex", "-shell-escape", "-interaction=nonstopmode", "main"],
                 ["bibtex", "main"],
                 ["pdflatex", "-shell-escape", "-interaction=nonstopmode", "main"],
                 ["pdflatex", "-shell-escape", "-interaction=nonstopmode", "main"]):
        subprocess.run(step, cwd=OUT, capture_output=True, text=True)
    log = open(os.path.join(OUT, "main.log"), errors="replace").read()
    bad = re.findall(r"(?:Reference|Citation) `[^']*' on page \d+ undefined", log)
    pages = re.search(r"\((\d+) pages", log)
    print(f"  {pages.group(1) if pages else '?'} pages, "
          f"{len(bad)} undefined references or citations")
    if bad:
        raise SystemExit("archive does not build cleanly:\n  " + "\n  ".join(bad[:5]))

    # epstopdf leaves a converted copy beside each EPS, and the build leaves its
    # scratch files; neither belongs in the upload. main.bbl stays because
    # Elsevier's pipeline uses it, and main.pdf as the reference rendering.
    for f in sorted(os.listdir(OUT)):
        if f.endswith("-eps-converted-to.pdf") or os.path.splitext(f)[1] in (
                ".aux", ".log", ".blg", ".out", ".spl", ".fls", ".fdb_latexmk"):
            os.remove(os.path.join(OUT, f))
    # The compile is the check that the archive builds, not a deliverable: the
    # publisher recompiles from source, and at 750 dpi the PDF is half the
    # archive. It is kept beside the package for reading rather than inside it.
    pdf = os.path.join(OUT, "main.pdf")
    if os.path.exists(pdf):
        shutil.move(pdf, os.path.join(REPO, "submission_main.pdf"))
        print("  main.pdf moved out of the package (kept as submission_main.pdf)")


    print("\n" + "=" * 78)
    w = max(len(r[1]) for r in report)
    for num, base, fmt, px, eff, size, note in report:
        print(f"  {num:<9s} {base:<{w}s} .{fmt:<4s} {px:>10s}  {eff:<26s} "
              f"{size:>8s}  {note}")
    print("=" * 78)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
