"""Generate OpenFOAM cases for the cylinder grid-independence study.

The question this answers: are the 16 wake-probe observations that Test 4 feeds to
the PINN and to the adjoint converged with respect to the mesh, or do they carry a
discretization error comparable to the inversion error they are used to certify?

Geometry, boundary conditions and probe layout are taken verbatim from
cylinder_config.CylinderRunConfig so that the OpenFOAM solution is an independent
discretization of the *same* continuous problem, not of a similar one:

    domain      [-3, 10] x [-3, 3], cylinder of D = 1 at the origin
    inlet       u = (1, 0)                      (x = -3)
    cylinder    no slip
    walls       u_y = 0, u_x free               (y = +-3)   -> OpenFOAM `slip`
    outlet      natural / do-nothing            (x = 10)    -> zeroGradient U, p = 0
    nu          0.01  (Re = U D / nu = 100)
    probes      4 x 4 over [1,3] x [-1,1]

Mesh: a 12-block structured grid. Four blocks form an O-grid annulus between the
cylinder and the square [-a, a]^2; eight rectangular blocks fill the rest of the
domain. All resolutions come from one scale factor, so the family is geometrically
similar and a Richardson/GCI analysis is meaningful. The block counts below are
rounded to integers, which perturbs the effective refinement ratio slightly; the
analysis script recomputes r from the realized cell counts rather than assuming it.

Solver: `pimpleFoam`. The flow at Re = 100 is a Karman limit cycle, so the
observations the PINN consumes are transient by construction and a steady solver
cannot produce them; PIMPLE's outer corrector loop is the SIMPLE algorithm applied
within each time step. Reported: Strouhal number, C_D, C_L amplitude, the
time-averaged and RMS probe velocities (all phase-free, so they compare across
meshes directly), and the probe time series over a window matching the paper's
[0, 5] observation window.

Symmetry is broken identically on every mesh by a transverse inlet gust of 0.1 for
the first 2 time units, mirroring the blob perturbation the FEM solver injects, so
the trigger cannot be a source of between-mesh differences.
"""

from __future__ import annotations

import math
import os
import shutil

# ----------------------------------------------------------------- geometry
X0, X1, Y0, Y1 = -3.0, 10.0, -3.0, 3.0
R = 0.5                 # cylinder radius, D = 1
A = 1.5                 # half-width of the square bounding the O-grid annulus
ZMIN, ZMAX = -0.05, 0.05
NU = 0.01               # Re = 100
U_INF = 1.0

GUST_V = 0.1            # transverse perturbation amplitude (FEM blob_amp)
RAMP_T = 2.0            # inlet ramp 0 -> U_inf          (FEM ramp_T)
BLOB_T = 8.0            # perturbation injected here     (FEM blob_t)

# probe grid: must match cylinder_config.probe_xy() exactly
PROBE_XY = [(x, y)
            for x in [1.0 + (3.0 - 1.0) * i / 3 for i in range(4)]
            for y in [-1.0 + 2.0 * j / 3 for j in range(4)]]

# --------------------------------------------------------------- resolution
# (Nc, Nr, NxL, NxR, NyB) at scale 1; every level multiplies and rounds.
BASE = dict(Nc=24, Nr=20, NxL=16, NxR=72, NyB=16)
LEVELS = {"L1": 1.0, "L2": 1.5, "L3": 2.25, "L4": 3.375}

# grading, held fixed across levels so the family stays geometrically similar
G_RAD = 4.0             # annulus, cells grow outward from the cylinder
G_XL = 1.0 / 3.0        # inlet block, cells shrink toward the cylinder
G_XR = 6.0              # wake block, cells grow downstream
G_Y = 3.0               # cross-stream, cells grow away from the cylinder


def counts(scale):
    n = {k: max(4, int(round(v * scale))) for k, v in BASE.items()}
    return n


# ----------------------------------------------------------------- vertices
def vertices():
    c = R / math.sqrt(2.0)
    xy = [
        (-c, -c), (c, -c), (c, c), (-c, c),          # 0-3  circle diagonals
        (-A, -A), (A, -A), (A, A), (-A, A),          # 4-7  inner square
        (X0, -A), (X0, A), (X1, -A), (X1, A),        # 8-11 left/right columns
        (-A, Y0), (A, Y0), (-A, Y1), (A, Y1),        # 12-15 bottom/top rows
        (X0, Y0), (X1, Y0), (X0, Y1), (X1, Y1),      # 16-19 domain corners
    ]
    return xy


NV = 20


def hexblock(quad, cells, grading):
    """quad = 4 xy-plane vertex ids, counterclockwise seen from +z."""
    a, b, cc, d = quad
    lo = f"{a} {b} {cc} {d}"
    hi = f"{a+NV} {b+NV} {cc+NV} {d+NV}"
    g = " ".join(f"{v:g}" for v in grading)
    n = " ".join(str(v) for v in cells)
    return f"    hex ({lo} {hi}) ({n}) simpleGrading ({g})"


def face(p, q):
    """Vertical patch face on the xy-edge p->q. Its normal is z_hat x (q-p), so
    the caller picks the traversal direction that points out of the domain."""
    return f"        ({p} {p+NV} {q+NV} {q})"


def block_mesh_dict(scale):
    n = counts(scale)
    Nc, Nr, NxL, NxR, NyB = n["Nc"], n["Nr"], n["NxL"], n["NxR"], n["NyB"]
    NyT = NyB

    v = vertices()
    vs = "\n".join(f"    ({x:.10g} {y:.10g} {ZMIN:g})" for x, y in v)
    vs += "\n" + "\n".join(f"    ({x:.10g} {y:.10g} {ZMAX:g})" for x, y in v)

    blocks = [
        # --- O-grid annulus: local x is radial, local y is circumferential ---
        hexblock((0, 4, 5, 1), (Nr, Nc, 1), (G_RAD, 1, 1)),      # below cylinder
        hexblock((1, 5, 6, 2), (Nr, Nc, 1), (G_RAD, 1, 1)),      # right
        hexblock((2, 6, 7, 3), (Nr, Nc, 1), (G_RAD, 1, 1)),      # above
        hexblock((3, 7, 4, 0), (Nr, Nc, 1), (G_RAD, 1, 1)),      # left
        # --- outer strips ---
        hexblock((8, 4, 7, 9), (NxL, Nc, 1), (G_XL, 1, 1)),      # inlet block
        hexblock((5, 10, 11, 6), (NxR, Nc, 1), (G_XR, 1, 1)),    # wake block
        hexblock((12, 13, 5, 4), (Nc, NyB, 1), (1, 1 / G_Y, 1)),  # below square
        hexblock((7, 6, 15, 14), (Nc, NyT, 1), (1, G_Y, 1)),     # above square
        # --- corners ---
        hexblock((16, 12, 4, 8), (NxL, NyB, 1), (G_XL, 1 / G_Y, 1)),
        hexblock((13, 17, 10, 5), (NxR, NyB, 1), (G_XR, 1 / G_Y, 1)),
        hexblock((9, 7, 14, 18), (NxL, NyT, 1), (G_XL, G_Y, 1)),
        hexblock((6, 11, 19, 15), (NxR, NyT, 1), (G_XR, G_Y, 1)),
    ]

    # circular edges: the four quarter-arcs, at both z planes
    arcs = []
    for (p, q), mid in (((0, 1), (0, -R)), ((1, 2), (R, 0)),
                        ((2, 3), (0, R)), ((3, 0), (-R, 0))):
        for off in (0, NV):
            arcs.append(f"    arc {p+off} {q+off} ({mid[0]:.10g} {mid[1]:.10g} "
                        f"{ZMIN if off == 0 else ZMAX:g})")

    # Traversal directions chosen so each normal points out of the fluid domain.
    inlet = [face(16, 8), face(8, 9), face(9, 18)]          # +y edges at x = X0
    outlet = [face(19, 11), face(11, 10), face(10, 17)]     # -y edges at x = X1
    top = [face(18, 14), face(14, 15), face(15, 19)]        # +x edges at y = Y1
    bottom = [face(17, 13), face(13, 12), face(12, 16)]     # -x edges at y = Y0
    cyl = [face(0, 1), face(1, 2), face(2, 3), face(3, 0)]  # ccw -> into the solid

    def patch(name, kind, faces):
        return (f"    {name}\n    {{\n        type {kind};\n"
                f"        faces\n        (\n" + "\n".join(faces) +
                "\n        );\n    }")

    boundary = "\n".join([
        patch("inlet", "patch", inlet),
        patch("outlet", "patch", outlet),
        patch("top", "symmetryPlane", top),
        patch("bottom", "symmetryPlane", bottom),
        patch("cylinder", "wall", cyl),
    ])

    ncells = (4 * Nr * Nc + Nc * (NxL + NxR + NyB + NyT)
              + NxL * NyB + NxR * NyB + NxL * NyT + NxR * NyT)

    txt = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale 1;

// resolution scale {scale:g}: Nc={Nc} Nr={Nr} NxL={NxL} NxR={NxR} NyB={NyB}
// nominal cell count {ncells}

vertices
(
{vs}
);

blocks
(
{chr(10).join(blocks)}
);

edges
(
{chr(10).join(arcs)}
);

boundary
(
{boundary}
);

defaultPatch
{{
    name  frontAndBack;
    type  empty;
}}
"""
    return txt, ncells, n


# ------------------------------------------------------------ field files
def U_file(steady):
    if steady:
        inlet_bc = f"        type            fixedValue;\n" \
                   f"        value           uniform ({U_INF} 0 0);"
    else:
        # Mirrors the FEM reference: a smooth inlet ramp over [0, ramp_T] and a
        # symmetry-breaking transverse perturbation at blob_t. Starting the flow
        # impulsively instead put a singular vorticity sheet on the cylinder at
        # t = 0, which on the two finest meshes drove the local Courant number
        # past 3 within three steps and diverged. Identical on every mesh, so the
        # trigger cannot be a source of between-mesh differences.
        inlet_bc = (
            "        type            uniformFixedValue;\n"
            "        uniformValue    table\n"
            "        (\n"
            "            (0      (0 0 0))\n"
            f"            ({RAMP_T:g}      ({U_INF} 0 0))\n"
            f"            ({BLOB_T:g}      ({U_INF} 0 0))\n"
            f"            ({BLOB_T + 0.01:g}   ({U_INF} {GUST_V} 0))\n"
            f"            ({BLOB_T + 2:g}     ({U_INF} {GUST_V} 0))\n"
            f"            ({BLOB_T + 2.01:g}  ({U_INF} 0 0))\n"
            f"            (1000   ({U_INF} 0 0))\n"
            "        );")
    return f"""FoamFile
{{
    version 2.0; format ascii; class volVectorField; object U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
{inlet_bc}
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    top
    {{
        type            symmetryPlane;
    }}
    bottom
    {{
        type            symmetryPlane;
    }}
    cylinder
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""


P_FILE = """FoamFile
{
    version 2.0; format ascii; class volScalarField; object p;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    top
    {
        type            symmetryPlane;
    }
    bottom
    {
        type            symmetryPlane;
    }
    cylinder
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""

# ESI OpenFOAM (v2512) reads transportProperties/turbulenceProperties; the
# physicalProperties/momentumTransport spelling is the Foundation convention.
# Both names are written so the case runs under either build.
TRANSPORT = f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object transportProperties;
}}

transportModel  Newtonian;
viscosityModel  constant;
nu              {NU};
"""

MOMENTUM = """FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}

simulationType  laminar;
"""


def control_dict(steady, end_time, dt, write_interval, probe_every):
    app = "simpleFoam" if steady else "pimpleFoam"
    if steady:
        timing = (f"deltaT          1;\nendTime         {end_time};\n"
                  f"writeControl    timeStep;\nwriteInterval   {write_interval};\n"
                  "adjustTimeStep  no;\n")
    else:
        timing = (f"deltaT          {dt};\nendTime         {end_time};\n"
                  f"writeControl    runTime;\nwriteInterval   {write_interval};\n"
                  "adjustTimeStep  no;\n")

    probe_pts = "\n".join(f"            ({x:.12g} {y:.12g} 0)" for x, y in PROBE_XY)

    # Aref = D * span; the mesh is one cell thick with span 0.1.
    span = ZMAX - ZMIN

    return f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object controlDict;
}}

application     {app};
startFrom       startTime;
startTime       0;
stopAt          endTime;
{timing}
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable false;

functions
{{
    probes
    {{
        type            probes;
        libs            (sampling);
        writeControl    timeStep;
        writeInterval   {probe_every};
        fields          (U p);
        probeLocations
        (
{probe_pts}
        );
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   {probe_every};
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {U_INF};
        lRef            1;
        Aref            {span:g};
    }}
}}
"""


FVSCHEMES_STEADY = """FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSchemes;
}

ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

# Central differencing on the transient case: second order and non-dissipative,
# the closest match to the Galerkin P2-P1 discretization it is being checked
# against. The steady case keeps linearUpwind for SIMPLE robustness.
FVSCHEMES_TRANS = """FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSchemes;
}

ddtSchemes      { default backward; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""

FVSOLUTION_STEADY = """FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-10;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-11;
        relTol          0.01;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {
        p               1e-9;
        U               1e-9;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
    }
    fields
    {
        p               1.0;
    }
}
"""

FVSOLUTION_TRANS = """FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-9;
        relTol          0.001;
        smoother        GaussSeidel;
    }
    pFinal
    {
        $p;
        tolerance       1e-11;
        relTol          0;
    }
    "(U|UFinal)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-12;
        relTol          0;
    }
}

PIMPLE
{
    // The outer loop is the SIMPLE algorithm applied within the time step.
    // Residual control lets it exit as soon as the outer iteration has
    // converged, which on a smooth laminar limit cycle is typically after two
    // passes; a fixed count of 3 spent 12 pressure solves per step for no
    // change in the answer.
    nOuterCorrectors 4;
    nCorrectors      2;
    nNonOrthogonalCorrectors 1;   // max non-orthogonality is 44 deg
    momentumPredictor yes;
    outerCorrectorResidualControl
    {
        p  { tolerance 1e-7; relTol 0; }
        U  { tolerance 1e-8; relTol 0; }
    }
}
"""


def decompose_dict(np_):
    return f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}}

numberOfSubdomains {np_};
method          scotch;
"""


def write_case(root, level, scale, steady, end_time, dt, np_=8):
    tag = f"{level}_{'steady' if steady else 'trans'}"
    case = os.path.join(root, tag)
    if os.path.exists(case):
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        os.makedirs(os.path.join(case, d))

    bm, ncells, n = block_mesh_dict(scale)
    W = lambda rel, txt: open(os.path.join(case, rel), "w").write(txt)

    W("system/blockMeshDict", bm)
    W("0/U", U_file(steady))
    W("0/p", P_FILE)
    W("constant/transportProperties", TRANSPORT)
    W("constant/turbulenceProperties", MOMENTUM)
    W("constant/physicalProperties", TRANSPORT.replace("transportProperties",
                                                       "physicalProperties"))
    W("constant/momentumTransport", MOMENTUM.replace("turbulenceProperties",
                                                     "momentumTransport"))
    if steady:
        # steady: "time" is the iteration index; write only the final state
        W("system/controlDict", control_dict(True, 20000, 1, 20000, 20))
        W("system/fvSchemes", FVSCHEMES_STEADY)
        W("system/fvSolution", FVSOLUTION_STEADY)
    else:
        W("system/controlDict", control_dict(False, end_time, dt, 5.0, 2))
        W("system/fvSchemes", FVSCHEMES_TRANS)
        W("system/fvSolution", FVSOLUTION_TRANS)
    W("system/decomposeParDict", decompose_dict(np_))
    return case, ncells, n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)) + "/cases")
    ap.add_argument("--end-time", type=float, default=85.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--np", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    print(f"{'case':16s} {'cells':>8s}  Nc  Nr  NxL  NxR  NyB   h_theta   h_r1")
    for level, scale in LEVELS.items():
        case, ncells, n = write_case(args.root, level, scale, False,
                                     args.end_time, args.dt, args.np)
        k = G_RAD ** (1.0 / (n["Nr"] - 1))
        h_r1 = (A - R) * (k - 1) / (k ** n["Nr"] - 1)   # first radial cell
        h_th = 2 * math.pi * R / (4 * n["Nc"])          # cell size along the wall
        print(f"{level:16s} {ncells:8d} {n['Nc']:3d} {n['Nr']:3d} "
              f"{n['NxL']:4d} {n['NxR']:4d} {n['NyB']:4d}  "
              f"{h_th:.5f}  {h_r1:.5f}")
    # A dt-refinement arm on the medium mesh, to show the spatial study is not
    # limited by the time step.
    for dt in (0.0025,):
        case, ncells, n = write_case(args.root, f"L3dt{dt:g}", LEVELS["L3"], False,
                                     args.end_time, dt, args.np)
        print(f"{'L3dt'+format(dt,'g'):16s} {ncells:8d}  (dt = {dt})")
