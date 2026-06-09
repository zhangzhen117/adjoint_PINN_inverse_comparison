"""
gmsh P2/P1 Taylor-Hood mesh for flow past a circular cylinder.

Domain: [x0, x1] x [y0, y1] with a disk of diameter D at the origin removed.
Refined boundary layer on the cylinder and a refined wake box downstream.

Produces the SAME (coords_p2, elem_p2, elem_p1) layout the cavity FEMAssembler
expects, so the element-level assembly routines are reusable verbatim:
  - coords_p2 : (n_p2, 2) coordinates of all P2 nodes (vertices + edge midpoints)
  - elem_p2   : (n_elem, 6) int, P2 connectivity. Local order matches cavity:
                0,1,2 = vertices; 3 = mid(0,1); 4 = mid(1,2); 5 = mid(2,0).
                (gmsh order-2 triangle node ordering is exactly this.)
  - elem_p1   : (n_elem, 3) int, vertex connectivity in a compact P1 numbering.
  - boundary node sets (P2 numbering) per physical group.
"""

import numpy as np


class CylinderMesh:
    """Unstructured P2/P1 mesh of a rectangle minus a centered disk, via gmsh."""

    def __init__(self, x0=-3.0, x1=10.0, y0=-3.0, y1=3.0, D=1.0,
                 h_cyl=0.035, h_wake=0.08, h_far=0.4,
                 wake_box=(0.5, 8.0, -1.5, 1.5), verbose=False):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.D = D
        self.R = 0.5 * D
        self.h_cyl, self.h_wake, self.h_far = h_cyl, h_wake, h_far
        self.wake_box = wake_box
        self._build(verbose=verbose)
        self._build_function_spaces()

    # ------------------------------------------------------------------ gmsh
    def _build(self, verbose=False):
        import gmsh
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
            occ = gmsh.model.occ

            # Rectangle minus disk
            rect = occ.addRectangle(self.x0, self.y0, 0.0,
                                    self.x1 - self.x0, self.y1 - self.y0)
            disk = occ.addDisk(0.0, 0.0, 0.0, self.R, self.R)
            occ.cut([(2, rect)], [(2, disk)])
            occ.synchronize()

            # --- classify boundary curves by geometry -> physical groups ---
            inlet, outlet, walls, cyl = [], [], [], []
            eps = 1e-6
            for dim, tag in gmsh.model.getEntities(1):
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
                if abs(xmin - self.x0) < eps and abs(xmax - self.x0) < eps:
                    inlet.append(tag)
                elif abs(xmin - self.x1) < eps and abs(xmax - self.x1) < eps:
                    outlet.append(tag)
                elif (abs(ymin - self.y0) < eps and abs(ymax - self.y0) < eps) or \
                     (abs(ymin - self.y1) < eps and abs(ymax - self.y1) < eps):
                    walls.append(tag)
                else:
                    cyl.append(tag)  # the circle arc(s)

            self._pg = {}
            for name, tags in [("inlet", inlet), ("outlet", outlet),
                               ("walls", walls), ("cylinder", cyl)]:
                pg = gmsh.model.addPhysicalGroup(1, tags)
                gmsh.model.setPhysicalName(1, pg, name)
                self._pg[name] = pg
            surf_pg = gmsh.model.addPhysicalGroup(2, [rect])
            gmsh.model.setPhysicalName(2, surf_pg, "fluid")

            # --- size field: fine on cylinder (distance+threshold) + wake box ---
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", cyl)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)

            f_th = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_th, "SizeMin", self.h_cyl)
            gmsh.model.mesh.field.setNumber(f_th, "SizeMax", self.h_far)
            gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.15)
            gmsh.model.mesh.field.setNumber(f_th, "DistMax", 2.0)

            xa, xb, ya, yb = self.wake_box
            f_box = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(f_box, "VIn", self.h_wake)
            gmsh.model.mesh.field.setNumber(f_box, "VOut", self.h_far)
            gmsh.model.mesh.field.setNumber(f_box, "XMin", xa)
            gmsh.model.mesh.field.setNumber(f_box, "XMax", xb)
            gmsh.model.mesh.field.setNumber(f_box, "YMin", ya)
            gmsh.model.mesh.field.setNumber(f_box, "YMax", yb)
            gmsh.model.mesh.field.setNumber(f_box, "Thickness", 0.5)

            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th, f_box])
            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(2)

            self._extract(gmsh)
        finally:
            gmsh.finalize()

    def _extract(self, gmsh):
        # --- all nodes (P2 nodes for an order-2 mesh) ---
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)[:, :2]
        # contiguous remap of gmsh tags -> 0..n_p2-1
        tag2p2 = {int(t): i for i, t in enumerate(node_tags)}
        self.coords_p2 = node_coords.copy()
        self.n_p2 = self.coords_p2.shape[0]

        # --- order-2 triangles (gmsh element type 9, 6 nodes) ---
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        elem6 = None
        for et, en in zip(etypes, enodes):
            if et == 9:  # 6-node 2nd order triangle
                elem6 = np.array([tag2p2[int(t)] for t in en], dtype=np.int64
                                 ).reshape(-1, 6)
        if elem6 is None:
            raise RuntimeError("No order-2 triangles found in mesh.")
        self.elem_p2 = elem6
        self.n_elem = elem6.shape[0]

        # --- P1 space: corner nodes only, compact numbering ---
        corner_global = np.unique(elem6[:, :3].ravel())
        p2_to_p1 = -np.ones(self.n_p2, dtype=np.int64)
        p2_to_p1[corner_global] = np.arange(corner_global.size)
        self.elem_p1 = p2_to_p1[elem6[:, :3]]
        self.n_p1 = corner_global.size
        self.coords_p1 = self.coords_p2[corner_global]
        self._p1_to_p2 = corner_global  # P1 idx -> P2 idx

        # --- boundary node sets (in P2 numbering) ---
        self.bnd = {}
        for name, pg in self._pg.items():
            btags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg)
            self.bnd[name] = np.array(sorted(tag2p2[int(t)] for t in btags),
                                      dtype=np.int64)

    # -------------------------------------------------------- DOF bookkeeping
    def _build_function_spaces(self):
        self.n_u = 2 * self.n_p2   # [ux (n_p2) ; uy (n_p2)]
        self.n_p = self.n_p1

        # Dirichlet DOFs for the cylinder problem:
        #   inlet:    ux, uy fixed (=1, 0)
        #   cylinder: ux, uy fixed (=0, 0)
        #   walls:    only uy fixed (=0)  -> slip / symmetry
        npx = self.n_p2
        inlet, cyl, walls = self.bnd["inlet"], self.bnd["cylinder"], self.bnd["walls"]

        # avoid double-counting nodes shared between groups (corners)
        dir_x = np.unique(np.concatenate([inlet, cyl]))            # ux constrained
        dir_y = np.unique(np.concatenate([inlet, cyl, walls]))    # uy constrained

        self.dir_ux_nodes = dir_x
        self.dir_uy_nodes = dir_y
        # global velocity DOF indices
        self.bnd_u = np.concatenate([dir_x, npx + dir_y])

        # prescribed values (lift) on those DOFs
        g = np.zeros(self.n_u)
        g[inlet] = 1.0          # ux = 1 at inlet (cylinder/walls -> 0)
        # uy components all zero
        self.u_dirichlet = g

    # ----------------------------------------------------------------- utils
    def summary(self):
        return (f"CylinderMesh: n_elem={self.n_elem}, n_p2={self.n_p2}, "
                f"n_p1={self.n_p1}, n_u={self.n_u}\n"
                f"  bnd: inlet={self.bnd['inlet'].size}, outlet={self.bnd['outlet'].size}, "
                f"walls={self.bnd['walls'].size}, cylinder={self.bnd['cylinder'].size}\n"
                f"  Dirichlet u-DOFs={self.bnd_u.size}")


if __name__ == "__main__":
    m = CylinderMesh(h_cyl=0.05, h_wake=0.12, h_far=0.5, verbose=False)
    print(m.summary())
    # sanity: cylinder nodes should sit on radius R
    rc = np.linalg.norm(m.coords_p2[m.bnd["cylinder"]], axis=1)
    print(f"  cylinder node radius: min={rc.min():.4f} max={rc.max():.4f} (R={m.R})")
    # inlet x ~ x0
    print(f"  inlet x range: [{m.coords_p2[m.bnd['inlet'],0].min():.3f}, "
          f"{m.coords_p2[m.bnd['inlet'],0].max():.3f}]")
