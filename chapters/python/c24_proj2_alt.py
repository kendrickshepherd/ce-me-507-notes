"""
Bridge between the Class 24 application models and the Project 2 code.

The c24_proj2_alt_*.py files repeat the analyses of the Class 24 pages with the
finite element code built in Homeworks 6-16 and Project 2.  That code is
imported from the homework folder, not copied:

  * the mesh is read by CreateBasisMesh, as the brick is in Project 2;
  * boundary conditions are the objects of Boundary_Conditions.py (Homework 13);
  * the system is assembled and solved by HeatConductionProblem;
  * fluxes, values, and gradients come from ExtractTotalBoundaryFlux,
    EvaluateSolution, and EvaluateSolutionGradient, and element matrices
    from the Homework 15 and 16 routines.

This file supplies only what Project 2 does not:

  * the mesh of each application, written as an OBJ file (the element layout
    comes from the matching c24_*.py file, so both codes use the same mesh);
  * boundary sides selected by the position of their midpoint and their
    outward normal (Project 2 uses the whole closed boundary curves);
  * nodes that share one unknown, such as a rigid boss, by giving them one
    equation number in the ID array;
  * post-processing: reactions, energy, and integrals over the domain or
    along boundary sides.

The Project 2 code is found in CE_ME_507_Homework/projects/Current, through
the symbolic link at the top of the notes repository.  To use another copy,
for example a student's, set the environment variables CE507_PROJECT2_DIR (the
folder) and CE507_PROJECT2_MODULE (the file name without .py).  Project 2
needs scipy and the homework folder, neither of which the publishing workflow
has, so no page imports these files.
"""

import contextlib
import importlib
import os
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np

_NOTES = Path(__file__).resolve().parents[2]
PROJECT2_DIR = Path(os.environ.get("CE507_PROJECT2_DIR",
                                   _NOTES / "CE_ME_507_Homework" / "projects" / "Current"))
PROJECT2_MODULE = os.environ.get("CE507_PROJECT2_MODULE", "CE_ME_507_Project_2_Solution")
if not (PROJECT2_DIR / (PROJECT2_MODULE + ".py")).is_file():
    raise ImportError(f"{PROJECT2_MODULE}.py was not found in {PROJECT2_DIR}; "
                      "set CE507_PROJECT2_DIR to the folder that holds the Project 2 code")
if str(PROJECT2_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT2_DIR))
P2 = importlib.import_module(PROJECT2_MODULE)

bnd = P2.bnd                                   # Boundary_Conditions.py (Homework 13)
Face = P2.q1d.BoundaryFace                     # BoundaryFace (Homework 14)
DIRICHLET = bnd.BoundaryConditionType.DIRICHLET

# parent coordinates of the middle of each face
_FACE_MID = {Face.BOTTOM: [0.0, -1.0], Face.TOP: [0.0, 1.0],
             Face.LEFT: [-1.0, 0.0], Face.RIGHT: [1.0, 0.0]}

Side = namedtuple("Side", "e face x y n nodes")
Side.__doc__ = """A boundary face: element e, its BoundaryFace, the midpoint (x, y),
the outward unit normal n, and the global indices of its basis functions."""


# --------------------------------------------------------------------- mesh
def basis_mesh(grid, p):
    """A Project 2 BasisMesh of degree p on the elements of grid.

    grid is a c24_fem.Mesh of degree 1, from the make_mesh function of a
    c24_*.py file.  It is written as an OBJ file, one quadrilateral face per
    element listed counterclockwise from its lower-left corner, and read back
    with CreateBasisMesh.  No boundary file is given, so the boundaries of the
    BasisMesh are its closed boundary curves; boundary conditions are placed on
    parts of them with sides_where."""
    if grid.p != 1:
        raise ValueError("the grid must have degree 1; BasisMesh adds the other nodes")
    with tempfile.TemporaryDirectory(prefix="c24_proj2_") as tmp:
        obj = Path(tmp) / "mesh.obj"
        with open(obj, "w") as fh:
            for x, y in grid.X:
                fh.write(f"v {float(x)!r} {float(y)!r} 0\n")
            for c in grid.IEN:                     # local nodes a = i + 2 j
                fh.write(f"f {c[0] + 1} {c[1] + 1} {c[3] + 1} {c[2] + 1}\n")
        bm = P2.CreateBasisMesh(obj, "", p)
    P2.AppendSideBases(bm.basis)
    bm.X = np.array(bm.pts, float)
    ien = np.asarray(bm.IEN)
    # every element is an axis-aligned rectangle from its first to its last node
    bm.lo, bm.hi = bm.X[ien[:, 0]], bm.X[ien[:, -1]]
    if (np.abs(bm.lo - grid.X[grid.IEN[:, 0]]).max() > 0
            or np.abs(bm.hi - grid.X[grid.IEN[:, 3]]).max() > 0):
        raise RuntimeError("the elements of the BasisMesh do not follow the grid")
    bm.sides = _boundary_sides(bm)
    return bm


def _boundary_sides(bm):
    """Element faces that no other element shares."""
    count = {}
    for c in bm.IEN:
        for loc in bm.basis.side_bfs.values():
            key = frozenset(c[a] for a in loc)
            count[key] = count.get(key, 0) + 1
    sides = []
    for e, c in enumerate(bm.IEN):
        x_pts = [bm.pts[A] for A in c]
        for face, loc in bm.basis.side_bfs.items():
            nodes = [c[a] for a in loc]
            if count[frozenset(nodes)] > 1:
                continue
            xi = _FACE_MID[face]
            x = bm.basis.EvaluateSpatialMapping(x_pts, xi)
            n = P2.ExtractOutwardElementNormal(bm.basis, face, x_pts, xi)
            n = tuple(float(v) + 0.0 for v in np.round(n, 12))
            sides.append(Side(e, face, float(x[0]), float(x[1]), n, nodes))
    return sides


def sides_where(bm, pred):
    """The boundary sides for which pred(x, y, n) is true, with (x, y) the
    midpoint of the side and n its outward unit normal."""
    return [s for s in bm.sides if pred(s.x, s.y, s.n)]


def nodes_of(sides):
    """Global indices of the basis functions on a list of sides."""
    return sorted({A for s in sides for A in s.nodes})


def conductivity(k):
    """kappa(x, y) as Project 2 expects it, a 2 x 2 matrix, from a number or a
    pair (k11, k22)."""
    m = np.diag(np.asarray(k, float)) if np.ndim(k) == 1 else float(k) * np.eye(2)
    return lambda x, y: m


# ------------------------------------------------------------------- solve
class Solution:
    """d (the coefficient of every basis function) and what produced it."""

    def __init__(self, bm, d, kappa, f, boundaries, gauss, gauss_1d):
        self.bm, self.d, self.kappa, self.f = bm, d, kappa, f
        self.boundaries, self.gauss, self.gauss_1d = boundaries, gauss, gauss_1d


def solve(bm, kappa, f, boundaries, tied=()):
    """Solve -div(kappa grad u) = f with HeatConductionProblem.

    kappa(x, y) returns a 2 x 2 matrix and f(x, y) a number, as in Project 2.
    boundaries is a list of Dirichlet, Neumann, and Robin objects.  tied is a
    list of node lists; the nodes of each list share one unknown.  The
    quadrature is the one used for the brick, max(2, p + 1) points in each
    direction, which is exact for every problem here."""
    p = bm.basis.degs[0]
    nq = max(2, p + 1)
    gauss = P2.quad.GaussQuadratureQuadrilateral(nq)
    gauss_1d = P2.quad.GaussQuadrature1D(nq)
    with tied_nodes(tied):
        d = P2.HeatConductionProblem(bm, gauss, gauss_1d, f, kappa, boundaries)
    return Solution(bm, np.asarray(d, float), kappa, f, boundaries, gauss, gauss_1d)


@contextlib.contextmanager
def tied_nodes(groups):
    """Within this block, the ID array gives every node of each group the
    same equation number, so that the nodes of a group share one unknown.

    HeatConductionProblem builds its ID array with GetIDArray (Homework 13).
    This wraps that function: the Dirichlet nodes still receive -1, the nodes
    of each group receive the number of the first node of the group, and the
    remaining numbers are closed up so that they run 0, 1, 2, ...  Assembly
    then adds the equations of the group together, and ExtractTotalD gives
    every node of the group the same coefficient."""
    groups = [list(g) for g in groups if len(g)]
    if not groups:
        yield
        return
    get_id = P2.GetIDArray

    def GetIDArray(basis_mesh, boundaries):
        ID = np.array(get_id(basis_mesh, boundaries))
        for group in groups:
            if (ID[group] < 0).any():
                raise ValueError("a node that shares an unknown is also a Dirichlet node")
            ID[group] = ID[group[0]]
        used = np.unique(ID[ID >= 0])
        renumber = np.full(ID.max() + 1, -1)
        renumber[used] = np.arange(used.size)
        return np.where(ID >= 0, renumber[ID], -1)

    P2.GetIDArray = GetIDArray
    try:
        yield
    finally:
        P2.GetIDArray = get_id


# ---------------------------------------------------------- post-processing
def _x_pts(bm, e):
    return [bm.pts[A] for A in bm.IEN[e]]


def element_arrays(sol, e):
    """The element stiffness matrix (with Robin terms) and the element force
    vector from f and the Neumann and Robin data, without the Dirichlet
    values that HeatConductionProblem moves to the right-hand side."""
    bm = sol.bm
    x_pts = _x_pts(bm, e)
    bdries = P2.ExtractElementBoundaries(e, bm.IEN, bm.basis, sol.boundaries)
    ke = P2.la.LocalStiffnessMatrix(sol.kappa, bm.basis, x_pts, sol.gauss)
    if bdries:
        ke += P2.labnd.LocalStiffnessBoundaryTerms(bm.basis, bdries, x_pts, sol.gauss_1d)
    fe = P2.la.LocalForceVector(sol.f, bm.basis, x_pts, sol.gauss)
    natural = [b for b in bdries if b[1].type != DIRICHLET]
    if natural:
        fe += P2.labnd.LocalForceBoundaryTerms(bm.basis, natural, x_pts, sol.gauss_1d)
    return ke, fe


def reactions(sol):
    """K d - F at every node, where K and F are assembled over all nodes,
    including the Dirichlet ones.  At a Dirichlet node A this is the integral
    of N_A (kappa grad u).n over the boundary: the flow into the domain there,
    or the reaction.  At other nodes it is zero up to roundoff.  The energy
    (1/2) d.K d is found in the same pass; both are kept on sol."""
    if not hasattr(sol, "R"):
        R = np.zeros(len(sol.d))
        energy = 0.0
        for e, c in enumerate(sol.bm.IEN):
            ke, fe = element_arrays(sol, e)
            de = sol.d[c]
            np.add.at(R, c, ke @ de - fe)
            energy += 0.5 * de @ ke @ de
        sol.R, sol.energy = R, energy
    return sol.R


def energy(sol):
    """(1/2) d.K d, with K the stiffness matrix of all nodes."""
    reactions(sol)
    return sol.energy


def integrate(sol, fun=None):
    """Integral over the domain of fun(u, x, y) (default: u)."""
    bm, g = sol.bm, sol.gauss
    total = 0.0
    for e in range(len(bm.IEN)):
        x_pts = _x_pts(bm, e)
        for xi, w in zip(g.quad_pts, g.quad_wts):
            u = P2.EvaluateSolution(e, bm.basis, sol.d, bm.IEN, xi)
            if fun is not None:
                x = bm.basis.EvaluateSpatialMapping(x_pts, xi)
                u = fun(u, x[0], x[1])
            total += w * u * bm.basis.EvaluateJacobian(x_pts, xi)
    return float(total)


def side_integral(sol, sides, fun=None):
    """Integral along boundary sides of fun(u, x, y) (default: u)."""
    bm, g = sol.bm, sol.gauss_1d
    total = 0.0
    for s in sides:
        x_pts = _x_pts(bm, s.e)
        for xi, w in zip(P2.q1d.GetFaceQuadraturePoints(g, s.face), g.quad_wts):
            u = P2.EvaluateSolution(s.e, bm.basis, sol.d, bm.IEN, xi)
            if fun is not None:
                x = bm.basis.EvaluateSpatialMapping(x_pts, xi)
                u = fun(u, x[0], x[1])
            total += w * u * P2.q1d.JacobianOneD(xi, x_pts, bm.basis, s.face)
    return float(total)


def outflow(sol, sides):
    """Integral of (-kappa grad u).n over boundary sides, the flow out of the
    domain, from ExtractTotalBoundaryFlux."""
    boundary = bnd.NeumannBoundaryCondition(nodes_of(sides))
    return float(P2.ExtractTotalBoundaryFlux(sol.bm, boundary, sol.d, sol.kappa, sol.gauss_1d)["total"])


def locate(bm, x, y):
    """The first element that contains (x, y), and the parent coordinates
    (xi, eta) of the point.  Each element is an axis-aligned rectangle, so the
    map from the parent domain is inverted directly."""
    c, h = 0.5 * (bm.lo + bm.hi), bm.hi - bm.lo
    inside = np.all(np.abs(np.array([x, y]) - c) <= h / 2 * (1 + 1e-9), axis=1)
    if not inside.any():
        raise ValueError("point outside the mesh")
    e = int(np.argmax(inside))
    return e, 2 * (x - c[e, 0]) / h[e, 0], 2 * (y - c[e, 1]) / h[e, 1]


def value(sol, e, xi, eta):
    """u at the parent point (xi, eta) of element e."""
    return float(P2.EvaluateSolution(e, sol.bm.basis, sol.d, sol.bm.IEN, [xi, eta]))


def gradient(sol, e, xi, eta):
    """grad u at the parent point (xi, eta) of element e."""
    bm = sol.bm
    return np.asarray(P2.EvaluateSolutionGradient(e, bm.basis, sol.d, bm.IEN, _x_pts(bm, e), [xi, eta]))


def nodal_gradient(sol):
    """grad u at the nodes, averaged over the elements that share each node."""
    bm = sol.bm
    p = bm.basis.degs[0]
    nd = np.linspace(-1.0, 1.0, p + 1)
    G = np.zeros((len(bm.pts), 2))
    count = np.zeros(len(bm.pts))
    for e, c in enumerate(bm.IEN):
        x_pts = _x_pts(bm, e)
        for a, A in enumerate(c):
            xi = [nd[a % (p + 1)], nd[a // (p + 1)]]
            G[A] += P2.EvaluateSolutionGradient(e, bm.basis, sol.d, bm.IEN, x_pts, xi)
            count[A] += 1
    return G / count[:, None]


# ------------------------------------------------- figures and comparisons
class NotesSolution:
    """A Project 2 solution renumbered onto a c24_fem.Mesh, in the form that the
    figure functions of the c24_*.py files expect."""


def to_notes(sol, mesh):
    """sol on mesh, a c24_fem.Mesh of the same degree and element layout.
    Element e and local node a are the same in both meshes, so the coefficient
    of local node a of element e is copied across."""
    ien = np.asarray(sol.bm.IEN)
    if mesh.p != sol.bm.basis.degs[0] or mesh.IEN.shape != ien.shape:
        raise ValueError("the meshes do not match")
    scale = np.abs(mesh.X).max()
    if np.abs(mesh.X[mesh.IEN] - sol.bm.X[ien]).max() > 1e-12 * scale:
        raise ValueError("the nodes of the two meshes are not in the same places")
    out = NotesSolution()
    out.mesh, out.p = mesh, mesh.p
    out.u = np.zeros(mesh.nnp)
    out.u[mesh.IEN] = sol.d[ien]
    out.Kap = np.array([sol.kappa(x, y) for x, y in mesh.cent])
    return out


def view(model):
    """A copy of model in which the mesh is the c24_fem.Mesh of the notes and
    every Project 2 solution is renumbered onto it, so that the model can be
    passed to the figure functions of the matching c24_*.py file."""
    mesh = model["notes_mesh"]()
    out = dict(model)
    out["mesh"] = mesh
    for k, v in model.items():
        if isinstance(v, Solution):
            out[k] = to_notes(v, mesh)
    return out


def report(title, rows):
    """Print (label, notes value, Project 2 value) rows with their relative difference."""
    print(f"\n{title}")
    print(f"  {'quantity':<44}{'notes (c24_fem)':>18}{'Project 2':>18}{'rel. diff.':>12}")
    for label, a, b in rows:
        rel = abs(b - a) / abs(a) if a != 0 else abs(b - a)
        print(f"  {label:<44}{a:>18.10g}{b:>18.10g}{rel:>12.1e}")
