"""
Two-dimensional linear elasticity by Galerkin's method, for the two
elasticity appendix pages.

Q4 (bilinear) and Q9 (biquadratic) isoparametric elements on block-structured
meshes of four bodies: a cantilever (for which an exact solution is known), a
plate with a hole (whole, and a quarter that uses symmetry), and the cross
section of a W-shape.  elast2d.js is a port of this file for the explorer on
the second page -- change the two together.

Hughes' conventions: nodes A = 1..nnp, local nodes a = 1..nen, elements
e = 1..nel, directions i = 1, 2, the array ID(i, A) with 0 marking a
prescribed displacement component, and LM(i, a, e) = ID(i, IEN(a, e)).  The
arrays below are 0-based Python arrays that hold those 1-based numbers.
Local nodes: corners counter-clockwise from (xi, eta) = (-1, -1), then for Q9
the mid-side nodes counter-clockwise from the bottom, then the centre.

Units: millimetres, newtons, and megapascals (N/mm^2).  Plane stress uses a
thickness t; plane strain uses t = 1 mm, so forces are per millimetre of
length (N/mm = kN/m).

Only numpy is used, so the pages render on the CI image.
"""

import math

import numpy as np

# ------------------------------------------------------------------ elements
LOCAL_IJ = {1: [(0, 0), (1, 0), (1, 1), (0, 1)],
            2: [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)]}
SIDE_PARAM = {"bottom": lambda t: (t, -1.0), "right": lambda t: (1.0, t),
              "top": lambda t: (t, 1.0), "left": lambda t: (-1.0, t)}


def gauss(n):
    return np.polynomial.legendre.leggauss(n)


def lagrange1d(p, s):
    """Values and derivatives of the p + 1 Lagrange polynomials on [-1, 1]."""
    nodes = np.linspace(-1.0, 1.0, p + 1)
    N, dN = np.ones(p + 1), np.zeros(p + 1)
    for a in range(p + 1):
        for b in range(p + 1):
            if b != a:
                N[a] *= (s - nodes[b]) / (nodes[a] - nodes[b])
        for m in range(p + 1):
            if m == a:
                continue
            t = 1.0 / (nodes[a] - nodes[m])
            for b in range(p + 1):
                if b != a and b != m:
                    t *= (s - nodes[b]) / (nodes[a] - nodes[b])
            dN[a] += t
    return N, dN


def shape(p, xi, eta):
    """N_a and dN_a/d(xi, eta) of the element's nen = (p + 1)^2 functions."""
    Nx, dx = lagrange1d(p, xi)
    Ny, dy = lagrange1d(p, eta)
    N = np.array([Nx[i] * Ny[j] for i, j in LOCAL_IJ[p]])
    dN = np.array([[dx[i] * Ny[j], Nx[i] * dy[j]] for i, j in LOCAL_IJ[p]])
    return N, dN


def dmatrix(E, nu, plane="stress"):
    """D, the elasticity matrix in Voigt form: sigma = D eps, with
    sigma = (s11, s22, s12) and eps = (e11, e22, 2 e12)."""
    if plane == "stress":
        c = E / (1.0 - nu * nu)
        return c * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]])


def bmatrix(dNx):
    """B (3 x 2 nen) from the spatial derivatives dNx (nen x 2):
    columns 2a and 2a + 1 hold B_a = [[N,1, 0], [0, N,2], [N,2, N,1]]."""
    nen = dNx.shape[0]
    B = np.zeros((3, 2 * nen))
    B[0, 0::2] = dNx[:, 0]
    B[1, 1::2] = dNx[:, 1]
    B[2, 0::2] = dNx[:, 1]
    B[2, 1::2] = dNx[:, 0]
    return B


# ---------------------------------------------------------------- meshing
def graded(n, ratio=1.0):
    """n + 1 points from 0 to 1, each interval `ratio` times the one before."""
    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)
    w = ratio ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(w) / w.sum()])


def ratio_for(first, total, n):
    """The growth ratio that splits `total` into n intervals starting at `first`."""
    if n * first >= total:
        return 1.0
    lo, hi = 1.0 + 1e-12, 50.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        s = first * (r ** n - 1.0) / (r - 1.0)
        lo, hi = (r, hi) if s < total else (lo, r)
    return 0.5 * (lo + hi)


def line(P, Q):
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    return lambda t: (1.0 - t) * P + t * Q


def arc(r, a0, a1, c=(0.0, 0.0)):
    return lambda t: np.array([c[0] + r * math.cos(a0 + t * (a1 - a0)),
                               c[1] + r * math.sin(a0 + t * (a1 - a0))])


class Patch:
    """A four-sided region mapped from [0, 1]^2 by transfinite interpolation.
    Corners P00, P10, P11, P01 must run counter-clockwise.  s runs from the
    left side to the right side, t from the bottom to the top; sx and sy are
    the element boundaries in s and t; names gives each side a boundary name
    (None for a side inside the body)."""

    def __init__(self, P00, P10, P11, P01, sx, sy, curves=None, names=None):
        self.P = [np.asarray(v, float) for v in (P00, P10, P11, P01)]
        self.sx, self.sy = np.asarray(sx, float), np.asarray(sy, float)
        c = dict(bottom=line(P00, P10), right=line(P10, P11), top=line(P01, P11), left=line(P00, P01))
        c.update(curves or {})
        self.c = c
        self.names = dict(bottom=None, right=None, top=None, left=None)
        self.names.update(names or {})

    def point(self, s, t):
        c, (P00, P10, P11, P01) = self.c, self.P
        return ((1 - t) * c["bottom"](s) + t * c["top"](s) + (1 - s) * c["left"](t) + s * c["right"](t)
                - ((1 - s) * (1 - t) * P00 + s * (1 - t) * P10 + s * t * P11 + (1 - s) * t * P01))


def _params(v, p):
    if p == 1:
        return list(v)
    out = [v[0]]
    for k in range(len(v) - 1):
        out += [0.5 * (v[k] + v[k + 1]), v[k + 1]]
    return out


class Mesh:
    """Nodes X (nnp x 2), connectivity IEN (nel x nen, 0-based node indices),
    and the boundary sides: sides[name] = list of (e, local side)."""

    def __init__(self, patches, p, info=None):
        self.p, self.nen = p, (p + 1) ** 2
        scale = max(float(np.abs(Q).max()) for pt in patches for Q in pt.P)
        tol = 1e-7 * scale
        coords, buckets, ien, sides = [], {}, [], {}

        def node(x):
            k = (round(x[0] / tol), round(x[1] / tol))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for A in buckets.get((k[0] + dx, k[1] + dy), ()):
                        if abs(coords[A][0] - x[0]) <= tol and abs(coords[A][1] - x[1]) <= tol:
                            return A
            coords.append((float(x[0]), float(x[1])))
            buckets.setdefault(k, []).append(len(coords) - 1)
            return len(coords) - 1

        for pt in patches:
            S, T = _params(pt.sx, p), _params(pt.sy, p)
            grid = [[node(pt.point(s, t)) for t in T] for s in S]
            nx, ny = len(pt.sx) - 1, len(pt.sy) - 1
            for ey in range(ny):
                for ex in range(nx):
                    e = len(ien)
                    ien.append([grid[p * ex + i][p * ey + j] for i, j in LOCAL_IJ[p]])
                    for side, edge in (("bottom", ey == 0), ("right", ex == nx - 1),
                                       ("top", ey == ny - 1), ("left", ex == 0)):
                        if edge and pt.names[side]:
                            sides.setdefault(pt.names[side], []).append((e, side))
        X, IEN = np.array(coords), np.array(ien)
        perm = rcm(len(X), IEN)                     # new node k is old node perm[k]
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        self.X, self.IEN = X[perm], inv[IEN]
        self.nnp, self.nel = len(self.X), len(self.IEN)
        self.sides = sides
        self.info = info or {}

    def side_nodes(self, name):
        out = set()
        for e, sd in self.sides.get(name, ()):
            for a in side_local(self.p, sd):
                out.add(int(self.IEN[e, a]))
        return sorted(out)


def side_local(p, side):
    """Local node numbers (0-based) on one side of the parent square."""
    cs = {"bottom": [0, 1], "right": [1, 2], "top": [2, 3], "left": [3, 0]}[side]
    if p == 1:
        return cs
    mid = {"bottom": 4, "right": 5, "top": 6, "left": 7}[side]
    return [cs[0], mid, cs[1]]


def rcm(nnp, IEN):
    """Reverse Cuthill-McKee order of the nodes, to keep the band of K narrow.
    Ties are broken by node number, so elast2d.js gets the same order."""
    nbrs = [set() for _ in range(nnp)]
    for row in IEN:
        for A in row:
            nbrs[A].update(int(B) for B in row)
    adj = [sorted(s - {A}) for A, s in enumerate(nbrs)]
    deg = [len(a) for a in adj]
    seen, order = [False] * nnp, []
    while len(order) < nnp:
        start = min((A for A in range(nnp) if not seen[A]), key=lambda A: (deg[A], A))
        seen[start] = True
        queue, head = [start], 0
        while head < len(queue):
            A = queue[head]
            head += 1
            for B in sorted((B for B in adj[A] if not seen[B]), key=lambda B: (deg[B], B)):
                seen[B] = True
                queue.append(B)
        order += queue
    return np.array(order[::-1])


# ------------------------------------------------------------------ geometries
CANTILEVER = dict(L=400.0, D=100.0, t=10.0, P=8000.0)
PLATE = dict(L=200.0, W=100.0, a=12.5, t=10.0, s=50.0)
WSHAPE = dict(name="W250×73 (W10×49)", d=253.0, bf=254.0, tf=14.2, tw=8.6, q=1.0)


def cantilever_mesh(p=1, ny=2, nx=None, L=CANTILEVER["L"], D=CANTILEVER["D"]):
    nx = nx or 4 * ny
    pt = Patch((0, -D / 2), (L, -D / 2), (L, D / 2), (0, D / 2), graded(nx), graded(ny),
               names=dict(left="left", right="right", bottom="bottom", top="top"))
    return Mesh([pt], p, info=dict(kind="cantilever", L=L, D=D))


PLATE_LEVELS = {1: (2, 4), 2: (3, 6), 3: (4, 8), 4: (6, 12)}       # (m, radial divisions)


def plate_mesh(p=1, level=2, quarter=False, L=PLATE["L"], W=PLATE["W"], a=PLATE["a"]):
    """The rectangle [-L/2, L/2] x [-W/2, W/2] with a hole of radius a at the
    centre, as eight patches between the hole and the outer boundary (two for
    the quarter [0, L/2] x [0, W/2]).  In every patch s runs out from the hole
    and t runs counter-clockwise around it."""
    m, nr = PLATE_LEVELS[level]
    Lh, Wh = L / 2, W / 2
    tc = math.atan2(Wh, Lh)
    first = a * tc / m                                      # elements at the hole: about square
    ratio = ratio_for(first, Lh - a, nr)
    sx = graded(nr, ratio)

    def outer(th):
        c, s = math.cos(th), math.sin(th)
        k = min(Lh / abs(c) if abs(c) > 1e-12 else 1e30, Wh / abs(s) if abs(s) > 1e-12 else 1e30)
        return np.array([k * c, k * s])

    def side_of(th0, th1):
        mid = 0.5 * (th0 + th1)
        P = outer(mid)
        if abs(P[0] - Lh) < 1e-9:
            return "right"
        if abs(P[0] + Lh) < 1e-9:
            return "left"
        return "top" if P[1] > 0 else "bottom"

    cuts = [0.0, tc, math.pi / 2, math.pi - tc, math.pi, math.pi + tc, 1.5 * math.pi, 2 * math.pi - tc, 2 * math.pi]
    nts = [m, 2 * m, 2 * m, m, m, 2 * m, 2 * m, m]
    if quarter:
        cuts, nts = cuts[:3], nts[:2]
    patches = []
    for k in range(len(cuts) - 1):
        a0, a1 = cuts[k], cuts[k + 1]
        names = dict(left="hole", right=side_of(a0, a1))
        if quarter and k == 0:
            names["bottom"] = "bottom"
        if quarter and k == len(cuts) - 2:
            names["top"] = "left"
        P00, P01 = np.array([a * math.cos(a0), a * math.sin(a0)]), np.array([a * math.cos(a1), a * math.sin(a1)])
        P10, P11 = outer(a0), outer(a1)
        patches.append(Patch(P00, P10, P11, P01, sx, graded(nts[k]),
                             curves=dict(left=arc(a, a0, a1)), names=names))
    return Mesh(patches, p, info=dict(kind="plate", quarter=quarter, L=L, W=W, a=a))


def wshape_mesh(p=1, level=2, d=WSHAPE["d"], bf=WSHAPE["bf"], tf=WSHAPE["tf"], tw=WSHAPE["tw"]):
    """The W-shape's cross-section as seven rectangles: three in each flange
    (the outstands and the part above or below the web) and the web."""
    h = tf / level
    ntf = level
    ntw = max(1, round(tw / h))
    no = max(2, round((bf - tw) / 2 / h))
    nh = max(2, round((d - 2 * tf) / h))
    xs = [-bf / 2, -tw / 2, tw / 2, bf / 2]
    nxs = [no, ntw, no]
    patches = []
    for (y0, y1, which) in ((0.0, tf, "bot"), (d - tf, d, "top")):
        for k in range(3):
            x0, x1 = xs[k], xs[k + 1]
            names = {}
            if which == "bot":
                names["bottom"] = "bottom face"
                if k != 1:
                    names["top"] = "bottom flange, inside left" if k == 0 else "bottom flange, inside right"
            else:
                names["top"] = "top face"
                if k != 1:
                    names["bottom"] = "top flange, inside left" if k == 0 else "top flange, inside right"
            if k == 0:
                names["left"] = "bottom-left tip" if which == "bot" else "top-left tip"
            if k == 2:
                names["right"] = "bottom-right tip" if which == "bot" else "top-right tip"
            patches.append(Patch((x0, y0), (x1, y0), (x1, y1), (x0, y1), graded(nxs[k]), graded(ntf), names=names))
    patches.insert(3, Patch((-tw / 2, tf), (tw / 2, tf), (tw / 2, d - tf), (-tw / 2, d - tf), graded(ntw), graded(nh),
                            names=dict(left="web, left face", right="web, right face")))
    return Mesh(patches, p, info=dict(kind="wshape", d=d, bf=bf, tf=tf, tw=tw))


# ---------------------------------------------------------- the solution
class Solution:
    pass


def element_arrays(mesh, e, D, t=1.0, nq=None):
    """k^e and the Gauss-point data of element e."""
    p = mesh.p
    xg, wg = gauss(nq or p + 1)
    xe = mesh.X[mesh.IEN[e]]
    ke = np.zeros((2 * mesh.nen, 2 * mesh.nen))
    pts = []
    for i, xi in enumerate(xg):
        for j, eta in enumerate(xg):
            N, dN = shape(p, xi, eta)
            DF = xe.T @ dN                        # dx/dxi, 2 x 2
            J = np.linalg.det(DF)
            dNx = dN @ np.linalg.inv(DF)          # dN/dx = dN/dxi . dxi/dx
            B = bmatrix(dNx)
            ke += (wg[i] * wg[j] * J * t) * (B.T @ D @ B)
            pts.append((xi, eta, wg[i] * wg[j], J))
    return ke, pts


def side_load(mesh, e, side, h, t=1.0, nq=None):
    """f_a,i = integral over one element side of N_a h_i t dGamma, for a
    traction h(x, y) -> (h1, h2).  Returns a (nen, 2) array."""
    p = mesh.p
    xg, wg = gauss(nq or p + 2)
    xe = mesh.X[mesh.IEN[e]]
    fe = np.zeros((mesh.nen, 2))
    for s, w in zip(xg, wg):
        xi, eta = SIDE_PARAM[side](s)
        N, dN = shape(p, xi, eta)
        DF = xe.T @ dN
        tang = DF[:, 0] if side in ("bottom", "top") else DF[:, 1]
        x = N @ xe
        fe += np.outer(N, np.asarray(h(x[0], x[1]), float)) * (w * np.hypot(*tang) * t)
    return fe


def solve(mesh, E, nu, plane="stress", t=None, dirichlet=None, traction=None, body=None):
    """Solve the plane elasticity problem.

    dirichlet: {side name: {0 or 1: g(x, y)}}, the prescribed u_i (mm).
    traction:  {side name: h(x, y) -> (h1, h2)}, the prescribed traction (MPa).
    body:      f(x, y) -> (f1, f2), the body force (N/mm^3), or None.
    t:         the thickness (mm) in plane stress; 1 in plane strain.
    """
    t = (t if t is not None else 1.0) if plane == "stress" else 1.0
    D = dmatrix(E, nu, plane)
    dirichlet, traction = dirichlet or {}, traction or {}
    nnp, nen = mesh.nnp, mesh.nen
    # prescribed values; a node on two Dirichlet sides gets the average
    gsum, gcnt = np.zeros((2, nnp)), np.zeros((2, nnp))
    for name, comps in dirichlet.items():
        for A in mesh.side_nodes(name):
            for i, g in comps.items():
                gsum[i, A] += g(*mesh.X[A])
                gcnt[i, A] += 1
    fixed = gcnt > 0
    gval = np.where(fixed, gsum / np.maximum(gcnt, 1), 0.0)
    ID = np.zeros((2, nnp), dtype=int)
    neq = 0
    for A in range(nnp):                                   # node by node, i = 1, 2 at each node
        for i in range(2):
            if not fixed[i, A]:
                neq += 1
                ID[i, A] = neq
    LM = np.array([[ID[i, mesh.IEN[e, a]] for a in range(nen) for i in range(2)] for e in range(mesh.nel)])
    # element arrays and the all-node load vector
    Ffull = np.zeros((nnp, 2))
    for name, h in traction.items():
        for e, sd in mesh.sides.get(name, ()):
            Ffull[mesh.IEN[e]] += side_load(mesh, e, sd, h, t)
    kes = []
    rows, cols, vals = [], [], []
    F = np.zeros(neq)
    minJ = np.inf
    for e in range(mesh.nel):
        ke, pts = element_arrays(mesh, e, D, t)
        minJ = min(minJ, min(q[3] for q in pts))
        if body is not None:
            Ffull[mesh.IEN[e]] += body_load(mesh, e, body, t)
        kes.append(ke)
        lm = LM[e]
        ge = np.array([gval[i, mesh.IEN[e, a]] for a in range(nen) for i in range(2)])
        fe = -ke @ ge
        for q in range(2 * nen):
            P = lm[q]
            if P:
                F[P - 1] += fe[q]
                for r in range(2 * nen):
                    Q = lm[r]
                    if Q:
                        rows.append(P - 1)
                        cols.append(Q - 1)
                        vals.append(ke[q, r])
    for A in range(nnp):
        for i in range(2):
            if ID[i, A]:
                F[ID[i, A] - 1] += Ffull[A, i]
    sol = Solution()
    sol.mesh, sol.D, sol.t, sol.E, sol.nu, sol.plane = mesh, D, t, E, nu, plane
    sol.ID, sol.LM, sol.neq, sol.kes, sol.F, sol.Ffull, sol.gval, sol.fixed = ID, LM, neq, kes, F, Ffull, gval, fixed
    sol.minJ = minJ
    try:
        d = banded_solve(np.array(rows), np.array(cols), np.array(vals), neq, F) if neq else np.zeros(0)
        sol.ok = bool(np.all(np.isfinite(d)))
    except np.linalg.LinAlgError:
        d, sol.ok = np.full(neq, np.nan), False
    U = gval.T.copy()
    for A in range(nnp):
        for i in range(2):
            if ID[i, A]:
                U[A, i] = d[ID[i, A] - 1]
    sol.d, sol.U = d, U
    # reactions: K_full U - F_full at the prescribed components
    R = -Ffull.copy()
    for e in range(mesh.nel):
        ue = U[mesh.IEN[e]].reshape(-1)
        R[mesh.IEN[e]] += (kes[e] @ ue).reshape(-1, 2)
    sol.R = np.where(fixed.T, R, 0.0)
    return sol


def body_load(mesh, e, f, t=1.0):
    p = mesh.p
    xg, wg = gauss(p + 1)
    xe = mesh.X[mesh.IEN[e]]
    fe = np.zeros((mesh.nen, 2))
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, dN = shape(p, xi, eta)
            J = np.linalg.det(xe.T @ dN)
            x = N @ xe
            fe += np.outer(N, np.asarray(f(x[0], x[1]), float)) * (wi * wj * J * t)
    return fe


def banded_solve(rows, cols, vals, n, b):
    """Solve A x = b for a symmetric positive definite A given as coordinate
    triples, by Cholesky factorization of its block-tridiagonal form (block
    size = half-bandwidth + 1).  Raises LinAlgError if A is singular, which
    happens when the supports do not stop every rigid-body motion."""
    bw = int(np.max(np.abs(rows - cols))) if len(rows) else 0
    m = max(bw + 1, 1)
    nb = -(-n // m)
    D = np.zeros((nb, m, m))
    S = np.zeros((max(nb - 1, 1), m, m))
    bi, bj, li, lj = rows // m, cols // m, rows % m, cols % m
    on = bi == bj
    np.add.at(D, (bi[on], li[on], lj[on]), vals[on])
    lo = bi == bj + 1
    np.add.at(S, (bj[lo], li[lo], lj[lo]), vals[lo])
    for k in range(n, nb * m):
        D[k // m, k % m, k % m] = 1.0
    scale = np.abs(np.diagonal(D, axis1=1, axis2=2)).max()
    L, C = np.zeros_like(D), np.zeros_like(S)

    def chol(M):
        Lm = np.linalg.cholesky(M)
        if np.min(np.diag(Lm)) < 1e-6 * math.sqrt(scale):
            raise np.linalg.LinAlgError("singular")
        return Lm

    L[0] = chol(D[0])
    for k in range(1, nb):
        C[k - 1] = np.linalg.solve(L[k - 1], S[k - 1].T).T
        L[k] = chol(D[k] - C[k - 1] @ C[k - 1].T)
    r = np.zeros(nb * m)
    r[:n] = b
    r = r.reshape(nb, m)
    y = np.zeros((nb, m))
    y[0] = np.linalg.solve(L[0], r[0])
    for k in range(1, nb):
        y[k] = np.linalg.solve(L[k], r[k] - C[k - 1] @ y[k - 1])
    x = np.zeros((nb, m))
    x[-1] = np.linalg.solve(L[-1].T, y[-1])
    for k in range(nb - 2, -1, -1):
        x[k] = np.linalg.solve(L[k].T, y[k] - C[k].T @ x[k + 1])
    return x.reshape(-1)[:n]


# ------------------------------------------------------------- evaluation
def at(sol, e, xi, eta):
    """Position, displacement, strain (Voigt), and stress at a parent point."""
    mesh = sol.mesh
    N, dN = shape(mesh.p, xi, eta)
    xe = mesh.X[mesh.IEN[e]]
    DF = xe.T @ dN
    dNx = dN @ np.linalg.inv(DF)
    ue = sol.U[mesh.IEN[e]]
    eps = bmatrix(dNx) @ ue.reshape(-1)
    sig = sol.D @ eps
    return dict(x=N @ xe, u=N @ ue, eps=eps, sig=sig, s33=s33(sol, sig))


def s33(sol, sig):
    return sol.nu * (sig[0] + sig[1]) if sol.plane == "strain" else 0.0


def von_mises(s11, s22, s12, s33=0.0):
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12 ** 2)


def nodal_stress(sol):
    """sigma at the nodes, averaged over the elements that share each node."""
    mesh = sol.mesh
    pn = np.linspace(-1, 1, mesh.p + 1)
    S, cnt = np.zeros((mesh.nnp, 3)), np.zeros(mesh.nnp)
    for e in range(mesh.nel):
        for a, (i, j) in enumerate(LOCAL_IJ[mesh.p]):
            r = at(sol, e, pn[i], pn[j])
            S[mesh.IEN[e, a]] += r["sig"]
            cnt[mesh.IEN[e, a]] += 1
    return S / cnt[:, None]


def locate(mesh, x, y, tries=8):
    """The element containing (x, y) and the parent coordinates there (Newton)."""
    best = None
    for e in range(mesh.nel):
        xe = mesh.X[mesh.IEN[e]]
        lo, hi = xe.min(axis=0) - 1e-9, xe.max(axis=0) + 1e-9
        if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
            continue
        z = np.zeros(2)
        for _ in range(tries * 3):
            N, dN = shape(mesh.p, *z)
            r = N @ xe - (x, y)
            z = z - np.linalg.solve(xe.T @ dN, r)
        if np.all(np.abs(z) <= 1 + 1e-7):
            return e, z[0], z[1]
        if best is None or np.max(np.abs(z)) < np.max(np.abs(best[1:])):
            best = (e, *z)
    return best


def reaction_sum(sol, name):
    out = np.zeros(2)
    for A in sol.mesh.side_nodes(name):
        out += sol.R[A]
    return out * 1.0


def energy(sol):
    """(1/2) d.K d, the strain energy (N mm)."""
    U = sol.U
    return 0.5 * sum(U[sol.mesh.IEN[e]].reshape(-1) @ sol.kes[e] @ U[sol.mesh.IEN[e]].reshape(-1)
                     for e in range(sol.mesh.nel))


# ----------------------------------------------- the exact cantilever solution
def cantilever_exact(x, y, E, nu, plane="stress", L=CANTILEVER["L"], D=CANTILEVER["D"],
                     t=CANTILEVER["t"], P=CANTILEVER["P"]):
    """Timoshenko and Goodier's solution for a cantilever of length L and
    depth D, fixed at x = 0 and loaded at x = L by a downward force P,
    distributed over the end as a parabolic shear traction.  y is measured
    from the mid-depth.  Plane strain replaces E by E/(1 - nu^2) and nu by
    nu/(1 - nu), with t = 1 (P per unit length)."""
    if plane == "strain":
        E, nu, t = E / (1 - nu * nu), nu / (1 - nu), 1.0
    I = t * D ** 3 / 12.0
    u1 = P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y * y - D * D / 4))
    u2 = -P / (6 * E * I) * (3 * nu * y * y * (L - x) + (4 + 5 * nu) * D * D * x / 4 + (3 * L - x) * x * x)
    s11 = P * (L - x) * y / I
    s12 = -P / (2 * I) * (D * D / 4 - y * y)
    return dict(u1=u1, u2=u2, s11=s11, s22=0.0 * x, s12=s12)


def cantilever_bc(E, nu, plane="stress", P=CANTILEVER["P"], clamp=False):
    """The boundary data for the exact solution: on the left end, the exact
    displacements (or u = 0 if clamp); on the right end, the parabolic shear."""
    ex = lambda x, y, k: cantilever_exact(x, y, E, nu, plane, P=P)[k]
    if clamp:
        dirichlet = {"left": {0: lambda x, y: 0.0, 1: lambda x, y: 0.0}}
    else:
        dirichlet = {"left": {0: lambda x, y: ex(x, y, "u1"), 1: lambda x, y: ex(x, y, "u2")}}
    traction = {"right": lambda x, y: (0.0, ex(x, y, "s12"))}
    return dirichlet, traction


def energy_error(sol, exact, nq=None):
    """|| u - u^h ||_E = sqrt(integral of (eps - eps^h).D.(eps - eps^h) t dOmega),
    and the energy of the exact solution, for the relative error."""
    mesh = sol.mesh
    xg, wg = gauss(nq or mesh.p + 3)
    Dinv = np.linalg.inv(sol.D)
    err = tot = 0.0
    for e in range(mesh.nel):
        xe = mesh.X[mesh.IEN[e]]
        for xi, wi in zip(xg, wg):
            for eta, wj in zip(xg, wg):
                N, dN = shape(mesh.p, xi, eta)
                J = np.linalg.det(xe.T @ dN)
                r = at(sol, e, xi, eta)
                ex = exact(*r["x"])
                sig = np.array([ex["s11"], ex["s22"], ex["s12"]])
                ds = sig - r["sig"]
                err += wi * wj * J * sol.t * ds @ Dinv @ ds
                tot += wi * wj * J * sol.t * sig @ Dinv @ sig
    return math.sqrt(err), math.sqrt(tot)


# --------------------------------------------------- the plate with a hole
def kirsch(x, y, s=PLATE["s"], a=PLATE["a"]):
    """Kirsch's stresses around a hole of radius a in an infinite plate under a
    remote tension s along x1 (polar formulas converted to x1, x2)."""
    r, th = math.hypot(x, y), math.atan2(y, x)
    q = (a / r) ** 2
    srr = s / 2 * (1 - q) + s / 2 * (1 - 4 * q + 3 * q * q) * math.cos(2 * th)
    stt = s / 2 * (1 + q) - s / 2 * (1 + 3 * q * q) * math.cos(2 * th)
    srt = -s / 2 * (1 + 2 * q - 3 * q * q) * math.sin(2 * th)
    c, sn = math.cos(th), math.sin(th)
    s11 = srr * c * c + stt * sn * sn - 2 * srt * sn * c
    s22 = srr * sn * sn + stt * c * c + 2 * srt * sn * c
    s12 = (srr - stt) * sn * c + srt * (c * c - sn * sn)
    return s11, s22, s12


def heywood(d, W, s):
    """Heywood's approximate peak stress for a finite plate of width W with a
    central hole of diameter d under a remote (gross) tension s:
    Kt = 2 + (1 - d/W)^3, applied to the net-section stress s W / (W - d)."""
    kt = 2 + (1 - d / W) ** 3
    return kt, kt * s * W / (W - d)
