"""
Finite elements for the Class 24 application pages.

Solves the scalar problem used throughout the course,

    -div(kappa grad u) = f                   in Omega
    u = g                                    on Gamma_g
    (kappa grad u) . n = h                   on Gamma_h
    a_hat u + (kappa grad u) . n = c_hat     on Gamma_r

on a tensor-product grid of axis-aligned rectangles, with Lagrange elements of
degree p = 1 (Q4) or p = 2 (Q9).  The grid may have

  * uneven spacing (graded toward corners and edges where the solution is
    not smooth),
  * elements switched off, to cut a shape such as an I-section out of its
    bounding box, and
  * slits: a grid line along which the elements on the two sides do not share
    nodes, to represent a thin impermeable wall such as a cutoff.

Because every element is an axis-aligned rectangle, the isoparametric map is
affine and diagonal, and the element matrices follow from a few reference
integrals.  Local nodes use the course ordering a = i + (p + 1) j.

Only numpy is used, so the pages render on the CI image.
"""

import numpy as np


# ---------------------------------------------------------------- 1D grids
def graded(a, b, n, ratio=1.0, toward="a"):
    """n + 1 points from a to b.  Successive spacings change by 'ratio';
    the smallest spacing is at 'a', at 'b', or at both ends ('both')."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if toward == "both":
        if n % 2:
            raise ValueError("use an even n when grading toward both ends")
        m = 0.5 * (a + b)
        left = graded(a, m, n // 2, ratio, "a")
        right = graded(m, b, n // 2, ratio, "b")
        return np.concatenate([left, right[1:]])
    w = ratio ** np.arange(n)
    if toward == "b":
        w = w[::-1]
    s = np.concatenate([[0.0], np.cumsum(w)]) / w.sum()
    return a + (b - a) * s


def join(*segments):
    """Concatenate 1D point arrays that share their end points."""
    out = [np.asarray(segments[0], float)]
    for s in segments[1:]:
        s = np.asarray(s, float)
        if abs(s[0] - out[-1][-1]) > 1e-12 * max(1.0, abs(s[0])):
            raise ValueError("segments do not meet")
        out.append(s[1:])
    return np.concatenate(out)


# ---------------------------------------------------------- reference element
def lagrange1d(p, s):
    s = np.atleast_1d(np.asarray(s, float))
    nd = np.linspace(-1.0, 1.0, p + 1)
    N = np.ones((p + 1, s.size))
    dN = np.zeros((p + 1, s.size))
    for i in range(p + 1):
        for k in range(p + 1):
            if k != i:
                N[i] *= (s - nd[k]) / (nd[i] - nd[k])
        for m in range(p + 1):
            if m == i:
                continue
            t = np.full(s.size, 1.0 / (nd[i] - nd[m]))
            for k in range(p + 1):
                if k not in (i, m):
                    t *= (s - nd[k]) / (nd[i] - nd[k])
            dN[i] += t
    return N, dN


def shape2d(p, xi, eta):
    """N_a, dN_a/dxi, dN_a/deta at points, rows a = i + (p+1) j."""
    Nx, dNx = lagrange1d(p, xi)
    Ny, dNy = lagrange1d(p, eta)
    idx = [(i, j) for j in range(p + 1) for i in range(p + 1)]
    N = np.array([Nx[i] * Ny[j] for i, j in idx])
    Nxi = np.array([dNx[i] * Ny[j] for i, j in idx])
    Neta = np.array([Nx[i] * dNy[j] for i, j in idx])
    return N, Nxi, Neta


def _reference(p, nq):
    g, w = np.polynomial.legendre.leggauss(nq)
    XI, ETA = np.meshgrid(g, g, indexing="ij")
    W = np.outer(w, w).ravel()
    N, Nxi, Neta = shape2d(p, XI.ravel(), ETA.ravel())
    Sxx = (Nxi * W) @ Nxi.T
    Syy = (Neta * W) @ Neta.T
    Sxy = (Nxi * W) @ Neta.T
    return dict(g=g, w=w, XI=XI.ravel(), ETA=ETA.ravel(), W=W, N=N,
                Nxi=Nxi, Neta=Neta, Sxx=Sxx, Syy=Syy, Sxy=Sxy)


SIDES = ("bottom", "right", "top", "left")


def side_local_nodes(p, side):
    if side == "bottom":
        return [i for i in range(p + 1)]
    if side == "top":
        return [i + (p + 1) * p for i in range(p + 1)]
    if side == "left":
        return [(p + 1) * j for j in range(p + 1)]
    return [p + (p + 1) * j for j in range(p + 1)]


# --------------------------------------------------------------------- mesh
class Mesh:
    """Tensor-product grid with element boundaries xs (nx+1) and ys (ny+1).

    active(xc, yc) -> bool decides whether the element centered at (xc, yc)
    exists.  slits is a list of ('v', x, y0, y1) or ('h', y, x0, x1): along
    that segment the elements on the two sides do not share nodes, except at
    an end of the segment that lies inside the domain (the tip)."""

    def __init__(self, xs, ys, p=1, active=None, slits=()):
        self.xs, self.ys, self.p = np.asarray(xs, float), np.asarray(ys, float), p
        nx, ny = len(xs) - 1, len(ys) - 1
        self.nx, self.ny = nx, ny
        # node grid: p sub-intervals per element
        gx = [xs[0]]
        for i in range(nx):
            gx += list(np.linspace(xs[i], xs[i + 1], p + 1)[1:])
        gy = [ys[0]]
        for j in range(ny):
            gy += list(np.linspace(ys[j], ys[j + 1], p + 1)[1:])
        self.gx, self.gy = np.array(gx), np.array(gy)
        xc = 0.5 * (self.xs[:-1] + self.xs[1:])
        yc = 0.5 * (self.ys[:-1] + self.ys[1:])
        act = np.ones((nx, ny), bool)
        if active is not None:
            for i in range(nx):
                for j in range(ny):
                    act[i, j] = bool(active(xc[i], yc[j]))
        self.act = act
        tol = 1e-9 * max(xs[-1] - xs[0], ys[-1] - ys[0])

        def on_slit(I, J):
            x, y = self.gx[I], self.gy[J]
            for s in slits:
                if s[0] == "v" and abs(x - s[1]) < tol and min(s[2], s[3]) - tol < y < max(s[2], s[3]) + tol:
                    tip = [v for v in (s[2], s[3]) if ys[0] + tol < v < ys[-1] - tol]
                    if any(abs(y - t) < tol for t in tip):
                        return None
                    return s
                if s[0] == "h" and abs(y - s[1]) < tol and min(s[2], s[3]) - tol < x < max(s[2], s[3]) + tol:
                    tip = [v for v in (s[2], s[3]) if xs[0] + tol < v < xs[-1] - tol]
                    if any(abs(x - t) < tol for t in tip):
                        return None
                    return s
            return None

        # element connectivity by node keys (I, J, copy)
        p1 = p + 1
        keys = {}
        elems, ecent, esize = [], [], []
        for i in range(nx):
            for j in range(ny):
                if not act[i, j]:
                    continue
                conn = []
                for jj in range(p1):
                    for ii in range(p1):
                        I, J = p * i + ii, p * j + jj
                        s = on_slit(I, J)
                        copy = 0
                        if s is not None:
                            if s[0] == "v":
                                copy = 1 if xc[i] > s[1] else 0
                            else:
                                copy = 1 if yc[j] > s[1] else 0
                        conn.append((I, J, copy))
                elems.append(conn)
                ecent.append((xc[i], yc[j]))
                esize.append((self.xs[i + 1] - self.xs[i], self.ys[j + 1] - self.ys[j]))
        # number nodes along the shorter grid direction to keep the band narrow
        allkeys = sorted({k for c in elems for k in c},
                         key=(lambda k: (k[0], k[1], k[2])) if len(gy) <= len(gx)
                         else (lambda k: (k[1], k[0], k[2])))
        num = {k: n for n, k in enumerate(allkeys)}
        self.keys = allkeys
        self.X = np.array([(self.gx[k[0]], self.gy[k[1]]) for k in allkeys])
        self.IEN = np.array([[num[k] for k in c] for c in elems])
        self.cent = np.array(ecent)
        self.size = np.array(esize)
        self.nel, self.nnp = len(elems), len(allkeys)
        # boundary sides: sides whose node set is not shared by another element
        count = {}
        for e in range(self.nel):
            for sd in SIDES:
                k = frozenset(self.IEN[e, side_local_nodes(p, sd)])
                count[k] = count.get(k, 0) + 1
        self.bsides = []
        for e in range(self.nel):
            for sd in SIDES:
                nodes = self.IEN[e, side_local_nodes(p, sd)]
                if count[frozenset(nodes)] == 1:
                    self.bsides.append((e, sd))

    # -- geometry helpers
    def side_geometry(self, e, sd):
        """Midpoint, length, and outward unit normal of a boundary side."""
        (xc, yc), (hx, hy) = self.cent[e], self.size[e]
        if sd == "bottom":
            return (xc, yc - hy / 2), hx, (0.0, -1.0)
        if sd == "top":
            return (xc, yc + hy / 2), hx, (0.0, 1.0)
        if sd == "left":
            return (xc - hx / 2, yc), hy, (-1.0, 0.0)
        return (xc + hx / 2, yc), hy, (1.0, 0.0)

    def sides_where(self, pred):
        out = []
        for e, sd in self.bsides:
            (xm, ym), L, n = self.side_geometry(e, sd)
            if pred(xm, ym, n):
                out.append((e, sd))
        return out

    def nodes_of_sides(self, sides):
        s = set()
        for e, sd in sides:
            s.update(self.IEN[e, side_local_nodes(self.p, sd)].tolist())
        return sorted(s)

    def triangles(self):
        """Split every element (and every Q9 sub-square) into two triangles."""
        p, p1 = self.p, self.p + 1
        tri = []
        for e in range(self.nel):
            c = self.IEN[e]
            for j in range(p):
                for i in range(p):
                    a, b = c[i + p1 * j], c[i + 1 + p1 * j]
                    d, q = c[i + p1 * (j + 1)], c[i + 1 + p1 * (j + 1)]
                    tri += [(a, b, q), (a, q, d)]
        return np.array(tri)


# ------------------------------------------------------------------- solver
def _block_cholesky_solve(rows, cols, vals, n, b):
    """Solve A x = b for a symmetric positive definite banded A given in
    coordinate form.  A is stored as dense diagonal and sub-diagonal blocks
    (block tridiagonal, block size = bandwidth + 1) and factored blockwise."""
    bw = int(np.max(np.abs(rows - cols))) if len(rows) else 0
    m = max(bw + 1, 1)
    nb = -(-n // m)
    npad = nb * m
    D = np.zeros((nb, m, m))
    S = np.zeros((max(nb - 1, 1), m, m))     # S[k] = A[block k+1, block k]
    bi, bj = rows // m, cols // m
    li, lj = rows % m, cols % m
    on = bi == bj
    np.add.at(D, (bi[on], li[on], lj[on]), vals[on])
    lo = bi == bj + 1
    np.add.at(S, (bj[lo], li[lo], lj[lo]), vals[lo])
    for k in range(n, npad):                  # pad with the identity
        D[k // m, k % m, k % m] = 1.0
    rhs = np.zeros(npad)
    rhs[:n] = b
    L = np.zeros_like(D)
    C = np.zeros_like(S)
    L[0] = np.linalg.cholesky(D[0])
    for k in range(1, nb):
        C[k - 1] = np.linalg.solve(L[k - 1], S[k - 1].T).T
        L[k] = np.linalg.cholesky(D[k] - C[k - 1] @ C[k - 1].T)
    y = np.zeros((nb, m))
    r = rhs.reshape(nb, m)
    y[0] = np.linalg.solve(L[0], r[0])
    for k in range(1, nb):
        y[k] = np.linalg.solve(L[k], r[k] - C[k - 1] @ y[k - 1])
    x = np.zeros((nb, m))
    x[-1] = np.linalg.solve(L[-1].T, y[-1])
    for k in range(nb - 2, -1, -1):
        x[k] = np.linalg.solve(L[k].T, y[k] - C[k].T @ x[k + 1])
    return x.ravel()[:n]


class Solution:
    pass


def solve(mesh, kappa=1.0, f=0.0, dirichlet=(), neumann=(), robin=(), nodes_fixed=None, tied=()):
    """Assemble and solve.

    kappa: number, (k11, k22), or 2x2 array, or a function of the element
           center (x, y) returning any of these.
    f:     number or function f(x, y).
    dirichlet: list of (sides, g) where sides is a list from mesh.sides_where
           and g is a number or g(x, y).
    nodes_fixed: optional dict {node: value} for Dirichlet nodes that are not
           on a boundary side (an embedded electrode, for example).
    neumann: list of (sides, h) with h a number or h(x, y).
    robin:   list of (sides, a_hat, c_hat) with numbers.
    tied:    list of node lists; all nodes in a list share one unknown value
             (a rigid region that is free to move, such as the boss of a
             membrane).  Their equations are added together.
    """
    p, nen = mesh.p, (mesh.p + 1) ** 2
    ref = _reference(p, p + 2)
    hx, hy = mesh.size[:, 0], mesh.size[:, 1]

    def kap(xc, yc):
        k = kappa(xc, yc) if callable(kappa) else kappa
        k = np.asarray(k, float)
        if k.ndim == 0:
            return np.diag([float(k), float(k)])
        if k.ndim == 1:
            return np.diag(k)
        return k
    Kap = np.array([kap(*c) for c in mesh.cent])           # (nel, 2, 2)
    ke = (Kap[:, 0, 0, None, None] * (hy / hx)[:, None, None] * ref["Sxx"]
          + Kap[:, 1, 1, None, None] * (hx / hy)[:, None, None] * ref["Syy"]
          + Kap[:, 0, 1, None, None] * (ref["Sxy"] + ref["Sxy"].T))
    # body force
    xq = mesh.cent[:, 0, None] + 0.5 * hx[:, None] * ref["XI"][None, :]
    yq = mesh.cent[:, 1, None] + 0.5 * hy[:, None] * ref["ETA"][None, :]
    fq = f(xq, yq) if callable(f) else np.full_like(xq, float(f))
    fe = (fq * ref["W"][None, :]) @ ref["N"].T * (0.25 * hx * hy)[:, None]
    rows = np.repeat(mesh.IEN, nen, axis=1).ravel()
    cols = np.tile(mesh.IEN, (1, nen)).ravel()
    vals = ke.reshape(mesh.nel, -1).ravel()
    F = np.zeros(mesh.nnp)
    np.add.at(F, mesh.IEN.ravel(), fe.ravel())
    # boundary terms on element sides
    g1, w1 = np.polynomial.legendre.leggauss(p + 2)
    N1, _ = lagrange1d(p, g1)
    M1 = (N1 * w1) @ N1.T                                  # int N_a N_b dt on [-1,1]
    rr, cc, vv = [rows], [cols], [vals]

    def side_points(e, sd):
        (xm, ym), Ls, n = mesh.side_geometry(e, sd)
        if sd in ("bottom", "top"):
            xs_ = xm + 0.5 * Ls * g1
            ys_ = np.full_like(g1, ym)
        else:
            xs_ = np.full_like(g1, xm)
            ys_ = ym + 0.5 * Ls * g1
        return xs_, ys_, Ls

    for sides, h in neumann:
        for e, sd in sides:
            xs_, ys_, Ls = side_points(e, sd)
            hv = h(xs_, ys_) if callable(h) else np.full_like(g1, float(h))
            loc = side_local_nodes(p, sd)
            F[mesh.IEN[e, loc]] += 0.5 * Ls * (N1 * w1) @ hv
    for sides, ahat, chat in robin:
        for e, sd in sides:
            xs_, ys_, Ls = side_points(e, sd)
            loc = mesh.IEN[e, side_local_nodes(p, sd)]
            F[loc] += 0.5 * Ls * chat * (N1 * w1).sum(axis=1)
            r_, c_ = np.meshgrid(loc, loc, indexing="ij")
            rr.append(r_.ravel()); cc.append(c_.ravel())
            vv.append((0.5 * Ls * ahat * M1).ravel())
    rows, cols, vals = np.concatenate(rr), np.concatenate(cc), np.concatenate(vv)
    # tied nodes: map every node of a group onto one representative node
    rep = np.arange(mesh.nnp)
    dropped = np.zeros(mesh.nnp, bool)
    for group in tied:
        group = np.sort(np.asarray(group, int))
        r0 = group[len(group) // 2]
        rep[group] = r0
        dropped[group] = True
        dropped[r0] = False
    if len(tied):
        rows, cols = rep[rows], rep[cols]
        F = np.bincount(rep, weights=F, minlength=mesh.nnp)
    # Dirichlet values
    g = {}
    for sides, gv in dirichlet:
        for A in mesh.nodes_of_sides(sides):
            x, y = mesh.X[A]
            g[A] = gv(x, y) if callable(gv) else float(gv)
    if nodes_fixed:
        g.update(nodes_fixed)
    fixed = np.zeros(mesh.nnp, bool)
    ug = np.zeros(mesh.nnp)
    for A, v in g.items():
        fixed[A] = True
        ug[A] = v
    free = np.where(~fixed & ~dropped)[0]
    newnum = -np.ones(mesh.nnp, int)
    newnum[free] = np.arange(free.size)
    # right-hand side: F - K[:, fixed] g
    Kg = np.bincount(rows, weights=vals * ug[cols], minlength=mesh.nnp)
    rhs = (F - Kg)[free]
    keep = (~fixed[rows]) & (~fixed[cols])
    r2, c2 = newnum[rows[keep]], newnum[cols[keep]]
    u = ug.copy()
    u[free] = _block_cholesky_solve(r2, c2, vals[keep], free.size, rhs)
    u[dropped] = u[rep[dropped]]
    sol = Solution()
    sol.mesh, sol.u, sol.p, sol.Kap, sol.fixed = mesh, u, p, Kap, fixed
    # reactions: (K u - F) at the Dirichlet nodes = int N_A (kappa grad u).n
    Ku = np.bincount(rows, weights=vals * u[cols], minlength=mesh.nnp)
    sol.reaction = np.where(fixed, Ku - F, 0.0)
    sol.energy = 0.5 * u @ Ku          # (1/2) u^T K u, with the boundary terms included
    sol.F = F
    return sol


# --------------------------------------------------------------- post-process
def gradient(sol, e, xi, eta):
    """grad u at parent points of element e (arrays)."""
    mesh = sol.mesh
    N, Nxi, Neta = shape2d(sol.p, np.atleast_1d(xi), np.atleast_1d(eta))
    hx, hy = mesh.size[e]
    ue = sol.u[mesh.IEN[e]]
    return np.vstack([(2.0 / hx) * ue @ Nxi, (2.0 / hy) * ue @ Neta])


def value(sol, e, xi, eta):
    N, _, _ = shape2d(sol.p, np.atleast_1d(xi), np.atleast_1d(eta))
    return sol.u[sol.mesh.IEN[e]] @ N


def locate(mesh, x, y):
    """Element containing (x, y) and the parent coordinates there."""
    for e in range(mesh.nel):
        (xc, yc), (hx, hy) = mesh.cent[e], mesh.size[e]
        if abs(x - xc) <= hx / 2 * (1 + 1e-9) and abs(y - yc) <= hy / 2 * (1 + 1e-9):
            return e, 2 * (x - xc) / hx, 2 * (y - yc) / hy
    raise ValueError("point outside the mesh")


def nodal_gradient(sol):
    """grad u at the nodes, averaged over the elements that share each node."""
    mesh, p = sol.mesh, sol.p
    nd = np.linspace(-1, 1, p + 1)
    XI = np.array([nd[i] for j in range(p + 1) for i in range(p + 1)])
    ETA = np.array([nd[j] for j in range(p + 1) for i in range(p + 1)])
    G = np.zeros((mesh.nnp, 2))
    cnt = np.zeros(mesh.nnp)
    N, Nxi, Neta = shape2d(p, XI, ETA)
    for e in range(mesh.nel):
        hx, hy = mesh.size[e]
        ue = sol.u[mesh.IEN[e]]
        gx = (2 / hx) * ue @ Nxi
        gy = (2 / hy) * ue @ Neta
        np.add.at(G[:, 0], mesh.IEN[e], gx)
        np.add.at(G[:, 1], mesh.IEN[e], gy)
        np.add.at(cnt, mesh.IEN[e], 1)
    return G / cnt[:, None]


def integrate(sol, fun=None):
    """Integral over the domain of fun(u, x, y) (default: u)."""
    mesh, p = sol.mesh, sol.p
    ref = _reference(p, p + 2)
    hx, hy = mesh.size[:, 0], mesh.size[:, 1]
    uq = sol.u[mesh.IEN] @ ref["N"]                       # (nel, nq)
    xq = mesh.cent[:, 0, None] + 0.5 * hx[:, None] * ref["XI"][None, :]
    yq = mesh.cent[:, 1, None] + 0.5 * hy[:, None] * ref["ETA"][None, :]
    vq = uq if fun is None else fun(uq, xq, yq)
    return float(np.sum(vq * ref["W"][None, :] * (0.25 * hx * hy)[:, None]))


def side_flux(sol, sides, npts=None):
    """Direct integral of (kappa grad u).n over boundary sides (the h-type,
    inward flux).  Uses the element gradient at Gauss points of each side."""
    mesh, p = sol.mesh, sol.p
    g1, w1 = np.polynomial.legendre.leggauss(npts or p + 2)
    tot = 0.0
    for e, sd in sides:
        (xm, ym), Ls, n = mesh.side_geometry(e, sd)
        if sd == "bottom":
            xi, eta = g1, -np.ones_like(g1)
        elif sd == "top":
            xi, eta = g1, np.ones_like(g1)
        elif sd == "left":
            xi, eta = -np.ones_like(g1), g1
        else:
            xi, eta = np.ones_like(g1), g1
        G = gradient(sol, e, xi, eta)
        q = sol.Kap[e] @ G
        tot += 0.5 * Ls * np.sum(w1 * (n[0] * q[0] + n[1] * q[1]))
    return tot


def side_integral(sol, sides, fun=None, npts=None):
    """Integral along boundary sides of fun(u, x, y) (default: u)."""
    mesh, p = sol.mesh, sol.p
    g1, w1 = np.polynomial.legendre.leggauss(npts or p + 2)
    tot = 0.0
    for e, sd in sides:
        (xm, ym), Ls, n = mesh.side_geometry(e, sd)
        if sd in ("bottom", "top"):
            xi, eta = g1, np.full_like(g1, -1.0 if sd == "bottom" else 1.0)
            xs_, ys_ = xm + 0.5 * Ls * g1, np.full_like(g1, ym)
        else:
            xi, eta = np.full_like(g1, -1.0 if sd == "left" else 1.0), g1
            xs_, ys_ = np.full_like(g1, xm), ym + 0.5 * Ls * g1
        uv = value(sol, e, xi, eta)
        vv = uv if fun is None else fun(uv, xs_, ys_)
        tot += 0.5 * Ls * np.sum(w1 * vv)
    return tot


def sci(v, digits=1):
    """A number in scientific notation for text: 3.8e-07 -> '3.8 × 10⁻⁷'."""
    if v == 0:
        return "0"
    m, e = f"{v:.{digits}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exp = str(int(e)).translate(sup)
    return f"10{exp}" if m in ("1", "1.0") else f"{m} × 10{exp}"
