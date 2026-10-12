"""
Two-dimensional Galerkin finite elements for steady heat conduction,

    -div(kappa grad u) = f   on the rectangle ]0, W[ x ]0, H[,

with Lagrange quadrilaterals of degree p = 1 (Q4, bilinear) or p = 2 (Q9,
biquadratic), in Hughes' notation: global nodes A = 1..n_np, local nodes
a = 1..n_en, elements e = 1..n_el, and the ID, IEN, LM arrays (1-based, with
ID = 0 marking a Dirichlet node).  Local nodes follow Hughes and Class 10:
corners counter-clockwise from (xi, eta) = (-1, -1), then, for Q9, the
mid-side nodes counter-clockwise from the bottom edge, then the centre.

The mesh is structured, n1 elements along x1 and n2 along x2, and can be
distorted smoothly inside (boundary nodes stay put), which makes the
isoparametric map non-trivial and, for Q9, curves the element edges.

Used by Poisson2D_Heat_FEM.qmd for its static figures and printed arrays.
The explorer on that page runs fem2d.js, a port of this file -- change the
two together.  Only numpy is used, so the page renders on the CI image.
"""

import math

import numpy as np

EDGES = ("left", "right", "bottom", "top")

# (i, j): the 1D node indices along xi and eta of local node a = 1..n_en
LOCAL_IJ = {
    1: [(0, 0), (1, 0), (1, 1), (0, 1)],
    2: [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1)],
}
# the local nodes on each side of the parent square, in order of increasing t
# (the side is parametrized so that t runs along +xi or +eta)
SIDES = {"bottom": lambda t: (t, -np.ones_like(t)),
         "right": lambda t: (np.ones_like(t), t),
         "top": lambda t: (t, np.ones_like(t)),
         "left": lambda t: (-np.ones_like(t), t)}


def gauss(n):
    """Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def lagrange1d(p, s):
    """The p+1 Lagrange polynomials on equally spaced nodes in [-1, 1] and
    their derivatives, at the points s.  Rows are the node index i."""
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
            term = np.full(s.size, 1.0 / (nd[i] - nd[m]))
            for k in range(p + 1):
                if k not in (i, m):
                    term *= (s - nd[k]) / (nd[i] - nd[k])
            dN[i] += term
    return N, dN


def shape(p, xi, eta):
    """N_a, N_a,xi, N_a,eta at the points (xi, eta): the tensor products
    N_a(xi, eta) = N_i(xi) N_j(eta).  Rows are a = 1..n_en (index a-1)."""
    xi, eta = np.broadcast_arrays(np.atleast_1d(np.asarray(xi, float)),
                                  np.atleast_1d(np.asarray(eta, float)))
    Nx, dNx = lagrange1d(p, xi.ravel())
    Ny, dNy = lagrange1d(p, eta.ravel())
    ij = LOCAL_IJ[p]
    N = np.array([Nx[i] * Ny[j] for i, j in ij])
    Nxi = np.array([dNx[i] * Ny[j] for i, j in ij])
    Neta = np.array([Nx[i] * dNy[j] for i, j in ij])
    return N, Nxi, Neta


def parent_nodes(p):
    """(xi_a, eta_a) of the local nodes."""
    nd = np.linspace(-1.0, 1.0, p + 1)
    return np.array([(nd[i], nd[j]) for i, j in LOCAL_IJ[p]])


class Mesh:
    """A structured mesh of the rectangle.  Node (I, J) of the (p n1 + 1) by
    (p n2 + 1) grid sits at (W I / (p n1), H J / (p n2)) before distortion.
    order = "x1" numbers nodes (and elements) along x1 first, "x2" along x2."""

    def __init__(self, W, H, n1, n2, p=1, distort=0.0, order="x1"):
        self.W, self.H, self.n1, self.n2, self.p = W, H, n1, n2, p
        self.order, self.distort = order, distort
        N1, N2 = p * n1 + 1, p * n2 + 1
        self.N1, self.N2 = N1, N2
        self.nnp, self.nel, self.nen = N1 * N2, n1 * n2, (p + 1) ** 2
        I, J = np.meshgrid(np.arange(N1), np.arange(N2), indexing="ij")
        I, J = I.ravel(), J.ravel()
        A = self.node_number(I, J)
        self.IJ = np.zeros((self.nnp, 2), int)
        self.IJ[A - 1] = np.column_stack([I, J])
        X1, X2 = W * I / (N1 - 1), H * J / (N2 - 1)
        h1, h2 = W / n1, H / n2
        d1 = distort * h1 * np.sin(2 * np.pi * X1 / W) * np.sin(np.pi * X2 / H)
        d2 = distort * h2 * np.sin(np.pi * X1 / W) * np.sin(2 * np.pi * X2 / H)
        self.X = np.zeros((self.nnp, 2))
        self.X[A - 1] = np.column_stack([X1 + d1, X2 + d2])
        # elements
        self.EIJ = np.zeros((self.nel, 2), int)
        self.IEN = np.zeros((self.nen, self.nel), int)
        for e1 in range(n1):
            for e2 in range(n2):
                e = self.element_number(e1, e2)
                self.EIJ[e - 1] = (e1, e2)
                for a, (i, j) in enumerate(LOCAL_IJ[p]):
                    self.IEN[a, e - 1] = self.node_number(p * e1 + i, p * e2 + j)
        # boundary nodes on each edge of the domain, in order along the edge
        self.edge_nodes = {
            "left": self.node_number(np.zeros(N2, int), np.arange(N2)),
            "right": self.node_number(np.full(N2, N1 - 1), np.arange(N2)),
            "bottom": self.node_number(np.arange(N1), np.zeros(N1, int)),
            "top": self.node_number(np.arange(N1), np.full(N1, N2 - 1)),
        }
        # element sides that lie on the boundary: (e, side, domain edge)
        self.boundary_sides = []
        for e in range(1, self.nel + 1):
            e1, e2 = self.EIJ[e - 1]
            if e2 == 0:
                self.boundary_sides.append((e, "bottom", "bottom"))
            if e1 == n1 - 1:
                self.boundary_sides.append((e, "right", "right"))
            if e2 == n2 - 1:
                self.boundary_sides.append((e, "top", "top"))
            if e1 == 0:
                self.boundary_sides.append((e, "left", "left"))

    def node_number(self, I, J):
        if self.order == "x1":
            return 1 + I + self.N1 * J
        return 1 + J + self.N2 * I

    def element_number(self, e1, e2):
        if self.order == "x1":
            return 1 + e1 + self.n1 * e2
        return 1 + e2 + self.n2 * e1

    def xe(self, e):
        """Coordinates of element e's nodes, rows a = 1..n_en."""
        return self.X[self.IEN[:, e - 1] - 1]

    def map(self, e, xi, eta):
        """x(xi, eta), the Jacobian matrix DF = dx/dxi (shape (m, 2, 2)), and
        J = det DF, at the points (xi, eta) of element e."""
        N, Nxi, Neta = shape(self.p, xi, eta)
        xe = self.xe(e)
        x = N.T @ xe
        DF = np.empty((N.shape[1], 2, 2))
        DF[:, :, 0] = Nxi.T @ xe          # column 1: dx/dxi
        DF[:, :, 1] = Neta.T @ xe         # column 2: dx/deta
        J = DF[:, 0, 0] * DF[:, 1, 1] - DF[:, 0, 1] * DF[:, 1, 0]
        return x, DF, J


class Problem:
    """kappa, f(x1, x2), and the Neumann data: h[edge] = (function of the
    position s along the edge, breakpoints where it jumps)."""

    def __init__(self, kappa, f, h=None):
        self.kappa, self.f = kappa, f
        self.h = h or {}

    def h_on(self, edge):
        return self.h.get(edge, (lambda s: 0.0 * s, []))


def tab_flux(h_tab, tabs):
    """Neumann data for the top edge: h_tab on the tab footprints."""
    def h(s):
        s = np.asarray(s, float)
        out = np.zeros_like(s)
        for a, b in tabs:
            out[(s >= a) & (s <= b)] = h_tab
        return out
    return h, sorted(v for ab in tabs for v in ab)


def dirichlet(mesh, g, overrides=None):
    """{A: g_A} for the nodes on Gamma_g.  g maps each Dirichlet edge to a
    function g(x1, x2); a corner on two Dirichlet edges takes its value from
    the edge that comes first in EDGES.  overrides maps node numbers to a
    value (fix that node) or None (free it)."""
    gd = {}
    for edge in EDGES:
        if edge not in g:
            continue
        for A in mesh.edge_nodes[edge]:
            A = int(A)
            if A not in gd:
                x1, x2 = mesh.X[A - 1]
                gd[A] = float(g[edge](x1, x2))
    for A, v in (overrides or {}).items():
        if v is None:
            gd.pop(A, None)
        else:
            gd[A] = float(v)
    return gd


def element_arrays(mesh, e, pb, gd, nint=None):
    """k^e and f^e = body + h - g for element e, with the data at its
    quadrature points kept for display."""
    p, nen = mesh.p, mesh.nen
    nint = nint or p + 1
    gx, gw = gauss(nint)
    XI, ETA = np.meshgrid(gx, gx, indexing="ij")
    WQ = np.outer(gw, gw).ravel()
    xi, eta = XI.ravel(), ETA.ravel()
    N, Nxi, Neta = shape(p, xi, eta)
    x, DF, J = mesh.map(e, xi, eta)
    ke = np.zeros((nen, nen))
    body = np.zeros(nen)
    fq = pb.f(x[:, 0], x[:, 1]) * np.ones(len(xi))
    grads = []
    for q in range(len(xi)):
        DFinvT = np.linalg.inv(DF[q]).T
        Gx = DFinvT @ np.vstack([Nxi[:, q], Neta[:, q]])      # 2 x nen
        grads.append(Gx)
        ke += pb.kappa * (Gx.T @ Gx) * J[q] * WQ[q]
        body += N[:, q] * fq[q] * J[q] * WQ[q]
    # Neumann sides of this element
    fh = np.zeros(nen)
    xe = mesh.xe(e)
    for (ee, side, edge) in mesh.boundary_sides:
        if ee != e:
            continue
        hfun, breaks = pb.h_on(edge)
        axis = 1 if edge in ("left", "right") else 0
        tq, tw = gauss(p + 2)
        xs, ys = SIDES[side](np.array([-1.0, 1.0]))
        s0, s1 = mesh.map(e, xs, ys)[0][:, axis]
        cuts = [-1.0] + [2 * (b - s0) / (s1 - s0) - 1 for b in breaks
                         if min(s0, s1) < b < max(s0, s1)] + [1.0]
        cuts = sorted(cuts)
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * tq
            wt = 0.5 * (t1 - t0) * tw
            a_xi, a_eta = SIDES[side](t)
            Ns, Nsxi, Nseta = shape(p, a_xi, a_eta)
            xpts = Ns.T @ xe
            dxi = 1.0 if side in ("bottom", "top") else 0.0
            deta = 1.0 - dxi
            tang = (dxi * Nsxi + deta * Nseta).T @ xe
            jl = np.hypot(tang[:, 0], tang[:, 1])
            hv = hfun(xpts[:, axis])
            fh += Ns @ (hv * jl * wt)
    ge = np.array([gd.get(int(A), 0.0) for A in mesh.IEN[:, e - 1]])
    fg = ke @ ge
    return dict(ke=ke, body=body, h=fh, g=fg, total=body + fh - fg,
                xq=x, J=J, DF=DF, grads=grads, WQ=WQ, xi=xi, eta=eta)


def solve(mesh, pb, gd, nint=None, keep_elements=True):
    """Assemble and solve.  Returns the ID/IEN/LM arrays, the element arrays,
    the reduced system K d = F, the full (all-node) system Kfull, Ffull
    before any Dirichlet row or column is removed, the nodal values U, and
    the reactions R = Kfull U - Ffull (non-zero only at Dirichlet nodes:
    the heat flowing INTO the body there, per unit depth)."""
    nnp = mesh.nnp
    ID = np.zeros(nnp, int)
    neq = 0
    for A in range(1, nnp + 1):
        if A not in gd:
            neq += 1
            ID[A - 1] = neq
    LM = ID[mesh.IEN - 1]
    Kfull = np.zeros((nnp, nnp))
    Ffull = np.zeros(nnp)
    K = np.zeros((neq, neq))
    F = np.zeros(neq)
    els = []
    for e in range(1, mesh.nel + 1):
        ea = element_arrays(mesh, e, pb, gd, nint)
        if keep_elements:
            els.append(ea)
        ien = mesh.IEN[:, e - 1] - 1
        Kfull[np.ix_(ien, ien)] += ea["ke"]
        Ffull[ien] += ea["body"] + ea["h"]
        lm = LM[:, e - 1]
        for a in range(mesh.nen):
            P = lm[a]
            if P == 0:
                continue
            F[P - 1] += ea["total"][a]
            for b in range(mesh.nen):
                Q = lm[b]
                if Q:
                    K[P - 1, Q - 1] += ea["ke"][a, b]
    d = np.linalg.solve(K, F) if neq else np.zeros(0)
    U = np.array([d[ID[A - 1] - 1] if ID[A - 1] else gd[A]
                  for A in range(1, nnp + 1)])
    R = Kfull @ U - Ffull
    R[ID > 0] = 0.0
    return dict(mesh=mesh, ID=ID, LM=LM, IEN=mesh.IEN, els=els, K=K, F=F,
                Kfull=Kfull, Ffull=Ffull, d=d, U=U, R=R, gd=gd, neq=neq)


def evaluate(sol, k=6):
    """u^h and grad u^h on a k-by-k grid of each element's parent square:
    arrays of shape (n_el, k, k) for x1, x2, uh, uh_1, uh_2."""
    mesh, U = sol["mesh"], sol["U"]
    s = np.linspace(-1, 1, k)
    XI, ETA = np.meshgrid(s, s, indexing="ij")
    N, Nxi, Neta = shape(mesh.p, XI.ravel(), ETA.ravel())
    out = {key: np.zeros((mesh.nel, k, k)) for key in ("x1", "x2", "uh", "g1", "g2")}
    for e in range(1, mesh.nel + 1):
        x, DF, J = mesh.map(e, XI.ravel(), ETA.ravel())
        Ue = U[mesh.IEN[:, e - 1] - 1]
        uh = N.T @ Ue
        gxi, geta = Nxi.T @ Ue, Neta.T @ Ue
        inv = np.linalg.inv(DF)                      # (m, 2, 2)
        g1 = inv[:, 0, 0] * gxi + inv[:, 1, 0] * geta
        g2 = inv[:, 0, 1] * gxi + inv[:, 1, 1] * geta
        for key, v in (("x1", x[:, 0]), ("x2", x[:, 1]), ("uh", uh),
                       ("g1", g1), ("g2", g2)):
            out[key][e - 1] = v.reshape(k, k)
    return out


def errors(sol, exact, nq=None):
    """||u - u^h||_{L2} and |u - u^h|_{H1}; exact(x1, x2) returns (u, u_1, u_2)."""
    mesh, U = sol["mesh"], sol["U"]
    nq = nq or mesh.p + 3
    gx, gw = gauss(nq)
    XI, ETA = np.meshgrid(gx, gx, indexing="ij")
    WQ = np.outer(gw, gw).ravel()
    N, Nxi, Neta = shape(mesh.p, XI.ravel(), ETA.ravel())
    xs, uhs, g1s, g2s, ws = [], [], [], [], []
    for e in range(1, mesh.nel + 1):
        x, DF, J = mesh.map(e, XI.ravel(), ETA.ravel())
        Ue = U[mesh.IEN[:, e - 1] - 1]
        gxi, geta = Nxi.T @ Ue, Neta.T @ Ue
        inv = np.linalg.inv(DF)
        xs.append(x)
        uhs.append(N.T @ Ue)
        g1s.append(inv[:, 0, 0] * gxi + inv[:, 1, 0] * geta)
        g2s.append(inv[:, 0, 1] * gxi + inv[:, 1, 1] * geta)
        ws.append(WQ * J)
    x = np.vstack(xs)
    u, u1, u2 = exact(x[:, 0], x[:, 1])
    w = np.concatenate(ws)
    l2 = math.sqrt(np.sum(w * (u - np.concatenate(uhs)) ** 2))
    h1 = math.sqrt(np.sum(w * ((u1 - np.concatenate(g1s)) ** 2
                               + (u2 - np.concatenate(g2s)) ** 2)))
    return l2, h1


# ------------------------------------------------------------ LaTeX output
def num(v, sig=4):
    if abs(v) < 1e-12:
        return "0"
    s = "%.*g" % (sig, v)
    if "e" in s:
        m, ex = s.split("e")
        return r"%s\times10^{%d}" % (m, int(ex))
    return s


def tex_matrix(M, sig=4, color=None, gray_zero=True):
    """A bmatrix of every entry; color(i, j) may return a colour for an entry."""
    M = np.atleast_2d(M)
    rows = []
    for i in range(M.shape[0]):
        cells = []
        for j in range(M.shape[1]):
            c = color(i, j) if color else None
            v = M[i, j]
            if abs(v) < 1e-12 and gray_zero and not c:
                cells.append(r"\color{#bbbbbb}{0}")
            elif c:
                cells.append(r"\color{%s}{%s}" % (c, num(v, sig)))
            else:
                cells.append(num(v, sig))
        rows.append(" & ".join(cells))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def tex_vector(v, sig=4, color=None):
    return tex_matrix(np.atleast_2d(v).T, sig=sig, gray_zero=False,
                      color=(lambda i, j: color(i)) if color else None)
