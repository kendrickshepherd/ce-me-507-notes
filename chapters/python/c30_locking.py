"""
Figures and numbers for Class 30: convergence in elasticity, and volumetric locking.

Two plane strain problems with exact solutions:
- a thick-walled cylinder under internal pressure (Lame's solution), modeled as
  a quarter by symmetry, with rollers on the two cuts;
- the end-loaded cantilever of Timoshenko and Goodier, with the exact
  displacements prescribed on the left end and the parabolic shear traction on
  the right end, as on the elasticity appendix pages.

Elements of degree 1 and 2 (and 3, for a few numbers) on structured meshes, with
full Gauss quadrature, (p + 1) points in each direction.  For degree 1 there is
also the option of evaluating the lambda term at the element center only
(selective reduced integration).  The JavaScript solver of the widget on the page
follows the same conventions; change the two together.

Every index is numbered from 0, and the local nodes follow the stride rule
a = i + (p + 1) j.  Units: millimetres, newtons, and megapascals; plane strain
forces are per millimetre of length.  Only numpy is used, so the page renders on
the CI image.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge, Rectangle

import elast2d as EL
import elast_figs as EF
from notes_style import INK, LOADCLR, BDRYCLR, MESHCLR, REACTCLR
from c29_supports import roller

E_RUB = 10.0                                   # Young's modulus (MPa); it only scales the displacements
A_IN, B_OUT, P_IN = 50.0, 100.0, 0.5           # cylinder radii (mm) and internal pressure q (MPa)
LB, DB, PB = 400.0, 100.0, 0.4                 # cantilever length, depth (mm), end load (N per mm of length)
NUS = (0.3, 0.49, 0.499, 0.4999, 0.49999)
BODY = "#e3eef8"
COLORS = {(1, False): "#c0392b", (2, False): "#1f6feb", (1, True): "#0d9488", (3, False): "#7c3aed"}
LABELS = {(1, False): "degree 1", (2, False): "degree 2", (1, True): "degree 1, $\\lambda$ term at one point",
          (3, False): "degree 3"}


def lame(nu, E=E_RUB):
    """lambda and mu from E and nu."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# ------------------------------------------------------------------ meshes
class Grid:
    """A structured mesh of degree p: n0 by n1 elements on the unit square
    [0, 1]^2 of (s, t), mapped to the body by mapf(s, t) -> (x0, x1).  Nodes are
    numbered with the index along the short direction running fastest, to keep
    the band of K narrow."""

    def __init__(self, p, n0, n1, mapf, fast):
        self.p, self.n0, self.n1 = p, n0, n1
        m0, m1 = p * n0 + 1, p * n1 + 1
        self.m0, self.m1 = m0, m1
        self.node = (lambda I, J: I + m0 * J) if fast == 0 else (lambda I, J: J + m1 * I)
        S, T = np.zeros(m0 * m1), np.zeros(m0 * m1)
        for I in range(m0):
            for J in range(m1):
                S[self.node(I, J)], T[self.node(I, J)] = I / (m0 - 1), J / (m1 - 1)
        self.S, self.T = S, T
        self.X = np.stack(mapf(S, T), axis=1)
        self.IEN = np.array([[self.node(p * e0 + i, p * e1 + j) for j in range(p + 1) for i in range(p + 1)]
                             for e1 in range(n1) for e0 in range(n0)])
        self.nnp, self.nel, self.nen = len(self.X), len(self.IEN), (p + 1) ** 2

    def elem(self, e0, e1):
        return e0 + self.n0 * e1


def cylinder_grid(p, n):
    """A quarter of the cylinder: n elements through the wall (s, radial) and
    2n around the quarter (t, from theta = 0 to 90 degrees)."""
    def mapf(s, t):
        r, th = A_IN + (B_OUT - A_IN) * s, 0.5 * math.pi * t
        return r * np.cos(th), r * np.sin(th)
    return Grid(p, n, 2 * n, mapf, fast=0)


def cantilever_grid(p, n):
    """The cantilever: 4n elements along the length (s) and n through the depth (t),
    with x1 measured from the mid-depth."""
    return Grid(p, 4 * n, n, lambda s, t: (LB * s, DB * (t - 0.5)), fast=1)


# --------------------------------------------------------------- elements
def lagrange1d(p, s):
    N, dN = EL.lagrange1d(p, s)
    return np.asarray(N), np.asarray(dN)


def shape(p, xi, eta):
    """N_a and dN_a/d(xi, eta), local nodes by the stride rule."""
    Nx, dx = lagrange1d(p, xi)
    Ny, dy = lagrange1d(p, eta)
    ij = [(i, j) for j in range(p + 1) for i in range(p + 1)]
    return (np.array([Nx[i] * Ny[j] for i, j in ij]),
            np.array([[dx[i] * Ny[j], Nx[i] * dy[j]] for i, j in ij]))


def _geometry(g, xi, eta):
    """At one parent point, for every element: N, B (nel x 3 x 2 nen), J, and x."""
    N, dN = shape(g.p, xi, eta)
    xe = g.X[g.IEN]
    DF = np.einsum("eai,aj->eij", xe, dN)
    J = DF[:, 0, 0] * DF[:, 1, 1] - DF[:, 0, 1] * DF[:, 1, 0]
    inv = np.empty_like(DF)
    inv[:, 0, 0], inv[:, 0, 1] = DF[:, 1, 1] / J, -DF[:, 0, 1] / J
    inv[:, 1, 0], inv[:, 1, 1] = -DF[:, 1, 0] / J, DF[:, 0, 0] / J
    dNx = np.einsum("aj,eji->eai", dN, inv)
    B = np.zeros((g.nel, 3, 2 * g.nen))
    B[:, 0, 0::2], B[:, 1, 1::2] = dNx[:, :, 0], dNx[:, :, 1]
    B[:, 2, 0::2], B[:, 2, 1::2] = dNx[:, :, 1], dNx[:, :, 0]
    return N, B, J, np.einsum("a,eai->ei", N, xe)


M_VEC = np.array([1.0, 1.0, 0.0])                      # m^T eps = u_0,0 + u_1,1, the divergence


def stiffness(g, lam, mu, sri=False):
    """k^e for every element: the mu part of D with (p + 1)^2 Gauss points, and the
    lambda part, D = mu diag(2, 2, 1) + lambda m m^T, with the same points or, if
    sri, at the element center only."""
    xg, wg = np.polynomial.legendre.leggauss(g.p + 1)
    Dmu = mu * np.diag([2.0, 2.0, 1.0])
    ke = np.zeros((g.nel, 2 * g.nen, 2 * g.nen))
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, B, J, x = _geometry(g, xi, eta)
            D = Dmu if sri else Dmu + lam * np.outer(M_VEC, M_VEC)
            ke += np.einsum("e,eIp,IJ,eJq->epq", wi * wj * J, B, D, B)
    if sri:
        N, B, J, x = _geometry(g, 0.0, 0.0)
        mB = np.einsum("I,eIp->ep", M_VEC, B)
        ke += lam * 4.0 * J[:, None, None] * mB[:, :, None] * mB[:, None, :]
    return ke


def side_load(g, elems, side, h):
    """The nodal forces of a traction h(x0, x1) -> (h0, h1) on one side
    ('left', xi = -1, or 'right', xi = 1) of the listed elements."""
    F = np.zeros(2 * g.nnp)
    xg, wg = np.polynomial.legendre.leggauss(g.p + 3)
    xi = -1.0 if side == "left" else 1.0
    for e in elems:
        xe = g.X[g.IEN[e]]
        for eta, w in zip(xg, wg):
            N, dN = shape(g.p, xi, eta)
            x, tang = N @ xe, xe.T @ dN[:, 1]
            hv = h(*x)
            F[2 * g.IEN[e]] += N * hv[0] * w * math.hypot(*tang)
            F[2 * g.IEN[e] + 1] += N * hv[1] * w * math.hypot(*tang)
    return F


class Solution:
    pass


def solve(g, lam, mu, fixed, gval, F, sri=False):
    """Solve with the components in `fixed` (2A + i) prescribed to gval."""
    ke = stiffness(g, lam, mu, sri)
    n2 = 2 * g.nnp
    ID = -np.ones(n2, dtype=int)
    free = np.setdiff1d(np.arange(n2), fixed)
    ID[free] = np.arange(len(free))
    U = np.zeros(n2)
    U[fixed] = gval
    dof = np.stack([2 * g.IEN, 2 * g.IEN + 1], axis=-1).reshape(g.nel, -1)
    fe = -np.einsum("epq,eq->ep", ke, U[dof])            # the prescribed values move to the right-hand side
    P = ID[dof]
    b = F[free].copy()
    ok = P >= 0
    np.add.at(b, P[ok], fe[ok])
    both = ok[:, :, None] & ok[:, None, :]
    rows = np.broadcast_to(P[:, :, None], both.shape)[both]
    cols = np.broadcast_to(P[:, None, :], both.shape)[both]
    U[free] = EL.banded_solve(rows, cols, ke[both], len(free), b)
    s = Solution()
    s.g, s.U, s.lam, s.mu, s.sri, s.neq = g, U, lam, mu, sri, len(free)
    return s


def volumetric_stress_range(s):
    """The smallest and largest lambda div u^h over the Gauss points of the stiffness rule."""
    g = s.g
    xg = [0.0] if s.sri else np.polynomial.legendre.leggauss(g.p + 1)[0]
    ue = np.stack([s.U[2 * g.IEN], s.U[2 * g.IEN + 1]], -1).reshape(g.nel, -1)
    vals = np.concatenate([s.lam * np.einsum("I,eIp,ep->e", M_VEC, _geometry(g, xi, eta)[1], ue)
                           for xi in xg for eta in xg])
    return vals.min(), vals.max()


def displacement(s, e, xi, eta):
    N, dN = shape(s.g.p, xi, eta)
    A = s.g.IEN[e]
    return np.array([N @ s.U[2 * A], N @ s.U[2 * A + 1]])


def errors(s, exact_u, exact_grad):
    """The relative L2 error of the displacement, and the relative error in the
    energy norm, sqrt(a(e, e) / a(u, u)), with (p + 3)^2 Gauss points."""
    g = s.g
    xg, wg = np.polynomial.legendre.leggauss(g.p + 3)
    ue = np.stack([s.U[2 * g.IEN], s.U[2 * g.IEN + 1]], -1)
    eL2 = uL2 = eE = uE = 0.0
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, dN = shape(g.p, xi, eta)
            xe = g.X[g.IEN]
            DF = np.einsum("eai,aj->eij", xe, dN)
            J = np.linalg.det(DF)
            dNx = np.einsum("aj,eji->eai", dN, np.linalg.inv(DF))
            x = np.einsum("a,eai->ei", N, xe)
            u, G = exact_u(x[:, 0], x[:, 1]), exact_grad(x[:, 0], x[:, 1])
            e, Ge = u - np.einsum("a,eai->ei", N, ue), G - np.einsum("eai,eaj->eij", ue, dNx)
            w = wi * wj * J

            def energy(H):
                eps = 0.5 * (H + np.transpose(H, (0, 2, 1)))
                tr = eps[:, 0, 0] + eps[:, 1, 1]
                return 2 * s.mu * np.sum(eps * eps, axis=(1, 2)) + s.lam * tr * tr
            eL2 += np.sum(w * np.sum(e * e, 1))
            uL2 += np.sum(w * np.sum(u * u, 1))
            eE += np.sum(w * energy(Ge))
            uE += np.sum(w * energy(G))
    return math.sqrt(eL2 / uL2), math.sqrt(eE / uE)


# ------------------------------------------------------------ the cylinder
def cylinder_exact(nu, E=E_RUB):
    """Lame's solution in plane strain: u_r = c ((1 - 2 nu) r + b^2 / r), with
    c = (1 + nu) q a^2 / (E (b^2 - a^2))."""
    c = (1 + nu) * P_IN * A_IN ** 2 / (E * (B_OUT ** 2 - A_IN ** 2))
    ur = lambda r: c * ((1 - 2 * nu) * r + B_OUT ** 2 / r)
    dur = lambda r: c * ((1 - 2 * nu) - B_OUT ** 2 / r ** 2)

    def u(x, y):
        r = np.hypot(x, y)
        return np.stack([ur(r) * x / r, ur(r) * y / r], 1)

    def grad(x, y):
        r = np.hypot(x, y)
        f = ur(r) / r
        fp = (dur(r) - f) / r                              # d(u_r / r)/dr
        G = np.zeros((len(x), 2, 2))
        G[:, 0, 0], G[:, 0, 1] = f + fp * x * x / r, fp * x * y / r
        G[:, 1, 0], G[:, 1, 1] = fp * x * y / r, f + fp * y * y / r
        return G
    return u, grad, ur


def cylinder(p, n, nu, sri=False):
    lam, mu = lame(nu)
    g = cylinder_grid(p, n)
    fixed = np.concatenate([[2 * g.node(I, 0) + 1 for I in range(g.m0)],          # theta = 0: u_1 = 0
                            [2 * g.node(I, g.m1 - 1) for I in range(g.m0)]])       # theta = 90: u_0 = 0
    inner = [g.elem(0, e1) for e1 in range(g.n1)]
    F = side_load(g, inner, "left", lambda x, y: (P_IN * x / math.hypot(x, y), P_IN * y / math.hypot(x, y)))
    s = solve(g, lam, mu, fixed, np.zeros(len(fixed)), F, sri)
    u, grad, ur = cylinder_exact(nu)
    s.eL2, s.eE = errors(s, u, grad)
    q = displacement(s, g.elem(0, g.n1 // 2), -1.0, -1.0)                          # at r = a, 45 degrees
    s.qoi = (q @ np.array([1.0, 1.0]) / math.sqrt(2.0)) / ur(A_IN)
    return s


# ---------------------------------------------------------- the cantilever
def cantilever_exact(nu, E=E_RUB):
    """Timoshenko and Goodier's solution in plane strain (per mm of length)."""
    Ep, nup, I = E / (1 - nu * nu), nu / (1 - nu), DB ** 3 / 12.0
    k = PB / (6 * Ep * I)

    def u(x, y):
        return np.stack([k * y * ((6 * LB - 3 * x) * x + (2 + nup) * (y * y - DB * DB / 4)),
                         -k * (3 * nup * y * y * (LB - x) + (4 + 5 * nup) * DB * DB * x / 4 + (3 * LB - x) * x * x)], 1)

    def grad(x, y):
        G = np.zeros((len(x), 2, 2))
        G[:, 0, 0] = k * y * (6 * LB - 6 * x)
        G[:, 0, 1] = k * ((6 * LB - 3 * x) * x + (2 + nup) * (3 * y * y - DB * DB / 4))
        G[:, 1, 0] = -k * (-3 * nup * y * y + (4 + 5 * nup) * DB * DB / 4 + 6 * LB * x - 3 * x * x)
        G[:, 1, 1] = -k * 6 * nup * y * (LB - x)
        return G
    s01 = lambda x, y: -PB / (2 * I) * (DB * DB / 4 - y * y)
    return u, grad, s01


def cantilever(p, n, nu, sri=False):
    lam, mu = lame(nu)
    g = cantilever_grid(p, n)
    u, grad, s01 = cantilever_exact(nu)
    left = np.array([g.node(0, J) for J in range(g.m1)])
    ul = u(g.X[left, 0], g.X[left, 1])
    fixed = np.concatenate([2 * left, 2 * left + 1])
    gval = np.concatenate([ul[:, 0], ul[:, 1]])
    F = side_load(g, [g.elem(g.n0 - 1, e1) for e1 in range(g.n1)], "right", lambda x, y: (0.0, s01(x, y)))
    s = solve(g, lam, mu, fixed, gval, F, sri)
    s.eL2, s.eE = errors(s, u, grad)
    e1 = min(g.n1 // 2, g.n1 - 1)
    eta = 2.0 * (0.5 * g.n1 - e1) - 1.0
    s.tip = displacement(s, g.elem(g.n0 - 1, e1), 1.0, eta)[1]
    s.qoi = s.tip / u(np.array([LB]), np.array([0.0]))[0, 1]
    return s


def cantilever_tip(nu):
    return cantilever_exact(nu)[0](np.array([LB]), np.array([0.0]))[0, 1]


# ---------------------------------------------------------------- studies
def study(problem, p, nu, ns, sri=False):
    f = cylinder if problem == "cylinder" else cantilever
    out = [f(p, n, nu, sri) for n in ns]
    return dict(n=np.array(ns), eL2=np.array([s.eL2 for s in out]), eE=np.array([s.eE for s in out]),
                qoi=np.array([s.qoi for s in out]), sols=out)


def rates(err):
    return np.log2(err[:-1] / err[1:])


# ---------------------------------------------------------------- drawing
def _parabolic_load(ax, x0, half, scale, color=LOADCLR):
    yy = np.linspace(-half, half, 81)
    env = x0 + scale * (1 - (yy / half) ** 2)
    ax.fill_betweenx(yy, x0, env, color=color, alpha=0.12, lw=0)
    ax.plot(env, yy, color=color, lw=1.2)
    ax.plot([x0, x0], [-half, half], color=color, lw=0.9)
    for y in np.linspace(-half * 0.75, half * 0.75, 5):
        x = x0 + 0.5 * scale * (1 - (y / half) ** 2)
        EF.arrow(ax, (x, y + 0.08 * half), (x, y - 0.08 * half), color, 1.2, 8)


def fig_problems():
    """The two test problems: (a) the quarter cylinder, (b) the cantilever."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), gridspec_kw=dict(width_ratios=[1, 1.75]))
    ax = axes[0]
    ax.add_patch(Wedge((0, 0), B_OUT, 0, 90, width=B_OUT - A_IN, fc=BODY, ec=INK, lw=1.3))
    th = np.linspace(0.08, 0.5 * math.pi - 0.08, 7)
    for t in th:
        c, s = math.cos(t), math.sin(t)
        EF.arrow(ax, (0.55 * A_IN * c, 0.55 * A_IN * s), (A_IN * c, A_IN * s), LOADCLR, 1.3, 9)
    ax.text(10, 10, "$q$", color=LOADCLR, fontsize=12)
    for x in np.linspace(A_IN + 8, B_OUT - 8, 4):
        roller(ax, x, 0, (0, 1), 8.0)
    for y in np.linspace(A_IN + 8, B_OUT - 8, 4):
        roller(ax, 0, y, (1, 0), 8.0)
    ax.annotate("", xy=(A_IN, -22), xytext=(0, -22), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(A_IN / 2, -26, "$a$ = 50 mm", fontsize=10, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(B_OUT, -40), xytext=(0, -40), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(B_OUT / 2, -44, "$b$ = 100 mm", fontsize=10, ha="center", va="top", color="0.3")
    ax.text(74, 84, "free", fontsize=10, color="#d97706")
    ax.text(B_OUT + 4, 46, "rollers on the\ntwo cuts, by\nsymmetry", fontsize=9.5, color=BDRYCLR, va="center")
    ax.set_xlim(-22, B_OUT + 52)
    ax.set_ylim(-60, B_OUT + 8)
    EF.clean(ax)
    ax.set_anchor("N")
    ax.set_title("(a) A quarter of a thick-walled cylinder\nunder an internal pressure $q$ = 0.5 MPa", fontsize=10.5, loc="left")
    ax = axes[1]
    ax.add_patch(Rectangle((0, -DB / 2), LB, DB, fc=BODY, ec=INK, lw=1.3))
    ax.plot([0, 0], [-DB / 2 - 10, DB / 2 + 10], color=BDRYCLR, lw=4)
    for y in np.linspace(-DB / 2 - 4, DB / 2 + 4, 8):
        ax.plot([0, -11], [y, y - 9], color=BDRYCLR, lw=1.0)
    _parabolic_load(ax, LB + 6, DB / 2, 34)
    EF.arrow(ax, (LB + 66, 26), (LB + 66, -26), LOADCLR, 2.2, 13)
    ax.text(LB + 72, 0, "$P$ = 0.4 N per mm\nof length, as a\nparabolic shear", color=LOADCLR, fontsize=9.5, va="center")
    ax.annotate("", xy=(LB, -DB / 2 - 18), xytext=(0, -DB / 2 - 18), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(LB / 2, -DB / 2 - 23, "$L$ = 400 mm", fontsize=10, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(LB * 0.72, DB / 2), xytext=(LB * 0.72, -DB / 2), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(LB * 0.72 + 5, 16, "$D$ = 100 mm", fontsize=10, color="0.3")
    ax.text(-6, DB / 2 + 16, "exact displacements prescribed", fontsize=9.5, color=BDRYCLR)
    ax.text(LB * 0.35, -DB / 2 + 8, "free top and bottom", fontsize=9.5, color="#d97706")
    ax.set_xlim(-30, LB + 150)
    ax.set_ylim(-DB / 2 - 48, DB / 2 + 30)
    EF.clean(ax)
    ax.set_anchor("N")
    ax.set_title("(b) The end-loaded cantilever of Timoshenko and Goodier\n", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


def _slope(ax, x0, y0, rate, width=0.5, color="0.35", below=True):
    """A slope triangle on log-log axes: run of `width` decades in x."""
    x1 = x0 * 10 ** width
    y1 = y0 * 10 ** (rate * width)
    if below:
        ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color=color, lw=0.8)
        ax.text(x1 * 1.08, math.sqrt(y0 * y1), str(rate), fontsize=9, color=color, va="center")
    else:
        ax.plot([x0, x0, x1, x0], [y0, y1, y1, y0], color=color, lw=0.8)
        ax.text(x0 / 1.08, math.sqrt(y0 * y1), str(rate), fontsize=9, color=color, va="center", ha="right")


H_OF = lambda ns: (B_OUT - A_IN) / np.asarray(ns, float)


def fig_review(nu=0.3):
    """The cylinder at nu = 0.3: L2 and energy-norm errors against h."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    st = {p: study("cylinder", p, nu, (1, 2, 4, 8, 16, 32) if p == 1 else (1, 2, 4, 8, 16)) for p in (1, 2)}
    for ax, key, name in ((axes[0], "eL2", "relative $L^2$ error of $\\underline{u}^h$"),
                          (axes[1], "eE", "relative error in the energy norm")):
        for p in (1, 2):
            d = st[p]
            r = rates(d[key])[-1]
            ax.loglog(H_OF(d["n"]), d[key], "o-", color=COLORS[(p, False)], lw=1.8, ms=5,
                      label="degree %d, last rate %.2f" % (p, r))
        EF._graph(ax, "element size $h$ through the wall (mm)", name)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.minorticks_off()
        ax.set_xticks(H_OF((1, 2, 4, 8, 16, 32)))
        ax.set_xticklabels(["50", "25", "12.5", "6.25", "3.13", "1.56"])
    _slope(axes[0], 2.5, 1.0e-4, 2)
    _slope(axes[0], 4.0, 1.5e-6, 3)
    _slope(axes[1], 2.5, 6.0e-3, 1)
    _slope(axes[1], 4.0, 2.5e-4, 2)
    axes[0].set_title("(a) Displacement, $L^2$ norm: rate $p + 1$", fontsize=10.5, loc="left")
    axes[1].set_title("(b) Energy norm: rate $p$", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig, st


def fig_locking_rates():
    """The cylinder: the L2 error against h for several nu, (a) degree 1, (b) degree 2."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    data = {}
    for ax, p, ns in ((axes[0], 1, (1, 2, 4, 8, 16, 32)), (axes[1], 2, (1, 2, 4, 8, 16))):
        for k, nu in enumerate(NUS):
            d = study("cylinder", p, nu, ns)
            data[(p, nu)] = d
            ax.loglog(H_OF(d["n"]), d["eL2"], "o-", color=cmap(0.9 * k / (len(NUS) - 1)), lw=1.7, ms=4.5,
                      label="$\\nu$ = %g,  $\\lambda/\\mu$ = %s" % (nu, "{:,.0f}".format(2 * nu / (1 - 2 * nu)) if nu > 0.3 else "1.5"))
        EF._graph(ax, "element size $h$ through the wall (mm)", "relative $L^2$ error of $\\underline{u}^h$" if p == 1 else "")
        ax.minorticks_off()
        ax.set_xticks(H_OF(ns))
        ax.set_xticklabels(["50", "25", "12.5", "6.25", "3.13", "1.56"][:len(ns)])
        ax.set_title("(%s) Degree %d" % ("a" if p == 1 else "b", p), fontsize=10.5, loc="left")
    axes[0].set_ylim(1e-7, 2.0)
    _slope(axes[0], 2.5, 3e-5, 2)
    _slope(axes[1], 3.6, 3e-7, 3)
    axes[1].legend(fontsize=8.6, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig, data


def fig_constraints():
    """Unknowns and constraints on a patch of elements."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    G1, G2 = 1 / math.sqrt(3), math.sqrt(0.6)
    cases = ((axes[0], 1, [-G1, G1], "(a) Degree 1", "per element: 2 unknowns,\n4 points where $\\nabla\\cdot\\underline{u}^h \\approx 0$"),
             (axes[1], 2, [-G2, 0.0, G2], "(b) Degree 2", "per element: 8 unknowns,\n9 points where $\\nabla\\cdot\\underline{u}^h \\approx 0$"),
             (axes[2], 1, [0.0], "(c) Degree 1, $\\lambda$ term at one point", "per element: 2 unknowns,\n1 point where $\\nabla\\cdot\\underline{u}^h \\approx 0$"))
    for ax, p, pts, title, note in cases:
        n = 3
        for k in range(n + 1):
            ax.plot([0, n], [k, k], color=MESHCLR, lw=1.0)
            ax.plot([k, k], [0, n], color=MESHCLR, lw=1.0)
        ax.add_patch(Rectangle((1, 1), 1, 1, fc="#fde3c8", ec=MESHCLR, lw=1.4, zorder=0))
        g = np.linspace(0, n, p * n + 1)
        for x in g:
            for y in g:
                ax.plot(x, y, "o", color=INK, ms=5.5 if p == 1 else 4.6, zorder=4)
        for ex in range(n):
            for ey in range(n):
                for a in pts:
                    for b in pts:
                        ax.plot(ex + 0.5 + 0.5 * a, ey + 0.5 + 0.5 * b, "x", color=LOADCLR, ms=5.5, mew=1.5, zorder=5)
        ax.text(n / 2, -0.35, note, ha="center", va="top", fontsize=10)
        ax.set_xlim(-0.3, n + 0.3)
        ax.set_ylim(-1.35, n + 0.25)
        EF.clean(ax)
        ax.set_title(title, fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


def fig_false_convergence(nu=0.49999):
    """(a) The cantilever on 16 x 4 elements of degree 1, against the exact
    solution; (b) the computed displacement over the exact one as the mesh is refined."""
    fig = plt.figure(figsize=(11.6, 4.2))
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.18)
    ax = fig.add_subplot(gsp[0, 0])
    s = cantilever(1, 4, nu)
    tip = cantilever_tip(nu)
    scale = 0.45 * DB / abs(tip)
    u, grad, s01 = cantilever_exact(nu)
    ax.add_patch(Rectangle((0, -DB / 2), LB, DB, fill=False, ec="0.65", lw=0.9, ls=(0, (4, 3))))
    g = s.g
    for e in range(g.nel):
        P = _outline_elem(g, s.U, e, scale)
        ax.add_patch(Polygon(P, closed=True, fc=BODY, ec=MESHCLR, lw=0.6, zorder=2))
    xs = np.linspace(0, LB, 120)
    for y in (-DB / 2, DB / 2):
        ue = u(xs, np.full_like(xs, y))
        ax.plot(xs + scale * ue[:, 0], y + scale * ue[:, 1], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ys = np.linspace(-DB / 2, DB / 2, 30)
    ue = u(np.full_like(ys, LB), ys)
    ax.plot(LB + scale * ue[:, 0], ys + scale * ue[:, 1], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(LB * 0.6, -DB / 2 + scale * u(np.array([LB * 0.6]), np.array([-DB / 2]))[0, 1] - 10,
            "exact", fontsize=10, color=INK, va="top")
    ax.text(LB + 8, scale * s.tip + 30, "degree 1,\n16 $\\times$ 4", fontsize=10, color=MESHCLR, va="center")
    ax.set_xlim(-20, LB + 80)
    ax.set_ylim(-DB / 2 - 0.45 * DB - 20, DB / 2 + 15)
    EF.clean(ax)
    ax.set_anchor("N")
    ax.set_title("(a) $\\nu$ = %g: the deformed beam, magnified %.0f times" % (nu, scale), fontsize=10.5, loc="left")
    ax = fig.add_subplot(gsp[0, 1])
    ns = (1, 2, 4, 8, 16)
    curves = [("cantilever", 1, False, nu, "-", "o"), ("cantilever", 2, False, nu, "-", "s"),
              ("cantilever", 1, True, nu, "-", "^"), ("cylinder", 1, False, nu, "--", "o")]
    out = {}
    for prob, p, sri, nv, ls, mk in curves:
        d = study(prob, p, nv, ns, sri)
        out[(prob, p, sri)] = d
        lab = ("cantilever, " if prob == "cantilever" else "cylinder, ") + LABELS[(p, sri)].replace("degree 1, $\\lambda$ term at one point", "$\\lambda$ term at one point")
        ax.plot(ns, d["qoi"], ls, marker=mk, color=COLORS[(p, sri)], lw=1.8, ms=5, label=lab)
    ax.axhline(1.0, color="0.45", lw=0.9, ls=(0, (4, 3)))
    ax.text(2.6, 0.94, "exact", fontsize=9, color="0.4", va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_ylim(0, 1.4)
    EF._graph(ax, "elements through the depth or the wall", "computed displacement / exact")
    ax.legend(fontsize=8.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.43))
    ax.set_title("(b) $\\nu$ = %g, as the mesh is refined" % nu, fontsize=10.5, loc="left")
    return fig, out


def _outline_elem(g, U, e, scale, m=4):
    xs = np.linspace(-1, 1, m + 1)
    path = [(xi, -1.0) for xi in xs[:-1]] + [(1.0, eta) for eta in xs[:-1]] + \
           [(xi, 1.0) for xi in xs[:0:-1]] + [(-1.0, eta) for eta in xs[:0:-1]]
    A = g.IEN[e]
    pts = []
    for xi, eta in path:
        N, dN = shape(g.p, xi, eta)
        pts.append(N @ g.X[A] + scale * np.array([N @ U[2 * A], N @ U[2 * A + 1]]))
    return np.array(pts)
