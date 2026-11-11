"""
Figures and numbers for Class 31: thermoelasticity.

One-way coupled plane problems on structured meshes: a steady heat conduction
problem is solved for the temperature, and its thermal strain is the load of an
elasticity problem on the same elements.  The temperature and the displacement
may have different polynomial degrees on the same elements.  Material properties
may differ from element to element (a two-layer strip).

The structured meshes and shape functions are those of c30_locking.py.  Every
index is numbered from 0, and the local nodes follow the stride rule.  Units:
millimetres, newtons, megapascals, and degrees Celsius.  Only numpy is used, so
the page renders on the CI image.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch

import elast2d as EL
import elast_figs as EF
from c30_locking import Grid, shape, _geometry, M_VEC
from c29_supports import pin, roller
from notes_style import INK, LOADCLR, BDRYCLR, MESHCLR

MATERIALS = {                      # E (MPa), nu, alpha (1/degC)
    "steel": (200000.0, 0.30, 12e-6),
    "aluminum": (70000.0, 0.33, 23e-6),
    "concrete": (30000.0, 0.20, 10e-6),
}
PL, PH = 400.0, 100.0              # the plate of the figures and the widget (mm)


# ------------------------------------------------------------------ meshes
def plate_grid(p, nx, ny, L=PL, H=PH):
    """nx by ny elements of degree p on [0, L] x [0, H]; nodes numbered up each column."""
    return Grid(p, nx, ny, lambda s, t: (L * s, H * t), fast=1)


def nodes_where(g, test):
    return [A for A in range(g.nnp) if test(*g.X[A])]


# ------------------------------------------------------------- heat problem
def solve_heat(g, kappa=1.0, source=None, dirichlet=None):
    """Steady conduction, -div(kappa grad T) = source, with T prescribed at the
    nodes in `dirichlet` ({A: value}); every other boundary is insulated."""
    xg, wg = np.polynomial.legendre.leggauss(g.p + 2)
    nen = g.nen
    ke = np.zeros((g.nel, nen, nen))
    fe = np.zeros((g.nel, nen))
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, dN = shape(g.p, xi, eta)
            xe = g.X[g.IEN]
            DF = np.einsum("eai,aj->eij", xe, dN)
            J = np.linalg.det(DF)
            G = np.einsum("aj,eji->eai", dN, np.linalg.inv(DF))
            ke += (wi * wj * J * kappa)[:, None, None] * np.einsum("eai,ebi->eab", G, G)
            if source is not None:
                x = np.einsum("a,eai->ei", N, xe)
                fe += (wi * wj * J * source(x[:, 0], x[:, 1]))[:, None] * N[None, :]
    dirichlet = dirichlet or {}
    fixed = np.array(sorted(dirichlet), dtype=int)
    T = np.zeros(g.nnp)
    T[fixed] = [dirichlet[A] for A in fixed]
    free = np.setdiff1d(np.arange(g.nnp), fixed)
    ID = -np.ones(g.nnp, dtype=int)
    ID[free] = np.arange(len(free))
    P = ID[g.IEN]
    b = np.zeros(len(free))
    lift = fe - np.einsum("eab,eb->ea", ke, T[g.IEN])
    ok = P >= 0
    np.add.at(b, P[ok], lift[ok])
    both = ok[:, :, None] & ok[:, None, :]
    rows = np.broadcast_to(P[:, :, None], both.shape)[both]
    cols = np.broadcast_to(P[:, None, :], both.shape)[both]
    T[free] = EL.banded_solve(rows, cols, ke[both], len(free), b)
    return T


# --------------------------------------------------------- elastic problem
def dmat(E, nu, plane):
    """D in Voigt form for plane stress or plane strain."""
    if plane == "stress":
        c = E / (1 - nu * nu)
        return c * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])


def thermal_strain(alpha, nu, dT, plane):
    """The in-plane free strain that D multiplies: alpha dT (1, 1, 0) in plane
    stress, and (1 + nu) alpha dT (1, 1, 0) in plane strain, where the
    out-of-plane strain is held at zero."""
    return (alpha * dT if plane == "stress" else (1 + nu) * alpha * dT)[..., None] * M_VEC


class Solution:
    pass


def solve_thermoelastic(gu, gT, T, mat, fixed, plane="stress", h=None):
    """Displacements on grid gu from the temperature T (nodal values on grid gT,
    the same elements).  mat: (E, nu, alpha) or a function of the element center
    (x, y) giving (E, nu, alpha).  fixed: the prescribed components 2A + i, held
    at zero."""
    nq = gu.p + 1 if h is None else h
    xg, wg = np.polynomial.legendre.leggauss(nq)
    xc = gu.X[gu.IEN].mean(axis=1)
    props = np.array([mat(*c) for c in xc]) if callable(mat) else np.tile(mat, (gu.nel, 1))
    Ds = np.array([dmat(E, nu, plane) for E, nu, _ in props])
    nd = 2 * gu.nen
    ke = np.zeros((gu.nel, nd, nd))
    fe = np.zeros((gu.nel, nd))
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, B, J, x = _geometry(gu, xi, eta)
            NT = shape(gT.p, xi, eta)[0]
            dT = T[gT.IEN] @ NT
            eps0 = thermal_strain(props[:, 2], props[:, 1], dT, plane)
            w = wi * wj * J
            ke += np.einsum("e,eIp,eIJ,eJq->epq", w, B, Ds, B)
            fe += np.einsum("e,eIp,eIJ,eJ->ep", w, B, Ds, eps0)
    n2 = 2 * gu.nnp
    fixed = np.array(sorted(set(fixed)), dtype=int)
    free = np.setdiff1d(np.arange(n2), fixed)
    ID = -np.ones(n2, dtype=int)
    ID[free] = np.arange(len(free))
    dof = np.stack([2 * gu.IEN, 2 * gu.IEN + 1], axis=-1).reshape(gu.nel, -1)
    P = ID[dof]
    ok = P >= 0
    b = np.zeros(len(free))
    np.add.at(b, P[ok], fe[ok])
    both = ok[:, :, None] & ok[:, None, :]
    rows = np.broadcast_to(P[:, :, None], both.shape)[both]
    cols = np.broadcast_to(P[:, None, :], both.shape)[both]
    U = np.zeros(n2)
    U[free] = EL.banded_solve(rows, cols, ke[both], len(free), b)
    s = Solution()
    s.gu, s.gT, s.T, s.U, s.props, s.Ds, s.plane = gu, gT, T, U, props, Ds, plane
    return s


def gauss_stress(s, n=None):
    """sigma = D (B d - eps_th) at the n x n Gauss points of every element:
    arrays of shape (nel, n*n, 3) for the stress and (nel, n*n, 2) for the points."""
    n = n or s.gu.p + 1
    xg = np.polynomial.legendre.leggauss(n)[0]
    dof = np.stack([2 * s.gu.IEN, 2 * s.gu.IEN + 1], axis=-1).reshape(s.gu.nel, -1)
    ue = s.U[dof]
    sig, pts = [], []
    for eta in xg:
        for xi in xg:
            N, B, J, x = _geometry(s.gu, xi, eta)
            dT = s.T[s.gT.IEN] @ shape(s.gT.p, xi, eta)[0]
            eps = np.einsum("eIp,ep->eI", B, ue) - thermal_strain(s.props[:, 2], s.props[:, 1], dT, s.plane)
            sig.append(np.einsum("eIJ,eJ->eI", s.Ds, eps))
            pts.append(x)
    return np.stack(sig, 1), np.stack(pts, 1)


def principal_max(sig):
    a, b, c = sig[..., 0], sig[..., 1], sig[..., 2]
    return 0.5 * (a + b) + np.sqrt(0.25 * (a - b) ** 2 + c * c)


def von_mises(sig):
    a, b, c = sig[..., 0], sig[..., 1], sig[..., 2]
    return np.sqrt(a * a - a * b + b * b + 3 * c * c)


# ------------------------------------------------------- temperature fields
def temperature(kind, gT, dT=50.0):
    """Nodal temperature rises (above the stress-free temperature) on grid gT,
    scaled so that the largest is dT."""
    X = gT.X
    L, H = X[:, 0].max(), X[:, 1].max()
    if kind == "uniform":
        return np.full(gT.nnp, dT)
    if kind == "linear":                         # hot top, reference temperature at the bottom
        T = solve_heat(gT, dirichlet={A: dT * (X[A, 1] / H) for A in nodes_where(gT, lambda x, y: y < 1e-9 or y > H - 1e-9)})
    elif kind == "edge":                         # a smooth hot patch on the top edge, the rest of the boundary at the reference
        bd = {A: 0.0 for A in nodes_where(gT, lambda x, y: x < 1e-9 or x > L - 1e-9 or y < 1e-9 or y > H - 1e-9)}
        for A in nodes_where(gT, lambda x, y: y > H - 1e-9):
            r = (X[A, 0] - L / 2) / (L / 4)
            bd[A] = math.cos(0.5 * math.pi * r) ** 2 if abs(r) < 1 else 0.0
        T = solve_heat(gT, dirichlet=bd)
    elif kind == "source":                       # uniform heat generation, the boundary at the reference
        bd = {A: 0.0 for A in nodes_where(gT, lambda x, y: x < 1e-9 or x > L - 1e-9 or y < 1e-9 or y > H - 1e-9)}
        T = solve_heat(gT, source=lambda x, y: np.ones_like(x), dirichlet=bd)
    elif kind == "spot":                         # heat generated in a small disk at the center
        bd = {A: 0.0 for A in nodes_where(gT, lambda x, y: x < 1e-9 or x > L - 1e-9 or y < 1e-9 or y > H - 1e-9)}
        T = solve_heat(gT, source=lambda x, y: (np.hypot(x - L / 2, y - H / 2) < 0.2 * H).astype(float), dirichlet=bd)
    else:
        raise ValueError(kind)
    return dT * T / np.abs(T).max()


def supports(gu, kind):
    """Prescribed components 2A + i for the support arrangements of the widget."""
    X = gu.X
    L, H = X[:, 0].max(), X[:, 1].max()
    at = lambda x, y: int(np.argmin(np.hypot(X[:, 0] - x, X[:, 1] - y)))
    if kind == "free":                           # a pin at the bottom left, a roller at the bottom right
        return [2 * at(0, 0), 2 * at(0, 0) + 1, 2 * at(L, 0) + 1]
    if kind == "clamped":                        # both ends clamped
        ends = nodes_where(gu, lambda x, y: x < 1e-9 or x > L - 1e-9)
        return [2 * A + i for A in ends for i in (0, 1)]
    if kind == "rollers":                        # the bottom on rollers, with a pin at the bottom left
        bottom = nodes_where(gu, lambda x, y: y < 1e-9)
        return [2 * A + 1 for A in bottom] + [2 * at(0, 0)]
    raise ValueError(kind)


def plate(kind, sup, mat="steel", pu=2, pT=1, nx=48, ny=12, dT=50.0, plane="stress"):
    gu, gT = plate_grid(pu, nx, ny), plate_grid(pT, nx, ny)
    T = temperature(kind, gT, dT)
    m = bimetal if mat == "bimetal" else MATERIALS[mat]
    return solve_thermoelastic(gu, gT, T, m, supports(gu, sup), plane)


def bimetal(x, y):
    """Steel over aluminum, each half of the depth."""
    return MATERIALS["steel"] if y > PH / 2 else MATERIALS["aluminum"]


def bimetal_curvature(dT, h=PH, m=1.0):
    """Timoshenko's curvature of a two-layer strip, steel (top, thickness h/2) on
    aluminum, for a uniform temperature rise dT (positive bends the aluminum side out)."""
    E1, _, a1 = MATERIALS["steel"]
    E2, _, a2 = MATERIALS["aluminum"]
    n = E1 / E2
    return 6 * (a2 - a1) * dT * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m * m + 1 / (m * n))))


# ---------------------------------------------------------------- numbers
THERMAL = {"steel": (7850.0, 460.0), "aluminum": (2700.0, 900.0)}      # density (kg/m^3), specific heat (J/(kg K))
T0_K = 293.0


def coupling_constant(mat):
    """The thermoelastic coupling constant (3 lambda + 2 mu)^2 alpha^2 T0 / (rho c (lambda + 2 mu)), in SI units."""
    E, nu, a = MATERIALS[mat]
    E = E * 1e6
    rho, c = THERMAL[mat]
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    return (3 * lam + 2 * mu) ** 2 * a * a * T0_K / (rho * c * (lam + 2 * mu))


def kelvin_dT(mat, dsigma_kk):
    """The temperature change (K) of a fast (adiabatic) change in the sum of the normal stresses (MPa)."""
    E, nu, a = MATERIALS[mat]
    rho, c = THERMAL[mat]
    return -a * T0_K * dsigma_kk * 1e6 / (rho * c)


def pipe(n=16, dT=50.0, whole=True, plane="strain", pu=2):
    """A steady temperature dT ln(b/r)/ln(b/a) through the wall of the quarter pipe of
    c30_locking (a = 50, b = 100 mm), steel.  whole: rollers on the two cuts, which
    represent the whole pipe by symmetry; otherwise three statically determinate
    supports, which represent a pipe cut along its length."""
    from c30_locking import cylinder_grid, A_IN, B_OUT
    gu, gT = cylinder_grid(pu, n), cylinder_grid(1, n)
    r = np.hypot(*gT.X.T)
    T = dT * np.log(B_OUT / r) / np.log(B_OUT / A_IN)
    if whole:
        fixed = [2 * gu.node(I, 0) + 1 for I in range(gu.m0)] + [2 * gu.node(I, gu.m1 - 1) for I in range(gu.m0)]
    else:
        o, t = gu.node(gu.m0 - 1, 0), gu.node(gu.m0 - 1, gu.m1 - 1)
        fixed = [2 * o, 2 * o + 1, 2 * t]
    return solve_thermoelastic(gu, gT, T, MATERIALS["steel"], fixed, plane)


def pipe_hoop_exact(r, dT=50.0, plane="strain"):
    """The hoop stress of the whole pipe (Timoshenko and Goodier)."""
    from c30_locking import A_IN as a, B_OUT as b
    E, nu, al = MATERIALS["steel"]
    k = E * al * dT / (2 * math.log(b / a)) / ((1 - nu) if plane == "strain" else 1.0)
    return k * (1 - np.log(b / r) - a * a / (b * b - a * a) * (1 + b * b / r ** 2) * math.log(b / a))


def hoop(sig, x):
    th = np.arctan2(x[..., 1], x[..., 0])
    s, c = np.sin(th), np.cos(th)
    return sig[..., 0] * s * s + sig[..., 1] * c * c - 2 * sig[..., 2] * s * c


# ---------------------------------------------------------------- drawing
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri

DIVMAP = "RdBu_r"
TMAP = "RdBu_r"


def deformed(g, U, mag, xi, eta):
    N = shape(g.p, xi, eta)[0]
    X = g.X + mag * np.stack([U[0::2], U[1::2]], 1)
    return np.einsum("a,eai->ei", N, X[g.IEN])


def gauss_cells(s, values, mag=0.0, n=None):
    """Polygons for the n x n Gauss-point cells of every element, on the deformed
    mesh, with the values (nel, n*n) for their colors."""
    n = n or s.gu.p + 1
    xg = np.polynomial.legendre.leggauss(n)[0]
    cuts = np.concatenate([[-1.0], 0.5 * (xg[:-1] + xg[1:]), [1.0]])
    polys, vals = [], []
    corners = {}
    for j in range(n + 1):
        for i in range(n + 1):
            corners[(i, j)] = deformed(s.gu, s.U, mag, cuts[i], cuts[j])
    for j in range(n):
        for i in range(n):
            P = np.stack([corners[(i, j)], corners[(i + 1, j)], corners[(i + 1, j + 1)], corners[(i, j + 1)]], 1)
            polys.append(P)
            vals.append(values[:, j * n + i])
    return np.concatenate(polys), np.concatenate(vals)


def draw_cells(ax, s, values, mag, cmap, vmin, vmax, n=None, lw=0.0):
    P, v = gauss_cells(s, values, mag, n)
    pc = PolyCollection(P, array=v, cmap=cmap, edgecolors="face", linewidths=0.2)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    return pc


def outline(ax, g, U=None, mag=0.0, **kw):
    """The boundary of a structured grid (deformed if U is given)."""
    pts = []
    for side in range(4):
        if side == 0:
            seq = [(g.elem(e0, 0), xi, -1.0) for e0 in range(g.n0) for xi in np.linspace(-1, 1, 5)[:-1]]
        elif side == 1:
            seq = [(g.elem(g.n0 - 1, e1), 1.0, eta) for e1 in range(g.n1) for eta in np.linspace(-1, 1, 5)[:-1]]
        elif side == 2:
            seq = [(g.elem(e0, g.n1 - 1), xi, 1.0) for e0 in range(g.n0 - 1, -1, -1) for xi in np.linspace(1, -1, 5)[:-1]]
        else:
            seq = [(g.elem(0, e1), -1.0, eta) for e1 in range(g.n1 - 1, -1, -1) for eta in np.linspace(1, -1, 5)[:-1]]
        for e, xi, eta in seq:
            N = shape(g.p, xi, eta)[0]
            X = g.X[g.IEN[e]]
            if U is not None:
                X = X + mag * np.stack([U[0::2][g.IEN[e]], U[1::2][g.IEN[e]]], 1)
            pts.append(N @ X)
    pts.append(pts[0])
    P = np.array(pts)
    ax.plot(P[:, 0], P[:, 1], **kw)


def temperature_plot(ax, gT, T, vmax):
    tris = []
    for q in gT.IEN:                                   # degree 1: stride order 0 1 / 2 3
        tris += [[q[0], q[1], q[3]], [q[0], q[3], q[2]]]
    tri = mtri.Triangulation(gT.X[:, 0], gT.X[:, 1], tris)
    return ax.tripcolor(tri, T, shading="gouraud", cmap=TMAP, vmin=-vmax, vmax=vmax)


def magnification(s, size):
    u = np.hypot(s.U[0::2], s.U[1::2]).max()
    return size / u if u > 0 else 0.0


def fig_expansion():
    """(a) a free bar, (b) a bar between rigid walls, (c) a strip of two bonded layers."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.0))
    body, hot = "#e3eef8", "#f6d2c4"
    # (a)
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), 10, 1.4, fill=False, ec="0.5", lw=1.0, ls=(0, (4, 3))))
    ax.add_patch(Rectangle((0, 0), 11.4, 1.4, fc=hot, ec=INK, lw=1.2))
    ax.annotate("", xy=(11.4, -0.6), xytext=(10, -0.6), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text(10.7, -0.85, r"$\alpha\,\Delta T\,L$", ha="center", va="top", fontsize=11)
    ax.annotate("", xy=(10, -1.9), xytext=(0, -1.9), arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
    ax.text(5, -2.15, "$L$", ha="center", va="top", fontsize=11)
    ax.text(5.7, 0.7, r"$\sigma = 0$", ha="center", va="center", fontsize=11)
    ax.set_title("(a) Free to expand: strain $\\alpha\\,\\Delta T$, no stress", fontsize=10.5, loc="left")
    # (b)
    ax = axes[1]
    ax.add_patch(Rectangle((0, 0), 10, 1.4, fc="#c9dcf0", ec=INK, lw=1.2))
    for x0, sgn in ((0, -1), (10, 1)):
        ax.plot([x0, x0], [-0.8, 2.2], color=BDRYCLR, lw=3)
        for y in np.linspace(-0.7, 2.0, 7):
            ax.plot([x0, x0 + 0.5 * sgn], [y, y - 0.35], color=BDRYCLR, lw=0.9)
    EF.arrow(ax, (-1.6, 0.7), (-0.15, 0.7), LOADCLR, 1.6, 11)
    EF.arrow(ax, (11.6, 0.7), (10.15, 0.7), LOADCLR, 1.6, 11)
    ax.text(5, 0.7, r"$\sigma = -E\,\alpha\,\Delta T$", ha="center", va="center", fontsize=11)
    ax.text(5, -0.9, "the walls prevent the expansion", ha="center", va="top", fontsize=9.5, color="0.35")
    ax.set_title("(b) Held between rigid walls: compression", fontsize=10.5, loc="left")
    # (c)
    ax = axes[2]
    R, th = 14.0, np.linspace(-0.36, 0.36, 60)
    for r0, r1, fc, lab in ((R, R + 0.7, "#d9dde3", "steel"), (R + 0.7, R + 1.4, "#f2c9a6", "aluminum")):
        P = np.concatenate([np.stack([r0 * np.sin(th), r0 * np.cos(th)], 1), np.stack([r1 * np.sin(th[::-1]), r1 * np.cos(th[::-1])], 1)])
        ax.add_patch(Polygon(P - [0, R + 0.7], closed=True, fc=fc, ec=INK, lw=1.0))
    ax.text(0, 0.35, "aluminum, larger $\\alpha$", ha="center", va="center", fontsize=9)
    ax.text(0, -0.35, "steel", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((-5, 2.0), 10, 0.7, fc="#f2c9a6", ec="0.5", lw=0.8, ls=(0, (4, 3)), fill=False))
    ax.text(0, 3.2, "before heating, straight", ha="center", fontsize=9, color="0.35")
    ax.set_title("(c) Two bonded layers: the strip bends", fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.axis("off")
    axes[0].set_xlim(-1, 12.5)
    axes[0].set_ylim(-3, 2.6)
    axes[1].set_xlim(-2, 12)
    axes[1].set_ylim(-3, 2.6)
    axes[2].set_xlim(-6.5, 6.5)
    axes[2].set_ylim(-3.2, 3.8)
    fig.tight_layout()
    return fig


def fig_pipeline(dT=50.0):
    """The one-way coupled procedure on a free plate heated at a spot: the
    temperature, the displacement, and the two normal stresses."""
    s = plate("spot", "free", dT=dT)
    E, nu, a = MATERIALS["steel"]
    ref = E * a * dT
    sig, x = gauss_stress(s)
    mag = magnification(s, 0.12 * PH)
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.8))
    tp = temperature_plot(axes[0], s.gT, s.T, dT)
    cb = fig.colorbar(tp, ax=axes[0], fraction=0.035, pad=0.02)
    cb.set_label(r"$\Delta T$ (°C)", fontsize=9)
    axes[0].set_title("(a) The temperature, from steady heat conduction with heat generated in the dashed disk", fontsize=10.5, loc="left")
    displacement_panel(axes[1], s, mag, every=(2, 2), arrows=(6, 3))
    spacer = fig.colorbar(tp, ax=axes[1], fraction=0.035, pad=0.02)   # hidden, to keep the panels the same width
    spacer.ax.set_visible(False)
    axes[1].set_title("(b) The displacement, magnified %.0f times, measured from the pin" % mag, fontsize=10.5, loc="left")
    axes[0].add_patch(plt.Circle((PL / 2, PH / 2), 0.2 * PH, fill=False, ec=INK, lw=0.8, ls=(0, (3, 2))))
    for k, (ax, comp, lab) in enumerate(((axes[2], 0, "$\\sigma_{00}$"), (axes[3], 1, "$\\sigma_{11}$"))):
        pc = draw_cells(ax, s, sig[..., comp], mag, DIVMAP, -0.6 * ref, 0.6 * ref)
        outline(ax, s.gu, color="0.55", lw=0.9, ls=(0, (4, 3)))
        cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(lab + " (MPa)", fontsize=9)
        ax.set_title("(%s) The deformed plate, magnified %.0f times, colored by %s" % ("cd"[k], mag, lab), fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-14, PL + 20)
        ax.set_ylim(-17, PH + 15)
        ax.axis("off")
    pin(axes[1], 0, 0, (0, 1), 9.0)
    roller(axes[1], PL + mag * s.U[2 * int(np.argmin(np.hypot(s.gu.X[:, 0] - PL, s.gu.X[:, 1])))], 0, (0, 1), 9.0)
    fig.tight_layout()
    return fig, s, sig


def displacement_panel(ax, s, mag, every=(4, 3), arrows=(8, 6)):
    """The undeformed outline (dashed), a coarse grid of element lines drawn in its
    deformed position, and arrows that carry points of the undeformed plate to
    their deformed positions, all with the same magnification."""
    g = s.gu
    X = g.X + mag * np.stack([s.U[0::2], s.U[1::2]], 1)
    outline(ax, g, color="0.55", lw=0.8, ls=(0, (4, 3)))
    for I in range(0, g.m0, g.p * every[0]):
        P = X[[g.node(I, J) for J in range(g.m1)]]
        ax.plot(P[:, 0], P[:, 1], color=MESHCLR, lw=0.7)
    for J in range(0, g.m1, g.p * every[1]):
        P = X[[g.node(I, J) for I in range(g.m0)]]
        ax.plot(P[:, 0], P[:, 1], color=MESHCLR, lw=0.7)
    pts = [g.node(I, J) for I in range(0, g.m0, g.p * arrows[0]) for J in range(0, g.m1, g.p * arrows[1])]
    x0 = g.X[pts]
    d = mag * np.stack([s.U[0::2][pts], s.U[1::2][pts]], 1)
    ax.quiver(x0[:, 0], x0[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1.0,
              color="#1f6feb", width=0.004, headwidth=4, headlength=5, headaxislength=4.5, zorder=5)


def fig_free_plate(dT=50.0):
    """A free plate under three temperature distributions: the temperature, the
    displacement, and the stress."""
    E, nu, a = MATERIALS["steel"]
    ref = E * a * dT
    kinds = (("linear", "(a) Linear: hot top, cool bottom"),
             ("edge", "(b) Steady conduction from a hot patch on the top edge"),
             ("source", "(c) Heat generated uniformly inside, edges at the reference"))
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 6.0))
    out = {}
    for row, (kind, title) in enumerate(kinds):
        s = plate(kind, "free", dT=dT)
        sig, x = gauss_stress(s)
        pmax = principal_max(sig)
        out[kind] = (s, sig)
        tp = temperature_plot(axes[row, 0], s.gT, s.T, dT)
        mag = magnification(s, 0.1 * PH)
        displacement_panel(axes[row, 1], s, mag)
        pc = draw_cells(axes[row, 2], s, pmax, mag, DIVMAP, -0.6 * ref, 0.6 * ref)
        outline(axes[row, 2], s.gu, color="0.55", lw=0.8, ls=(0, (4, 3)))
        axes[row, 0].set_title(title, fontsize=9.5, loc="left")
        axes[row, 1].set_title("displacement, magnified %.0f times" % mag, fontsize=9.5, loc="left")
        axes[row, 2].set_title("largest principal stress: %.1f to %.1f MPa" % (round(pmax.min(), 1) + 0.0, round(pmax.max(), 1) + 0.0),
                               fontsize=9.5, loc="left")
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlim(-8, PL + 14)
        ax.set_ylim(-16, PH + 16)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.93, top=0.94, bottom=0.03, hspace=0.32, wspace=0.04)
    c1 = fig.add_axes([0.94, 0.55, 0.009, 0.36])
    cb = fig.colorbar(tp, cax=c1)
    cb.set_label(r"$\Delta T$ (°C)", fontsize=9)
    c2 = fig.add_axes([0.94, 0.08, 0.009, 0.36])
    cb = fig.colorbar(pc, cax=c2)
    cb.set_label("stress (MPa)", fontsize=9)
    return fig, out


def fig_pipe(dT=50.0, n=16):
    """A pipe with heat flowing out through its wall: cut along its length, and whole."""
    cut, whole = pipe(n, dT, whole=False), pipe(n, dT, whole=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3))
    vmax = 110.0
    tp = temperature_plot(axes[0], whole.gT, whole.T, dT)
    axes[0].set_title("(a) The steady temperature,\nhot inside, cool outside", fontsize=10.5, loc="left")
    res = {}
    for ax, s, title in ((axes[1], cut, "(b) Cut along its length:\nthe cut opens, with no stress"),
                         (axes[2], whole, "(c) The whole pipe (by symmetry):\nhoop stress")):
        sig, x = gauss_stress(s)
        h = hoop(sig, x)
        res[id(s)] = (sig, x, h)
        mag = magnification(cut, 12.0)
        pc = draw_cells(ax, s, h, mag, DIVMAP, -vmax, vmax)
        outline(ax, s.gu, color="0.55", lw=0.8, ls=(0, (4, 3)))
        ax.set_title(title, fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-8, 115)
        ax.set_ylim(-8, 115)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.9, top=0.86, bottom=0.03, wspace=0.08)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.915, 0.12, 0.012, 0.66]))
    cb.set_label("hoop stress (MPa)", fontsize=9)
    return fig, cut, whole, res[id(cut)], res[id(whole)]


def fig_degree(dT=50.0, nx=24, ny=6):
    """Spurious stress in a free plate with a linear temperature, for two pairs of degrees."""
    E, nu, a = MATERIALS["steel"]
    ref = E * a * dT
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0))
    out = {}
    for ax, (pu, title) in zip(axes, ((1, "(a) Displacement of degree 1, temperature of degree 1"),
                                       (2, "(b) Displacement of degree 2, temperature of degree 1"))):
        s = plate("linear", "free", pu=pu, pT=1, nx=nx, ny=ny, dT=dT)
        sig, x = gauss_stress(s)
        vm = von_mises(sig)
        out[pu] = (vm.max() / ref, np.abs(sig).max() / ref)
        pc = draw_cells(ax, s, sig[..., 1], 0.0, DIVMAP, -0.06 * ref, 0.06 * ref)
        for q in s.gu.IEN:
            c = s.gu.X[[q[0], q[s.gu.p], q[-1], q[-1 - s.gu.p]]]
            ax.add_patch(Polygon(c, closed=True, fill=False, ec="0.6", lw=0.4))
        ax.set_title(title + r": $\sigma_{11}$ between %.1f and %.1f MPa" % (round(sig[..., 1].min(), 1) + 0.0, round(sig[..., 1].max(), 1) + 0.0), fontsize=10.5, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-5, PL + 5)
        ax.set_ylim(-5, PH + 5)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.04, hspace=0.3)
    cb = fig.colorbar(pc, cax=fig.add_axes([0.9, 0.12, 0.015, 0.75]))
    cb.set_label(r"$\sigma_{11}$ (MPa)", fontsize=9)
    return fig, out


def fig_blocks():
    """The block structure of the coupled equations, one-way and two-way."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    zero, blk, cpl = "#f4f5f7", "#cfe3f5", "#f6d2c4"

    def box(ax, x, y, w, h, fc, text, fs=12):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    for ax, two in zip(axes, (False, True)):
        box(ax, 0, 2, 2, 2, blk, r"$\mathbf{K}_{\theta\theta}$" if not two else
            r"$\frac{1}{\Delta t}\mathbf{C} + \mathbf{K}_{\theta\theta}$", 15 if not two else 13)
        box(ax, 2, 2, 2.6, 2, cpl if two else zero, r"$\frac{T_0}{\Delta t}\mathbf{K}_{u\theta}^{T}$" if two else r"$\mathbf{0}$", 15)
        box(ax, 0, 0, 2, 2, cpl, r"$-\mathbf{K}_{u\theta}$", 15)
        box(ax, 2, 0, 2.6, 2, blk, r"$\mathbf{K}_{uu}$", 15)
        ax.text(4.9, 3.0, r"$\boldsymbol{\theta}$", fontsize=15, va="center")
        ax.text(4.9, 1.0, r"$\mathbf{d}$", fontsize=15, va="center")
        ax.text(5.45, 2.0, "$=$", fontsize=15, va="center")
        ax.text(5.95, 3.0, r"$\mathbf{Q}$" if not two else r"$\mathbf{Q} + \ldots$", fontsize=15, va="center")
        ax.text(5.95, 1.0, r"$\mathbf{F}$", fontsize=15, va="center")
        ax.set_xlim(-0.3, 7.6)
        ax.set_ylim(-0.4, 4.4)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].set_title("(a) One-way: block lower triangular,\nso the temperature is solved first", fontsize=10.5, loc="left")
    axes[1].set_title("(b) Two-way, one time step: both coupling blocks\nare nonzero, so the two fields are solved together", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig
