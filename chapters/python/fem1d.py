"""
One-dimensional Galerkin finite elements for Hughes' model problem

    -(K u_,x)_,x = f   on ]0, L[,   plus one boundary condition at each end,

with Lagrange elements of degree p = 1, 2, 3, in Hughes' notation: global
nodes A = 1..n_np, local nodes a = 1..p+1, and the ID, IEN, LM arrays (all
1-based, with ID = 0 marking a Dirichlet node).

Used by Poisson1D_Heat_FEM.qmd and Poisson1D_Rod_FEM.qmd for their static
figures and printed matrices.  The in-browser explorers on those pages run
fem1d.js, a port of this file -- change the two together.

Boundary-condition arrangements, as on the strong-form pages:
    "a"  Neumann h at x = 0,   Dirichlet g at x = L   (Hughes)
    "b"  Dirichlet g at x = 0, Neumann h at x = L
    "c"  Dirichlet g0 at x = 0 and g at x = L
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from notes_style import (INK, LOADCLR, REACTCLR, DIMCLR, BDRYCLR, FLUXCLR,
                         MESHCLR, REFCLR, BASISCLRS, save, tidy)


# ------------------------------------------------------------------ basics
def gauss(n):
    """Gauss-Legendre points and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def parent_nodes(p):
    return np.linspace(-1.0, 1.0, p + 1)


def shape(p, xi):
    """N_a(xi) and N_a,xi(xi) for the p+1 Lagrange functions on the parent
    element, equally spaced nodes numbered left to right.  Rows are a."""
    xi = np.atleast_1d(np.asarray(xi, float))
    nd = parent_nodes(p)
    N = np.ones((p + 1, xi.size))
    dN = np.zeros((p + 1, xi.size))
    for a in range(p + 1):
        for b in range(p + 1):
            if b != a:
                N[a] *= (xi - nd[b]) / (nd[a] - nd[b])
        for m in range(p + 1):
            if m == a:
                continue
            term = np.full(xi.size, 1.0 / (nd[a] - nd[m]))
            for b in range(p + 1):
                if b not in (a, m):
                    term *= (xi - nd[b]) / (nd[a] - nd[b])
            dN[a] += term
    return N, dN


def make_mesh(L, nel, kind="uniform", custom=None, grade=2.0):
    """Element end points.  'left' crowds elements toward x = 0, 'right'
    toward x = L; 'custom' takes a list of interior vertex positions."""
    s = np.linspace(0.0, 1.0, nel + 1)
    if kind == "left":
        return L * s ** grade
    if kind == "right":
        return L * (1.0 - (1.0 - s) ** grade)
    if kind == "custom":
        inner = sorted(c for c in custom if 0.0 < c < L)
        return np.array([0.0] + inner + [L])
    return L * s


class Problem:
    def __init__(self, L, K, f, bc="a", h=0.0, g=0.0, g0=0.0):
        self.L, self.K, self.f, self.bc = L, K, f, bc
        self.h, self.g, self.g0 = h, g, g0


# ---------------------------------------------------------- exact solution
def exact(pb, nsub=64, nq=8):
    """Exact u and u_,x for constant K, from the repeated integrals
    F1(x) = int_0^x f   and   F2(x) = int_0^x (x - s) f(s) ds."""
    gx, gw = gauss(nq)

    def F(x):
        if x <= 0.0:
            return 0.0, 0.0
        m = max(1, int(math.ceil(nsub * x / pb.L)))
        e = np.linspace(0.0, x, m + 1)
        a, b = e[:-1, None], e[1:, None]
        s = 0.5 * (a + b) + 0.5 * (b - a) * gx[None, :]
        w = 0.5 * (b - a) * gw[None, :]
        fs = pb.f(s)
        return float((w * fs).sum()), float((w * (x - s) * fs).sum())

    K, L = pb.K, pb.L
    F1L, F2L = F(L)
    if pb.bc == "a":
        C2 = -pb.h / K
        C1 = pb.g - C2 * L + F2L / K
    elif pb.bc == "b":
        C1 = pb.g
        C2 = (pb.h + F1L) / K
    else:
        C1 = pb.g0
        C2 = (pb.g - pb.g0 + F2L / K) / L

    def u(x):
        x = np.atleast_1d(np.asarray(x, float))
        return np.array([C1 + C2 * xv - F(xv)[1] / K for xv in x])

    def du(x):
        x = np.atleast_1d(np.asarray(x, float))
        return np.array([C2 - F(xv)[0] / K for xv in x])

    return u, du


# ------------------------------------------------------------------- solve
def dirichlet_nodes(pb, nnp):
    """{A: prescribed value} for the nodes on Gamma_g."""
    if pb.bc == "a":
        return {nnp: pb.g}
    if pb.bc == "b":
        return {1: pb.g}
    return {1: pb.g0, nnp: pb.g}


def solve(pb, p, verts, nint=None):
    verts = np.asarray(verts, float)
    nel, nen = len(verts) - 1, p + 1
    nnp = p * nel + 1
    X = np.empty(nnp)
    IEN = np.zeros((nen, nel), int)
    for e in range(nel):
        for a in range(nen):
            A = p * e + a + 1
            IEN[a, e] = A
            X[A - 1] = verts[e] + (verts[e + 1] - verts[e]) * a / p
    gnode = dirichlet_nodes(pb, nnp)
    ID = np.zeros(nnp, int)
    neq = 0
    for A in range(1, nnp + 1):
        if A not in gnode:
            neq += 1
            ID[A - 1] = neq
    LM = ID[IEN - 1]

    nint = nint or p + 3
    gx, gw = gauss(nint)
    N, dN = shape(p, gx)
    Kg = np.zeros((neq, neq))
    F = np.zeros(neq)
    ke_all, fe_all = [], []
    for e in range(nel):
        he = verts[e + 1] - verts[e]
        jac = he / 2.0                                   # x_,xi
        xq = verts[e] + (gx + 1.0) * jac
        ke = pb.K * (dN * gw / jac) @ dN.T
        f_body = (N * gw * jac) @ pb.f(xq)
        f_h = np.zeros(nen)
        if pb.bc == "a" and e == 0:
            f_h[0] = pb.h
        if pb.bc == "b" and e == nel - 1:
            f_h[-1] = pb.h
        ge = np.array([gnode.get(IEN[a, e], 0.0) for a in range(nen)])
        f_g = ke @ ge
        fe = f_body + f_h - f_g
        ke_all.append(ke)
        fe_all.append(dict(body=f_body, h=f_h, g=f_g, total=fe))
        for a in range(nen):
            P = LM[a, e]
            if P == 0:
                continue
            F[P - 1] += fe[a]
            for b in range(nen):
                Q = LM[b, e]
                if Q:
                    Kg[P - 1, Q - 1] += ke[a, b]
    d = np.linalg.solve(Kg, F) if neq else np.zeros(0)
    U = np.array([d[ID[A - 1] - 1] if ID[A - 1] else gnode[A]
                  for A in range(1, nnp + 1)])
    return dict(p=p, verts=verts, nel=nel, nnp=nnp, neq=neq, X=X, IEN=IEN,
                ID=ID, LM=LM, ke=ke_all, fe=fe_all, K=Kg, F=F, d=d, U=U,
                gnode=gnode)


def evaluate(sol, m=40):
    """u^h and u^h_,x sampled inside each element.  Returns a list of
    (x, uh, duh) arrays, one per element, so plots can break at element
    boundaries where u^h_,x jumps."""
    p, verts, IEN, U = sol["p"], sol["verts"], sol["IEN"], sol["U"]
    xi = np.linspace(-1, 1, m)
    N, dN = shape(p, xi)
    out = []
    for e in range(sol["nel"]):
        jac = (verts[e + 1] - verts[e]) / 2.0
        Ue = U[IEN[:, e] - 1]
        out.append((verts[e] + (xi + 1) * jac, Ue @ N, (Ue @ dN) / jac))
    return out


def errors(sol, u, du, nq=10):
    """||u - u^h||_{L2} and |u - u^h|_{H1} (the energy-type seminorm)."""
    p, verts, IEN, U = sol["p"], sol["verts"], sol["IEN"], sol["U"]
    gx, gw = gauss(nq)
    N, dN = shape(p, gx)
    l2 = h1 = 0.0
    for e in range(sol["nel"]):
        jac = (verts[e + 1] - verts[e]) / 2.0
        xq = verts[e] + (gx + 1) * jac
        Ue = U[IEN[:, e] - 1]
        l2 += np.sum(gw * jac * (u(xq) - Ue @ N) ** 2)
        h1 += np.sum(gw * jac * (du(xq) - (Ue @ dN) / jac) ** 2)
    return math.sqrt(l2), math.sqrt(h1)


# ------------------------------------------------------------ LaTeX output
def num(v, sig=4):
    if abs(v) < 1e-12:
        return "0"
    s = "%.*g" % (sig, v)
    if "e" in s:
        m, ex = s.split("e")
        return r"%s\times10^{%d}" % (m, int(ex))
    return s


def tex_matrix(M, sig=4, maxn=14, show=5, zero_gray=True):
    """A bmatrix; beyond maxn rows it shows the first `show` rows and columns
    and the last one, with dots between."""
    M = np.atleast_2d(M)
    n, m = M.shape

    def cell(i, j):
        v = M[i, j]
        if zero_gray and abs(v) < 1e-12:
            return r"\color{#bbbbbb}{0}"
        return num(v, sig)

    def idx(k):
        return list(range(k)) if k <= maxn else list(range(show)) + [None, k - 1]

    rows = []
    for i in idx(n):
        if i is None:
            rows.append(" & ".join(r"\ddots" if (j is None) else r"\vdots"
                                   for j in idx(m)))
            continue
        rows.append(" & ".join(r"\cdots" if j is None else cell(i, j)
                               for j in idx(m)))
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def tex_vector(v, sig=4, maxn=14, show=5):
    return tex_matrix(np.atleast_2d(v).T, sig=sig, maxn=maxn, show=show,
                      zero_gray=False)


# ----------------------------------------------------------------- figures
def graph(ax, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9.5)


def fig_basis(verts, bc, name, xlabel="$x$"):
    """Top: the parent-element Lagrange functions N_a(xi) for p = 1, 2, 3.
    Bottom: the global functions N_A they build on a mesh; the ones on
    Gamma_g (which carry g, not an unknown) are dashed."""
    verts = np.asarray(verts, float)
    nel = len(verts) - 1
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.6),
                             gridspec_kw=dict(height_ratios=[1.0, 1.05]))
    xi = np.linspace(-1, 1, 200)
    for j, p in enumerate((1, 2, 3)):
        ax = axes[0, j]
        N, _ = shape(p, xi)
        for a in range(p + 1):
            c = BASISCLRS[a % len(BASISCLRS)]
            ax.plot(xi, N[a], color=c, lw=2.0)
            xa = parent_nodes(p)[a]
            ax.plot([xa], [1.0], "o", color=c, ms=5)
            ax.text(xa, 1.08, "$N_{%d}$" % (a + 1), color=c, ha="center",
                    va="bottom", fontsize=11)
        ax.axhline(0, color=DIMCLR, lw=0.7)
        ax.plot(parent_nodes(p), 0 * parent_nodes(p), "|", color=REFCLR,
                ms=10, mew=1.6)
        ax.set_title("degree $p = %d$: %d functions" % (p, p + 1),
                     fontsize=11.5)
        ax.set_ylim(-0.35, 1.3)
        ax.set_xticks([-1, 0, 1])
        graph(ax, r"parent coordinate $\xi$", "$N_a(\\xi)$" if j == 0 else None)

        ax = axes[1, j]
        sol_nodes = np.concatenate([np.linspace(verts[e], verts[e + 1], p + 1)[:-1]
                                    for e in range(nel)] + [[verts[-1]]])
        nnp = len(sol_nodes)
        gset = {"a": {nnp}, "b": {1}, "c": {1, nnp}}[bc]
        for e in range(nel):
            xs = np.linspace(verts[e], verts[e + 1], 80)
            Ne, _ = shape(p, 2 * (xs - verts[e]) / (verts[e + 1] - verts[e]) - 1)
            for a in range(p + 1):
                A = p * e + a + 1
                c = BASISCLRS[(A - 1) % len(BASISCLRS)]
                ax.plot(xs, Ne[a], color=c, lw=1.8,
                        ls="--" if A in gset else "-")
        for A, xv in enumerate(sol_nodes, start=1):
            c = BASISCLRS[(A - 1) % len(BASISCLRS)]
            ax.text(xv, 1.08, "$N_{%d}$" % A, color=c, ha="center",
                    va="bottom", fontsize=9.5 if nnp < 9 else 8)
            ax.plot([xv], [0], "o", ms=5, color=INK if A in gset else "white",
                    mec=INK, zorder=5)
        for xv in verts:
            ax.axvline(xv, color=MESHCLR, lw=0.8, ls=":")
        ax.axhline(0, color=DIMCLR, lw=0.7)
        ax.set_ylim(-0.35, 1.3)
        ax.set_xticks(verts)
        ax.set_xticklabels(["%g" % v for v in verts], fontsize=9)
        graph(ax, xlabel, "$N_A(x)$" if j == 0 else None)
    fig.tight_layout(h_pad=1.2, w_pad=1.4)
    return save(fig, name)


def fig_assembly(bc, name, nel=4):
    """How element matrices tile the global matrix, for p = 1, 2, 3.  Each
    coloured square is one element's k^e; overlaps are where two elements
    share a node.  The hatched row and column belong to a Dirichlet node:
    they are not equations, and their k^e entries move into F as -k^e g."""
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
    for j, p in enumerate((1, 2, 3)):
        ax = axes[j]
        nnp = p * nel + 1
        gset = {"a": {nnp}, "b": {1}, "c": {1, nnp}}[bc]
        for e in range(nel):
            A0 = p * e
            c = BASISCLRS[e % len(BASISCLRS)]
            ax.add_patch(Rectangle((A0, A0), p + 1, p + 1, facecolor=c,
                                   alpha=0.28, edgecolor=c, lw=1.6, zorder=2))
            ax.text(A0 + (p + 1) / 2, A0 + (p + 1) / 2, "$k^{%d}$" % (e + 1),
                    ha="center", va="center", fontsize=11 if p > 1 else 9.5,
                    color=c, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
        for A in gset:
            ax.add_patch(Rectangle((0, A - 1), nnp, 1, facecolor="none",
                                   edgecolor=DIMCLR, hatch="////", lw=0,
                                   zorder=3))
            ax.add_patch(Rectangle((A - 1, 0), 1, nnp, facecolor="none",
                                   edgecolor=DIMCLR, hatch="////", lw=0,
                                   zorder=3))
        for k in range(nnp + 1):
            ax.plot([0, nnp], [k, k], color="0.85", lw=0.5, zorder=1)
            ax.plot([k, k], [0, nnp], color="0.85", lw=0.5, zorder=1)
        ax.add_patch(Rectangle((0, 0), nnp, nnp, facecolor="none",
                               edgecolor=INK, lw=1.2, zorder=5))
        ax.set_xlim(-0.3, nnp + 0.3)
        ax.set_ylim(nnp + 0.3, -1.2)
        step = 1 if nnp <= 9 else 2
        for A in range(1, nnp + 1, step):
            ax.text(A - 0.5, -0.35, str(A), ha="center", va="bottom",
                    fontsize=8, color=DIMCLR)
        ax.set_title("$p = %d$: %d nodes, %d equations" % (p, nnp, nnp - len(gset)),
                     fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(w_pad=1.5)
    return save(fig, name)


def fig_parent_map(name, verts=(0.10, 0.22), p=2, xunit="m"):
    """The isoparametric map from the parent element [-1, 1] onto one
    physical element, with the quadrature points carried along."""
    x1, x2 = verts
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    gx, _ = gauss(p + 1)
    # parent
    ax.plot([-1, 1], [1.2, 1.2], color=REFCLR, lw=2.6)
    for a, xa in enumerate(parent_nodes(p)):
        ax.plot([xa], [1.2], "o", color=REFCLR, ms=7, zorder=5)
        ax.text(xa, 1.38, r"$\xi_{%d} = %g$" % (a + 1, xa), ha="center",
                va="bottom", fontsize=10.5, color=REFCLR)
    for gq in gx:
        ax.plot([gq], [1.2], "x", color=LOADCLR, ms=8, mew=2, zorder=6)
    ax.text(-1.2, 1.2, "parent\nelement", ha="right", va="center",
            fontsize=10.5, color=REFCLR)
    # physical, drawn in a scaled copy underneath
    X0, X1 = -0.55, 1.55
    ax.plot([X0, X1], [0, 0], color=INK, lw=2.6)
    for a in range(p + 1):
        xa = X0 + (X1 - X0) * a / p
        ax.plot([xa], [0], "o", color=INK, ms=7, zorder=5)
        xv = x1 + (x2 - x1) * a / p
        ax.text(xa, -0.2, "$x^e_{%d}$ = %g %s" % (a + 1, round(xv, 4), xunit),
                ha="center", va="top", fontsize=10.5)
    for gq in gx:
        xa = X0 + (X1 - X0) * (gq + 1) / 2
        ax.plot([xa], [0], "x", color=LOADCLR, ms=8, mew=2, zorder=6)
        ax.annotate("", xy=(xa, 0.12), xytext=(gq, 1.08),
                    arrowprops=dict(arrowstyle="-|>", color=LOADCLR, lw=0.9,
                                    ls=":", mutation_scale=9))
    ax.text(-1.2, 0.0, "element $e$", ha="right", va="center", fontsize=10.5)
    ax.text(1.9, 0.62,
            r"$x(\xi) = \sum_a N_a(\xi)\,x^e_a$" + "\n"
            + r"$x_{,\xi} = h^e/2$" + "\n"
            + r"$\times$ = Gauss points $\tilde\xi_l$",
            ha="left", va="center", fontsize=11)
    tidy(ax)
    ax.set_xlim(-2.0, 3.6)
    ax.set_ylim(-0.7, 1.85)
    return save(fig, name)


def fig_build(pb, verts, name, uscale=1.0, ulabel="$u$", xlabel="$x$",
              degrees=(1, 2)):
    """u^h as the sum of its scaled basis functions, next to the exact u."""
    u, _ = exact(pb)
    fig, axes = plt.subplots(1, len(degrees), figsize=(10.4, 3.9), sharey=True)
    xx = np.linspace(0, pb.L, 300)
    for ax, p in zip(np.atleast_1d(axes), degrees):
        sol = solve(pb, p, verts)
        ax.plot(xx, u(xx) * uscale, color="0.75", lw=6, solid_capstyle="round",
                label="exact $u$", zorder=1)
        for e in range(sol["nel"]):
            v0, v1 = sol["verts"][e], sol["verts"][e + 1]
            xs = np.linspace(v0, v1, 60)
            Ne, _ = shape(p, 2 * (xs - v0) / (v1 - v0) - 1)
            for a in range(p + 1):
                A = sol["IEN"][a, e]
                c = BASISCLRS[(A - 1) % len(BASISCLRS)]
                gnodal = A in sol["gnode"]
                ax.plot(xs, sol["U"][A - 1] * Ne[a] * uscale, color=c, lw=1.2,
                        ls="--" if gnodal else "-", zorder=2)
        for (xs, uh, _) in evaluate(sol):
            ax.plot(xs, uh * uscale, color=INK, lw=2.2, zorder=3)
        ax.plot(sol["X"], sol["U"] * uscale, "o", ms=5, color=INK, zorder=4)
        for xv in sol["verts"]:
            ax.axvline(xv, color=MESHCLR, lw=0.7, ls=":")
        ax.set_title("$p = %d$, %d elements: %d unknowns $d_A$"
                     % (p, sol["nel"], sol["neq"]), fontsize=11)
        graph(ax, xlabel, ulabel if p == degrees[0] else None)
    ax.plot([], [], color=INK, lw=2.2, label=r"$u^h = \sum_A d_A N_A + g^h$")
    ax.plot([], [], color="0.45", lw=1.2, label=r"each $d_A N_A$ (coloured by $A$)")
    ax.plot([], [], color="0.45", lw=1.2, ls="--", label=r"$g\,N_A$ on $\Gamma_g$")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
    return save(fig, name)


def fig_result(pb, p, verts, name, flux, uscale=1.0, fscale=1.0,
               ulabel="$u$", flabel="flux", xlabel="$x$", neu_label=None):
    """Strong-form picture, redrawn for the FEM: u^h over u, and the
    piecewise flux of u^h over the exact flux.  `flux(du)` turns a slope
    into the physical flux (heat: -K du; rod: K du)."""
    u, du = exact(pb)
    sol = solve(pb, p, verts)
    xx = np.linspace(0, pb.L, 300)
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.6), sharex=True)
    ax = axes[0]
    ax.plot(xx, u(xx) * uscale, color="0.75", lw=6, solid_capstyle="round",
            label="exact $u$")
    for k, (xs, uh, _) in enumerate(evaluate(sol)):
        ax.plot(xs, uh * uscale, color=LOADCLR, lw=2.2,
                label=r"$u^h$" if k == 0 else None)
    ax.plot(sol["X"], sol["U"] * uscale, "o", ms=5, color=LOADCLR)
    for xv in sol["verts"]:
        ax.axvline(xv, color=MESHCLR, lw=0.7, ls=":")
    ax.legend(frameon=False, fontsize=10.5)
    graph(ax, None, ulabel)
    ax = axes[1]
    ax.plot(xx, flux(du(xx)) * fscale, color="0.75", lw=6,
            solid_capstyle="round", label="exact")
    for k, (xs, _, duh) in enumerate(evaluate(sol)):
        ax.plot(xs, flux(duh) * fscale, color=FLUXCLR, lw=2.2,
                label="from $u^h$" if k == 0 else None)
    for xv in sol["verts"]:
        ax.axvline(xv, color=MESHCLR, lw=0.7, ls=":")
    if neu_label is not None:
        xn, val, txt = neu_label
        ax.plot([xn], [val * fscale], "o", color=INK, ms=7, zorder=6)
        ax.annotate(txt, xy=(xn, val * fscale),
                    xytext=(0.5 * pb.L, val * fscale), ha="center",
                    va="center", fontsize=10.5,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.legend(frameon=False, fontsize=10.5)
    graph(ax, xlabel, flabel)
    fig.tight_layout(h_pad=1.0)
    return save(fig, name), sol


def fig_convergence(pb, name, hlabel="element size $h^e$", nels=(1, 2, 4, 8, 16, 32)):
    """Errors on uniform meshes, log-log, for p = 1, 2, 3."""
    u, du = exact(pb)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    rates = {}
    for p in (1, 2, 3):
        hs, e0, e1 = [], [], []
        for n in nels:
            sol = solve(pb, p, make_mesh(pb.L, n))
            a, b = errors(sol, u, du)
            hs.append(pb.L / n)
            e0.append(a)
            e1.append(b)
        c = BASISCLRS[p - 1]
        rates[p] = (math.log(e0[-2] / e0[-1]) / math.log(2),
                    math.log(e1[-2] / e1[-1]) / math.log(2))
        axes[0].loglog(hs, e0, "o-", color=c, lw=1.8, ms=5,
                       label="$p = %d$: slope %.1f" % (p, rates[p][0]))
        axes[1].loglog(hs, e1, "o-", color=c, lw=1.8, ms=5,
                       label="$p = %d$: slope %.1f" % (p, rates[p][1]))
    axes[0].set_title(r"$\|u - u^h\|_{L^2}$: slope $p + 1$", fontsize=11.5)
    axes[1].set_title(r"$|u - u^h|_{H^1}$ (slope error): slope $p$", fontsize=11.5)
    for ax in axes:
        ax.legend(frameon=False, fontsize=10)
        graph(ax, hlabel, None)
        ax.grid(True, which="major", color="0.9", lw=0.6)
    fig.tight_layout(w_pad=2.5)
    return save(fig, name), rates
