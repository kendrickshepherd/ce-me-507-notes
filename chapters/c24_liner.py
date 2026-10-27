"""
Diffusion through a landfill liner beneath a tear in its geomembrane
(Class 24 application page).

A geomembrane lies on a compacted clay liner of thickness L.  Leachate of
concentration c0 stands on the geomembrane, which stops the contaminant except
at a long tear of width a.  Below the clay, a drainage layer carries away what
arrives, so c = 0 there.  The concentration satisfies -div(n D grad c) = 0.
Tears repeat every 2 X1 meters, so the half cell 0 <= x <= X1 is modeled with
zero-flux (symmetry) conditions on both sides.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle

import c24_fem as F
from notes_style import INK, BDRYCLR, FLUXCLR, LOADCLR

LCLAY = 0.6                  # m
A_TEAR = 0.02                # m
X1 = 3.0                     # m, half the spacing between tears
POROSITY = 0.35
DIFF = 6.0e-10               # m^2/s, effective diffusion coefficient of chloride in the clay
NDIFF = POROSITY * DIFF      # the coefficient in Fick's law, J = -n D grad c
C0 = 1.0                     # kg/m^3 (1000 mg/L of chloride in the leachate)
YEAR = 365.25 * 86400.0
CONCMAP = "Greens"
WASTE, GRAVEL, CLAY, DRAIN = "#a58b6f", "#d9d6cf", "#e9d8b4", "#cfe3f5"
TOL = 1e-12


def flux_1d():
    """Flux through a liner with no geomembrane, kg/(m^2 s)."""
    return NDIFF * C0 / LCLAY


def _count(length, h0, ratio, nmin):
    """Number of geometrically graded elements whose smallest is at most h0."""
    return max(nmin, math.ceil(math.log(1 + length * (ratio - 1) / h0) / math.log(ratio)))


def build(p=2, nref=1, a=A_TEAR, x1=X1, r0=1.35):
    """The elements next to the edge of the tear are scaled with its width:
    about a/8 tall (at most 3 mm) at the top of the clay, and half that wide
    beside the tear, so that narrow tears are resolved as well as wide ones."""
    r = r0 ** (1.0 / nref)
    n = lambda k: max(1, int(round(k * nref)))
    h_top = min(a / 8, 0.003)
    if a / 2 < x1 - 1e-12:
        nx = _count(x1 - a / 2, h_top / 2, r0, 26)
        xs = F.join(F.graded(0, a / 2, n(4), r, "b"), F.graded(a / 2, x1, n(nx), r, "a"))
    else:
        xs = F.graded(0, x1, n(12), 1.0, "a")
    ys = F.graded(0, LCLAY, n(_count(LCLAY, h_top, r0, 14)), r, "b")
    mesh = F.Mesh(xs, ys, p=p)
    tear = mesh.sides_where(lambda x, y, nn: abs(y - LCLAY) < 1e-9 and x < a / 2 and nn[1] > 0)
    drain = mesh.sides_where(lambda x, y, nn: abs(y) < 1e-9 and nn[1] < 0)
    sol = F.solve(mesh, kappa=NDIFF, dirichlet=[(tear, C0), (drain, 0.0)])
    out = dict(mesh=mesh, sol=sol, a=a, x1=x1)
    # mass flux per meter length of the tear (both halves), kg/(s m)
    out["Q_in"] = 2 * sol.reaction[mesh.nodes_of_sides(tear)].sum()
    out["Q_out"] = -2 * sol.reaction[mesh.nodes_of_sides(drain)].sum()
    out["a_eff"] = out["Q_in"] / flux_1d()
    return out


def _ellipk(m):
    """Complete elliptic integral of the first kind, parameter m, by the arithmetic-geometric mean."""
    x, y = 1.0, math.sqrt(1.0 - m)
    for _ in range(60):
        x, y = (x + y) / 2, math.sqrt(x * y)
    return math.pi / (2 * x)


def a_eff_exact(a, L=LCLAY):
    """Exact effective width of a tear of width a over a layer of thickness L
    that extends without limit on both sides (conformal mapping):
    a_eff = L K(m) / K(1 - m), with m = 1 - exp(-pi a / L)."""
    mp = math.exp(-math.pi * a / L)
    return L * _ellipk(1 - mp) / _ellipk(mp)


def base_flux(model, xs):
    """Flux leaving through the bottom of the clay, kg/(m^2 s)."""
    out = []
    for x in xs:
        e, xi, _ = F.locate(model["mesh"], x, 1e-12)
        out.append(NDIFF * F.gradient(model["sol"], e, xi, -1.0)[1, 0])
    return np.array(out)


# ---------------------------------------------------------------- figures
def fig_setting():
    fig, ax = plt.subplots(figsize=(10.2, 2.9))
    xl, xr = -3.0, 3.0
    lay = [(-0.9, -0.6, DRAIN, "drainage layer (carries away what arrives): $c$ = 0"),
           (-0.6, 0.0, CLAY, "compacted clay liner, 0.6 m"),
           (0.0, 0.3, GRAVEL, "leachate collection gravel, leachate at $c_0$"),
           (0.3, 1.2, WASTE, "waste")]
    for y0, y1, col, lab in lay:
        ax.add_patch(Rectangle((xl, y0), xr - xl, y1 - y0, facecolor=col, edgecolor="none"))
        ax.text(xr + 0.12, (y0 + y1) / 2, lab, ha="left", va="center", fontsize=10, color=INK)
    rng = np.random.default_rng(3)
    gx, gy = rng.uniform(xl, xr, 260), rng.uniform(0.03, 0.27, 260)
    ax.plot(gx, gy, "o", ms=2.2, color="0.55")
    ax.plot([xl, -A_TEAR * 6], [0, 0], color=INK, lw=3.2, solid_capstyle="butt")
    ax.plot([A_TEAR * 6, xr], [0, 0], color=INK, lw=3.2, solid_capstyle="butt")
    ax.annotate("tear in the geomembrane", xy=(0, 0.02), xytext=(-1.6, 0.55), fontsize=10.5, color=LOADCLR,
                ha="center", arrowprops=dict(arrowstyle="-|>", color=LOADCLR, lw=1.0))
    ax.text(-2.2, 0.06, "geomembrane", fontsize=10, color="white", va="bottom",
            bbox=dict(fc=INK, ec="none", pad=1.2))
    # the spreading plume, schematically
    for ang in np.linspace(-65, 65, 7):
        t = np.radians(ang)
        ax.annotate("", xy=(0.45 * np.sin(t) * 1.6, -0.45 * np.cos(t)), xytext=(0, -0.02),
                    arrowprops=dict(arrowstyle="-|>", color="#2f7d4f", lw=1.1, mutation_scale=9))
    ax.set_xlim(xl - 0.05, xr + 3.6)
    ax.set_ylim(-1.0, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_domain(model):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.2, 5.2))
    a, x1 = model["a"], model["x1"]
    ax.add_patch(Rectangle((0, 0), x1, LCLAY, facecolor="#f4f1ea", edgecolor="none"))
    lw = 3.2
    ax.plot([0, a / 2 + 0.03], [LCLAY, LCLAY], color=BDRYCLR, lw=lw + 1.5, solid_capstyle="butt")
    ax.plot([a / 2 + 0.03, x1], [LCLAY, LCLAY], color=FLUXCLR, lw=lw, solid_capstyle="butt")
    ax.plot([0, x1], [0, 0], color=BDRYCLR, lw=lw, solid_capstyle="butt")
    ax.plot([0, 0], [0, LCLAY], color=FLUXCLR, lw=lw)
    ax.plot([x1, x1], [0, LCLAY], color=FLUXCLR, lw=lw)
    k = dict(fontsize=10.5)
    ax.annotate("tear: $c = c_0$ (half of it, $a/2$ = 10 mm)", xy=(0.01, LCLAY), xytext=(0.25, LCLAY + 0.17),
                color=BDRYCLR, arrowprops=dict(arrowstyle="-", color=BDRYCLR, lw=0.9), **k)
    ax.text(1.9, LCLAY + 0.05, "intact geomembrane: no flux", ha="center", va="bottom", color=FLUXCLR, **k)
    ax.text(x1 / 2, -0.05, "drainage layer: $c = 0$", ha="center", va="top", color=BDRYCLR, **k)
    ax.text(-0.05, LCLAY / 2, "symmetry", ha="right", va="center", color=FLUXCLR, rotation=90, **k)
    ax.text(x1 + 0.05, LCLAY / 2, "symmetry\n(half way to\nthe next tear)", ha="left", va="center",
            color=FLUXCLR, **k)
    ax.text(1.6, LCLAY / 2, r"$-\nabla\cdot(nD\,\nabla c) = 0$", ha="center", va="center", fontsize=12)
    ax.set_title("(a) Half of one cell of the liner", fontsize=11.5, loc="left")
    m = model["mesh"]
    for e in range(m.nel):
        (xc, yc), (hx, hy) = m.cent[e], m.size[e]
        bx.add_patch(Rectangle((xc - hx / 2, yc - hy / 2), hx, hy, facecolor="none", edgecolor="0.5", lw=0.35))
    bx.set_title("(b) The biquadratic mesh, graded toward the edge of the tear", fontsize=11.5, loc="left")
    for a_ in (ax, bx):
        a_.set_xlim(-0.35, x1 + 0.75)
        a_.set_ylim(-0.25, LCLAY + 0.32)
        a_.set_aspect("equal")
        a_.axis("off")
    fig.tight_layout(h_pad=0.8)
    return fig


def fig_solution(model):
    m, sol = model["mesh"], model["sol"]
    on_axis = np.abs(m.X[:, 0]) < 1e-12
    new = np.where(on_axis, np.arange(m.nnp), m.nnp + np.cumsum(~on_axis) - 1)
    X = np.vstack([m.X, (m.X * np.array([-1, 1]))[~on_axis]])
    T = m.triangles()
    T = np.vstack([T, new[T][:, ::-1]])
    U = np.concatenate([sol.u, sol.u[~on_axis]]) / C0
    tri = mtri.Triangulation(X[:, 0], X[:, 1], T)
    G = F.nodal_gradient(sol)
    Gm = np.vstack([G, (G * np.array([-1, 1]))[~on_axis]])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.2, 6.4), gridspec_kw=dict(height_ratios=[1.15, 1]))
    lev = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 1.0]
    ax.tricontourf(tri, U, levels=np.linspace(0, 1, 21), cmap=CONCMAP)
    cs = ax.tricontour(tri, U, levels=lev[1:-1], colors=INK, linewidths=0.6)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    gx = np.linspace(-1.6, 1.6, 400)
    gy = np.linspace(0.002, LCLAY - 0.002, 90)
    GX, GY = np.meshgrid(gx, gy)
    jx = mtri.LinearTriInterpolator(tri, -Gm[:, 0])(GX, GY).filled(0)
    jy = mtri.LinearTriInterpolator(tri, -Gm[:, 1])(GX, GY).filled(0)
    seeds = np.array([[s, LCLAY - 0.004] for s in np.linspace(-model["a"] / 2 * 0.9, model["a"] / 2 * 0.9, 9)])
    ax.streamplot(gx, gy, jx, jy, start_points=seeds, color="#14532d", linewidth=0.8, arrowsize=0.7,
                  density=6, broken_streamlines=False)
    ax.plot([-1.6, -model["a"] / 2], [LCLAY, LCLAY], color=INK, lw=3)
    ax.plot([model["a"] / 2, 1.6], [LCLAY, LCLAY], color=INK, lw=3)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(0, LCLAY + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("distance from the center of the tear (m)")
    ax.set_ylabel("height above\nthe drainage layer (m)")
    ax.set_title("(a) Concentration $c/c_0$, with lines along which the contaminant moves", fontsize=11.5,
                 loc="left")
    xs = np.linspace(0, 1.6, 300)
    J = base_flux(model, xs) * YEAR * 1e3
    bx.plot(np.concatenate([-xs[::-1], xs]), np.concatenate([J[::-1], J]), color="#2f7d4f", lw=2)
    bx.axhline(flux_1d() * YEAR * 1e3, color="0.5", lw=1.0, ls="--")
    bx.text(1.55, flux_1d() * YEAR * 1e3 + 0.15, "with no geomembrane at all", ha="right", va="bottom",
            fontsize=9.5, color="0.4")
    bx.set_xlim(-1.6, 1.6)
    bx.set_ylim(0, None)
    bx.set_xlabel("distance from the center of the tear (m)")
    bx.set_ylabel("flux into the drainage\nlayer, g/(m² yr)")
    bx.set_title("(b) Where the contaminant arrives", fontsize=11.5, loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    bx.grid(True, color="0.9", lw=0.7)
    fig.tight_layout(h_pad=1.2)
    return fig


def fig_width(widths, eff):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    w = np.array(widths) * 1e3
    ax.loglog(w, np.array(eff) * 1e3, "o-", color="#2f7d4f", lw=1.6, ms=5, label="effective width $a_{eff}$")
    ax.loglog(w, w, color="0.5", lw=1.0, ls="--", label="actual width $a$")
    ax.set_xlabel("width of the tear, $a$ (mm)")
    ax.set_ylabel("width (mm)")
    ax.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="both", color="0.92", lw=0.6)
    fig.tight_layout()
    return fig
