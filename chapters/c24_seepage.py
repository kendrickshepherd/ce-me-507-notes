"""
Seepage under a concrete gravity dam (Class 24 application page).

A dam with a 20 m wide base sits on a 20 m layer of sand over impermeable
rock.  The reservoir stands 12 m above the ground upstream and the tailwater
2 m above it downstream.  A cutoff wall 8 m deep may be placed at the heel
(upstream) or at the toe (downstream).  The total head h satisfies
-div(K grad h) = 0 in the sand.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle

import c24_fem as F
from notes_style import INK, BDRYCLR, FLUXCLR, LOADCLR

H1, H2 = 12.0, 2.0           # m, reservoir and tailwater levels above the ground
B = 20.0                     # m, dam base
D = 20.0                     # m, sand layer depth
DCUT = 8.0                   # m, cutoff depth
LSIDE = 40.0                 # m, extent of the model upstream and downstream
K = 2.0e-5                   # m/s, hydraulic conductivity of the sand
GW = 9.81                    # kN/m^3
IC = 1.0                     # critical gradient (G_s = 2.65, e = 0.65)
WIN = 2.0                    # m, window over which the exit gradient is averaged
DAY = 86400.0
SAND, ROCK, WATER, CONC = "#efe3cf", "#c9c3b8", "#cfe3f5", "#d6d6d6"
HEADMAP = "Blues"
TOL = 1e-9


def build(p=2, nref=1, cut=None, kappa=None, depth=D, side=LSIDE, r0=1.2):
    """Solve for the head.  cut is None, 'up', or 'down'."""
    r = r0 ** (1.0 / nref)
    n = lambda k: max(1, int(round(k * nref)))
    xs = F.join(F.graded(-side, 0, n(12), r, "b"), F.graded(0, B, 2 * n(8), r, "both"),
                F.graded(B, B + WIN, n(4), r, "a"), F.graded(B + WIN, B + side, n(10), r, "a"))
    if cut:
        ys = F.join(F.graded(-depth, -DCUT, n(8), r, "b"), F.graded(-DCUT, 0, 2 * n(5), r, "both"))
    else:
        ys = F.graded(-depth, 0, n(14), r, "b")
    xcut = {"up": 0.0, "down": B}.get(cut)
    mesh = F.Mesh(xs, ys, p=p, slits=[("v", xcut, -DCUT, 0.0)] if cut else [])
    up = mesh.sides_where(lambda x, y, nn: abs(y) < TOL and x < 0 and nn[1] > 0)
    dn = mesh.sides_where(lambda x, y, nn: abs(y) < TOL and x > B and nn[1] > 0)
    base = mesh.sides_where(lambda x, y, nn: abs(y) < TOL and 0 < x < B and nn[1] > 0)
    kap = K if kappa is None else kappa
    sol = F.solve(mesh, kappa=kap, dirichlet=[(up, H1), (dn, H2)])
    out = dict(mesh=mesh, sol=sol, cut=cut, kappa=kap, up=up, dn=dn, base=base, depth=depth, side=side)
    out["q"] = sol.reaction[mesh.nodes_of_sides(up)].sum()            # m^3/s per m, into the sand
    out["q_out"] = -sol.reaction[mesh.nodes_of_sides(dn)].sum()
    out["U"] = GW * F.side_integral(sol, base)                         # kN per m of dam
    win = mesh.sides_where(lambda x, y, nn: abs(y) < TOL and B < x < B + WIN and nn[1] > 0)
    kv = kap if np.ndim(kap) == 0 else kap[1]
    out["kv"] = kv
    # average upward gradient over the window, two ways: by integrating the
    # computed gradient, and from the flow that crosses the window (psi)
    out["i_avg_grad"] = -F.side_flux(sol, win) / (kv * WIN)
    psi = stream_function(out)
    e, xi, _ = F.locate(mesh, B + WIN, -1e-9)
    out["psi"] = psi
    out["i_avg"] = F.value(psi, e, xi, 1.0)[0] / (kv * WIN)
    return out


def stream_function(model):
    """The stream function psi (m^3/s per m): v_x = dpsi/dy, v_y = -dpsi/dx.
    psi = 0 on the dam base and the cutoff, and psi = q on the rock and the
    far sides.  For K = diag(k_h, k_v), psi satisfies
    div(diag(1/k_v, 1/k_h) grad psi) = 0."""
    mesh = model["mesh"]
    kap = model["kappa"]
    kh, kv = (kap, kap) if np.ndim(kap) == 0 else (kap[0], kap[1])
    q = model["q"]
    top = mesh.sides_where(lambda x, y, nn: (abs(y) < TOL and 0 < x < B and nn[1] > 0)
                           or (model["cut"] is not None and abs(x - {"up": 0.0, "down": B}[model["cut"]]) < TOL))
    outer = mesh.sides_where(lambda x, y, nn: abs(y + model["depth"]) < TOL
                             or abs(x + model["side"]) < TOL or abs(x - B - model["side"]) < TOL)
    return F.solve(mesh, kappa=(1.0 / kv, 1.0 / kh), dirichlet=[(top, 0.0), (outer, q)])


def surface_gradient(model, xs):
    """Upward exit gradient -dh/dy along the ground downstream of the toe."""
    sol, mesh = model["sol"], model["mesh"]
    out = []
    for x in xs:
        e, xi, _ = F.locate(mesh, x, -1e-9)
        out.append(-F.gradient(sol, e, xi, 1.0)[1, 0])
    return np.array(out)


def base_head(model, xs):
    sol, mesh = model["sol"], model["mesh"]
    out = []
    for x in xs:
        e, xi, _ = F.locate(mesh, x, -1e-9)
        out.append(F.value(sol, e, xi, 1.0)[0])
    return np.array(out)


# ---------------------------------------------------------------- figures
def _rock(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=ROCK, edgecolor="none",
                           hatch="xx", zorder=0))


def fig_setting(model_up):
    """The dam, reservoir, sand, rock, and cutoff, with computed flow lines."""
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    xl, xr, yb = -LSIDE, B + LSIDE, -D
    ax.add_patch(Rectangle((xl, yb), xr - xl, D, facecolor=SAND, edgecolor="none", zorder=0))
    _rock(ax, xl, xr, yb - 4, yb)
    ax.add_patch(Rectangle((xl, 0), -xl, H1, facecolor=WATER, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((B, 0), LSIDE, H2, facecolor=WATER, edgecolor="none", zorder=0))
    ax.plot([xl, 0], [H1, H1], color="#2b5fa8", lw=1.4)
    ax.plot([B, xr], [H2, H2], color="#2b5fa8", lw=1.4)
    for xw, yw in ((-20, H1), (B + 20, H2)):
        ax.plot([xw - 1.2, xw + 1.2, xw, xw - 1.2], [yw + 1.4, yw + 1.4, yw + 0.2, yw + 1.4],
                color="#2b5fa8", lw=1.0)
    dam = [(0, 0), (B, 0), (4.5, 15.0), (0, 15.0)]
    ax.add_patch(Polygon(dam, closed=True, facecolor=CONC, edgecolor=INK, lw=1.3, zorder=3))
    ax.plot([0, 0], [0, -DCUT], color=INK, lw=3.2, zorder=4, solid_capstyle="butt")
    # computed flow lines (isotropic, upstream cutoff)
    psi, qK = model_up["psi"], model_up["q"]
    m = model_up["mesh"]
    tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
    ax.tricontour(tri, psi.u, levels=np.linspace(0, qK, 6)[1:-1], colors="#2b5fa8",
                  linewidths=1.0, linestyles="-", zorder=2)
    ax.annotate("", xy=(B + 6, -0.6), xytext=(B + 4.5, -3.2),
                arrowprops=dict(arrowstyle="-|>", color="#2b5fa8", lw=1.2), zorder=5)
    ax.plot([xl, xr], [0, 0], color="#8a6d3b", lw=1.2, zorder=1)
    t = dict(fontsize=10.5, color=INK)
    ax.text(-20, H1 / 2, "reservoir", ha="center", va="center", **t)
    ax.text(B + 22, H2 + 2.2, "tailwater", ha="center", va="bottom", **t)
    ax.text(6.0, 6.0, "concrete\ngravity dam", ha="center", va="center", fontsize=10, color=INK, zorder=5)
    ax.text(-26, -12, "sand foundation\nhydraulic conductivity $K$", ha="center", va="center", **t)
    ax.text(B + 30, -22, "impermeable rock", ha="center", va="center", fontsize=10.5, color="#4a443a",
            bbox=dict(fc=ROCK, ec="none", pad=1.5))
    ax.annotate("cutoff wall", xy=(0, -6.5), xytext=(-9, -16.5), ha="center", va="center", fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    ax.text(B + 7, -6.0, "seepage", ha="left", va="center", fontsize=10.5, color="#2b5fa8")
    # dimensions
    def dim(x0, y0, x1, y1, lab, off=(0, 0), ha="center", va="center"):
        ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], lab, ha=ha, va=va, fontsize=10, color="0.25")
    dim(-37, 0, -37, H1, "$H_1$ = 12 m", off=(1.0, 0), ha="left")
    dim(B + 37, 0, B + 37, H2, "$H_2$ = 2 m", off=(-1.0, 3.2), ha="right")
    dim(0, 17.0, B, 17.0, "$B$ = 20 m", off=(0, 1.1))
    dim(-3, 0, -3, -DCUT, "$d$ = 8 m", off=(-0.8, 0), ha="right")
    dim(B + 37, 0, B + 37, -D, "$D$ = 20 m", off=(-1.0, 0), ha="right")
    ax.set_xlim(xl - 1, xr + 1)
    ax.set_ylim(yb - 4.5, 19.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_domain(model_up):
    """(a) the domain and its boundary conditions, (b) the mesh."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.4, 6.2))
    xl, xr = -LSIDE, B + LSIDE
    ax.add_patch(Rectangle((xl, -D), xr - xl, D, facecolor="#f4f1ea", edgecolor="none"))
    lw = 3.2
    ax.plot([xl, 0], [0, 0], color=BDRYCLR, lw=lw, solid_capstyle="butt")
    ax.plot([B, xr], [0, 0], color=BDRYCLR, lw=lw, solid_capstyle="butt")
    for seg in (([0, B], [0, 0]), ([xl, xl], [-D, 0]), ([xr, xr], [-D, 0]), ([xl, xr], [-D, -D])):
        ax.plot(*seg, color=FLUXCLR, lw=lw, solid_capstyle="butt")
    ax.plot([-0.25, -0.25], [0, -DCUT], color=FLUXCLR, lw=2.2)
    ax.plot([0.25, 0.25], [0, -DCUT], color=FLUXCLR, lw=2.2)
    t = dict(fontsize=10.5)
    ax.text(-20, 1.0, "$h = H_1$ = 12 m", ha="center", va="bottom", color=BDRYCLR, **t)
    ax.text(B + 20, 1.0, "$h = H_2$ = 2 m", ha="center", va="bottom", color=BDRYCLR, **t)
    ax.text(B / 2, 1.0, "dam base: no flow", ha="center", va="bottom", color=FLUXCLR, **t)
    ax.text(1.2, -DCUT / 2, "cutoff: no flow\non either face", ha="left", va="center", color=FLUXCLR, **t)
    ax.text((xl + xr) / 2, -D - 1.0, "rock: no flow", ha="center", va="top", color=FLUXCLR, **t)
    ax.text(xl - 1.0, -D / 2, "no flow", ha="right", va="center", color=FLUXCLR, rotation=90, **t)
    ax.text(xr + 1.0, -D / 2, "no flow", ha="left", va="center", color=FLUXCLR, rotation=90, **t)
    ax.text(-22, -11, r"$-\nabla\cdot(K\,\nabla h) = 0$", ha="center", va="center", fontsize=13, color=INK)
    ax.set_title("(a) The sand layer and its boundary conditions", fontsize=11.5, loc="left")
    for a in (ax, bx):
        a.set_xlim(xl - 4, xr + 4)
        a.set_ylim(-D - 3.5, 4.5)
        a.set_aspect("equal")
        a.axis("off")
    m = model_up["mesh"]
    for e in range(m.nel):
        (xc, yc), (hx, hy) = m.cent[e], m.size[e]
        bx.add_patch(Rectangle((xc - hx / 2, yc - hy / 2), hx, hy, facecolor="none",
                               edgecolor="0.55", lw=0.35))
    bx.plot([0, 0], [0, -DCUT], color=INK, lw=2.0)
    bx.set_title("(b) The biquadratic mesh, graded toward the heel, the toe, and the tip of the cutoff",
                 fontsize=11.5, loc="left")
    fig.tight_layout(h_pad=1.0)
    return fig


def fig_flownets(models):
    """Equipotentials every meter of head and flow lines every meter of psi,
    so that the flow net is made of approximately square cells."""
    titles = {None: "(a) No cutoff", "up": "(b) Cutoff at the heel (upstream)",
              "down": "(c) Cutoff at the toe (downstream)"}
    fig, axes = plt.subplots(len(models), 1, figsize=(10.4, 2.75 * len(models) + 0.4))
    for ax, md in zip(np.atleast_1d(axes), models):
        m, sol = md["mesh"], md["sol"]
        tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
        ax.tricontourf(tri, sol.u, levels=np.linspace(H2, H1, 21), cmap=HEADMAP, alpha=0.75)
        ax.tricontour(tri, sol.u, levels=np.arange(H2 + 1, H1, 1.0), colors="#1e3a5f", linewidths=0.8,
                      linestyles="--")
        qK = md["q"] / K                                  # psi in meters of head
        ax.tricontour(tri, md["psi"].u / K, levels=np.arange(1.0, qK, 1.0), colors=INK, linewidths=1.0)
        ax.add_patch(Rectangle((0, 0), B, 1.6, facecolor=CONC, edgecolor=INK, lw=1.0))
        if md["cut"]:
            xc = 0.0 if md["cut"] == "up" else B
            ax.plot([xc, xc], [0, -DCUT], color=INK, lw=2.6)
        nf = qK / (H1 - H2) * 10
        ax.set_title(titles[md["cut"]] + f":  $q$ = {md['q'] * DAY:.2f} m³/day per m,  "
                     f"$N_f/N_d$ = {nf:.2f}/10", fontsize=11, loc="left")
        ax.set_xlim(-LSIDE, B + LSIDE)
        ax.set_ylim(-D, 1.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(h_pad=0.6)
    return fig


def fig_uplift(models):
    """(a) Uplift pressure on the base, (b) exit gradient downstream of the toe."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    labs = {None: "no cutoff", "up": "cutoff at the heel", "down": "cutoff at the toe"}
    cols = {None: "0.35", "up": "#2b5fa8", "down": LOADCLR}
    xb = np.linspace(0.02, B - 0.02, 300)
    for md in models:
        ax.plot(xb, GW * base_head(md, xb), color=cols[md["cut"]], lw=2.0,
                label=f"{labs[md['cut']]} ($U$ = {md['U']:.0f} kN/m)")
    ax.plot([0, B], [GW * H1, GW * H2], color="0.6", lw=1.2, ls=":",
            label="straight-line assumption")
    ax.set_xlabel("distance from the heel (m)")
    ax.set_ylabel("uplift pressure on the base (kPa)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_xlim(0, B)
    ax.set_ylim(0, 130)
    xd = np.linspace(B + 0.01, B + 10, 400)
    for md in models:
        bx.plot(xd - B, surface_gradient(md, xd), color=cols[md["cut"]], lw=2.0,
                label=f"{labs[md['cut']]}: {md['i_avg']:.2f}")
    bx.axhline(IC, color=LOADCLR, lw=1.0, ls="--")
    bx.text(5.0, IC + 0.04, "critical gradient $i_c$ = 1.0", ha="left", va="bottom", fontsize=9.5,
            color=LOADCLR)
    bx.axvspan(0, WIN, color="0.92", zorder=0)
    bx.text(WIN / 2, 2.35, "averaging\nwindow", ha="center", va="top", fontsize=9, color="0.35")
    bx.set_xlabel("distance downstream of the toe (m)")
    bx.set_ylabel("upward exit gradient  $-\\partial h/\\partial y$")
    bx.set_ylim(0, 2.5)
    bx.set_xlim(0, 10)
    bx.legend(frameon=False, fontsize=9.5, loc="upper right", title="average over the window",
              title_fontsize=9.5)
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, color="0.9", lw=0.7)
        a.set_axisbelow(True)
    ax.set_title("(a) Uplift pressure", fontsize=11.5, loc="left")
    bx.set_title("(b) Exit gradient", fontsize=11.5, loc="left")
    fig.tight_layout(w_pad=2.5)
    return fig


def fig_benchmark(deep):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xb = np.linspace(0.02, B - 0.02, 300)
    exact = H2 + (H1 - H2) / math.pi * np.arccos((2 * xb - B) / B)
    ax.plot(xb, exact, color=INK, lw=2.4, label="closed form, infinitely deep sand")
    ax.plot(xb, base_head(deep, xb), color=LOADCLR, lw=1.6, ls="--", label="finite elements, sand 200 m deep")
    ax.plot([0, B], [H1, H2], color="0.6", lw=1.0, ls=":", label="straight line")
    ax.set_xlabel("distance from the heel (m)")
    ax.set_ylabel("head under the base, $h$ (m)")
    ax.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="0.9", lw=0.7)
    ax.set_xlim(0, B)
    fig.tight_layout()
    err = np.max(np.abs(base_head(deep, xb) - exact))
    return fig, err


def fig_anisotropic(model):
    """Head contours and flow directions when k_h = 4 k_v."""
    m, sol = model["mesh"], model["sol"]
    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
    ax.tricontourf(tri, sol.u, levels=np.linspace(H2, H1, 21), cmap=HEADMAP, alpha=0.75)
    ax.tricontour(tri, sol.u, levels=np.arange(H2 + 1, H1, 1.0), colors="#1e3a5f", linewidths=0.8,
                  linestyles="--")
    kap = np.diag(model["kappa"]) if np.ndim(model["kappa"]) == 1 else np.eye(2) * model["kappa"]
    for x in np.arange(-34, B + 36, 6.0):
        for y in np.arange(-17, -1, 4.0):
            if model["cut"] == "up" and abs(x) < 1.5 and y > -DCUT:
                continue
            e, xi, eta = F.locate(m, x, y)
            g = F.gradient(sol, e, xi, eta)[:, 0]
            v = -kap @ g
            v = v / np.linalg.norm(v) * 2.4
            ax.annotate("", xy=(x + v[0], y + v[1]), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=8))
    ax.add_patch(Rectangle((0, 0), B, 1.6, facecolor=CONC, edgecolor=INK, lw=1.0))
    if model["cut"]:
        ax.plot([0, 0], [0, -DCUT], color=INK, lw=2.6)
    ax.set_xlim(-LSIDE, B + LSIDE)
    ax.set_ylim(-D, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
