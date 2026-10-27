"""
A pressurized membrane with a rigid center boss (Class 24 application page).

A square silicon-nitride membrane, side a, is held by a silicon frame and
pre-stressed so that it carries a uniform tension S (force per unit length).
A rigid square boss of side b is attached to its center.  A pressure p pushes
on it, and its deflection w satisfies -S lap(w) = p, with w = 0 on the frame.
Every point of the boss moves by the same unknown amount, so the boss nodes
share one unknown.  A quarter is modeled, with symmetry on the two axes.
Lengths are in meters.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle

import c24_fem as F
from notes_style import INK, BDRYCLR, FLUXCLR, LOADCLR

A_SIDE = 1.0e-3          # m, side of the membrane
B_BOSS = 0.4e-3          # m, side of the boss
THICK = 0.5e-6           # m, membrane thickness
SIGMA0 = 200.0e6         # Pa, residual tensile stress in the film
TENSION = SIGMA0 * THICK # N/m
E_FILM = 250.0e9         # Pa, Young's modulus of silicon nitride
PRESS = 1000.0           # Pa (1 kPa)
DEFLMAP = "BuPu"
SILICON, NITRIDE = "#b9bec6", "#7c5cb8"
TOL = 1e-12


def build(p=2, nref=1, b=B_BOSS, a=A_SIDE, S=TENSION, pr=PRESS, r0=1.15):
    r = r0 ** (1.0 / nref)
    n = lambda k: max(1, int(round(k * nref)))
    if b > 0:
        xs = F.join(F.graded(0, b / 2, n(6), r, "b"), F.graded(b / 2, a / 2, 2 * n(8), r, "both"))
    else:
        xs = F.graded(0, a / 2, n(16), 1.0, "a")
    mesh = F.Mesh(xs, xs, p=p)
    frame = mesh.sides_where(lambda x, y, nn: abs(x - a / 2) < TOL or abs(y - a / 2) < TOL)
    boss = [A for A in range(mesh.nnp) if mesh.X[A, 0] <= b / 2 + TOL and mesh.X[A, 1] <= b / 2 + TOL] if b > 0 else []
    sol = F.solve(mesh, kappa=S, f=pr, dirichlet=[(frame, 0.0)], tied=[boss] if boss else [])
    out = dict(mesh=mesh, sol=sol, a=a, b=b, S=S, p=pr, frame=frame)
    out["w0"] = sol.u[np.argmin(np.hypot(mesh.X[:, 0], mesh.X[:, 1]))]
    out["volume"] = 4 * F.integrate(sol)
    out["w_avg"] = out["volume"] / a ** 2
    out["frame_force"] = -4 * sol.reaction[mesh.nodes_of_sides(frame)].sum()
    return out


def exact_center(a=A_SIDE, S=TENSION, pr=PRESS):
    """Center deflection of a square membrane with no boss (series)."""
    al = a / 2
    s = sum((-1) ** ((n - 1) // 2) / (n ** 3 * math.cosh(n * math.pi / 2)) for n in range(1, 401, 2))
    phi0 = al ** 2 * (1 - 32 / math.pi ** 3 * s)          # the torsion function of the square, at its center
    return pr / (2 * S) * phi0


def edge_force(model, ys):
    """S dw/dn along the frame edge x = a/2 (N/m): the vertical force per unit
    length that the membrane applies to the frame."""
    a = model["a"]
    out = []
    for y in ys:
        e, _, eta = F.locate(model["mesh"], a / 2 - 1e-15, y)
        g = F.gradient(model["sol"], e, 1.0, eta)[:, 0]
        out.append(-model["S"] * g[0])
    return np.array(out)


def profile(model, xs):
    out = []
    for x in xs:
        e, xi, _ = F.locate(model["mesh"], x, 0.0)
        out.append(F.value(model["sol"], e, xi, -1.0)[0])
    return np.array(out)


def _full(model):
    m, sol = model["mesh"], model["sol"]
    T0 = m.triangles()
    Xs, Ts, Us, off = [], [], [], 0
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        Xs.append(m.X * np.array([sx, sy]))
        Ts.append(T0 + off if sx * sy > 0 else T0[:, ::-1] + off)
        Us.append(sol.u)
        off += m.nnp
    return np.vstack(Xs), np.vstack(Ts), np.concatenate(Us)


# ---------------------------------------------------------------- figures
def fig_setting():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    # (a) cross-section through the die (not to scale)
    a, b, H = 1.0, 0.4, 0.55
    ax.add_patch(Polygon([(-1.25, 0), (-a / 2 - 0.25, 0), (-a / 2, H), (-1.25, H)], closed=True,
                         facecolor=SILICON, edgecolor=INK, lw=1.0))
    ax.add_patch(Polygon([(1.25, 0), (a / 2 + 0.25, 0), (a / 2, H), (1.25, H)], closed=True,
                         facecolor=SILICON, edgecolor=INK, lw=1.0))
    ax.add_patch(Polygon([(-b / 2, H), (b / 2, H), (b / 2 - 0.08, H - 0.32), (-b / 2 + 0.08, H - 0.32)],
                         closed=True, facecolor=SILICON, edgecolor=INK, lw=1.0))
    ax.plot([-1.25, 1.25], [H + 0.012, H + 0.012], color=NITRIDE, lw=2.4)
    for x in np.linspace(-1.1, 1.1, 12):
        ax.annotate("", xy=(x, H + 0.03), xytext=(x, H + 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=LOADCLR, lw=1.2, mutation_scale=9))
    ax.text(0, H + 0.26, "pressure $p$", ha="center", va="bottom", fontsize=10.5, color=LOADCLR)
    ax.annotate("silicon nitride membrane,\n0.5 μm thick, tension $S$", xy=(-0.34, H + 0.012),
                xytext=(-0.42, -0.17), fontsize=9.5, color=NITRIDE, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=NITRIDE, lw=0.9))
    ax.annotate("rigid silicon boss", xy=(0.14, H - 0.24), xytext=(0.18, 0.08), fontsize=10, color=INK,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.9))
    ax.text(-1.05, 0.10, "silicon\nframe", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0, -0.18), xytext=(0, 0.18),
                arrowprops=dict(arrowstyle="-|>", color="#2b5fa8", lw=1.4))
    ax.text(0.06, -0.12, "the readout (optical or capacitive)\nmeasures how far the boss moves",
            ha="left", va="center", fontsize=9.5, color="#2b5fa8")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.32, H + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) A cross-section of the sensor (not to scale)", fontsize=11.5, loc="left")
    # (b) top view with dimensions
    bx.add_patch(Rectangle((-0.75, -0.75), 1.5, 1.5, facecolor=SILICON, edgecolor=INK, lw=1.0))
    bx.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="#ece6f5", edgecolor=INK, lw=1.0))
    bx.add_patch(Rectangle((-0.2, -0.2), 0.4, 0.4, facecolor="#cfd3d9", edgecolor=INK, lw=1.0, ls="--"))

    def dim(x0, y0, x1, y1, lab, off=(0, 0), ha="center", va="center"):
        bx.annotate("", xy=(x0, y0), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9, shrinkA=0, shrinkB=0))
        bx.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], lab, ha=ha, va=va, fontsize=10, color="0.25")
    dim(-0.5, 0.88, 0.5, 0.88, "$a$ = 1 mm", off=(0, 0.1))
    dim(-0.2, -0.32, 0.2, -0.32, "$b$ = 0.4 mm", off=(0, -0.1))
    bx.text(0, 0.0, "boss", ha="center", va="center", fontsize=9.5)
    bx.text(0.0, 0.36, "membrane", ha="center", va="center", fontsize=9.5, color=NITRIDE)
    bx.set_xlim(-0.85, 0.85)
    bx.set_ylim(-0.85, 1.1)
    bx.set_aspect("equal")
    bx.axis("off")
    bx.set_title("(b) Seen from above", fontsize=11.5, loc="left")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_domain(model):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    a, b = model["a"] * 1e3, model["b"] * 1e3
    ax.add_patch(Rectangle((0, 0), a / 2, a / 2, facecolor="#f4f1ea", edgecolor="none"))
    ax.add_patch(Rectangle((0, 0), b / 2, b / 2, facecolor="#cfd3d9", edgecolor="none"))
    lw = 3.0
    ax.plot([a / 2, a / 2, 0], [0, a / 2, a / 2], color=BDRYCLR, lw=lw)
    ax.plot([0, 0], [0, a / 2], color=FLUXCLR, lw=lw)
    ax.plot([0, a / 2], [0, 0], color=FLUXCLR, lw=lw)
    k = dict(fontsize=10.5)
    ax.text(a / 4, a / 2 + 0.015, "frame: $w = 0$", ha="center", va="bottom", color=BDRYCLR, **k)
    ax.text(a / 2 + 0.015, a / 4, "frame:\n$w = 0$", ha="left", va="center", color=BDRYCLR, **k)
    ax.text(-0.015, a / 3, "symmetry", ha="right", va="center", color=FLUXCLR, rotation=90, **k)
    ax.text(a / 3, -0.015, "symmetry", ha="center", va="top", color=FLUXCLR, **k)
    ax.text(b / 4, b / 4, "boss:\none unknown\nheight", ha="center", va="center", fontsize=9.5)
    ax.text(0.33, 0.30, r"$-S\,\nabla^2 w = p$", ha="center", va="center", fontsize=12)
    ax.set_title("(a) A quarter of the membrane", fontsize=11.5, loc="left")
    m = model["mesh"]
    for e in range(m.nel):
        (xc, yc), (hx, hy) = m.cent[e] * 1e3, m.size[e] * 1e3
        inb = xc < b / 2 and yc < b / 2
        bx.add_patch(Rectangle((xc - hx / 2, yc - hy / 2), hx, hy, facecolor="#cfd3d9" if inb else "none",
                               edgecolor="0.5", lw=0.35))
    bx.set_title("(b) The biquadratic mesh", fontsize=11.5, loc="left")
    for a_ in (ax, bx):
        a_.set_xlim(-0.09, a / 2 + 0.13)
        a_.set_ylim(-0.07, a / 2 + 0.07)
        a_.set_aspect("equal")
        a_.axis("off")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_shapes(plain, bossed):
    fig = plt.figure(figsize=(11.0, 4.2))
    zmax = max(plain["sol"].u.max(), bossed["sol"].u.max()) * 1e6
    for k, (md, title) in enumerate(((plain, "(a) No boss"), (bossed, "(b) With a boss"))):
        X, T, U = _full(md)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_trisurf(X[:, 0] * 1e3, X[:, 1] * 1e3, U * 1e6, triangles=T, cmap=DEFLMAP, vmin=0, vmax=zmax,
                        linewidth=0, antialiased=True)
        ax.set_zlim(0, zmax * 1.05)
        ax.set_box_aspect((1, 1, 0.45))
        ax.view_init(elev=24, azim=-55)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
        if k == 0:
            ax.set_zlabel("w (μm)", labelpad=2)
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title(title, fontsize=11, loc="left")
    cx = fig.add_subplot(1, 3, 3)
    xs = np.linspace(0, plain["a"] / 2, 300)
    for md, lab, col in ((plain, "no boss", "0.35"), (bossed, "with a boss", LOADCLR)):
        w = profile(md, xs) * 1e6
        cx.plot(np.concatenate([-xs[::-1], xs]) * 1e3, np.concatenate([w[::-1], w]), color=col, lw=2, label=lab)
    cx.axvspan(-bossed["b"] / 2 * 1e3, bossed["b"] / 2 * 1e3, color="0.92", zorder=0)
    cx.set_xlabel("x (mm), along the center line")
    cx.set_ylabel("deflection w (μm)")
    cx.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        cx.spines[s].set_visible(False)
    cx.grid(True, color="0.9", lw=0.7)
    cx.set_title("(c) Through the center", fontsize=11, loc="left")
    fig.tight_layout(w_pad=2.5)
    return fig


def fig_edge(plain, bossed):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ys = np.linspace(0, plain["a"] / 2, 300)
    for md, lab, col in ((plain, "no boss", "0.35"), (bossed, "with a boss", LOADCLR)):
        q = edge_force(md, ys) * 1e3
        ax.plot(np.concatenate([-ys[::-1], ys]) * 1e3, np.concatenate([q[::-1], q]), color=col, lw=2, label=lab)
    ax.set_xlabel("position along one side of the frame (mm)")
    ax.set_ylabel("force on the frame (mN/m)")
    ax.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="0.9", lw=0.7)
    fig.tight_layout()
    return fig


def fig_sweep(fracs, wb):
    """Deflection of the boss against its size, each point labeled with the
    share of the membrane that stays flat."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    f = np.array(fracs)
    w = np.array(wb) * 1e6
    ax.plot(f, w, "o-", color=LOADCLR, lw=1.8, ms=5)
    for fi, wi in zip(f, w):
        if fi > 0:
            ax.annotate(f"{100 * fi ** 2:.0f}% flat", xy=(fi, wi), xytext=(4, 6), textcoords="offset points",
                        fontsize=8.5, color="0.35")
    ax.set_xlabel("width of the boss / width of the membrane, $b/a$")
    ax.set_ylabel("deflection of the boss for 1 kPa (μm)")
    ax.set_ylim(0, 0.85)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.grid(True, color="0.9", lw=0.7)
    fig.tight_layout()
    return fig
