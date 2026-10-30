"""
Figures for the two elasticity appendix pages (Elasticity2D.qmd and
Elasticity2D_FEM.qmd).  The finite element results come from elast2d.py.
Units: mm, N, MPa.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch, Arc
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import elast2d as EL
from notes_style import INK, LOADCLR, REACTCLR, BDRYCLR, FLUXCLR, MESHCLR, DOMCLR

E_STEEL, NU_STEEL = 200000.0, 0.3
S11C, S22C, S33C, S12C, S13C, S23C = "#c0392b", "#1f6feb", "#7c3aed", "#0d9488", "#e67e22", "#8a5a20"
DEFCLR = "#e67e22"
BODY = "#e3eef8"
STRESSMAP = "RdBu_r"


def arrow(ax, p, q, color=INK, lw=1.8, ms=13, z=6, style="-|>"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=ms,
                                                    shrinkA=0, shrinkB=0), zorder=z)


def clean(ax, equal=True):
    if equal:
        ax.set_aspect("equal")
    ax.axis("off")


def axes_glyph(ax, o, s, labels=("$x_1$", "$x_2$"), fs=12):
    arrow(ax, o, (o[0] + s, o[1]), INK, 1.1, 10)
    arrow(ax, o, (o[0], o[1] + s), INK, 1.1, 10)
    ax.text(o[0] + 1.18 * s, o[1], labels[0], fontsize=fs, va="center")
    ax.text(o[0], o[1] + 1.2 * s, labels[1], fontsize=fs, ha="center")


# ------------------------------------------------------------ meshes on a plot
def triangles(mesh, k=None):
    """Points and triangles that split every element (each Q9 sub-square too)."""
    p = mesh.p
    idx = [[0, 4, 1], [7, 8, 5], [3, 6, 2]] if p == 2 else [[0, 1], [3, 2]]
    tri = []
    for row in mesh.IEN:
        g = [[row[idx[j][i]] for j in range(p + 1)] for i in range(p + 1)]
        for i in range(p):
            for j in range(p):
                a, b, c, d = g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]
                tri += [(a, b, c), (a, c, d)]
    return np.array(tri)


def element_outline(mesh, e, m=8, U=None, scale=0.0):
    pts = []
    sides = [lambda t: (t, -1.0), lambda t: (1.0, t), lambda t: (-t, 1.0), lambda t: (-1.0, -t)]
    xe = mesh.X[mesh.IEN[e]]
    ue = U[mesh.IEN[e]] if U is not None else np.zeros_like(xe)
    for s in sides:
        for i in range(m):
            N, _ = EL.shape(mesh.p, *s(-1 + 2 * i / m))
            pts.append(N @ (xe + scale * ue))
    return np.array(pts)


def draw_mesh(ax, mesh, U=None, scale=0.0, color=MESHCLR, lw=0.5, z=3):
    m = 1 if mesh.p == 1 else 6
    for e in range(mesh.nel):
        P = element_outline(mesh, e, m, U, scale)
        ax.add_patch(Polygon(P, closed=True, fill=False, edgecolor=color, lw=lw, zorder=z))


def fine_field(ax, mesh, fn, k=12, cmap="Blues", levels=None, norm=None, z=1):
    """Contour fn(e, xi, eta) by sampling each element on a k-by-k grid."""
    pts, vals, tri = [], [], []
    for e in range(mesh.nel):
        xe = mesh.X[mesh.IEN[e]]
        base = len(pts)
        for i in range(k + 1):
            for j in range(k + 1):
                xi, eta = -1 + 2 * i / k, -1 + 2 * j / k
                N, _ = EL.shape(mesh.p, xi, eta)
                pts.append(N @ xe)
                vals.append(fn(e, xi, eta))
        for i in range(k):
            for j in range(k):
                a, b = base + i * (k + 1) + j, base + (i + 1) * (k + 1) + j
                tri += [(a, b, b + 1), (a, b + 1, a + 1)]
    pts = np.array(pts)
    T = mtri.Triangulation(pts[:, 0], pts[:, 1], np.array(tri))
    return ax.tricontourf(T, np.array(vals), levels=levels, cmap=cmap, norm=norm, zorder=z)


def field_plot(ax, mesh, vals, U=None, scale=0.0, cmap=STRESSMAP, norm=None, levels=40, z=1):
    X = mesh.X + (scale * U if U is not None else 0)
    tri = mtri.Triangulation(X[:, 0], X[:, 1], triangles(mesh))
    return ax.tricontourf(tri, vals, levels=levels, cmap=cmap, norm=norm, zorder=z)


# =============================================================================
#  Page 1: stress, strain, and the strong form
# =============================================================================
def plate_solution(p=2, level=3):
    m = EL.plate_mesh(p=p, level=level)
    P = EL.PLATE
    zero = lambda x, y: 0.0
    s = EL.solve(m, E_STEEL, NU_STEEL, "stress", t=P["t"], dirichlet={"left": {0: zero, 1: zero}},
                 traction={"right": lambda x, y: (P["s"], 0.0)})
    return m, s


def fig_deformation():
    """(a) the displacement field of the plate with a hole; (b) a small square deforming."""
    m, s = plate_solution(2, 2)
    P = EL.PLATE
    scale = 400.0
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 3.9), gridspec_kw=dict(width_ratios=[1.75, 1]))
    Lh, Wh, a = P["L"] / 2, P["W"] / 2, P["a"]
    ax.add_patch(Rectangle((-Lh, -Wh), P["L"], P["W"], fill=False, ec="0.45", lw=1.0, ls="--", zorder=2))
    ax.add_patch(Circle((0, 0), a, fill=False, ec="0.45", lw=1.0, ls="--", zorder=2))
    umag = np.hypot(s.U[:, 0], s.U[:, 1])
    cf = field_plot(ax, m, umag, s.U, scale, cmap="YlOrBr", levels=30)
    for e in range(m.nel):
        ax.add_patch(Polygon(element_outline(m, e, 6, s.U, scale), closed=True, fill=False, ec="none", zorder=1))
    # displacement arrows at a few points, from the original to the moved position (exaggerated)
    for x in np.linspace(-80, 80, 5):
        for y in (-35, 0.0, 35):
            if math.hypot(x, y) < a + 8:
                continue
            e, xi, eta = EL.locate(m, x, y)
            u = EL.at(s, e, xi, eta)["u"]
            arrow(ax, (x, y), (x + scale * u[0], y + scale * u[1]), INK, 1.2, 9, z=7)
            ax.plot(x, y, "o", ms=2.5, color=INK, zorder=7)
    ax.plot([-Lh, -Lh], [-Wh, Wh], color=BDRYCLR, lw=4, solid_capstyle="butt", zorder=5)
    for y in np.linspace(-Wh + 8, Wh - 8, 5):
        arrow(ax, (Lh + scale * 0.054 + 2, y), (Lh + scale * 0.054 + 16, y), LOADCLR, 1.6, 11)
    ax.text(Lh + scale * 0.054 + 22, 0, "pulled", color=LOADCLR, fontsize=10.5, rotation=90, va="center")
    ax.text(-Lh - 6, 0, "held", color=BDRYCLR, fontsize=10.5, rotation=90, va="center", ha="right")
    # the small square, near the top of the hole
    sq0 = np.array([[-4.0, 13.5], [4.0, 13.5], [4.0, 21.5], [-4.0, 21.5]])
    ax.add_patch(Polygon(sq0, closed=True, fill=False, ec=INK, lw=1.6, zorder=8))
    ax.set_xlim(-Lh - 14, Lh + scale * 0.054 + 34)
    ax.set_ylim(-Wh - 8, Wh + 8)
    clean(ax)
    cb = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("$|\\underline{u}|$ (mm)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_title("(a) Every point moves: the displacement $\\underline{u}(\\underline{x})$, magnified 400 times",
                 fontsize=11, loc="left")
    # (b) the square, enlarged: original and deformed, magnified more
    mag = 400.0
    corners = []
    for (x, y) in sq0:
        e, xi, eta = EL.locate(m, x, y)
        corners.append(EL.at(s, e, xi, eta)["u"])
    corners = np.array(corners)
    c0 = sq0.mean(axis=0)
    rel = sq0 - c0
    urel = corners - corners.mean(axis=0)
    bx.add_patch(Polygon(rel + mag * urel, closed=True, fc="#fde9d2", ec=DEFCLR, lw=1.8, zorder=2))
    bx.add_patch(Polygon(rel, closed=True, fill=False, ec=INK, lw=1.4, ls="--", zorder=3))
    for k in range(4):
        arrow(bx, rel[k], rel[k] + mag * urel[k], INK, 1.0, 8)
    bx.text(0, -6.0, "original square (dashed)", fontsize=10, ha="center", va="top")
    bx.text(0, 6.6, "deformed square, with the motion\nof the square as a whole removed", fontsize=9.5, color=DEFCLR,
            ha="center", va="bottom")
    bx.set_xlim(-9, 9)
    bx.set_ylim(-8, 9.5)
    clean(bx)
    bx.set_title("(b) The small square at the top of\nthe hole, enlarged; motion magnified 400 times", fontsize=11, loc="left")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_strain():
    """(a) the derivation: a small rectangle O, A, B before and after; (b) the three strain modes and a rotation."""
    fig = plt.figure(figsize=(12.4, 4.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[3.1, 1, 1, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    dx1, dx2 = 3.0, 2.0
    O = np.array([0.0, 0.0])
    uO = np.array([1.1, 0.7])
    H = np.array([[0.22, 0.25], [0.2, 0.10]])                  # u_i,j, exaggerated
    A0, B0, C0 = np.array([dx1, 0.0]), np.array([0.0, dx2]), np.array([dx1, dx2])
    mv = lambda X: X + uO + H @ X
    ax.add_patch(Polygon([O, A0, C0, B0], closed=True, fc="#eef2f6", ec=INK, lw=1.2, ls="--"))
    ax.add_patch(Polygon([mv(O), mv(A0), mv(C0), mv(B0)], closed=True, fc="#fde9d2", ec=DEFCLR, lw=1.8))
    for P0, nm in ((O, "O"), (A0, "A"), (B0, "B")):
        ax.plot(*P0, "o", color=INK, ms=4)
        ax.text(P0[0] - 0.12, P0[1] - 0.12, nm, fontsize=12, ha="right", va="top")
        ax.plot(*mv(P0), "o", color=DEFCLR, ms=4)
        ax.text(mv(P0)[0] + 0.08, mv(P0)[1] + 0.1, nm + "$'$", fontsize=12, color=DEFCLR)
    arrow(ax, O, mv(O), INK, 1.4, 11)
    ax.text(*(0.5 * (O + mv(O)) + np.array([0.12, -0.3])), "$\\underline{u}(O)$", fontsize=12)
    ax.annotate("", xy=(dx1, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(dx1 / 2, -0.62, "$dx_1$", fontsize=11, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(-0.45, dx2), xytext=(-0.45, 0), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(-0.62, dx2 / 2, "$dx_2$", fontsize=11, ha="right", va="center", color="0.3")
    # the angles: OA turns by u2,1, OB turns by u1,2
    Op, Ap, Bp = mv(O), mv(A0), mv(B0)
    a1 = math.degrees(math.atan2(*(Ap - Op)[::-1]))
    a2 = math.degrees(math.atan2(*(Bp - Op)[::-1]))
    ax.plot([Op[0], Op[0] + 1.6 * dx1], [Op[1], Op[1]], color="0.55", lw=0.8, ls=":")
    ax.plot([Op[0], Op[0]], [Op[1], Op[1] + 1.35 * dx2], color="0.55", lw=0.8, ls=":")
    r1, r2 = 2.2, 1.7
    ax.add_patch(Arc(Op, 2 * r1, 2 * r1, theta1=0, theta2=a1, color=S12C, lw=2.0, zorder=7))
    ax.add_patch(Arc(Op, 2 * r2, 2 * r2, theta1=a2, theta2=90, color=S12C, lw=2.0, zorder=7))
    m1, m2 = math.radians(a1 / 2), math.radians((a2 + 90) / 2)
    ax.text(Op[0] + (r1 + 0.12) * math.cos(m1), Op[1] + (r1 + 0.12) * math.sin(m1) - 0.04, "$u_{2,1}$",
            fontsize=12, color=S12C, va="center", ha="left",
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85), zorder=8)
    ax.text(Op[0] + (r2 + 0.12) * math.cos(m2) + 0.02, Op[1] + (r2 + 0.12) * math.sin(m2) + 0.08, "$u_{1,2}$",
            fontsize=12, color=S12C, va="bottom", ha="center",
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85), zorder=8)
    ax.text(5.65, 0.75, "O$'$A$'$: length $(1 + u_{1,1})\\,dx_1$,\nturned up by $u_{2,1}$\n\n"
            "O$'$B$'$: length $(1 + u_{2,2})\\,dx_2$,\nturned right by $u_{1,2}$",
            fontsize=10.2, va="bottom", ha="left")
    ax.set_xlim(-1.2, 8.4)
    ax.set_ylim(-1.1, 4.4)
    clean(ax)
    ax.set_title("(a) A small rectangle, before and after", fontsize=11, loc="left")
    # (b) the modes
    modes = [("$\\varepsilon_{11}$: stretch along $x_1$", np.array([[0.3, 0], [0, 0]]), S11C),
             ("$\\varepsilon_{22}$: stretch along $x_2$", np.array([[0, 0], [0, 0.3]]), S22C),
             ("$\\varepsilon_{12}$: shear, a change\nof angle $\\gamma_{12} = 2\\varepsilon_{12}$", np.array([[0, 0.17], [0.17, 0]]), S12C),
             ("$\\varepsilon_{11} = \\varepsilon_{22} = 0$:\ncompressed along one diagonal", np.array([[0, 0.17], [0.17, 0]]), S12C),
             ("rotation $\\omega_{12}$:\nno strain at all", np.array([[0, -0.25], [0.25, 0]]), S22C),
             ("$\\varepsilon_{11} + \\varepsilon_{22}$:\na change of area", np.array([[0.18, 0], [0, 0.18]]), S11C)]
    keep = [0, 1, 2, 4, 5]
    pos = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]
    for (r, c), k in zip(pos, keep):
        title, Hm, col = modes[k]
        bxx = fig.add_subplot(gs[r, c])
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) - 0.5
        d = sq + sq @ Hm.T
        bxx.add_patch(Polygon(sq, closed=True, fill=False, ec=INK, lw=1.0, ls="--"))
        bxx.add_patch(Polygon(d, closed=True, fc=col, alpha=0.18, ec=col, lw=1.6))
        bxx.set_xlim(-0.95, 0.95)
        bxx.set_ylim(-0.95, 0.95)
        clean(bxx)
        bxx.set_title(title, fontsize=9.6)
    cx = fig.add_subplot(gs[1, 3])
    cx.axis("off")
    cx.text(0.0, 0.55, "Exaggerated.  In linear\nelasticity every $u_{i,j}$\nis much smaller than 1,\nso the strain is linear\nin $\\underline{u}$.",
            fontsize=9.8, va="center", transform=cx.transAxes)
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    return fig


def fig_traction():
    """(a) a body cut by a plane, with the traction on the cut; (b) the stress components on a small square."""
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.0, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1.0]))
    # (a) a body under loads, cut by a plane
    th = np.linspace(0, 2 * np.pi, 200)
    rr = 1 + 0.18 * np.cos(3 * th) + 0.08 * np.sin(2 * th)
    bodyx, bodyy = 1.6 * rr * np.cos(th), 1.1 * rr * np.sin(th)
    keep = bodyx <= 0.15
    ax.fill(np.r_[bodyx[keep], 0.15], np.r_[bodyy[keep], bodyy[keep][0]], color=BODY, zorder=1)
    ax.plot(bodyx, bodyy, color="0.6", lw=1, ls=":")
    ax.plot(bodyx[keep], bodyy[keep], color=INK, lw=1.4)
    ys = np.linspace(-0.95, 0.95, 9)
    ax.plot([0.15, 0.15], [-1.0, 1.0], color=INK, lw=1.8)
    for y in ys:
        t = np.array([0.55 - 0.25 * y * y, 0.18 + 0.12 * y])
        arrow(ax, (0.15, y), (0.15 + t[0], y + t[1]), LOADCLR, 1.2, 8)
    for ang in (2.2, 3.3, 4.25):
        r = 1 + 0.18 * math.cos(3 * ang) + 0.08 * math.sin(2 * ang)
        x0, y0 = 1.6 * r * math.cos(ang), 1.1 * r * math.sin(ang)
        d = np.array([x0 / 1.6, y0 / 1.1])
        d /= np.linalg.norm(d)
        arrow(ax, (x0 + 0.55 * d[0], y0 + 0.55 * d[1]), (x0, y0), INK, 1.5, 11)
    ax.text(-2.35, 0.12, "loads on\nthe body", fontsize=10, ha="left", va="bottom")
    ax.text(-0.75, 0.0, "$\\Omega$", fontsize=15)
    arrow(ax, (0.15, -1.25), (0.75, -1.25), INK, 1.4, 11)
    ax.text(0.8, -1.25, "$\\underline{n}$", fontsize=13, va="center")
    ax.text(0.7, 0.9, "traction $\\underline{t}$\n(force per area)", color=LOADCLR, fontsize=10.5)
    ax.set_xlim(-2.4, 1.9)
    ax.set_ylim(-1.6, 1.6)
    clean(ax)
    ax.set_title("(a) Cut the body: the part on the right\npulls on the part on the left", fontsize=10.5, loc="left")
    # (b) the 2D square with sigma_ij on each face
    h = 1.0
    bx.add_patch(Rectangle((-h, -h), 2 * h, 2 * h, fc=BODY, ec=INK, lw=1.4))
    g = 0.12
    arrow(bx, (h + g, 0), (h + g + 0.8, 0), S11C, 2.2)
    arrow(bx, (-h - g, 0), (-h - g - 0.8, 0), S11C, 2.2)
    arrow(bx, (0, h + g), (0, h + g + 0.8), S22C, 2.2)
    arrow(bx, (0, -h - g), (0, -h - g - 0.8), S22C, 2.2)
    o = h + 0.12
    arrow(bx, (o, -0.55), (o, 0.55), S12C, 2.2)
    arrow(bx, (-o, 0.55), (-o, -0.55), S12C, 2.2)
    arrow(bx, (-0.55, o), (0.55, o), S12C, 2.2)
    arrow(bx, (0.55, -o), (-0.55, -o), S12C, 2.2)
    bx.text(h + 0.98, 0.0, "$\\sigma_{11}$", color=S11C, fontsize=13, va="center")
    bx.text(0.0, h + 1.05, "$\\sigma_{22}$", color=S22C, fontsize=13, ha="center")
    bx.text(o + 0.1, 0.62, "$\\sigma_{21}$", color=S12C, fontsize=13)
    bx.text(0.62, o + 0.1, "$\\sigma_{12}$", color=S12C, fontsize=13)
    bx.text(0, 0, "$dx_1 \\times dx_2$", fontsize=10.5, ha="center", va="center", color="0.3")
    axes_glyph(bx, (-2.1, -2.1), 0.55)
    bx.set_xlim(-2.3, 2.6)
    bx.set_ylim(-2.3, 2.5)
    clean(bx)
    bx.set_title("(b) A small square: $\\sigma_{ij}$ is the $i$-component\nof the traction on the face whose\noutward normal is $\\underline{e}_j$",
                 fontsize=10.5, loc="left")
    # (c) the triangle that gives t = sigma n
    P0, P1, P2 = np.array([0.0, 0.0]), np.array([0.0, 2.0]), np.array([1.4, 0.0])
    cx.add_patch(Polygon([P0, P1, P2], closed=True, fc=BODY, ec=INK, lw=1.4))
    n = np.array([2.0, 1.4])
    n /= np.linalg.norm(n)
    mid = (P1 + P2) / 2
    arrow(cx, mid, mid + 0.75 * n, INK, 1.4, 11)
    cx.text(*(mid + 0.85 * n), "$\\underline{n}$", fontsize=13)
    t = np.array([0.9, 0.35])
    arrow(cx, mid, mid + t, LOADCLR, 2.0, 12)
    cx.text(*(mid + t + np.array([0.05, -0.18])), "$\\underline{t}$", fontsize=13, color=LOADCLR)
    arrow(cx, (-0.08, 1.0), (-0.75, 1.0), S11C, 1.8)
    cx.text(-0.8, 1.15, "$-\\sigma_{11}$", color=S11C, fontsize=11, ha="right")
    arrow(cx, (0.7, -0.08), (0.7, -0.75), S22C, 1.8)
    cx.text(0.85, -0.7, "$-\\sigma_{22}$", color=S22C, fontsize=11)
    cx.text(-0.3, -1.25, "$t_i = \\sigma_{ij}\\,n_j$", fontsize=14)
    cx.set_xlim(-1.5, 2.4)
    cx.set_ylim(-1.5, 2.6)
    clean(cx)
    cx.set_title("(c) Balance the forces on a small\ntriangle and let it shrink", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.2)
    return fig


def fig_equilibrium():
    """The forces in the x1 direction on a small rectangle."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    w, h = 3.2, 2.2
    ax.add_patch(Rectangle((0, 0), w, h, fc=BODY, ec=INK, lw=1.5))
    g = 0.12
    arrow(ax, (-g, h / 2), (-g - 1.2, h / 2), S11C, 2.2)
    ax.text(-g - 1.25, h / 2 + 0.22, "$\\sigma_{11}(x_1, x_2)\\,dx_2$", color=S11C, fontsize=12, ha="right")
    arrow(ax, (w + g, h / 2), (w + g + 1.45, h / 2), S11C, 2.2)
    ax.text(w + g + 0.1, h / 2 + 0.22, "$\\sigma_{11}(x_1 + dx_1, x_2)\\,dx_2$", color=S11C, fontsize=12)
    arrow(ax, (w / 2 + 0.7, -g - 0.04), (w / 2 - 0.7, -g - 0.04), S12C, 2.2)
    ax.text(w / 2, -g - 0.3, "$\\sigma_{12}(x_1, x_2)\\,dx_1$", color=S12C, fontsize=12, ha="center", va="top")
    arrow(ax, (w / 2 - 0.8, h + g + 0.04), (w / 2 + 0.8, h + g + 0.04), S12C, 2.2)
    ax.text(w / 2, h + g + 0.28, "$\\sigma_{12}(x_1, x_2 + dx_2)\\,dx_1$", color=S12C, fontsize=12, ha="center")
    arrow(ax, (w / 2 - 0.45, h / 2), (w / 2 + 0.45, h / 2), LOADCLR, 2.0)
    ax.text(w / 2, h / 2 - 0.22, "$f_1\\,dx_1\\,dx_2$", color=LOADCLR, fontsize=12, ha="center", va="top")
    ax.plot(0, 0, "o", color=INK, ms=4)
    ax.text(-0.08, -0.08, "$(x_1, x_2)$", fontsize=11, ha="right", va="top")
    ax.annotate("", xy=(w, -0.95), xytext=(0, -0.95), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(w / 2, -1.08, "$dx_1$", fontsize=11, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(w + 2.7, h), xytext=(w + 2.7, 0), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(w + 2.82, h / 2 - 0.5, "$dx_2$", fontsize=11, va="center", color="0.3")
    axes_glyph(ax, (-2.8, -1.1), 0.5)
    ax.set_xlim(-3.4, w + 3.3)
    ax.set_ylim(-1.45, h + 0.75)
    clean(ax)
    fig.tight_layout()
    return fig


def fig_hooke():
    """(a) a tension test: E and nu; (b) a shear test: G."""
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.0, 3.5), gridspec_kw=dict(width_ratios=[1.25, 1.0, 1.0]))
    L, H = 3.0, 1.0
    ex, ey = 0.22, -0.08
    ax.add_patch(Rectangle((0, -ey * H / 2), L * (1 + ex), H * (1 + ey), fc="#fde9d2", ec=DEFCLR, lw=1.6, zorder=2))
    ax.add_patch(Rectangle((0, 0), L, H, fill=False, ec=INK, lw=1.2, ls="--", zorder=3))
    for y in np.linspace(0.12, 0.84, 4):
        arrow(ax, (L * (1 + ex) + 0.05, y), (L * (1 + ex) + 0.5, y), S11C, 1.6, 10)
        arrow(ax, (-0.05, y), (-0.5, y), S11C, 1.6, 10)
    ax.text(L * (1 + ex) + 0.62, 0.5, "$\\sigma$", color=S11C, fontsize=13, va="center")
    ax.text(L * 0.45, 1.25, "lengthens by $\\varepsilon_{11} = \\sigma/E$", fontsize=10.5, ha="center")
    ax.text(L * 0.45, -0.45, "narrows by $\\varepsilon_{22} = -\\nu\\,\\varepsilon_{11}$", fontsize=10.5, ha="center")
    ax.set_xlim(-0.9, L * 1.25 + 0.9)
    ax.set_ylim(-0.8, 1.6)
    clean(ax)
    ax.set_title("(a) Pull a bar: Young's modulus $E$ and\nPoisson's ratio $\\nu$ (exaggerated)", fontsize=10.5, loc="left")
    # stress-strain line
    epsv = np.linspace(0, 1.5e-3, 10)
    bx.plot(epsv * 1e3, E_STEEL * epsv, color=INK, lw=2)
    bx.plot([0.5, 1.0, 1.0], [100, 100, 200], color=S11C, lw=1.2)
    bx.text(1.05, 150, "$E$ = 200 GPa", color=S11C, fontsize=10.5, va="center")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        bx.spines[s].set_linewidth(0.9)
    bx.set_xlabel("strain $\\varepsilon_{11}$ ($\\times 10^{-3}$)", fontsize=10.5)
    bx.set_ylabel("stress $\\sigma_{11}$ (MPa)", fontsize=10.5)
    bx.tick_params(labelsize=9)
    bx.set_title("(b) Steel, before it yields:\na straight line", fontsize=10.5, loc="left")
    # shear
    s = 1.2
    gam = 0.28
    cx.add_patch(Polygon([[0, 0], [s, 0.5 * gam * s], [s + 0.5 * gam * s, s + 0.5 * gam * s], [0.5 * gam * s, s]], closed=True,
                         fc="#fde9d2", ec=DEFCLR, lw=1.6, zorder=2))
    cx.add_patch(Polygon([[0, 0], [s, 0], [s, s], [0, s]], closed=True, fill=False, ec=INK, lw=1.2, ls="--", zorder=3))
    o = 0.12
    arrow(cx, (s + o + 0.18, 0.2), (s + o + 0.18, s - 0.1), S12C, 2.0)
    arrow(cx, (-o, s - 0.2), (-o, 0.2), S12C, 2.0)
    arrow(cx, (0.2, s + o + 0.18), (s - 0.1, s + o + 0.18), S12C, 2.0)
    arrow(cx, (s - 0.2, -o), (0.2, -o), S12C, 2.0)
    cx.text(s + 0.45, 0.6, "$\\tau$", color=S12C, fontsize=13)
    cx.text(0.6 * s, -0.55, "angle change $\\gamma = \\tau / G$,\n$G = E / (2(1 + \\nu))$", fontsize=10.2, ha="center", va="top")
    cx.set_xlim(-0.5, s + 0.9)
    cx.set_ylim(-1.15, s + 0.55)
    clean(cx)
    cx.set_title("(c) Shear a square:\nthe shear modulus $G$", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.6)
    return fig


def fig_c_patterns():
    """The 6 x 6 matrix D in Voigt form for isotropic, orthotropic, and anisotropic materials."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    labels = ["11", "22", "33", "23", "13", "12"]
    pal = ["#c0392b", "#1f6feb", "#0d9488", "#e67e22", "#7c3aed", "#8a5a20", "#d97706", "#2e7d32", "#ad1457",
           "#455a64", "#6d4c41", "#00838f", "#5e35b1", "#827717", "#b71c1c", "#1565c0", "#4e342e", "#283593",
           "#00695c", "#9e9d24", "#bf360c"]
    iso = [[1, 2, 2, 0, 0, 0], [2, 1, 2, 0, 0, 0], [2, 2, 1, 0, 0, 0], [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3]]
    ort = [[1, 4, 5, 0, 0, 0], [4, 2, 6, 0, 0, 0], [5, 6, 3, 0, 0, 0], [0, 0, 0, 7, 0, 0], [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 9]]
    ani = [[0] * 6 for _ in range(6)]
    k = 0
    for i in range(6):
        for j in range(i, 6):
            k += 1
            ani[i][j] = ani[j][i] = k
    titles = ["(a) Isotropic: 2 constants\n($\\lambda$ and $\\mu$, or $E$ and $\\nu$)",
              "(b) Orthotropic (wood, a rolled sheet,\na fiber composite): 9 constants",
              "(c) Fully anisotropic: 21 constants"]
    for ax, M, title in zip(axes, (iso, ort, ani), titles):
        for i in range(6):
            for j in range(6):
                v = M[i][j]
                fc = "white" if v == 0 else pal[(v - 1) % len(pal)]
                ax.add_patch(Rectangle((j, 5 - i), 1, 1, fc=fc, ec="0.75", lw=0.6, alpha=0.75 if v else 1))
                if v and M is not iso:
                    ax.text(j + 0.5, 5.5 - i, str(v), fontsize=7.5, ha="center", va="center", color="white")
        if M is iso:
            for i in range(6):
                for j in range(6):
                    t = {1: "$\\lambda{+}2\\mu$", 2: "$\\lambda$", 3: "$\\mu$"}.get(M[i][j])
                    if t:
                        ax.text(j + 0.5, 5.5 - i, t, fontsize=8.5, ha="center", va="center", color="white")
        for i, lb in enumerate(labels):
            ax.text(-0.15, 5.5 - i, lb, fontsize=8.5, ha="right", va="center", color="0.35")
            ax.text(i + 0.5, 6.15, lb, fontsize=8.5, ha="center", va="bottom", color="0.35")
        ax.set_xlim(-0.8, 6.2)
        ax.set_ylim(-0.2, 6.8)
        clean(ax)
        ax.set_title(title, fontsize=10.2, loc="left")
    fig.tight_layout(w_pad=1.5)
    return fig


def _extrude(ax, outline, depth, ang=30.0, k=0.55, front="#e3eef8", wall="#b9c6d4", back="#dfe6ee", hatch=False, z=0):
    """A 2D outline (outer, holes) extruded along x3 in an oblique projection."""
    a = math.radians(ang)
    P = lambda x, y, zz: (x + k * zz * math.cos(a), y + k * zz * math.sin(a))
    d = np.array([math.cos(a), math.sin(a)])
    rings = [outline["outer"]] + outline["holes"]
    for ring in rings:
        ax.add_patch(Polygon([P(x, y, depth) for x, y in ring], closed=True, fc=back, ec=INK, lw=0.7, zorder=z))
    walls = []
    for ring in rings:
        n = len(ring)
        for i in range(n):
            A, B = np.array(ring[i]), np.array(ring[(i + 1) % n])
            nrm = np.array([B[1] - A[1], -(B[0] - A[0])])
            if nrm @ d <= 0:
                continue
            walls.append(((A + B) @ d, A, B))
    walls.sort(key=lambda w: -w[0])
    for _, A, B in walls:
        ax.add_patch(Polygon([P(*A, 0), P(*B, 0), P(*B, depth), P(*A, depth)], closed=True, fc=wall, ec="#6b7b8c",
                             lw=0.5, zorder=z + 1))
    verts, codes = [], []
    for ring in rings:
        pts = [P(x, y, 0) for x, y in ring]
        verts += pts + [pts[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=front, ec=INK, lw=1.1, zorder=z + 2, hatch="////" if hatch else None))
    return P


def fig_planes():
    """Plane stress (a thin plate) and plane strain (a long body)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.6))
    P = EL.PLATE
    Lh, Wh, a = P["L"] / 2, P["W"] / 2, P["a"]
    hole = [(a * math.cos(-2 * math.pi * k / 48), a * math.sin(-2 * math.pi * k / 48)) for k in range(48)]
    plate = dict(outer=[(-Lh, -Wh), (Lh, -Wh), (Lh, Wh), (-Lh, Wh)], holes=[hole])
    Pp = _extrude(ax, plate, 14.0)
    for y in np.linspace(-Wh + 10, Wh - 10, 4):
        arrow(ax, Pp(Lh + 4, y, 7), Pp(Lh + 34, y, 7), LOADCLR, 1.6, 11, z=10)
        arrow(ax, Pp(-Lh - 4, y, 7), Pp(-Lh - 34, y, 7), LOADCLR, 1.6, 11, z=10)
    ax.text(-Lh, -Wh - 14, "thickness $t$ much smaller than the plate;\nloads in the plane, the faces $x_3 = \\pm t/2$ free",
            fontsize=10.2, ha="left", va="top")
    ax.text(-Lh, Wh + 30, "$\\sigma_{33} = \\sigma_{13} = \\sigma_{23} = 0$\n$\\varepsilon_{33} = -\\dfrac{\\nu}{E}(\\sigma_{11} + \\sigma_{22}) \\neq 0$",
            fontsize=11, va="bottom")
    gx, gy = Lh - 6, -Wh - 40
    o = Pp(gx, gy, 0)
    for v, lab, off in (((22, 0, 0), "$x_1$", (3, -2)), ((0, 22, 0), "$x_2$", (-3, 3)), ((0, 0, 40), "$x_3$", (2, 1))):
        q = Pp(gx + v[0], gy + v[1], v[2])
        arrow(ax, o, q, INK, 1.1, 9)
        ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=11, ha="left" if off[0] >= 0 else "right")
    ax.set_xlim(-Lh - 45, Lh + 55)
    ax.set_ylim(-Wh - 50, Wh + 75)
    clean(ax)
    ax.set_title("(a) Plane stress: a thin plate", fontsize=11, loc="left")
    W = EL.WSHAPE
    d, bf, tf, tw = W["d"], W["bf"], W["tf"], W["tw"]
    b, w = bf / 2, tw / 2
    ws = dict(outer=[(-b, 0), (b, 0), (b, tf), (w, tf), (w, d - tf), (b, d - tf), (b, d), (-b, d), (-b, d - tf), (-w, d - tf), (-w, tf), (-b, tf)], holes=[])
    Pw = _extrude(bx, ws, 520.0, hatch=True)
    for x in np.linspace(-b + 15, b - 15, 5):
        for zz in (60, 220, 380):
            arrow(bx, Pw(x, d + 45, zz), Pw(x, d + 4, zz), LOADCLR, 1.2, 9, z=10)
    bx.text(-b, -40, "long, with the same cross-section and loads\nall along $x_3$; the ends held (hatched)",
            fontsize=10.2, va="top")
    bx.text(-b - 20, d + 170, "$\\varepsilon_{33} = \\varepsilon_{13} = \\varepsilon_{23} = 0$\n$\\sigma_{33} = \\nu(\\sigma_{11} + \\sigma_{22}) \\neq 0$",
            fontsize=11, va="bottom")
    bx.set_xlim(-b - 30, b + 420)
    bx.set_ylim(-110, d + 245)
    clean(bx)
    bx.set_title("(b) Plane strain: a long body, here a W-shape\nloaded evenly along its length", fontsize=11, loc="left")
    fig.tight_layout(w_pad=1.2)
    return fig


def _support(ax, x, y, direction, size, color=BDRYCLR, filled=True, outward=(0, -1)):
    """A small triangle at (x, y) whose height lies along `direction` (0 for x1, 1 for x2)."""
    e = np.array([1.0, 0.0]) if direction == 0 else np.array([0.0, 1.0])
    o = np.array(outward, float)
    sg = 1 if o @ e >= 0 else -1
    off = 0.55 * size * o if abs(o @ e) < 0.5 else 0 * o
    apex = np.array([x, y]) + off
    base = apex + sg * e * size
    perp = np.array([-e[1], e[0]])
    ax.add_patch(Polygon([apex, base + 0.55 * size * perp, base - 0.55 * size * perp], closed=True,
                         fc=color if filled else "white", ec=color, lw=1.1, zorder=8))


def fig_bcs():
    """The boundary of the quarter plate, split once for direction 1 and once for direction 2."""
    P = EL.PLATE
    Lh, Wh, a = P["L"] / 2, P["W"] / 2, P["a"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    th = np.linspace(0, np.pi / 2, 40)
    edges = {"left": ([0, 0], [a, Wh]), "bottom": ([a, Lh], [0, 0]), "right": ([Lh, Lh], [0, Wh]),
             "top": ([0, Lh], [Wh, Wh]), "hole": (a * np.cos(th), a * np.sin(th))}
    for ax, i in zip(axes, (1, 2)):
        ax.add_patch(Polygon(np.c_[np.r_[a, Lh, Lh, 0, 0, a * np.cos(th[::-1])], np.r_[0, 0, Wh, Wh, a, a * np.sin(th[::-1])]],
                             closed=True, fc=BODY, ec="none"))
        gname = "left" if i == 1 else "bottom"
        for name, (xs, ys) in edges.items():
            col = BDRYCLR if name == gname else FLUXCLR
            ax.plot(xs, ys, color=col, lw=4.2 if name == gname else 3.0, solid_capstyle="butt", zorder=5)
        if i == 1:
            for y in np.linspace(a + 5, Wh - 4, 4):
                _support(ax, 0, y, 0, 5.5, outward=(-1, 0))
            ax.text(-9, (a + Wh) / 2, "$\\Gamma_{g_1}$: $u_1 = 0$\n(symmetry)", color=BDRYCLR, fontsize=11, ha="right", va="center")
            ax.text(Lh + 15, Wh / 2, "$h_1 = \\sigma$", color=FLUXCLR, fontsize=11, va="center")
            ax.text((a + Lh) / 2, -5, "$h_1 = 0$: no shear on a\nline of symmetry", color=FLUXCLR, fontsize=10.5, ha="center", va="top")
            ax.text(Lh / 2, Wh + 3, "$h_1 = 0$ (free)", color=FLUXCLR, fontsize=10.5, ha="center", va="bottom")
            for y in np.linspace(5, Wh - 5, 4):
                arrow(ax, (Lh + 1.5, y), (Lh + 13, y), LOADCLR, 1.5, 10)
        else:
            for x in np.linspace(a + 6, Lh - 6, 6):
                _support(ax, x, 0, 1, 5.5, outward=(0, -1))
            ax.text((a + Lh) / 2, -9, "$\\Gamma_{g_2}$: $u_2 = 0$ (symmetry)", color=BDRYCLR, fontsize=11, ha="center", va="top")
            ax.text(-3, (a + Wh) / 2, "$h_2 = 0$: no shear on a\nline of symmetry", color=FLUXCLR, fontsize=10.5, ha="right", va="center")
            ax.text(Lh + 4, Wh / 2, "$h_2 = 0$", color=FLUXCLR, fontsize=11, va="center")
            ax.text(Lh / 2, Wh + 3, "$h_2 = 0$ (free)", color=FLUXCLR, fontsize=10.5, ha="center", va="bottom")
        ax.text(a * 0.75, a * 0.75, "hole:\n$h_{%d} = 0$" % i, color=FLUXCLR, fontsize=10, ha="left", va="bottom")
        ax.set_xlim(-48, Lh + 34)
        ax.set_ylim(-26, Wh + 14)
        clean(ax)
        ax.set_title("(%s) Direction %d: $\\Gamma = \\overline{\\Gamma_{g_%d} \\cup \\Gamma_{h_%d}}$, with no overlap"
                     % ("a" if i == 1 else "b", i, i, i), fontsize=11, loc="left")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_supports():
    """Common supports, each a pair of conditions, one for each direction."""
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 2.7))
    items = [("held (clamped)", [("g", "g")], "$u_1 = g_1$, $u_2 = g_2$"),
             ("roller on a horizontal surface", [("h", "g")], "$h_1 = 0$, $u_2 = g_2$"),
             ("a line of symmetry\n(normal $x_1$)", [("g", "h")], "$u_1 = 0$, $h_2 = 0$"),
             ("loaded surface", [("h", "h")], "$h_1$ and $h_2$ given"),
             ("free surface", [("h0", "h0")], "$h_1 = h_2 = 0$")]
    for ax, (title, _, rule) in zip(axes, items):
        ax.add_patch(Rectangle((0, 0), 2, 1.2, fc=BODY, ec="none"))
        clean(ax)
        ax.set_xlim(-0.8, 2.5)
        ax.set_ylim(-0.9, 1.7)
        ax.text(0.85, -0.55, rule, fontsize=10.2, ha="center", va="top")
        ax.set_title(title, fontsize=10)
    a0, a1, a2, a3, a4 = axes
    a0.plot([0, 2], [0, 0], color=BDRYCLR, lw=4)
    for x in np.linspace(0.2, 1.8, 5):
        a0.plot([x, x - 0.18], [0, -0.22], color=BDRYCLR, lw=1.1)
    a1.plot([0, 2], [0, 0], color=BDRYCLR, lw=3)
    for x in np.linspace(0.3, 1.7, 4):
        a1.add_patch(Circle((x, -0.1), 0.09, fc="white", ec=BDRYCLR, lw=1.2))
    a1.plot([0, 2], [-0.21, -0.21], color=BDRYCLR, lw=1.0)
    a2.plot([0, 0], [0, 1.2], color=BDRYCLR, lw=3.5)
    a2.plot([0, 0], [-0.3, 1.5], color="0.4", lw=1.0, ls="-.")
    for y in np.linspace(0.2, 1.0, 3):
        a2.add_patch(Circle((-0.12, y), 0.09, fc="white", ec=BDRYCLR, lw=1.2))
    a3.plot([0, 2], [1.2, 1.2], color=FLUXCLR, lw=3)
    for x in np.linspace(0.2, 1.8, 6):
        arrow(a3, (x - 0.15, 1.62), (x, 1.24), LOADCLR, 1.4, 9)
    a4.plot([0, 2], [1.2, 1.2], color=FLUXCLR, lw=3)
    fig.tight_layout(w_pad=0.4)
    return fig


def fig_cantilever_setup():
    C = EL.CANTILEVER
    L, D = C["L"], C["D"]
    fig, ax = plt.subplots(figsize=(10.0, 3.3))
    ax.add_patch(Rectangle((0, -D / 2), L, D, fc=BODY, ec=INK, lw=1.3))
    ax.plot([0, 0], [-D / 2 - 12, D / 2 + 12], color=BDRYCLR, lw=4)
    for y in np.linspace(-D / 2 - 6, D / 2 + 6, 8):
        ax.plot([0, -11], [y, y - 9], color=BDRYCLR, lw=1.0)
    yy = np.linspace(-D / 2, D / 2, 81)
    env = L + 6 + 34 * (1 - 4 * yy * yy / (D * D))
    ax.fill_betweenx(yy, L + 6, env, color=LOADCLR, alpha=0.12, lw=0)
    ax.plot(env, yy, color=LOADCLR, lw=1.3)
    ax.plot([L + 6, L + 6], [-D / 2, D / 2], color=LOADCLR, lw=1.0)
    for y in np.linspace(-D / 2 + 12, D / 2 - 12, 6):
        x = L + 6 + 17 * (1 - 4 * y * y / (D * D))
        arrow(ax, (x, y + 5), (x, y - 5), LOADCLR, 1.3, 8)
    arrow(ax, (L + 72, 28), (L + 72, -28), LOADCLR, 2.4, 14)
    ax.text(L + 78, 0, "$P$ = 8 kN, the\nresultant of a\nparabolic shear\ntraction", color=LOADCLR, fontsize=10.2, va="center")
    ax.plot([0, L], [0, 0], color="0.55", lw=0.8, ls="-.")
    arrow(ax, (0, 0), (30, 0), INK, 1.1, 10)
    arrow(ax, (0, 0), (0, 30), INK, 1.1, 10)
    ax.text(34, 0, "$x_1$", fontsize=12, va="center")
    ax.text(4, 30, "$x_2$", fontsize=12, va="center")
    ax.annotate("", xy=(L, -D / 2 - 22), xytext=(0, -D / 2 - 22), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(L / 2, -D / 2 - 27, "$L$ = 400 mm", fontsize=10.5, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(L * 0.72, D / 2), xytext=(L * 0.72, -D / 2), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(L * 0.72 + 5, 18, "$D$ = 100 mm", fontsize=10.5, color="0.3")
    ax.text(-14, D / 2 + 18, "exact displacements\nprescribed", fontsize=10, color=BDRYCLR, ha="center")
    ax.text(L * 0.3, D / 2 + 8, "free:  $h_1 = h_2 = 0$", fontsize=10, color=FLUXCLR)
    ax.text(L * 0.3, -D / 2 - 8, "free", fontsize=10, color=FLUXCLR, va="top")
    ax.text(L * 0.06, -D / 2 + 10, "steel plate, $t$ = 10 mm thick (plane stress)", fontsize=10, color="0.3")
    ax.set_xlim(-40, L + 150)
    ax.set_ylim(-D / 2 - 45, D / 2 + 45)
    clean(ax)
    fig.tight_layout()
    return fig


def fig_cantilever_exact():
    C = EL.CANTILEVER
    L, D, t, P = C["L"], C["D"], C["t"], C["P"]
    I = t * D ** 3 / 12
    fig = plt.figure(figsize=(10.8, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, :])
    xs, ys = np.linspace(0, L, 161), np.linspace(-D / 2, D / 2, 41)
    XX, YY = np.meshgrid(xs, ys)
    ex = EL.cantilever_exact(XX, YY, E_STEEL, NU_STEEL)
    mag = 25.0
    norm = TwoSlopeNorm(0, -200, 200)
    cf = ax.contourf(XX + mag * ex["u1"], YY + mag * ex["u2"], ex["s11"], levels=np.linspace(-200, 200, 41), cmap=STRESSMAP, norm=norm)
    ax.add_patch(Rectangle((0, -D / 2), L, D, fill=False, ec="0.45", ls="--", lw=1))
    for x in np.linspace(0, L, 9):
        ax.plot(x + mag * ex["u1"][:, np.searchsorted(xs, x) - (1 if x == L else 0)], ys + mag * ex["u2"][:, np.searchsorted(xs, x) - (1 if x == L else 0)],
                color=INK, lw=0.6, alpha=0.6)
    cb = fig.colorbar(cf, ax=ax, fraction=0.025, pad=0.01, ticks=[-200, -100, 0, 100, 200])
    cb.set_label("$\\sigma_{11}$ (MPa)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    tip = EL.cantilever_exact(L, 0, E_STEEL, NU_STEEL)["u2"]
    ax.text(L + 6, mag * ex["u2"][20, -1], "tip: %s mm" % ("%.3f" % tip).replace("-", "\u2212"),
            fontsize=10, va="center")
    ax.set_xlim(-10, L + 75)
    ax.set_ylim(-D / 2 - 40, D / 2 + 8)
    clean(ax)
    ax.set_title("(a) The exact solution, displacements magnified 25 times: lines that were vertical stay nearly straight, but tilt and bend slightly",
                 fontsize=10.2, loc="left")
    # (b) sigma11 through the depth at x = L/2, (c) sigma12, (d) deflection
    bx, cx, dx = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    y = np.linspace(-D / 2, D / 2, 101)
    e2 = EL.cantilever_exact(L / 2 + 0 * y, y, E_STEEL, NU_STEEL)
    bx.plot(e2["s11"], y, color=S11C, lw=2.2, label="exact")
    bx.plot(P * (L / 2) * y / I, y, ls=(0, (4, 3)), color=INK, lw=1.2, label="beam theory, $My/I$")
    cx.plot(e2["s12"], y, color=S12C, lw=2.2, label="exact")
    cx.plot(-P / (2 * I) * (D * D / 4 - y * y), y, ls=(0, (4, 3)), color=INK, lw=1.2, label="beam theory, $VQ/(It)$")
    x = np.linspace(0, L, 101)
    w = EL.cantilever_exact(x, 0 * x, E_STEEL, NU_STEEL)["u2"]
    eb = -P * x * x * (3 * L - x) / (6 * E_STEEL * I)
    dx.plot(x, w, color=INK, lw=2.2, label="exact, at $x_2 = 0$")
    dx.plot(x, eb, ls=(0, (4, 3)), color=REACTCLR, lw=1.4, label="Euler–Bernoulli")
    for a_, xl, yl, title in ((bx, "$\\sigma_{11}$ at $x_1 = L/2$ (MPa)", "$x_2$ (mm)", "(b) Bending stress: linear"),
                              (cx, "$\\sigma_{12}$ (MPa), any $x_1$", "$x_2$ (mm)", "(c) Shear stress: parabolic"),
                              (dx, "$x_1$ (mm)", "$u_2$ (mm)", "(d) Deflection: bending + shear")):
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a_.spines[s].set_linewidth(0.8)
        a_.set_xlabel(xl, fontsize=10)
        a_.set_ylabel(yl, fontsize=10)
        a_.tick_params(labelsize=8.5)
        a_.legend(frameon=False, fontsize=8.5)
        a_.set_title(title, fontsize=10.5, loc="left")
    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    return fig


def fig_examples():
    """The two bodies without closed-form solutions: a plate with a hole and a W-shape."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw=dict(width_ratios=[1.45, 1]))
    P = EL.PLATE
    Lh, Wh, a = P["L"] / 2, P["W"] / 2, P["a"]
    ax.add_patch(Rectangle((-Lh, -Wh), P["L"], P["W"], fc=BODY, ec=INK, lw=1.3))
    ax.add_patch(Circle((0, 0), a, fc="white", ec=INK, lw=1.3))
    ax.plot([-Lh, -Lh], [-Wh, Wh], color=BDRYCLR, lw=4)
    for y in np.linspace(-Wh + 4, Wh - 4, 7):
        ax.plot([-Lh, -Lh - 7], [y, y - 6], color=BDRYCLR, lw=1)
    for y in np.linspace(-Wh + 6, Wh - 6, 6):
        arrow(ax, (Lh + 2, y), (Lh + 18, y), LOADCLR, 1.5, 10)
    ax.text(Lh + 22, 0, "$\\sigma$ = 50 MPa", color=LOADCLR, fontsize=10.5, va="center")
    ax.annotate("", xy=(Lh, -Wh - 10), xytext=(-Lh, -Wh - 10), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(0, -Wh - 13, "200 mm", fontsize=10, ha="center", va="top", color="0.3")
    ax.annotate("", xy=(Lh - 10, Wh), xytext=(Lh - 10, -Wh), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    ax.text(Lh - 13, 18, "100 mm", fontsize=10, ha="right", color="0.3")
    ax.text(0, a + 4, "$d$ = 25 mm", fontsize=10, ha="center", va="bottom")
    ax.text(-Lh + 4, Wh - 4, "steel, 10 mm thick:\nplane stress", fontsize=9.8, va="top", color="0.3")
    ax.text(-Lh - 10, -Wh - 4, "held", fontsize=10, color=BDRYCLR, ha="right", va="top")
    ax.set_xlim(-Lh - 25, Lh + 60)
    ax.set_ylim(-Wh - 28, Wh + 10)
    clean(ax)
    ax.set_title("(a) A plate with a hole, pulled at one end", fontsize=11, loc="left")
    W = EL.WSHAPE
    d, bf, tf, tw = W["d"], W["bf"], W["tf"], W["tw"]
    b, w = bf / 2, tw / 2
    shp = [(-b, 0), (b, 0), (b, tf), (w, tf), (w, d - tf), (b, d - tf), (b, d), (-b, d), (-b, d - tf), (-w, d - tf), (-w, tf), (-b, tf)]
    bx.add_patch(Polygon(shp, closed=True, fc=BODY, ec=INK, lw=1.3))
    bx.plot([-b, b], [0, 0], color=BDRYCLR, lw=4)
    for x in np.linspace(-b + 8, b - 8, 9):
        bx.plot([x, x - 7], [0, -9], color=BDRYCLR, lw=1)
    for x in np.linspace(-b + 10, b - 10, 7):
        arrow(bx, (x, d + 24), (x, d + 2), LOADCLR, 1.4, 9)
    bx.text(0, d + 28, "$q$ = 1 MPa over the top flange", color=LOADCLR, fontsize=10.2, ha="center", va="bottom")
    bx.annotate("", xy=(b + 16, d), xytext=(b + 16, 0), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    bx.text(b + 20, d / 2, "$d$ = 253", fontsize=10, va="center", color="0.3")
    bx.annotate("", xy=(b, -20), xytext=(-b, -20), arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.9))
    bx.text(0, -24, "$b_f$ = 254", fontsize=10, ha="center", va="top", color="0.3")
    bx.text(w + 6, d * 0.55, "$t_w$ = 8.6", fontsize=9.8)
    bx.text(-b, d - tf - 6, "$t_f$ = 14.2", fontsize=9.8, va="top")
    bx.text(-b - 6, 4, "held", fontsize=10, color=BDRYCLR, ha="right")
    bx.text(-b - 8, d * 0.5, "steel;\nplane\nstrain;\nmm", fontsize=9.8, ha="right", va="center", color="0.3")
    bx.set_xlim(-b - 50, b + 60)
    bx.set_ylim(-45, d + 50)
    clean(bx)
    bx.set_title("(b) The cross-section of a %s,\nloaded across its flanges" % W["name"], fontsize=11, loc="left")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_kirsch():
    """Kirsch's stress along the line through the hole, perpendicular to the load."""
    P = EL.PLATE
    a, s = P["a"], P["s"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.linspace(a, 6 * a, 200)
    k = np.array([EL.kirsch(0.0, yy, s, a)[0] for yy in y]) / s
    ax.plot(y / a, k, color=S11C, lw=2.2, label="Kirsch, infinite plate")
    ax.axhline(1, color="0.5", lw=0.9, ls=":")
    kt, peak = EL.heywood(2 * a, P["W"], s)
    ax.plot([1], [peak / s], "o", color=INK, ms=6, label="finite plate (Heywood), $W$ = 4$d$")
    ax.text(1.1, 2.93, "$3\\sigma$ at the edge of the hole", fontsize=10, va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel("distance from the center of the hole along $x_1 = 0$, $x_2/a$", fontsize=10.5)
    ax.set_ylabel("$\\sigma_{11}/\\sigma$", fontsize=10.5)
    ax.set_ylim(0.8, 3.4)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
#  Page 2: plane elasticity by Galerkin's method
# =============================================================================
ZERO = lambda x, y: 0.0


def _graph(ax, xl, yl):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel(yl, fontsize=10)
    ax.tick_params(labelsize=8.5)


def fig_vector_basis():
    """The two vector basis functions of one node: N_A e_1 and N_A e_2."""
    m = EL.cantilever_mesh(p=1, ny=2, nx=4, L=200.0, D=100.0)
    A = int(np.argmin(np.hypot(m.X[:, 0] - 100.0, m.X[:, 1])))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    # (a) N_A as a field
    ax = axes[0]
    def NA(e, xi, eta):
        row = list(m.IEN[e])
        return EL.shape(m.p, xi, eta)[0][row.index(A)] if A in row else 0.0
    fine_field(ax, m, NA, k=16, cmap="Blues", levels=np.linspace(-0.001, 1.0, 11))
    draw_mesh(ax, m, color="0.35", lw=0.8)
    ax.plot(*m.X[A], "o", color=INK, ms=6, zorder=6)
    ax.text(m.X[A][0] + 5, m.X[A][1] + 6, "$A$", fontsize=13, zorder=7)
    ax.set_title("(a) The scalar function $N_A$:\n1 at node $A$, 0 at every other node", fontsize=10.5, loc="left")
    for k, (ax, i, col) in enumerate(((axes[1], 0, S11C), (axes[2], 1, S22C))):
        U = np.zeros((m.nnp, 2))
        U[A, i] = 1.0
        sc = 28.0
        draw_mesh(ax, m, color="0.75", lw=0.7)
        for e in range(m.nel):
            if A in m.IEN[e]:
                P = element_outline(m, e, 1, U, sc)
                ax.add_patch(Polygon(P, closed=True, fc=col, alpha=0.15, ec="none", zorder=2))
        draw_mesh(ax, m, U=U, scale=sc, color=col, lw=1.3, z=4)
        arrow(ax, m.X[A], m.X[A] + sc * U[A], INK, 1.6, 11, z=7)
        ax.plot(*m.X[A], "o", color=INK, ms=5, zorder=7)
        ax.set_title("(%s) $N_A\\,\\underline{e}_%d$: node $A$ moves along $x_%d$,\nevery other node stays put"
                     % ("b" if i == 0 else "c", i + 1, i + 1), fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_xlim(-8, 208)
        ax.set_ylim(-58, 58)
        clean(ax)
    fig.tight_layout(w_pad=1.2)
    return fig


def fig_id_lm():
    """ID(i, A) on a small mesh with rollers on two sides, and LM for one element."""
    m = EL.cantilever_mesh(p=1, ny=2, nx=3, L=300.0, D=200.0)
    s = EL.solve(m, E_STEEL, NU_STEEL, "stress", t=10.0, dirichlet={"left": {0: ZERO}, "bottom": {1: ZERO}},
                 traction={"right": lambda x, y: (50.0, 0.0)})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax.add_patch(Rectangle((0, -100), 300, 200, fc=BODY, ec="none"))
    draw_mesh(ax, m, color="0.4", lw=1.0)
    eh = int(np.argmin([np.hypot(*(m.X[m.IEN[e]].mean(axis=0) - (50.0, -50.0))) for e in range(m.nel)]))
    ax.add_patch(Polygon(element_outline(m, eh, 1), closed=True, fill=False, ec=DEFCLR, lw=2.6, zorder=5))
    c = m.X[m.IEN[eh]].mean(axis=0)
    ax.text(c[0], c[1], "$e = %d$" % (eh + 1), fontsize=11.5, ha="center", va="center", color=DEFCLR, zorder=6)
    for y in np.linspace(-80, 80, 5):
        _support(ax, 0, y, 0, 13, outward=(-1, 0))
    for x in np.linspace(20, 280, 7):
        _support(ax, x, -100, 1, 13, outward=(0, -1))
    for y in np.linspace(-70, 70, 5):
        arrow(ax, (302, y), (335, y), LOADCLR, 1.4, 10)
    for A in range(m.nnp):
        x, y = m.X[A]
        ax.plot(x, y, "o", ms=6, color=INK, zorder=7)
        ids = s.ID[:, A]
        txt = "(%s, %s)" % tuple("$\\mathbf{0}$" if v == 0 else str(v) for v in ids)
        ax.text(x + 6, y + 7, txt, fontsize=10.5, zorder=8, color=BDRYCLR,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ax.text(x - 5, y + 7, "%d" % (A + 1), fontsize=9.5, ha="right", va="bottom", color="0.35", zorder=8)
    ax.set_xlim(-45, 370)
    ax.set_ylim(-135, 125)
    clean(ax)
    ax.set_title("(a) Node numbers $A$ (gray) and $(\\mathrm{ID}(1, A),\\ \\mathrm{ID}(2, A))$ (navy):\n"
                 "the left side fixes $u_1$ only, the bottom fixes $u_2$ only", fontsize=10.5, loc="left")
    # (b) the arrays for the highlighted element
    bx.axis("off")
    ien = m.IEN[eh] + 1
    lm = s.LM[eh]
    rows = [("$a$", ["1", "2", "3", "4"]),
            ("$\\mathrm{IEN}(a, e)$", [str(v) for v in ien])]
    y0 = 0.92
    bx.text(0.0, y0 + 0.06, "(b) The arrays of element $e = %d$" % (eh + 1), fontsize=10.5, transform=bx.transAxes)
    for k, (name, vals) in enumerate(rows):
        bx.text(0.0, y0 - 0.1 * (k + 1), name, fontsize=11, transform=bx.transAxes, va="center")
        for j, v in enumerate(vals):
            bx.text(0.36 + 0.16 * j, y0 - 0.1 * (k + 1), v, fontsize=11, transform=bx.transAxes, ha="center", va="center")
    bx.text(0.0, y0 - 0.36, "$p = 2(a - 1) + i$", fontsize=11, transform=bx.transAxes, va="center")
    for j in range(8):
        a, i = j // 2 + 1, j % 2 + 1
        bx.text(0.36 + 0.08 * j, y0 - 0.36, "%d" % (j + 1), fontsize=10.5, transform=bx.transAxes, ha="center", va="center")
        bx.text(0.36 + 0.08 * j, y0 - 0.46, "%d,%d" % (i, a), fontsize=9.5, transform=bx.transAxes, ha="center", va="center", color="0.4")
        v = lm[j]
        bx.text(0.36 + 0.08 * j, y0 - 0.56, "$\\mathbf{0}$" if v == 0 else "%d" % v, fontsize=11, transform=bx.transAxes,
                ha="center", va="center", color=LOADCLR if v == 0 else INK)
    bx.text(0.0, y0 - 0.46, "$(i, a)$", fontsize=11, transform=bx.transAxes, va="center", color="0.4")
    bx.text(0.0, y0 - 0.56, "$\\mathrm{LM}(p, e)$", fontsize=11, transform=bx.transAxes, va="center")
    bx.text(0.0, y0 - 0.72, "$\\mathrm{LM}(p, e) = \\mathrm{ID}(i, \\mathrm{IEN}(a, e))$.  A 0 marks a prescribed\n"
            "component: its row and column of $\\mathbf{k}^e$ are not assembled,\n"
            "and its known value moves to the right-hand side.", fontsize=10.2, transform=bx.transAxes, va="top")
    fig.tight_layout(w_pad=1.0)
    return fig, s


def ke_modes(E=E_STEEL, nu=NU_STEEL):
    """k^e of a square Q4 (side 1 mm, t = 1 mm, plane stress) and its eight
    modes, as an eigenbasis that separates the rigid and the bending modes."""
    m = EL.cantilever_mesh(p=1, ny=1, nx=1, L=1.0, D=1.0)
    ke, _ = EL.element_arrays(m, 0, EL.dmatrix(E, nu, "stress"), 1.0)
    X = m.X[m.IEN[0]] - m.X[m.IEN[0]].mean(axis=0)
    x, y = X[:, 0], X[:, 1]
    z = np.zeros(4)
    modes = [("translation along $x_1$", np.c_[1 + z, z]), ("translation along $x_2$", np.c_[z, 1 + z]),
             ("rotation", np.c_[-y, x]), ("uniform expansion", np.c_[x, y]),
             ("stretch $x_1$, shorten $x_2$", np.c_[x, -y]), ("shear", np.c_[y, x]),
             ("bending along $x_1$", np.c_[4 * x * y, z]), ("bending along $x_2$", np.c_[z, 4 * x * y])]
    out = []
    for name, U in modes:
        d = U.reshape(-1)
        d = d / np.linalg.norm(d)
        lam = d @ ke @ d
        res = np.linalg.norm(ke @ d - lam * d) / max(np.linalg.norm(ke), 1)
        out.append((name, d.reshape(-1, 2), lam, res))
    return ke, X, out


def fig_ke_modes():
    ke, X, modes = ke_modes()
    fig = plt.figure(figsize=(11.6, 5.0))
    gs = fig.add_gridspec(2, 5)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)] + [fig.add_subplot(gs[1, k]) for k in range(5)]
    tx = fig.add_subplot(gs[0, 3:])
    tx.axis("off")
    tx.text(0.02, 0.5, "Top row: the three rigid-body modes.\nThey store no energy, so $\\mathbf{k}^e\\mathbf{d} = \\mathbf{0}$.\n\n"
            "Bottom row: the five modes that strain the\nelement. The bending modes (red) are the\n"
            "softest, but a straight-sided element bends\nonly by shearing as well.",
            fontsize=10.2, va="center", transform=tx.transAxes)
    sq = X[[0, 1, 2, 3]]
    for ax, (name, U, lam, res) in zip(axes, modes):
        ax.add_patch(Polygon(sq, closed=True, fill=False, ec=INK, lw=1.0, ls="--"))
        rigid = lam < 1e-6 * E_STEEL
        col = REACTCLR if rigid else (LOADCLR if "bending" in name else DEFCLR)
        D = sq + 0.32 * U
        ax.add_patch(Polygon(D, closed=True, fc=col, alpha=0.16, ec=col, lw=1.8))
        for k in range(4):
            arrow(ax, sq[k], D[k], col, 1.1, 8)
        ax.set_xlim(-0.95, 0.95)
        ax.set_ylim(-0.95, 0.95)
        clean(ax)
        lab = "$\\lambda = 0$" if rigid else "$\\lambda = %.3f\\,Et$" % (lam / E_STEEL)
        ax.set_title("%s\n%s" % (name, lab), fontsize=10, color=col if rigid else INK)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    return fig


def cantilever_study(ps=(1, 2), nys=(1, 2, 4, 8), E=E_STEEL, nu=NU_STEEL):
    C = EL.CANTILEVER
    ex = lambda x, y: EL.cantilever_exact(x, y, E, nu)
    tip = EL.cantilever_exact(C["L"], 0, E, nu)["u2"]
    out = {}
    for p in ps:
        for ny in nys:
            m = EL.cantilever_mesh(p, ny)
            dr, tr = EL.cantilever_bc(E, nu)
            s = EL.solve(m, E, nu, "stress", t=C["t"], dirichlet=dr, traction=tr)
            e, xi, eta = EL.locate(m, C["L"], 0.0)
            err, tot = EL.energy_error(s, ex)
            out[(p, ny)] = dict(neq=s.neq, tip=EL.at(s, e, xi, eta)["u"][1], ratio=EL.at(s, e, xi, eta)["u"][1] / tip,
                                rel=err / tot, h=C["D"] / ny, sol=s)
    return out


def fig_cantilever_conv(study):
    C = EL.CANTILEVER
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.2, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a) the coarsest Q4 mesh against the exact deflection
    s = study[(1, 1)]["sol"]
    m = s.mesh
    mag = 60.0
    xs = np.linspace(0, C["L"], 81)
    for yv in (-C["D"] / 2, C["D"] / 2):
        exl = EL.cantilever_exact(xs, yv + 0 * xs, E_STEEL, NU_STEEL)
        ax.plot(xs + mag * exl["u1"], yv + mag * exl["u2"], color=INK, lw=1.4, ls=(0, (4, 3)))
    draw_mesh(ax, m, U=s.U, scale=mag, color=LOADCLR, lw=1.6)
    s4 = study[(2, 1)]["sol"]
    draw_mesh(ax, s4.mesh, U=s4.U, scale=mag, color=REACTCLR, lw=1.2)
    ax.add_patch(Rectangle((0, -C["D"] / 2), C["L"], C["D"], fill=False, ec="0.6", lw=0.8))
    ax.text(C["L"] + 8, mag * study[(1, 1)]["tip"], "Q4, 4 elements", color=LOADCLR, fontsize=9.5, va="center")
    ax.text(C["L"] + 8, mag * study[(2, 1)]["tip"] - 4, "Q9, 4 elements,\nand exact (dashed)", color=REACTCLR, fontsize=9.5, va="top")
    ax.set_xlim(-10, C["L"] + 120)
    ax.set_ylim(-125, 60)
    clean(ax)
    ax.set_title("(a) One element through the depth,\ndisplacements magnified %d times" % mag, fontsize=10.5, loc="left")
    nys = sorted({k[1] for k in study})
    for p, col, mk, lab in ((1, LOADCLR, "o", "Q4"), (2, REACTCLR, "s", "Q9")):
        bx.plot(nys, [study[(p, n)]["ratio"] for n in nys], "-" + mk, color=col, lw=1.8, ms=6, label=lab)
        hs = np.array([study[(p, n)]["h"] for n in nys])
        es = np.array([study[(p, n)]["rel"] for n in nys])
        cx.loglog(hs, es, "-" + mk, color=col, lw=1.8, ms=6, label=lab)
        rate = math.log(es[-2] / es[-1]) / math.log(hs[-2] / hs[-1])
        cx.text(math.sqrt(hs[-1] * hs[-2]) * 1.08, math.sqrt(es[-1] * es[-2]), "slope %.2f" % rate, color=col,
                fontsize=9.5, ha="left", va="top")
    bx.axhline(1.0, color="0.5", lw=0.9, ls=":")
    bx.set_xscale("log", base=2)
    bx.set_xticks(nys)
    bx.set_xticklabels([str(n) for n in nys])
    bx.set_ylim(0.6, 1.05)
    _graph(bx, "elements through the depth", "tip deflection / exact")
    bx.legend(frameon=False, fontsize=9.5, loc="lower right")
    bx.set_title("(b) Tip deflection", fontsize=10.5, loc="left")
    _graph(cx, "element size $h$ (mm)", "relative error in energy")
    hs = sorted({v["h"] for v in study.values()})
    cx.xaxis.set_major_locator(FixedLocator(hs))
    cx.xaxis.set_major_formatter(FixedFormatter(["%g" % h for h in hs]))
    cx.xaxis.set_minor_locator(NullLocator())
    cx.legend(frameon=False, fontsize=9.5, loc="upper left")
    cx.set_title("(c) Error in the energy norm", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.4)
    return fig


def plate_study(levels=(1, 2, 3, 4)):
    P = EL.PLATE
    out = {}
    for quarter in (False, True):
        for p in (1, 2):
            for lev in levels:
                m = EL.plate_mesh(p, lev, quarter)
                dr = {"left": {0: ZERO}, "bottom": {1: ZERO}} if quarter else {"left": {0: ZERO, 1: ZERO}}
                s = EL.solve(m, E_STEEL, NU_STEEL, "stress", t=P["t"], dirichlet=dr,
                             traction={"right": lambda x, y: (P["s"], 0.0)})
                e, xi, eta = EL.locate(m, 0.0, P["a"])
                out[(quarter, p, lev)] = dict(neq=s.neq, peak=EL.at(s, e, xi, eta)["sig"][0], sol=s,
                                              reac=EL.reaction_sum(s, "left"))
    return out


def fig_plate_results(study):
    P = EL.PLATE
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.0), gridspec_kw=dict(width_ratios=[1.45, 1]))
    s = study[(False, 2, 3)]["sol"]
    m = s.mesh
    S = EL.nodal_stress(s)
    mag = 150.0
    norm = Normalize(-20, 170)
    cf = field_plot(ax, m, S[:, 0], s.U, mag, cmap="RdBu_r", norm=TwoSlopeNorm(0, -60, 170), levels=np.linspace(-60, 170, 47))
    draw_mesh(ax, m, U=s.U, scale=mag, color="0.45", lw=0.2)
    ax.add_patch(Rectangle((-P["L"] / 2, -P["W"] / 2), P["L"], P["W"], fill=False, ec="0.5", lw=0.8, ls="--"))
    cb = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.01, ticks=[-50, 0, 50, 100, 150])
    cb.set_label("$\\sigma_{11}$ (MPa)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_xlim(-P["L"] / 2 - 6, P["L"] / 2 + 14)
    ax.set_ylim(-P["W"] / 2 - 6, P["W"] / 2 + 6)
    clean(ax)
    ax.set_title("(a) $\\sigma_{11}$ on the deformed plate (Q9, %d elements;\ndisplacements magnified %d times)" % (m.nel, mag),
                 fontsize=10.5, loc="left")
    kt, peak = EL.heywood(2 * P["a"], P["W"], P["s"])
    for quarter, ls in ((False, "-"), (True, "--")):
        for p, col, mk in ((1, LOADCLR, "o"), (2, REACTCLR, "s")):
            levs = sorted(k[2] for k in study if k[0] == quarter and k[1] == p)
            bx.semilogx([study[(quarter, p, l)]["neq"] for l in levs], [study[(quarter, p, l)]["peak"] for l in levs],
                        ls=ls, marker=mk, color=col, lw=1.6, ms=5,
                        label="%s, %s" % ("Q4" if p == 1 else "Q9", "quarter (symmetry)" if quarter else "whole plate"))
    bx.axhline(peak, color=INK, lw=1.0, ls=":")
    bx.text(8000, peak - 0.15, "Heywood's estimate, %.1f MPa" % peak, fontsize=9.5, va="top", ha="right")
    _graph(bx, "number of equations", "$\\sigma_{11}$ at the top of the hole (MPa)")
    bx.legend(frameon=False, fontsize=8.8, loc="lower right")
    bx.set_title("(b) The peak stress as the mesh is refined", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.2)
    return fig


def wshape_study(levels=(1, 2, 3, 4)):
    W = EL.WSHAPE
    out = {}
    for p in (1, 2):
        for lev in levels:
            m = EL.wshape_mesh(p, lev)
            s = EL.solve(m, E_STEEL, NU_STEEL, "strain", dirichlet={"bottom face": {0: ZERO, 1: ZERO}},
                         traction={"top face": lambda x, y: (0.0, -W["q"])})
            A = int(np.argmin(np.hypot(m.X[:, 0] + W["bf"] / 2, m.X[:, 1] - W["d"])))
            S = EL.nodal_stress(s)
            vm = EL.von_mises(S[:, 0], S[:, 1], S[:, 2], NU_STEEL * (S[:, 0] + S[:, 1]))
            out[(p, lev)] = dict(neq=s.neq, tip=s.U[A, 1], vm=vm.max(), at=m.X[int(np.argmax(vm))], sol=s, S=S, VM=vm)
    return out


def fig_wshape_results(study):
    W = EL.WSHAPE
    d, bf, tf, tw = W["d"], W["bf"], W["tf"], W["tw"]
    fig = plt.figure(figsize=(11.2, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.85, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    r = study[(2, 3)]
    s, m = r["sol"], r["sol"].mesh
    mag = 40.0
    cf = field_plot(ax, m, r["VM"], s.U, mag, cmap="magma_r", norm=Normalize(0, 240), levels=np.linspace(0, 240, 25))
    shp = [(-bf / 2, 0), (bf / 2, 0), (bf / 2, tf), (tw / 2, tf), (tw / 2, d - tf), (bf / 2, d - tf), (bf / 2, d),
           (-bf / 2, d), (-bf / 2, d - tf), (-tw / 2, d - tf), (-tw / 2, tf), (-bf / 2, tf)]
    ax.add_patch(Polygon(shp, closed=True, fill=False, ec="0.5", lw=0.8, ls="--"))
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.01, ticks=[0, 60, 120, 180, 240])
    cb.set_label("von Mises stress (MPa)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)
    ax.set_xlim(-bf / 2 - 10, bf / 2 + 10)
    ax.set_ylim(-10, d + 10)
    clean(ax)
    ax.set_title("(a) Q9, %d elements; displacements\nmagnified %d times" % (m.nel, mag), fontsize=10.5, loc="left")
    # (b) a close view of a re-entrant corner
    bx = fig.add_subplot(gs[:, 1])
    field_plot(bx, m, r["VM"], cmap="magma_r", norm=Normalize(0, 240), levels=np.linspace(0, 240, 25))
    draw_mesh(bx, m, color="0.6", lw=0.4)
    bx.set_xlim(-2, 40)
    bx.set_ylim(d - tf - 30, d + 2)
    clean(bx)
    bx.set_title("(b) Where the web meets\nthe top flange (undeformed)", fontsize=10.5, loc="left")
    cx, dx = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for p, col, mk, lab in ((1, LOADCLR, "o", "Q4"), (2, REACTCLR, "s", "Q9")):
        levs = sorted(k[1] for k in study if k[0] == p)
        n = [study[(p, l)]["neq"] for l in levs]
        cx.semilogx(n, [-study[(p, l)]["tip"] for l in levs], "-" + mk, color=col, lw=1.6, ms=5, label=lab)
        dx.semilogx(n, [study[(p, l)]["vm"] for l in levs], "-" + mk, color=col, lw=1.6, ms=5, label=lab)
    for a_ in (cx, dx):
        a_.xaxis.set_major_locator(FixedLocator([300, 1000, 3000]))
        a_.xaxis.set_major_formatter(FixedFormatter(["300", "1000", "3000"]))
        a_.xaxis.set_minor_locator(NullLocator())
    _graph(cx, "", "flange tip\ndeflection (mm)")
    cx.set_title("(c) The flange tip settles down ...", fontsize=10.5, loc="left")
    cx.legend(frameon=False, fontsize=9, loc="lower right")
    _graph(dx, "number of equations", "largest von\nMises stress (MPa)")
    dx.set_title("(d) ... but the peak stress keeps rising", fontsize=10.5, loc="left")
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    return fig
