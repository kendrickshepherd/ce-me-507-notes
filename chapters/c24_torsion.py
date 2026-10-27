"""
Saint-Venant torsion of an I-section (Class 24 application page).

The Prandtl stress function phi satisfies -lap(phi) = 2 G theta in the
cross-section with phi = 0 on its boundary.  We solve for Phi = phi / (G theta),
so that -lap(Phi) = 2, and recover
    J = 2 * integral of Phi,   tau = (T / J) |grad Phi|.
A quarter of the section is modeled, with symmetry (zero normal derivative)
on the two axes.  Lengths are in millimeters.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle, Circle

import c24_fem as F
from notes_style import INK, BDRYCLR, FLUXCLR, LOADCLR, iso

BF, DEPTH, TF, TW = 100.0, 200.0, 12.0, 8.0     # mm: flange width, depth, flange and web thickness
G_STEEL = 79.3e3                                 # MPa
TORQUE = 1.0e6                                   # N mm (1 kN m)
STRESSMAP = "Oranges"
STEEL = "#cdd3da"
TOL = 1e-9


def thin_walled_J(bf=BF, d=DEPTH, tf=TF, tw=TW):
    return (2 * bf * tf ** 3 + (d - 2 * tf) * tw ** 3) / 3.0


def polar_moment(bf=BF, d=DEPTH, tf=TF, tw=TW):
    """I_1 + I_2 about the centroid: the torsion constant if the section did not warp."""
    hw = d - 2 * tf
    ix = 2 * (bf * tf ** 3 / 12 + bf * tf * ((d - tf) / 2) ** 2) + tw * hw ** 3 / 12
    iy = 2 * tf * bf ** 3 / 12 + hw * tw ** 3 / 12
    return ix + iy


def rectangle_J(a, b):
    """Exact torsion constant of an a x b rectangle (a >= b), by series."""
    s = sum(math.tanh(n * math.pi * a / (2 * b)) / n ** 5 for n in range(1, 400, 2))
    return a * b ** 3 / 3 * (1 - 192 / math.pi ** 5 * (b / a) * s)


def _quarter_mesh(p, nref, bf, d, tf, tw, r0):
    r = r0 ** (1.0 / nref)
    n = lambda k: max(1, int(round(k * nref)))
    xw, yf = tw / 2, d / 2 - tf                     # the re-entrant corner (xw, yf)
    xs = F.join(F.graded(0, xw, n(4), r, "b"), F.graded(xw, bf / 2, n(16), r, "a"))
    ys = F.join(F.graded(0, yf, n(16), r, "b"), F.graded(yf, d / 2, 2 * n(4), r, "both"))
    mesh = F.Mesh(xs, ys, p=p, active=lambda xc, yc: yc > yf or xc < xw)
    return mesh, xw, yf


def build(p=2, nref=1, bf=BF, d=DEPTH, tf=TF, tw=TW, r0=1.3):
    mesh, xw, yf = _quarter_mesh(p, nref, bf, d, tf, tw, r0)
    edge = mesh.sides_where(lambda x, y, nn: not (abs(x) < TOL or abs(y) < TOL))
    sol = F.solve(mesh, kappa=1.0, f=2.0, dirichlet=[(edge, 0.0)])
    out = dict(mesh=mesh, sol=sol, corner=(xw, yf), dims=(bf, d, tf, tw))
    out["J"] = 4 * 2 * F.integrate(sol)
    return out


def warping(p=2, nref=1, bf=BF, d=DEPTH, tf=TF, tw=TW, r0=1.3):
    """Saint-Venant warping function omega on the same quarter mesh.

    -lap(omega) = 0, d(omega)/dn = x2 n1 - x1 n2 on the outer surface, and
    omega = 0 on the axes of symmetry (omega is odd in x1 and in x2).
    J = I1 + I2 - int |grad omega|^2, which is an upper bound when computed
    from the finite element solution (2 int Phi is a lower bound)."""
    mesh, xw, yf = _quarter_mesh(p, nref, bf, d, tf, tw, r0)
    on_axis = lambda x, y: abs(x) < TOL or abs(y) < TOL
    axes = mesh.sides_where(lambda x, y, nn: on_axis(x, y))
    neumann = []
    for nrm, h in (((1.0, 0.0), lambda x, y: y), ((-1.0, 0.0), lambda x, y: -y),
                   ((0.0, 1.0), lambda x, y: -x), ((0.0, -1.0), lambda x, y: x)):
        sides = mesh.sides_where(lambda x, y, nn, nrm=nrm: not on_axis(x, y) and tuple(nn) == nrm)
        if sides:
            neumann.append((sides, h))
    sol = F.solve(mesh, kappa=1.0, f=0.0, dirichlet=[(axes, 0.0)], neumann=neumann)
    grad2 = 4 * 2 * sol.energy                     # int |grad omega|^2 over the whole section
    return dict(mesh=mesh, sol=sol, dims=(bf, d, tf, tw), grad2=grad2,
                J=polar_moment(bf, d, tf, tw) - grad2)


def omega_at(model, x, y):
    """omega anywhere in the section, from the quarter (odd in x and in y)."""
    e, xi, eta = F.locate(model["mesh"], abs(x), abs(y))
    return float(np.sign(x) * np.sign(y) * F.value(model["sol"], e, xi, eta)[0])


def build_rectangle(a, b, p=2, nref=1):
    """Quarter of an a x b rectangle, for the benchmark."""
    n = lambda k: max(1, int(round(k * nref)))
    xs = np.linspace(0, a / 2, n(16) + 1)
    ys = np.linspace(0, b / 2, n(4) + 1)
    mesh = F.Mesh(xs, ys, p=p)
    edge = mesh.sides_where(lambda x, y, nn: abs(x - a / 2) < TOL or abs(y - b / 2) < TOL)
    sol = F.solve(mesh, kappa=1.0, f=2.0, dirichlet=[(edge, 0.0)])
    return dict(mesh=mesh, sol=sol, J=4 * 2 * F.integrate(sol))


def grad_mag(model, x, y):
    e, xi, eta = F.locate(model["mesh"], x, y)
    g = F.gradient(model["sol"], e, xi, eta)[:, 0]
    return float(np.hypot(*g))


def stresses(model, torque=TORQUE):
    """Shear stresses (MPa) at the middle of the flange face, the web face,
    and at the element corner next to the re-entrant corner."""
    bf, d, tf, tw = model["dims"]
    J = model["J"]
    xw, yf = model["corner"]
    t_fl = grad_mag(model, bf / 4, d / 2 - 1e-9)          # outer face of the flange, half way out
    t_web = grad_mag(model, tw / 2 - 1e-9, yf / 2)        # face of the web, mid-height of the quarter
    # largest |grad Phi| sampled on the elements touching the corner
    m, sol = model["mesh"], model["sol"]
    best = 0.0
    for e in range(m.nel):
        (xc, yc), (hx, hy) = m.cent[e], m.size[e]
        if abs(abs(xc - xw) - hx / 2) < 1e-9 * d or abs(abs(yc - yf) - hy / 2) < 1e-9 * d:
            near = (abs(xc - xw) <= hx / 2 + 1e-9) and (abs(yc - yf) <= hy / 2 + 1e-9)
            if near:
                for xi in (-1, 1):
                    for eta in (-1, 1):
                        x, y = xc + xi * hx / 2, yc + eta * hy / 2
                        if abs(x - xw) < 1e-9 and abs(y - yf) < 1e-9:
                            g = F.gradient(sol, e, xi, eta)[:, 0]
                            best = max(best, float(np.hypot(*g)))
    k = torque / J
    return dict(flange=k * t_fl, web=k * t_web, corner=k * best,
                flange_thin=torque * tf / J, web_thin=torque * tw / J)


def _full(model, odd=False):
    """The whole section by mirroring the quarter about both axes
    (odd: the field changes sign across each axis, as omega does)."""
    m, sol = model["mesh"], model["sol"]
    T0 = m.triangles()
    Xs, Ts, Us, off = [], [], [], 0
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        Xs.append(m.X * np.array([sx, sy]))
        Ts.append(T0 + off if sx * sy > 0 else T0[:, ::-1] + off)
        Us.append(sx * sy * sol.u if odd else sol.u)
        off += m.nnp
    return np.vstack(Xs), np.vstack(Ts), np.concatenate(Us)


# ---------------------------------------------------------------- figures
def fig_setting():
    """An I-beam twisted by end torques, with its cross-section."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    L = 700.0
    KW = dict(ang=22.0, k=0.75)

    def P(x, y, z):
        # the beam axis z recedes into the page; the cross-section faces the viewer
        return iso(x, z, y, **KW)

    sec = [(-50, 100), (50, 100), (50, 88), (4, 88), (4, -88), (50, -88), (50, -100), (-50, -100),
           (-50, -88), (-4, -88), (-4, 88), (-50, 88)]
    far = [P(x, y, L) for x, y in sec]
    near = [P(x, y, 0) for x, y in sec]
    ax.add_patch(Polygon(far, closed=True, facecolor="#dfe4ea", edgecolor="0.55", lw=0.8, zorder=1))
    for i in range(len(sec)):
        j = (i + 1) % len(sec)
        quad = [near[i], near[j], far[j], far[i]]
        ax.add_patch(Polygon(quad, closed=True, facecolor="#e9edf2", edgecolor="0.6", lw=0.6, zorder=2))
    ax.add_patch(Polygon(near, closed=True, facecolor=STEEL, edgecolor=INK, lw=1.3, zorder=3))
    for z, sgn in ((0, 1), (L, 1)):
        th = np.linspace(0.30 * math.pi, 1.70 * math.pi, 60)
        pts = np.array([P(150 * math.cos(t), 150 * math.sin(t), z) for t in th])
        if z > 0:
            pts = pts[::-1]
        ax.plot(pts[:, 0], pts[:, 1], color=LOADCLR, lw=1.8, zorder=4)
        ax.annotate("", xy=pts[-1], xytext=pts[-4],
                    arrowprops=dict(arrowstyle="-|>", color=LOADCLR, lw=1.8, mutation_scale=14), zorder=5)
    xs_, ys_ = P(-150, 0, 0)
    ax.text(xs_ - 12, ys_, "$T$", fontsize=13, color=LOADCLR, ha="right", va="center")
    xs_, ys_ = P(150, 0, L)
    ax.text(xs_ + 12, ys_, "$T$", fontsize=13, color=LOADCLR, ha="left", va="center")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) A steel I-beam twisted by end torques", fontsize=11.5, loc="left")
    # (b) the cross-section with dimensions
    bx.add_patch(Polygon(sec, closed=True, facecolor=STEEL, edgecolor=INK, lw=1.3))

    def dim(x0, y0, x1, y1, lab, off=(0, 0), ha="center", va="center"):
        bx.annotate("", xy=(x0, y0), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9, shrinkA=0, shrinkB=0))
        bx.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], lab, ha=ha, va=va, fontsize=10, color="0.25")
    dim(-50, 116, 50, 116, "$b_f$ = 100", off=(0, 9))
    dim(66, -100, 66, 100, "$d$ = 200", off=(6, 0), ha="left")
    dim(-62, 88, -62, 100, "$t_f$ = 12", off=(-6, 0), ha="right")
    dim(-4, -40, 4, -40, "", off=(0, 0))
    bx.text(10, -40, "$t_w$ = 8", ha="left", va="center", fontsize=10, color="0.25")
    bx.annotate("re-entrant corner", xy=(4, 88), xytext=(28, 45), fontsize=10, color=LOADCLR,
                arrowprops=dict(arrowstyle="-", color=LOADCLR, lw=0.9))
    bx.text(0, -128, "dimensions in mm", ha="center", va="top", fontsize=9.5, color="0.35")
    bx.set_xlim(-110, 125)
    bx.set_ylim(-140, 140)
    bx.set_aspect("equal")
    bx.axis("off")
    bx.set_title("(b) Its cross-section", fontsize=11.5, loc="left")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_domain(model):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 5.4))
    bf, d, tf, tw = model["dims"]
    xw, yf = model["corner"]
    poly = [(0, 0), (xw, 0), (xw, yf), (bf / 2, yf), (bf / 2, d / 2), (0, d / 2)]
    ax.add_patch(Polygon(poly, closed=True, facecolor="#f4f1ea", edgecolor="none"))
    lw = 3.0
    ax.plot([xw, xw, bf / 2, bf / 2, 0], [0, yf, yf, d / 2, d / 2], color=BDRYCLR, lw=lw)
    ax.plot([0, 0], [0, d / 2], color=FLUXCLR, lw=lw)
    ax.plot([0, xw], [0, 0], color=FLUXCLR, lw=lw)
    k = dict(fontsize=10.5)
    ax.text(bf / 4, d / 2 + 3, r"outer surface: $\Phi = 0$", ha="center", va="bottom", color=BDRYCLR, **k)
    ax.text(-3, d / 4, "symmetry\n" + r"$\nabla\Phi\cdot\underline{n} = 0$", ha="right", va="center", color=FLUXCLR, **k)
    ax.text(xw + 2, -4, "symmetry", ha="left", va="top", color=FLUXCLR, **k)
    ax.text(20, 40, r"$-\nabla^2\Phi = 2$", ha="left", va="center", fontsize=12)
    ax.set_title("(a) A quarter of the section", fontsize=11.5, loc="left")
    m = model["mesh"]
    for e in range(m.nel):
        (xc, yc), (hx, hy) = m.cent[e], m.size[e]
        bx.add_patch(Rectangle((xc - hx / 2, yc - hy / 2), hx, hy, facecolor="none", edgecolor="0.5", lw=0.35))
    bx.set_title("(b) The biquadratic mesh", fontsize=11.5, loc="left")
    for a in (ax, bx):
        a.set_xlim(-30, bf / 2 + 4)
        a.set_ylim(-10, d / 2 + 12)
        a.set_aspect("equal")
        a.axis("off")
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_solution(model):
    """(a) contours of the stress function = shear-stress trajectories;
    (b) the size of the shear stress; (c) the membrane (soap film) view."""
    X, T, U = _full(model)
    tri = mtri.Triangulation(X[:, 0], X[:, 1], T)
    sol, m = model["sol"], model["mesh"]
    Gq = F.nodal_gradient(sol)
    Gmag = np.tile(np.hypot(Gq[:, 0], Gq[:, 1]), 4)
    fig = plt.figure(figsize=(11.2, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.tricontourf(tri, U, levels=np.linspace(0, U.max(), 15), cmap=STRESSMAP)
    ax.tricontour(tri, U, levels=np.linspace(0, U.max(), 6)[1:-1], colors="white", linewidths=0.8)
    ax.set_title("(a) Contours of $\\Phi$", fontsize=11, loc="left")
    bx = fig.add_subplot(1, 3, 2)
    tau = Gmag * TORQUE / model["J"]
    cs = bx.tricontourf(tri, np.clip(tau, 0, 120), levels=np.linspace(0, 120, 13), cmap=STRESSMAP)
    cb = fig.colorbar(cs, ax=bx, fraction=0.06, pad=0.02)
    cb.set_label("shear stress (MPa) for $T$ = 1 kN m")
    bx.set_title("(b) Shear stress $|\\tau|$", fontsize=11, loc="left")
    for a in (ax, bx):
        a.set_aspect("equal")
        a.axis("off")
    cx = fig.add_subplot(1, 3, 3, projection="3d")
    cx.plot_trisurf(X[:, 0], X[:, 1], U, triangles=T, cmap=STRESSMAP, linewidth=0, antialiased=True)
    cx.set_box_aspect((1, 2, 0.55))
    cx.view_init(elev=28, azim=-58)
    cx.set_axis_off()
    cx.set_title("(c) $\\Phi$ as a membrane", fontsize=11, loc="left")
    fig.tight_layout(w_pad=0.5)
    return fig


def fig_corner(levels):
    """J and the stresses as the mesh is refined (levels = list of models)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    hmin = [min(m["mesh"].size.min(axis=0)) for m in levels]
    st = [stresses(m) for m in levels]
    ax.plot(hmin, [s["corner"] for s in st], "o-", color=LOADCLR, lw=1.6, label="at the re-entrant corner")
    ax.plot(hmin, [s["flange"] for s in st], "s-", color=INK, lw=1.6, label="flange face, half way out")
    ax.plot(hmin, [s["web"] for s in st], "^-", color="0.5", lw=1.6, label="web face, mid-height")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("smallest element size near the corner (mm)")
    ax.set_ylabel("shear stress (MPa), $T$ = 1 kN m")
    ax.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="0.9", lw=0.7)
    fig.tight_layout()
    return fig, st


def fig_warping(wm, lift=45.0, stub=150.0):
    """(a) omega over the section; (b) plan view of the end of the beam, warped."""
    bf, d, tf, tw = wm["dims"]
    X, Tr, U = _full(wm, odd=True)
    tri = mtri.Triangulation(X[:, 0], X[:, 1], Tr)
    vmax = 1000.0 * math.ceil(float(np.abs(U).max()) / 1000.0)
    sec = [(-bf / 2, d / 2), (bf / 2, d / 2), (bf / 2, d / 2 - tf), (tw / 2, d / 2 - tf), (tw / 2, -d / 2 + tf),
           (bf / 2, -d / 2 + tf), (bf / 2, -d / 2), (-bf / 2, -d / 2), (-bf / 2, -d / 2 + tf),
           (-tw / 2, -d / 2 + tf), (-tw / 2, d / 2 - tf), (-bf / 2, d / 2 - tf)]
    fig = plt.figure(figsize=(10.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45])
    ax = fig.add_subplot(gs[0])
    cs = ax.tricontourf(tri, U, levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r")
    ax.add_patch(Polygon(sec, closed=True, facecolor="none", edgecolor=INK, lw=1.0))
    cb = fig.colorbar(cs, ax=ax, shrink=0.75, pad=0.03, ticks=np.linspace(-vmax, vmax, 5))
    cb.set_label(r"$\omega$ (mm$^2$)")
    for xx, yy, lab, ha in ((bf / 2 + 4, d / 2 - tf / 2, "$+$", "left"), (-bf / 2 - 4, d / 2 - tf / 2, "$-$", "right"),
                            (bf / 2 + 4, -d / 2 + tf / 2, "$-$", "left"), (-bf / 2 - 4, -d / 2 + tf / 2, "$+$", "right")):
        ax.text(xx, yy, lab, fontsize=12, ha=ha, va="center")
    ax.set_xlim(-bf / 2 - 18, bf / 2 + 18)
    ax.set_ylim(-d / 2 - 8, d / 2 + 8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(a) The warping function $\omega$", fontsize=11, loc="left")
    # (b) plan view from above: x_3 to the right, x_1 up, x_2 out of the page
    bx = fig.add_subplot(gs[1])
    s = lift / vmax
    x1 = np.linspace(-bf / 2, bf / 2, 41)
    ym = d / 2 - tf / 2                                   # mid-thickness of the flanges
    top = np.array([s * omega_at(wm, x, ym) for x in x1])
    bot = np.array([s * omega_at(wm, x, -ym) for x in x1])
    bx.add_patch(Rectangle((0, -bf / 2), stub, bf, facecolor="#eef1f4", edgecolor="0.6", lw=0.8))
    bx.plot([0, stub], [0, 0], color="0.6", lw=0.7, ls="-.")
    bx.plot([0, 0], [-bf / 2, bf / 2], color="0.45", lw=1.2, ls="--")
    bx.plot(bot, x1, color="0.55", lw=2.2, solid_capstyle="round")
    bx.plot(top, x1, color=INK, lw=2.4, solid_capstyle="round")
    for k in (0, -1):
        bx.annotate("", xy=(top[k], x1[k]), xytext=(0, x1[k]),
                    arrowprops=dict(arrowstyle="-|>", color=LOADCLR, lw=1.5, shrinkA=0, shrinkB=0))
    bx.text(top[-1] + 6, x1[-1] - 2, "top flange", fontsize=10.5, ha="left", va="top")
    bx.text(bot[0] + 6, x1[0] + 2, "bottom flange\n(below)", fontsize=10.5, color="0.4", ha="left", va="bottom")
    bx.text(-6, -bf / 2 - 6, "plane end\nof the beam", fontsize=9.5, color="0.4", ha="center", va="top")
    o = np.array([-128.0, -bf / 2 + 12])
    for v, lab in (((32, 0), "$x_3$"), ((0, 32), "$x_1$")):
        e = o + np.array(v)
        bx.annotate("", xy=e, xytext=o, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, shrinkA=0, shrinkB=0))
        bx.text(*(o + 1.4 * np.array(v)), lab, fontsize=11, ha="center", va="center")
    bx.add_patch(Circle(o, 5.0, facecolor="white", edgecolor=INK, lw=1.0, zorder=5))
    bx.add_patch(Circle(o, 1.6, color=INK, zorder=6))
    bx.text(o[0] - 7, o[1] - 7, "$x_2$", fontsize=11, ha="right", va="top")
    bx.set_xlim(-155, stub + 10)
    bx.set_ylim(-bf / 2 - 34, bf / 2 + 14)
    bx.set_aspect("equal")
    bx.axis("off")
    bx.set_title(r"(b) Plan view of the end of the beam, $u_3 = \theta\,\omega$ (exaggerated)", fontsize=11, loc="left")
    fig.tight_layout(w_pad=1.5)
    return fig
