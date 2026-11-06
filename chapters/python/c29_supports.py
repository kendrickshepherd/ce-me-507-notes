"""
Figures for Class 29: supports, symmetry, and boundary conditions in elasticity.

Built on elast2d.py (meshes, element integrals, and the banded solver) and
elast_figs.py (plotting helpers).  The solver here adds what Class 29 needs and
elast2d.solve does not: supports at single nodes, point loads, a uniform free
strain such as a thermal strain, springs on a side (a Robin condition), and
nodes whose two unknowns are rotated to normal and tangential components.

Every index is numbered from 0.  Units: millimetres, newtons, and megapascals.
Only numpy is used, so the page renders on the CI image.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch, Wedge
from matplotlib.colors import Normalize

import elast2d as EL
import elast_figs as EF
from notes_style import INK, LOADCLR, REACTCLR, BDRYCLR, MESHCLR

E_STEEL, NU_STEEL, T_PLATE = 200000.0, 0.3, 10.0
BODY = "#e3eef8"
GHOST = "#e67e22"
STRESSMAP = "YlOrRd"


# ------------------------------------------------------------------ meshes
def rect_mesh(L, H, nx, ny, p=1, x0=0.0, y0=0.0):
    pt = EL.Patch((x0, y0), (x0 + L, y0), (x0 + L, y0 + H), (x0, y0 + H), EL.graded(nx), EL.graded(ny),
                  names=dict(left="left", right="right", bottom="bottom", top="top"))
    return EL.Mesh([pt], p)


def node_at(mesh, x, y):
    return int(np.argmin(np.hypot(mesh.X[:, 0] - x, mesh.X[:, 1] - y)))


def nodes_on(mesh, test):
    return [A for A in range(mesh.nnp) if test(*mesh.X[A])]


def hole_patch(nr, nt, a, b, t0, t1, ratio, names):
    """One eighth of a square plate [-b, b]^2 with a hole of radius a, between the
    angles t0 and t1 (radians, 45 degrees apart, from a multiple of 45 degrees).
    s runs out from the hole and t counterclockwise, so the hole is the patch's
    left side, the edge of the square its right side, and the two cuts its
    bottom and top."""
    def outer(th):
        c, s = math.cos(th), math.sin(th)
        m = max(abs(c), abs(s))
        return np.array([b * c / m, b * s / m])

    q0, q1 = a * np.array([math.cos(t0), math.sin(t0)]), outer(t0)
    q3, q2 = a * np.array([math.cos(t1), math.sin(t1)]), outer(t1)
    return EL.Patch(q0, q1, q2, q3, EL.graded(nr, ratio), EL.graded(nt),
                    curves=dict(left=EL.arc(a, t0, t1)), names=names)


def eighth_mesh(p, nr, nt, a, b, ratio=1.18):
    """The eighth 0 <= angle <= 45 degrees.  Sides: 'hole', 'axis' (on x1 = 0),
    'diagonal', and 'outer' (on x0 = b)."""
    pt = hole_patch(nr, nt, a, b, 0.0, math.pi / 4, ratio,
                    dict(left="hole", bottom="axis", top="diagonal", right="outer"))
    return EL.Mesh([pt], p)


def quarter_mesh(p, nr, nt, a, b, ratio=1.18):
    """The quarter 0 <= angle <= 90 degrees, as two eighths.  Sides: 'hole',
    'axis0' (on x1 = 0), 'axis1' (on x0 = 0), 'outer0' (on x0 = b), 'outer1' (on x1 = b)."""
    p0 = hole_patch(nr, nt, a, b, 0.0, math.pi / 4, ratio, dict(left="hole", bottom="axis0", right="outer0"))
    p1 = hole_patch(nr, nt, a, b, math.pi / 4, math.pi / 2, ratio, dict(left="hole", top="axis1", right="outer1"))
    return EL.Mesh([p0, p1], p)


# ------------------------------------------------------------------ solver
class Sol:
    pass


def side_spring(mesh, e, side, kmat, t=1.0):
    """The integral over one element side of N_a N_b kmat t dGamma, in the
    node-by-node order of k^e."""
    xg, wg = EL.gauss(mesh.p + 1)
    xe = mesh.X[mesh.IEN[e]]
    ks = np.zeros((2 * mesh.nen, 2 * mesh.nen))
    for s, w in zip(xg, wg):
        xi, eta = EL.SIDE_PARAM[side](s)
        N, dN = EL.shape(mesh.p, xi, eta)
        DF = xe.T @ dN
        tang = DF[:, 0] if side in ("bottom", "top") else DF[:, 1]
        ks += np.kron(np.outer(N, N), kmat) * (w * np.hypot(*tang) * t)
    return ks


def free_strain_load(mesh, e, D, eps0, t=1.0):
    """The integral of B^T D eps0 t over element e, for a uniform free strain eps0."""
    xg, wg = EL.gauss(mesh.p + 1)
    xe = mesh.X[mesh.IEN[e]]
    fe = np.zeros(2 * mesh.nen)
    for xi, wi in zip(xg, wg):
        for eta, wj in zip(xg, wg):
            N, dN = EL.shape(mesh.p, xi, eta)
            DF = xe.T @ dN
            B = EL.bmatrix(dN @ np.linalg.inv(DF))
            fe += (B.T @ D @ eps0) * (wi * wj * np.linalg.det(DF) * t)
    return fe


def solve(mesh, fixed=None, point=None, traction=None, eps0=None, springs=None, rotated=None,
          E=E_STEEL, nu=NU_STEEL, t=T_PLATE, plane="stress"):
    """fixed: {(A, i): g}.  i is the Cartesian component, or, at a node listed in
    `rotated`, 0 for the component along its normal and 1 for the tangential one.
    point: {A: (F0, F1)}; traction: {side name: h(x, y)}; eps0: a uniform free
    strain (Voigt); springs: [(side name, 2 x 2 stiffness per unit area)];
    rotated: {A: unit normal n}, whose unknowns become (u.n, u.t), t = (-n1, n0)."""
    t = t if plane == "stress" else 1.0
    D = EL.dmatrix(E, nu, plane)
    fixed, point, traction = fixed or {}, point or {}, traction or {}
    springs, rotated = springs or [], rotated or {}
    nnp, nen = mesh.nnp, mesh.nen
    Rot = {A: np.array([[n[0], n[1]], [-n[1], n[0]]], float) for A, n in rotated.items()}
    ID, g, neq = -np.ones((nnp, 2), int), np.zeros((nnp, 2)), 0
    for A in range(nnp):                                   # node by node
        for i in range(2):
            if (A, i) in fixed:
                g[A, i] = fixed[(A, i)]
            else:
                ID[A, i] = neq
                neq += 1
    Fn = np.zeros((nnp, 2))                                # applied loads, Cartesian
    for name, h in traction.items():
        for e, sd in mesh.sides.get(name, ()):
            Fn[mesh.IEN[e]] += EL.side_load(mesh, e, sd, h, t)
    for A, F in point.items():
        Fn[A] += F
    extra = {}
    for name, kmat in springs:
        for e, sd in mesh.sides.get(name, ()):
            extra[e] = extra.get(e, 0.0) + side_spring(mesh, e, sd, np.asarray(kmat, float), t)
    rows, cols, vals, F, arrays = [], [], [], np.zeros(neq), []
    for e in range(mesh.nel):
        ien = mesh.IEN[e]
        ke, _ = EL.element_arrays(mesh, e, D, t)
        ke = ke + extra.get(e, 0.0)
        fe = free_strain_load(mesh, e, D, eps0, t) if eps0 is not None else np.zeros(2 * nen)
        T = np.eye(2 * nen)
        for a, A in enumerate(ien):
            if A in Rot:
                T[2 * a:2 * a + 2, 2 * a:2 * a + 2] = Rot[A].T
        ke, fe = T.T @ ke @ T, T.T @ fe
        arrays.append((ke, fe))
        lm = ID[ien].reshape(-1)
        fl = fe - ke @ g[ien].reshape(-1)
        for q in range(2 * nen):
            P = lm[q]
            if P < 0:
                continue
            F[P] += fl[q]
            for r in range(2 * nen):
                if lm[r] >= 0:
                    rows.append(P)
                    cols.append(lm[r])
                    vals.append(ke[q, r])
    Fl = np.array([Rot[A] @ Fn[A] if A in Rot else Fn[A] for A in range(nnp)])
    for A in range(nnp):
        for i in range(2):
            if ID[A, i] >= 0:
                F[ID[A, i]] += Fl[A, i]
    sol = Sol()
    sol.mesh, sol.D, sol.t, sol.E, sol.nu, sol.plane = mesh, D, t, E, nu, plane
    sol.eps0 = np.zeros(3) if eps0 is None else np.asarray(eps0, float)
    sol.ID, sol.neq, sol.Fn = ID, neq, Fn
    try:
        d = EL.banded_solve(np.array(rows), np.array(cols), np.array(vals), neq, F) if neq else np.zeros(0)
        sol.ok = bool(np.all(np.isfinite(d)))
    except np.linalg.LinAlgError:
        d, sol.ok = np.full(neq, np.nan), False
    Ul = g.copy()
    for A in range(nnp):
        for i in range(2):
            if ID[A, i] >= 0:
                Ul[A, i] = d[ID[A, i]]
    sol.Ul = Ul
    sol.U = np.array([Rot[A].T @ Ul[A] if A in Rot else Ul[A] for A in range(nnp)])
    # reactions R = K d - F at the prescribed components, in each node's basis
    Rr = -Fl.copy()
    for e, (ke, fe) in enumerate(arrays):
        ien = mesh.IEN[e]
        Rr[ien] += (ke @ Ul[ien].reshape(-1) - fe).reshape(-1, 2)
    sol.R = np.where(ID < 0, Rr, 0.0)
    return sol


def stress(sol, e, xi, eta):
    """sigma = D (B u - eps0) at a parent point of element e."""
    mesh = sol.mesh
    N, dN = EL.shape(mesh.p, xi, eta)
    xe = mesh.X[mesh.IEN[e]]
    B = EL.bmatrix(dN @ np.linalg.inv(xe.T @ dN))
    return sol.D @ (B @ sol.U[mesh.IEN[e]].reshape(-1) - sol.eps0)


def gauss_stresses(sol):
    """Stresses at the Gauss points of every element, (nel * ng^2, 3)."""
    xg, _ = EL.gauss(sol.mesh.p + 1)
    return np.array([stress(sol, e, xi, eta) for e in range(sol.mesh.nel) for xi in xg for eta in xg])


# ------------------------------------------------------------------ drawing
def pin(ax, x, y, up=(0, 1), s=1.0, color=BDRYCLR):
    """A pin: a filled triangle whose apex is at (x, y), on ground hatching.
    `up` points from the support into the body."""
    u = np.asarray(up, float) / np.hypot(*up)
    v = np.array([-u[1], u[0]])
    P = np.array([x, y])
    base = P - 0.9 * s * u
    ax.add_patch(Polygon([P, base + 0.55 * s * v, base - 0.55 * s * v], closed=True, fc=color, ec=color, lw=1.0, zorder=8))
    ground(ax, base - 0.12 * s * u, v, 1.6 * s, u, color)


def roller(ax, x, y, up=(0, 1), s=1.0, color=BDRYCLR):
    """A roller: an open triangle on two wheels; it holds motion along `up` only."""
    u = np.asarray(up, float) / np.hypot(*up)
    v = np.array([-u[1], u[0]])
    P = np.array([x, y])
    base = P - 0.75 * s * u
    ax.add_patch(Polygon([P, base + 0.5 * s * v, base - 0.5 * s * v], closed=True, fc="white", ec=color, lw=1.3, zorder=8))
    for k in (-0.28, 0.28):
        c = base + k * s * v - 0.15 * s * u
        ax.add_patch(Circle(c, 0.13 * s, fc="white", ec=color, lw=1.1, zorder=8))
    ground(ax, base - 0.3 * s * u, v, 1.5 * s, u, color)


def ground(ax, c, v, w, u, color):
    a, b = c - 0.5 * w * v, c + 0.5 * w * v
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4, zorder=8)
    for k in np.linspace(0.05, 0.95, 6):
        q = a + k * (b - a)
        r = q - 0.22 * w * u / 1.5 - 0.12 * w * v / 1.5
        ax.plot([q[0], r[0]], [q[1], r[1]], color=color, lw=0.8, zorder=8)


def turn_arrow(ax, c, r, a0, a1, color=GHOST):
    """A curved arrow on a circle of radius r about c, from angle a0 to a1 (degrees)."""
    t = np.radians(np.linspace(a0, a1, 40))
    ax.plot(c[0] + r * np.cos(t[:-3]), c[1] + r * np.sin(t[:-3]), color=color, lw=1.8, zorder=9)
    EF.arrow(ax, (c[0] + r * np.cos(t[-4]), c[1] + r * np.sin(t[-4])), (c[0] + r * np.cos(t[-1]), c[1] + r * np.sin(t[-1])),
             color, 1.8, 13, z=9)


def fig_mechanisms():
    """Four ways to support a body: one holds it, three leave a rigid-body motion free."""
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4))
    L, H, s = 3.0, 1.4, 0.32
    rect = np.array([(0, 0), (L, 0), (L, H), (0, H)])

    def body(ax, P=rect, ghost=None):
        ax.add_patch(Polygon(P, closed=True, fc=BODY, ec=INK, lw=1.3, zorder=2))
        if ghost is not None:
            ax.add_patch(Polygon(ghost, closed=True, fill=False, ec=GHOST, lw=1.6, ls="--", zorder=3))

    def rot(P, c, a):
        a = math.radians(a)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return (np.asarray(P) - c) @ R.T + c

    # (a) a pin and a roller
    ax = axes[0]
    body(ax)
    pin(ax, 0, 0, (0, 1), s)
    roller(ax, L, 0, (0, 1), s)
    ax.set_title("(a) A pin and a roller:\nthe body is held", fontsize=10.5, loc="left")
    # (b) three parallel rollers
    ax = axes[1]
    body(ax, ghost=rect + (0.45, 0))
    for x in (0, L / 2, L):
        roller(ax, x, 0, (0, 1), s)
        ax.plot([x, x], [-0.9, H + 0.5], color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
    EF.arrow(ax, (L / 2 - 0.35, H + 0.28), (L / 2 + 0.45, H + 0.28), GHOST, 1.8, 13, z=9)
    ax.set_title("(b) Three rollers whose lines of action\nare parallel: the body slides", fontsize=10.5, loc="left")
    # (c) a pin, and a roller whose line of action passes through the pin
    ax = axes[2]
    body(ax, ghost=rot(rect, (0, 0), 9))
    pin(ax, 0, 0, (0, 1), s)
    roller(ax, L, 0, (-1, 0), s)
    ax.plot([-0.6, L + 0.6], [0, 0], color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
    turn_arrow(ax, (0, 0), 1.0, 12, 62)
    ax.set_title("(c) A pin, and a roller whose line of\naction passes through it: the body turns", fontsize=10.5, loc="left")
    # (d) a ring on three radial rollers
    ax = axes[3]
    c, ro, ri = np.array([1.5, 0.7]), 1.05, 0.55
    th = np.linspace(0, 2 * np.pi, 120)
    ring = np.r_[np.c_[c[0] + ro * np.cos(th), c[1] + ro * np.sin(th)], np.c_[c[0] + ri * np.cos(th[::-1]), c[1] + ri * np.sin(th[::-1])]]
    ax.add_patch(Polygon(ring, closed=True, fc=BODY, ec="none", zorder=2))
    for r in (ro, ri):
        ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color=INK, lw=1.3, zorder=3)
    for k, ang in enumerate((90, 210, 330)):
        a = math.radians(ang)
        n = np.array([math.cos(a), math.sin(a)])
        roller(ax, *(c + ro * n), -n, 0.8 * s)
        ax.plot([c[0], c[0] + 1.75 * n[0]], [c[1], c[1] + 1.75 * n[1]], color="0.55", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.plot(*c, "o", ms=4, color="0.4", zorder=4)
    turn_arrow(ax, c, 0.8, 100, 150)
    ax.set_title("(d) Three rollers whose lines of action\nmeet at a point: the ring spins", fontsize=10.5, loc="left")
    for ax in axes:
        ax.set_xlim(-0.9, L + 0.9)
        ax.set_ylim(-1.0, H + 0.95)
        EF.clean(ax)
    fig.tight_layout(w_pad=0.6)
    return fig


# ------------------------------------------------------------------ redundant supports
PL, PH, PS = 200.0, 100.0, 50.0                     # plate length, depth (mm), and the applied pull (MPa)


def pulled_plate(ny, clamp, p=1):
    """The plate pulled on its right end.  The left end is either on rollers that
    hold u0, with the node at its midpoint also held in u1, or clamped."""
    m = rect_mesh(PL, PH, 2 * ny, ny, p)
    left = nodes_on(m, lambda x, y: abs(x) < 1e-9)
    fixed = {(A, 0): 0.0 for A in left}
    if clamp:
        fixed.update({(A, 1): 0.0 for A in left})
    else:
        fixed[(node_at(m, 0, PH / 2), 1)] = 0.0
    return solve(m, fixed, traction={"right": lambda x, y: (PS, 0.0)})


def fig_redundant():
    """The pulled plate with its left end on rollers or clamped: sigma_00, and
    the largest sigma_00 as the mesh is refined."""
    from matplotlib.colors import TwoSlopeNorm
    fig = plt.figure(figsize=(12.0, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.82], height_ratios=[1, 0.06], hspace=0.35, wspace=0.12)
    levels = np.arange(-22.5, 22.6, 3.0)
    norm = TwoSlopeNorm(0.0, -21, 21)
    for k, clamp in enumerate((False, True)):
        ax = fig.add_subplot(gs[0, k])
        s = pulled_plate(16, clamp)
        cf = EF.fine_field(ax, s.mesh, lambda e, xi, eta: stress(s, e, xi, eta)[1], k=4, cmap="RdBu_r",
                           levels=levels, norm=norm)
        EF.draw_mesh(ax, s.mesh, color="0.6", lw=0.12)
        ax.add_patch(Rectangle((0, 0), PL, PH, fill=False, ec=INK, lw=1.0, zorder=4))
        for y in np.linspace(0, PH, 5):
            if clamp or abs(y - PH / 2) < 1e-9:
                pin(ax, 0, y, (1, 0), 9.0)
            else:
                roller(ax, 0, y, (1, 0), 9.0)
        for y in np.linspace(8, PH - 8, 6):
            EF.arrow(ax, (PL + 3, y), (PL + 28, y), LOADCLR, 1.4, 10)
        ax.set_xlim(-28, PL + 34)
        ax.set_ylim(-12, PH + 12)
        EF.clean(ax)
        ax.set_title("(b) Left end clamped: $u_0 = u_1 = 0$\n" if clamp else
                     "(a) Left end on rollers that hold $u_0$,\nwith its midpoint also held in $u_1$",
                     fontsize=10.5, loc="left")
    cax = fig.add_subplot(gs[1, 0:2])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=np.arange(-18, 19, 6), extend="both")
    cb.set_label("$\\sigma_{11}$ (MPa), the stress across the plate, which the pull alone does not cause", fontsize=10)
    # (c) the largest sigma_00 as the mesh is refined
    ax = fig.add_subplot(gs[:, 2])
    nys = (2, 4, 8, 16, 32)
    for clamp, col, mk, lab in ((False, REACTCLR, "s", "left end on rollers"), (True, LOADCLR, "o", "left end clamped")):
        peaks = [gauss_stresses(pulled_plate(ny, clamp))[:, 0].max() for ny in nys]
        ax.plot(nys, peaks, mk + "-", color=col, lw=1.8, ms=6, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(nys)
    ax.set_xticklabels([str(n) for n in nys])
    ax.minorticks_off()
    ax.set_ylim(45, 85)
    EF._graph(ax, "elements through the depth", "largest $\\sigma_{00}$ at a Gauss point (MPa)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(c) The largest stress as the mesh is refined", fontsize=10.5, loc="left")
    return fig


# ------------------------------------------------------------------ symmetry
BL, BH, BQ, BW = 600.0, 150.0, 10.0, 75.0           # beam length, depth (mm), patch load (MPa) and width (mm); the patch edges fall on element edges


def beam_supports(m):
    """A vertical support at each bottom corner and a horizontal one at the bottom
    midpoint: three components, placed symmetrically."""
    return {(node_at(m, 0, 0), 1): 0.0, (node_at(m, BL, 0), 1): 0.0, (node_at(m, BL / 2, 0), 0): 0.0}


def patch_load(centers):
    """Downward (+1) or upward (-1) patch loads on the top edge: {x center: sign}."""
    def h(x, y):
        return (0.0, -sum(sg * BQ for xc, sg in centers.items() if abs(x - xc) < BW / 2 - 1e-9))
    return h


def beam_solution(case, half=None, nx=32, ny=8):
    """case: 'sym' (both quarter points pushed down) or 'anti' (left down, right up).
    half: None for the whole beam, 'sym' or 'anti' for the left half with
    symmetry or antisymmetry conditions on the cut."""
    loads = {"sym": {BL / 4: 1, 3 * BL / 4: 1}, "anti": {BL / 4: 1, 3 * BL / 4: -1}}[case]
    if half is None:
        m = rect_mesh(BL, BH, nx, ny)
        fixed = beam_supports(m)
    else:
        m = rect_mesh(BL / 2, BH, nx // 2, ny)
        fixed = {(node_at(m, 0, 0), 1): 0.0}
        cut = nodes_on(m, lambda x, y: abs(x - BL / 2) < 1e-9)
        comp = 0 if half == "sym" else 1
        fixed.update({(A, comp): 0.0 for A in cut})
        fixed[(node_at(m, BL / 2, 0), 0)] = 0.0
        loads = {xc: sg for xc, sg in loads.items() if xc < BL / 2}
    return solve(m, fixed, traction={"top": patch_load(loads)})


def outline_of(mesh, U=None, scale=0.0):
    """The boundary of a rectangular mesh, in order, displaced by scale * U."""
    X = mesh.X + (scale * U if U is not None else 0.0)
    x0, x1 = mesh.X[:, 0].min(), mesh.X[:, 0].max()
    y0, y1 = mesh.X[:, 1].min(), mesh.X[:, 1].max()
    tol = 1e-9 * (x1 - x0)
    b = [A for A in range(mesh.nnp) if abs(mesh.X[A, 1] - y0) < tol]
    r = [A for A in range(mesh.nnp) if abs(mesh.X[A, 0] - x1) < tol]
    t = [A for A in range(mesh.nnp) if abs(mesh.X[A, 1] - y1) < tol]
    l = [A for A in range(mesh.nnp) if abs(mesh.X[A, 0] - x0) < tol]
    ring = (sorted(b, key=lambda A: mesh.X[A, 0]) + sorted(r, key=lambda A: mesh.X[A, 1])[1:]
            + sorted(t, key=lambda A: -mesh.X[A, 0])[1:] + sorted(l, key=lambda A: -mesh.X[A, 1])[1:-1])
    return X[ring]


def fig_symmetry():
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 5.6), gridspec_kw=dict(height_ratios=[1, 1]))
    for col, case in enumerate(("sym", "anti")):
        full = beam_solution(case)
        half = beam_solution(case, half=case)
        sc = 0.12 * BH / np.abs(full.U).max()
        loads = {"sym": {BL / 4: 1, 3 * BL / 4: 1}, "anti": {BL / 4: 1, 3 * BL / 4: -1}}[case]
        for row, s in enumerate((full, half)):
            ax = axes[row, col]
            Lm = s.mesh.X[:, 0].max()
            ax.add_patch(Polygon(outline_of(s.mesh), closed=True, fill=False, ec="0.6", lw=0.9, ls="--", zorder=1))
            ax.add_patch(Polygon(outline_of(s.mesh, s.U, sc), closed=True, fc=BODY, ec=INK, lw=1.2, zorder=2))
            EF.draw_mesh(ax, s.mesh, U=s.U, scale=sc, color="0.55", lw=0.25, z=3)
            for xc, sg in loads.items():
                if xc > Lm:
                    continue
                for x in np.linspace(xc - BW / 2 + 6, xc + BW / 2 - 6, 3):
                    tip = BH + 4 if sg > 0 else BH + 34
                    EF.arrow(ax, (x, BH + 34 if sg > 0 else BH + 4), (x, tip), LOADCLR, 1.3, 9)
            for A, i in fixed_of(s):
                if not on_cut(s, A):
                    EF._support(ax, *s.mesh.X[A], i, 13, outward=(0, -1))
            if row == 0:
                ax.plot([BL / 2, BL / 2], [-30, BH + 45], color="0.35", lw=1.0, ls="-.", zorder=5)
                # a mirror pair of points and their displacements, drawn to a common scale
                pair = [node_at(s.mesh, xa, BH) for xa in (75.0, BL - 75.0)]
                big = max(np.hypot(*s.U[A]) for A in pair)
                for A in pair:
                    P = s.mesh.X[A] + sc * s.U[A]
                    ax.plot(*P, "o", ms=5, color=GHOST, zorder=7)
                    EF.arrow(ax, P, P + 34.0 * s.U[A] / big, GHOST, 1.8, 12, z=7)
                ax.set_title(("(a) Symmetric loads: $u_0$ changes sign across the plane,\nso $u_0 = 0$ on it, and $u_1$ is mirrored"
                              if case == "sym" else
                              "(b) Antisymmetric loads: $u_1$ changes sign across the plane,\nso $u_1 = 0$ on it, and $u_0$ is mirrored"),
                             fontsize=10.5, loc="left")
            else:
                cut = nodes_on(s.mesh, lambda x, y: abs(x - BL / 2) < 1e-9)
                for A in cut[::1]:
                    x, y = s.mesh.X[A]
                    EF._support(ax, x, y, 0 if case == "sym" else 1, 11, outward=(1, 0))
                ax.set_title(("(c) The left half, with $u_0 = 0$ on the plane of symmetry"
                              if case == "sym" else
                              "(d) The left half, with $u_1 = 0$ on the plane of antisymmetry"),
                             fontsize=10.5, loc="left")
            ax.set_xlim(-30, BL + 30)
            ax.set_ylim(-45, BH + 62)
            EF.clean(ax)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    return fig


def fixed_of(s):
    return [(A, i) for A in range(s.mesh.nnp) for i in range(2) if s.ID[A, i] < 0]


def on_cut(s, A):
    return abs(s.mesh.X[A, 0] - BL / 2) < 1e-9 and s.mesh.X[:, 0].max() < BL - 1e-9


# ------------------------------------------------------------------ an inclined plane of symmetry
HA, HB, HS = 12.5, 100.0, 50.0                     # hole radius, half-width of the square (mm), pull on each edge (MPa)
DIAG_N = np.array([-1.0, 1.0]) / math.sqrt(2.0)     # outward normal of the eighth's diagonal side


def eighth_solution(method="rotate", k=None, p=2, nr=10, nt=6):
    """One eighth of the square plate under equal pulls on all four edges.
    method 'rotate': the diagonal nodes' unknowns are rotated and u.n = 0 is
    prescribed; 'spring': a spring of stiffness k (N/mm^3) along n holds the diagonal."""
    m = eighth_mesh(p, nr, nt, HA, HB)
    fixed = {(A, 1): 0.0 for A in m.side_nodes("axis")}
    rotated, springs = None, None
    if method == "rotate":
        rotated = {A: DIAG_N for A in m.side_nodes("diagonal")}
        fixed.update({(A, 0): 0.0 for A in m.side_nodes("diagonal")})
    else:
        springs = [("diagonal", k * np.outer(DIAG_N, DIAG_N))]
    return solve(m, fixed, traction={"outer": lambda x, y: (HS, 0.0)}, rotated=rotated, springs=springs)


def quarter_solution(p=2, nr=10, nt=6):
    m = quarter_mesh(p, nr, nt, HA, HB)
    fixed = {(A, 1): 0.0 for A in m.side_nodes("axis0")}
    fixed.update({(A, 0): 0.0 for A in m.side_nodes("axis1")})
    return solve(m, fixed, traction={"outer0": lambda x, y: (HS, 0.0), "outer1": lambda x, y: (0.0, HS)})


def hoop_at_hole(s, k=6):
    """(angle in degrees, sigma_theta_theta) along the hole, from the elements on it."""
    out = []
    for e, sd in s.mesh.sides["hole"]:
        for eta in np.linspace(-1, 1, k + 1):
            xi, et = EL.SIDE_PARAM[sd](eta)
            N, _ = EL.shape(s.mesh.p, xi, et)
            x = N @ s.mesh.X[s.mesh.IEN[e]]
            sg = stress(s, e, xi, et)
            th = math.atan2(x[1], x[0])
            out.append((math.degrees(th), sg[0] * math.sin(th) ** 2 + sg[1] * math.cos(th) ** 2
                        - 2 * sg[2] * math.sin(th) * math.cos(th)))
    return np.array(sorted(out))


def normal_gap(s):
    """The largest |u.n| on the diagonal, relative to the largest displacement."""
    diag = s.mesh.side_nodes("diagonal")
    return max(abs(s.U[A] @ DIAG_N) for A in diag) / np.abs(s.U).max()


def fig_inclined():
    fig = plt.figure(figsize=(12.6, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 0.95, 1.05, 1.05], wspace=0.5)
    # (a) the plate, its four planes of symmetry, and one eighth
    ax = fig.add_subplot(gs[0, 0])
    b, a = HB, HA * 2.4                              # the hole drawn larger in the sketch
    ax.add_patch(Rectangle((-b, -b), 2 * b, 2 * b, fc=BODY, ec=INK, lw=1.2, zorder=1))
    ax.add_patch(Polygon([(a, 0), (b, 0), (b, b), (a / math.sqrt(2), a / math.sqrt(2))], closed=True,
                         fc="#b9cfe6", ec="none", zorder=2))
    ax.add_patch(Circle((0, 0), a, fc="white", ec=INK, lw=1.2, zorder=3))
    for P, Q in (((-b - 12, 0), (b + 12, 0)), ((0, -b - 12), (0, b + 12)), ((-b - 8, -b - 8), (b + 8, b + 8)),
                 ((-b - 8, b + 8), (b + 8, -b - 8))):
        ax.plot(*zip(P, Q), color="0.35", lw=0.9, ls="-.", zorder=4)
    for v in np.linspace(-0.7 * b, 0.7 * b, 4):
        for (P, Q) in (((b + 3, v), (b + 30, v)), ((-b - 3, v), (-b - 30, v)), ((v, b + 3), (v, b + 30)), ((v, -b - 3), (v, -b - 30))):
            EF.arrow(ax, P, Q, LOADCLR, 1.2, 8)
    ax.text(0.62 * b, 0.22 * b, "one\neighth", fontsize=9.5, ha="center", va="center", zorder=5)
    ax.set_xlim(-b - 36, b + 36)
    ax.set_ylim(-b - 36, b + 36)
    EF.clean(ax)
    ax.set_title("(a) Equal pulls on all four edges:\nfour planes of symmetry", fontsize=10.5, loc="left")
    # (b) the eighth, with the diagonal held along its normal
    ax = fig.add_subplot(gs[0, 1])
    s = eighth_solution()
    sc = 0.1 * HB / np.abs(s.U).max()
    EF.draw_mesh(ax, s.mesh, color="0.75", lw=0.5, z=1)
    EF.draw_mesh(ax, s.mesh, U=s.U, scale=sc, color=REACTCLR, lw=0.9, z=3)
    ax.plot([0, HB + 6], [0, HB + 6], color="0.35", lw=1.0, ls="-.", zorder=2)
    for A in s.mesh.side_nodes("diagonal"):
        ax.plot(*(s.mesh.X[A] + sc * s.U[A]), "o", ms=3, color=REACTCLR, zorder=4)
    for A in s.mesh.side_nodes("axis")[::2]:
        EF._support(ax, *s.mesh.X[A], 1, 5.5, outward=(0, -1))
    for A in s.mesh.side_nodes("diagonal")[::2]:
        P = s.mesh.X[A]
        apex = P + 1.2 * DIAG_N
        base = apex + 5.5 * DIAG_N
        perp = np.array([-DIAG_N[1], DIAG_N[0]])
        ax.add_patch(Polygon([apex, base + 3.0 * perp, base - 3.0 * perp], closed=True, fc=BDRYCLR, ec=BDRYCLR, zorder=8))
    for y in np.linspace(8, HB - 8, 5):
        EF.arrow(ax, (HB + 2, y), (HB + 16, y), LOADCLR, 1.2, 8)
    ax.set_xlim(-6, HB + 22)
    ax.set_ylim(-12, HB + 14)
    EF.clean(ax)
    ax.set_title("(b) The eighth, deformed: $u_1 = 0$ on\nthe axis, $\\underline{u}\\cdot\\underline{n} = 0$ on the diagonal",
                 fontsize=10.5, loc="left")
    # (c) the stress around the hole, from the quarter and from the eighth
    ax = fig.add_subplot(gs[0, 2])
    hq, he = hoop_at_hole(quarter_solution()), hoop_at_hole(s)
    ax.plot(hq[:, 0], hq[:, 1], "-", color=REACTCLR, lw=2.2, label="quarter, Cartesian supports")
    ax.plot(he[::3, 0], he[::3, 1], "o", color=LOADCLR, ms=5, label="eighth, rotated unknowns")
    ax.axhline(2 * HS, color="0.4", lw=1.0, ls="--")
    ax.text(88, 2 * HS - 2, "$2s$, infinite plate", fontsize=9, ha="right", va="top", color="0.3")
    ax.set_xlim(0, 90)
    ax.set_ylim(80, 110)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    EF._graph(ax, "angle around the hole (degrees)", "$\\sigma_{\\theta\\theta}$ at the hole (MPa)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("(c) The stress around the hole", fontsize=10.5, loc="left")
    # (d) a stiff spring along n in place of the exact condition
    ax = fig.add_subplot(gs[0, 3])
    ratios = 10.0 ** np.arange(-2, 6)
    gaps = [normal_gap(eighth_solution("spring", r * E_STEEL / HA)) for r in ratios]
    ax.loglog(ratios, gaps, "o-", color=LOADCLR, lw=1.8, ms=5)
    EF._graph(ax, "spring stiffness $k$, as a multiple of $E/a$", "largest $|\\underline{u}\\cdot\\underline{n}|$ on the diagonal,\nrelative to the largest $|\\underline{u}|$")
    ax.set_title("(d) A stiff spring along $\\underline{n}$ instead", fontsize=10.5, loc="left")
    return fig


# ------------------------------------------------------------------ Saint-Venant's principle
def cantilever_pair(p=2, ny=8):
    """The cantilever of the elasticity appendices with the exact end conditions,
    and with its left end clamped instead."""
    m = EL.cantilever_mesh(p, ny)
    out = []
    for clamp in (False, True):
        dr, tr = EL.cantilever_bc(E_STEEL, NU_STEEL, clamp=clamp)
        out.append(EL.solve(m, E_STEEL, NU_STEEL, "stress", t=EL.CANTILEVER["t"], dirichlet=dr, traction=tr))
    return m, out


BLK_L, BLK_H, BLK_P, BLK_W = 200.0, 100.0, 10000.0, 50.0   # block length, depth (mm), load (N), patch width (mm)


def block_peak(ny, patch):
    """The largest |sigma_11| at a Gauss point of a block on rollers, pushed down at the
    middle of its top by BLK_P, either at one node or spread over a patch."""
    m = rect_mesh(BLK_L, BLK_H, 2 * ny, ny)
    fixed = {(A, 1): 0.0 for A in m.side_nodes("bottom")}
    fixed[(node_at(m, BLK_L / 2, 0), 0)] = 0.0
    if patch:
        q = BLK_P / (BLK_W * T_PLATE)
        s = solve(m, fixed, traction={"top": lambda x, y: (0.0, -q if abs(x - BLK_L / 2) < BLK_W / 2 - 1e-9 else 0.0)})
    else:
        s = solve(m, fixed, point={node_at(m, BLK_L / 2, BLK_H): (0.0, -BLK_P)})
    return np.abs(gauss_stresses(s)[:, 1]).max()


def fig_saint_venant():
    fig = plt.figure(figsize=(12.6, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.42)
    m, (ex, cl) = cantilever_pair()
    C = EL.CANTILEVER
    sref = C["P"] * C["L"] * (C["D"] / 2) / (C["t"] * C["D"] ** 3 / 12)
    diff = lambda e, xi, eta: EL.at(cl, e, xi, eta)["sig"][0] - EL.at(ex, e, xi, eta)["sig"][0]
    # (a) the change in sigma_00 near the wall
    ax = fig.add_subplot(gs[0, 0])
    from matplotlib.colors import TwoSlopeNorm
    lev = np.linspace(-0.12, 0.12, 25) * sref
    cf = EF.fine_field(ax, m, diff, k=6, cmap="RdBu_r", levels=lev, norm=TwoSlopeNorm(0, lev[0], lev[-1]))
    EF.draw_mesh(ax, m, color="0.6", lw=0.15)
    for y in np.linspace(-C["D"] / 2, C["D"] / 2, 5):
        pin(ax, 0, y, (1, 0), 7.0)
    for xd in (1, 2, 3):
        ax.plot([xd * C["D"]] * 2, [-C["D"] / 2 - 6, C["D"] / 2 + 6], color="0.3", lw=0.9, ls=":")
        ax.text(xd * C["D"], C["D"] / 2 + 9, "%d depth%s" % (xd, "" if xd == 1 else "s"), fontsize=8.5, ha="center", va="bottom")
    ax.set_xlim(-20, C["L"] + 5)
    ax.set_ylim(-C["D"] / 2 - 12, C["D"] / 2 + 26)
    EF.clean(ax)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04, ticks=np.linspace(-0.1, 0.1, 5) * sref, extend="both")
    cb.ax.set_xticklabels(["%+.0f%%" % v if v else "0" for v in np.linspace(-10, 10, 5)])
    cb.set_label("change in $\\sigma_{00}$, as a percentage of the largest bending stress", fontsize=9)
    ax.set_title("(a) The change in $\\sigma_{00}$ when the left end\nis clamped instead of given the exact conditions",
                 fontsize=10.5, loc="left")
    # (b) the largest change at each section
    ax = fig.add_subplot(gs[0, 1])
    pts = []
    for e in range(m.nel):
        for xi in np.linspace(-1, 1, 7):
            for eta in np.linspace(-1, 1, 7):
                a_ex, a_cl = EL.at(ex, e, xi, eta), EL.at(cl, e, xi, eta)
                pts.append((a_ex["x"][0], np.abs(a_cl["sig"] - a_ex["sig"]).max() / sref))
    pts = np.array(pts)
    bins = np.linspace(0, C["L"], 81)
    k = np.digitize(pts[:, 0], bins) - 1
    xb = 0.5 * (bins[:-1] + bins[1:])
    yb = np.array([pts[k == j, 1].max() if np.any(k == j) else np.nan for j in range(len(xb))])
    ax.semilogy(xb / C["D"], yb, "-", color=LOADCLR, lw=1.8)
    ax.axvline(1.0, color="0.4", lw=0.9, ls=":")
    ax.set_xlim(0, C["L"] / C["D"])
    ax.set_ylim(1e-6, 1.5)
    EF._graph(ax, "distance from the wall, in depths of the beam", "largest change in any stress component,\nrelative to the largest bending stress")
    ax.set_title("(b) The change dies out within about\none depth of the wall", fontsize=10.5, loc="left")
    # (c) a point load and a patch load as the mesh is refined
    ax = fig.add_subplot(gs[0, 2])
    nys = (4, 8, 16, 32)
    for patch, col, mk, lab in ((True, REACTCLR, "s", "spread over 50 mm"), (False, LOADCLR, "o", "at a single node")):
        ax.loglog(nys, [block_peak(ny, patch) for ny in nys], mk + "-", color=col, lw=1.8, ms=6, label=lab)
    ax.set_xticks(nys)
    ax.set_xticklabels([str(n) for n in nys])
    ax.set_yticks([20, 50, 100, 200])
    ax.set_yticklabels(["20", "50", "100", "200"])
    ax.minorticks_off()
    EF._graph(ax, "elements through the depth", "largest $|\\sigma_{11}|$ at a Gauss point (MPa)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(c) A 10 kN load on a block, at a point\nor spread out, as the mesh is refined", fontsize=10.5, loc="left")
    return fig


# ------------------------------------------------------------------ one element on an inclined roller
ONE_L, ONE_P = 100.0, 10000.0          # side of the square element (mm), and the load on its top (N)


def one_element(alpha_deg=30.0):
    """A square element, pinned at local node 0, with local node 1 on a roller on a
    surface inclined at alpha, and a downward load ONE_P shared by nodes 2 and 3.
    With one element, the global and local numbers are taken to be the same.
    The unknowns of node 1 are rotated to (u_n, u_t), with n = (sin a, -cos a) the
    outward normal of the surface and t = (-n1, n0) = (cos a, sin a)."""
    m = rect_mesh(ONE_L, ONE_L, 1, 1)
    X = m.X[m.IEN[0]]                                  # local nodes, by the stride rule
    D = EL.dmatrix(E_STEEL, NU_STEEL, "stress")
    ke, _ = EL.element_arrays(m, 0, D, T_PLATE)
    a = math.radians(alpha_deg)
    n = np.array([math.sin(a), -math.cos(a)])
    R = np.array([[n[0], n[1]], [-n[1], n[0]]])
    T = np.eye(8)
    T[2:4, 2:4] = R.T
    kr = T.T @ ke @ T
    F = np.zeros(8)
    F[5] = F[7] = -ONE_P / 2
    fr = T.T @ F
    free = [3, 4, 5, 6, 7]                             # p_hat 0, 1 (node 0) and 2 (u_n of node 1) are prescribed
    d = np.zeros(8)
    d[free] = np.linalg.solve(kr[np.ix_(free, free)], fr[free])
    return dict(X=X, ke=ke, T=T, kr=kr, K=kr[np.ix_(free, free)], F=fr[free], d=d, U=(T @ d).reshape(4, 2),
                R=kr @ d - fr, n=n, t=np.array([-n[1], n[0]]), alpha=alpha_deg)


def fig_one_element(alpha_deg=30.0):
    s = one_element(alpha_deg)
    X, n, t = s["X"], s["n"], s["t"]
    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1], wspace=0.22)
    rs = 11.0                                           # size of the roller

    def inclined_roller(ax, P):
        """An open triangle on two wheels at P, on a long surface along t with hatching below it."""
        base = P + 0.8 * rs * n
        v = t
        ax.add_patch(Polygon([P, base + 0.5 * rs * v, base - 0.5 * rs * v], closed=True, fc="white", ec=BDRYCLR, lw=1.3, zorder=8))
        for k in (-0.27, 0.27):
            ax.add_patch(Circle(base + k * rs * v + 0.16 * rs * n, 0.14 * rs, fc="white", ec=BDRYCLR, lw=1.1, zorder=8))
        g = base + 0.31 * rs * n
        Q0, Q1 = g - 4.5 * rs * t, g + 4.5 * rs * t
        ax.plot([Q0[0], Q1[0]], [Q0[1], Q1[1]], color=BDRYCLR, lw=1.6, zorder=7)
        for k in np.linspace(0.04, 0.96, 12):
            q = Q0 + k * (Q1 - Q0)
            r = q + 0.55 * rs * n - 0.45 * rs * t
            ax.plot([q[0], r[0]], [q[1], r[1]], color=BDRYCLR, lw=0.8, zorder=7)
        return g

    def element(ax, P, **kw):
        ax.add_patch(Polygon(P[[0, 1, 3, 2]], closed=True, **kw))

    # (a) the setup
    ax = fig.add_subplot(gs[0, 0])
    element(ax, X, fc=BODY, ec=INK, lw=1.4, zorder=2)
    pin(ax, 0, 0, (0, 1), 13)
    g = inclined_roller(ax, X[1])
    for a in (2, 3):
        EF.arrow(ax, X[a] + (0, 34), X[a] + (0, 3), LOADCLR, 1.6, 12)
    ax.text(50, 138, "5 kN at nodes 2 and 3", fontsize=9.5, ha="center", va="bottom", color=LOADCLR)
    for a in range(4):
        ax.text(*(X[a] + (8 if a in (0, 2) else -8, 8 if a in (0, 1) else -8)), "$a = %d$" % a, fontsize=11,
                ha="left" if a in (0, 2) else "right", va="bottom" if a in (0, 1) else "top", zorder=6)
    EF.arrow(ax, X[1], X[1] + 34 * n, "#7c3aed", 1.8, 12, z=9)
    EF.arrow(ax, X[1], X[1] + 34 * t, GHOST, 1.8, 12, z=9)
    ax.text(*(X[1] + 34 * n + (9, -2)), "$\\underline{n}$", fontsize=13, color="#7c3aed", ha="left", va="center")
    ax.text(*(X[1] + 41 * t), "$\\underline{t}$", fontsize=13, color=GHOST, ha="center", va="center")
    ang = np.radians(np.linspace(0, alpha_deg, 20))
    ax.plot(g[0] + 44 * np.cos(ang), g[1] + 44 * np.sin(ang), color="0.4", lw=0.9)
    ax.plot([g[0], g[0] + 52], [g[1], g[1]], color="0.4", lw=0.9, ls=":")
    ax.text(g[0] + 50 * math.cos(math.radians(alpha_deg / 2)), g[1] + 50 * math.sin(math.radians(alpha_deg / 2)),
            "$\\alpha$", fontsize=12, va="center")
    EF.axes_glyph(ax, (-58, 60), 20, fs=10.5)
    ax.set_xlim(-65, 185)
    ax.set_ylim(-70, 152)
    EF.clean(ax)
    ax.set_title("(a) Node 0 pinned; node 1 on a roller on a\nsurface at the angle $\\alpha$, with its own $\\underline{n}$ and $\\underline{t}$",
                 fontsize=10.5, loc="left")
    # (b) the blocks of T^T k^e T
    ax = fig.add_subplot(gs[0, 1])
    for a in range(4):
        for b in range(4):
            rot = 1 in (a, b)
            ax.add_patch(Rectangle((2 * b, -2 * a - 2), 2, 2, fc="#fde3c8" if rot else "white", ec=INK, lw=1.4, zorder=1))
            txt = ("$\\mathbf{R}\\mathbf{k}_{11}\\mathbf{R}^T$" if a == b == 1 else
                   "$\\mathbf{R}\\mathbf{k}_{1%d}$" % b if a == 1 else
                   "$\\mathbf{k}_{%d1}\\mathbf{R}^T$" % a if b == 1 else "$\\mathbf{k}_{%d%d}$" % (a, b))
            ax.text(2 * b + 1, -2 * a - 1, txt, fontsize=9 if rot else 9.5, ha="center", va="center", zorder=3,
                    color=INK if rot else "0.45")
    for k in range(3):                                  # the prescribed rows and columns
        ax.add_patch(Rectangle((k, -8), 1, 8, fc="0.5", alpha=0.16, ec="none", zorder=2))
        ax.add_patch(Rectangle((0, -k - 1), 8, 1, fc="0.5", alpha=0.16, ec="none", zorder=2))
    names = ["$(0, 0)$", "$(0, 1)$", "$(1, n)$", "$(1, t)$", "$(2, 0)$", "$(2, 1)$", "$(3, 0)$", "$(3, 1)$"]
    for p in range(8):
        ax.text(-0.25, -p - 0.5, "%d   %s" % (p, names[p]), fontsize=9, ha="right", va="center",
                color="0.55" if p < 3 else INK)
        ax.text(p + 0.5, 0.2, "%d" % p, fontsize=9, ha="center", va="bottom", color="0.55" if p < 3 else INK)
    ax.text(-0.25, 0.2, "$\\hat{p}$     $(a, i)$", fontsize=9, ha="right", va="bottom")
    ax.text(4, -8.5, "gray: the rows and columns of prescribed components,\nwhich the ID array removes, leaving a $5 \\times 5$ system",
            fontsize=9.5, ha="center", va="top", color="0.35")
    ax.set_xlim(-3.4, 8.2)
    ax.set_ylim(-10.3, 1.1)
    EF.clean(ax)
    ax.set_title("(b) $\\mathbf{T}^T\\mathbf{k}^e\\mathbf{T}$: only the blocks in the row and column\nof node 1 change (shaded)",
                 fontsize=10.5, loc="left")
    # (c) the deformed element
    ax = fig.add_subplot(gs[0, 2])
    U = s["U"]
    sc = 16.0 / np.abs(U).max()
    Xd = X + sc * U
    element(ax, X, fill=False, ec="0.45", lw=1.1, ls="--", zorder=2)
    element(ax, Xd, fc=BODY, ec=INK, lw=1.4, zorder=3, alpha=0.9)
    inclined_roller(ax, Xd[1])
    pin(ax, 0, 0, (0, 1), 13)
    Rn = s["R"][2]                                      # the force on the element along n
    fdir = np.sign(Rn) * n                             # the direction in which the surface pushes
    EF.arrow(ax, Xd[1] + 3 * fdir, Xd[1] + 38 * fdir, "#0d9488", 2.0, 12, z=9)
    ax.text(Xd[1][0] + 14, Xd[1][1] + 30, "%.2f kN, normal\nto the surface" % abs(Rn / 1000), fontsize=9.5,
            color="#0d9488", ha="left", va="center")
    for a in (2, 3):
        EF.arrow(ax, Xd[a] + (0, 30), Xd[a] + (0, 3), LOADCLR, 1.4, 10)
    ax.set_xlim(-30, 200)
    ax.set_ylim(-70, 152)
    EF.clean(ax)
    ax.set_title("(c) The deformed element, magnified: node 1\nslides along the surface, held only along $\\underline{n}$",
                 fontsize=10.5, loc="left")
    return fig
