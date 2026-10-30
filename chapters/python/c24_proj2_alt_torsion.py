"""
Saint-Venant torsion of an I-section, computed with the Project 2 code.

This repeats the analyses of c24_torsion.py (the torsion page of Class 24)
with the finite element code of Project 2, through c24_proj2_alt.  The
dimensions, the mesh layout, the closed-form checks, and the figures come from
c24_torsion.py.  The stress function satisfies -lap(Phi) = 2 in a quarter of
the section with Phi = 0 on its outer surface, and J = 2 * integral of Phi.

Run this file to compare the two codes.  To draw a figure of the page from a
Project 2 model m, pass view(m) to the figure function, for example
T.fig_solution(view(build())).
"""

import sys

import numpy as np

import c24_proj2_alt as A
import c24_torsion as T

ONE = A.conductivity(1.0)
TWO = lambda x, y: 2.0
ZERO = lambda x, y: 0.0
view = A.view


def build(p=2, nref=1, bf=T.BF, d=T.DEPTH, tf=T.TF, tw=T.TW, r0=1.3):
    grid, xw, yf = T.quarter_mesh(1, nref, bf, d, tf, tw, r0)
    bm = A.basis_mesh(grid, p)
    edge = A.sides_where(bm, lambda x, y, n: not (abs(x) < T.TOL or abs(y) < T.TOL))
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(edge), ZERO)]
    sol = A.solve(bm, ONE, TWO, boundaries)
    out = dict(bm=bm, sol=sol, corner=(xw, yf), dims=(bf, d, tf, tw),
               notes_mesh=lambda: T.quarter_mesh(p, nref, bf, d, tf, tw, r0)[0])
    out["J"] = 4 * 2 * A.integrate(sol)
    return out


def warping(p=2, nref=1, bf=T.BF, d=T.DEPTH, tf=T.TF, tw=T.TW, r0=1.3):
    """The warping function omega: -lap(omega) = 0, grad(omega).n = x2 n1 - x1 n2
    on the outer surface, and omega = 0 on the axes of symmetry.  Each part of
    the outer surface with one normal direction is a separate Neumann object."""
    grid, xw, yf = T.quarter_mesh(1, nref, bf, d, tf, tw, r0)
    bm = A.basis_mesh(grid, p)
    on_axis = lambda x, y: abs(x) < T.TOL or abs(y) < T.TOL
    axes = A.sides_where(bm, lambda x, y, n: on_axis(x, y))
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(axes), ZERO)]
    for nrm, h in (((1.0, 0.0), lambda x, y: y), ((-1.0, 0.0), lambda x, y: -y),
                   ((0.0, 1.0), lambda x, y: -x), ((0.0, -1.0), lambda x, y: x)):
        sides = A.sides_where(bm, lambda x, y, n, nrm=nrm: not on_axis(x, y) and n == nrm)
        if sides:
            boundaries.append(A.bnd.NeumannBoundaryCondition(A.nodes_of(sides), h))
    sol = A.solve(bm, ONE, ZERO, boundaries)
    grad2 = 4 * 2 * A.energy(sol)                  # int |grad omega|^2 over the whole section
    return dict(bm=bm, sol=sol, dims=(bf, d, tf, tw), grad2=grad2,
                J=T.polar_moment(bf, d, tf, tw) - grad2,
                notes_mesh=lambda: T.quarter_mesh(p, nref, bf, d, tf, tw, r0)[0])


def omega_at(model, x, y):
    """omega anywhere in the section, from the quarter (odd in x and in y)."""
    e, xi, eta = A.locate(model["bm"], abs(x), abs(y))
    return float(np.sign(x) * np.sign(y) * A.value(model["sol"], e, xi, eta))


def build_rectangle(a, b, p=2, nref=1):
    """Quarter of an a x b rectangle, for the benchmark."""
    bm = A.basis_mesh(T.rectangle_mesh(a, b, 1, nref), p)
    edge = A.sides_where(bm, lambda x, y, n: abs(x - a / 2) < T.TOL or abs(y - b / 2) < T.TOL)
    sol = A.solve(bm, ONE, TWO, [A.bnd.DirichletBoundaryCondition(A.nodes_of(edge), ZERO)])
    return dict(bm=bm, sol=sol, J=4 * 2 * A.integrate(sol),
                notes_mesh=lambda: T.rectangle_mesh(a, b, p, nref))


def grad_mag(model, x, y):
    e, xi, eta = A.locate(model["bm"], x, y)
    return float(np.hypot(*A.gradient(model["sol"], e, xi, eta)))


def stresses(model, torque=T.TORQUE):
    """Shear stresses (MPa) at the middle of the flange face, the web face,
    and at the element corner next to the re-entrant corner."""
    bf, d, tf, tw = model["dims"]
    J = model["J"]
    xw, yf = model["corner"]
    t_fl = grad_mag(model, bf / 4, d / 2 - 1e-9)
    t_web = grad_mag(model, tw / 2 - 1e-9, yf / 2)
    # largest |grad Phi| at the corner of the elements that meet the re-entrant corner
    bm, sol = model["bm"], model["sol"]
    best = 0.0
    for e in range(len(bm.IEN)):
        (x0, y0), (x1, y1) = bm.lo[e], bm.hi[e]
        for xi, x in ((-1, x0), (1, x1)):
            for eta, y in ((-1, y0), (1, y1)):
                if abs(x - xw) < 1e-9 and abs(y - yf) < 1e-9:
                    best = max(best, float(np.hypot(*A.gradient(sol, e, xi, eta))))
    k = torque / J
    return dict(flange=k * t_fl, web=k * t_web, corner=k * best,
                flange_thin=torque * tf / J, web_thin=torque * tw / J)


def compare(nref=1):
    """The quantities of the torsion page from both codes, on one mesh."""
    rows = []
    for p in (1, 2):
        a, b = T.build(p, nref), build(p, nref)
        rows.append((f"J (mm^4), p = {p}", a["J"], b["J"]))
    sa, sb = T.stresses(a), stresses(b)
    rows += [(f"stress at {k} (MPa), p = 2", sa[k], sb[k]) for k in ("flange", "web", "corner")]
    wa, wb = T.warping(2, nref), warping(2, nref)
    rows.append(("J from the warping function (mm^4)", wa["J"], wb["J"]))
    rows.append(("omega at (40, 94) mm", T.omega_at(wa, 40.0, 94.0), omega_at(wb, 40.0, 94.0)))
    ra, rb = T.build_rectangle(100.0, 12.0, 2, nref), build_rectangle(100.0, 12.0, 2, nref)
    rows.append(("J of a 100 x 12 mm rectangle (mm^4)", ra["J"], rb["J"]))
    A.report(f"Torsion of the I-section, nref = {nref}", rows)


if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
