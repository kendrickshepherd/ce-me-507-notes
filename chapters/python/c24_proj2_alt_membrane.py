"""
A pressurized membrane with a rigid center boss, computed with the Project 2 code.

This repeats the analyses of c24_membrane.py (the membrane page of Class 24)
with the finite element code of Project 2, through c24_proj2_alt.  The
dimensions, the mesh layout, the series solution, and the figures come from
c24_membrane.py.  The deflection satisfies -S lap(w) = p with w = 0 on the
frame.  Every node of the boss receives the same equation number in the ID
array, so the boss moves as one rigid piece (see c24_proj2_alt.tied_nodes).

Run this file to compare the two codes.  To draw a figure of the page from
Project 2 models, pass them through view, for example
Mb.fig_shapes(view(build(b=0.0)), view(build())).
"""

import sys

import numpy as np

import c24_proj2_alt as A
import c24_membrane as Mb

view = A.view


def build(p=2, nref=1, b=Mb.B_BOSS, a=Mb.A_SIDE, S=Mb.TENSION, pr=Mb.PRESS, r0=1.15):
    bm = A.basis_mesh(Mb.make_mesh(1, nref, b, a, r0), p)
    TOL = Mb.TOL
    frame = A.sides_where(bm, lambda x, y, n: abs(x - a / 2) < TOL or abs(y - a / 2) < TOL)
    X = bm.X
    boss = [node for node in range(len(X)) if X[node, 0] <= b / 2 + TOL and X[node, 1] <= b / 2 + TOL] if b > 0 else []
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(frame), lambda x, y: 0.0)]
    sol = A.solve(bm, A.conductivity(S), lambda x, y: pr, boundaries, tied=[boss] if boss else [])
    out = dict(bm=bm, sol=sol, a=a, b=b, S=S, p=pr, frame=frame,
               notes_mesh=lambda: Mb.make_mesh(p, nref, b, a, r0))
    out["w0"] = sol.d[np.argmin(np.hypot(X[:, 0], X[:, 1]))]
    out["volume"] = 4 * A.integrate(sol)
    out["w_avg"] = out["volume"] / a ** 2
    out["frame_force"] = -4 * A.reactions(sol)[A.nodes_of(frame)].sum()
    return out


def edge_force(model, ys):
    """S dw/dn along the frame edge x = a/2 (N/m): the vertical force per unit
    length that the membrane applies to the frame."""
    a = model["a"]
    out = []
    for y in ys:
        e, _, eta = A.locate(model["bm"], a / 2 - 1e-15, y)
        out.append(-model["S"] * A.gradient(model["sol"], e, 1.0, eta)[0])
    return np.array(out)


def profile(model, xs):
    out = []
    for x in xs:
        e, xi, _ = A.locate(model["bm"], x, 0.0)
        out.append(A.value(model["sol"], e, xi, -1.0))
    return np.array(out)


def compare(nref=1):
    """The quantities of the membrane page from both codes, on one mesh."""
    rows = []
    for b, name in ((0.0, "no boss"), (Mb.B_BOSS, "boss")):
        x, y = Mb.build(2, nref, b=b), build(2, nref, b=b)
        rows += [(f"{name}: center deflection (um)", x["w0"] * 1e6, y["w0"] * 1e6),
                 (f"{name}: volume (1e-12 m^3)", x["volume"] * 1e12, y["volume"] * 1e12),
                 (f"{name}: force on the frame (mN)", x["frame_force"] * 1e3, y["frame_force"] * 1e3),
                 (f"{name}: S dw/dn at the middle of an edge (N/m)",
                  Mb.edge_force(x, [0.0])[0], edge_force(y, [0.0])[0]),
                 (f"{name}: deflection at x = 0.3 mm (um)",
                  Mb.profile(x, [3e-4])[0] * 1e6, profile(y, [3e-4])[0] * 1e6)]
    A.report(f"Membrane with a rigid boss, p = 2, nref = {nref}", rows)


if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
