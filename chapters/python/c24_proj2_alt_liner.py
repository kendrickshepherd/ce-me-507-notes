"""
Diffusion through a landfill liner beneath a tear in its geomembrane,
computed with the Project 2 code.

This repeats the analyses of c24_liner.py (the diffusion page of Class 24)
with the finite element code of Project 2, through c24_proj2_alt.  The
material data, the mesh layout, the exact effective width, and the figures
come from c24_liner.py.  The concentration satisfies -div(n D grad c) = 0,
with c = c0 at the tear and c = 0 at the drainage layer.

Run this file to compare the two codes.  To draw a figure of the page from a
Project 2 model m, pass view(m) to the figure function, for example
D.fig_solution(view(build(2, 2))).
"""

import sys

import numpy as np

import c24_proj2_alt as A
import c24_liner as D

view = A.view


def build(p=2, nref=1, a=D.A_TEAR, x1=D.X1, r0=1.35):
    bm = A.basis_mesh(D.make_mesh(1, nref, a, x1, r0), p)
    tear = A.sides_where(bm, lambda x, y, n: abs(y - D.LCLAY) < 1e-9 and x < a / 2 and n[1] > 0)
    drain = A.sides_where(bm, lambda x, y, n: abs(y) < 1e-9 and n[1] < 0)
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(tear), lambda x, y: D.C0),
                  A.bnd.DirichletBoundaryCondition(A.nodes_of(drain), lambda x, y: 0.0)]
    sol = A.solve(bm, A.conductivity(D.NDIFF), lambda x, y: 0.0, boundaries)
    out = dict(bm=bm, sol=sol, a=a, x1=x1, notes_mesh=lambda: D.make_mesh(p, nref, a, x1, r0))
    # mass flux per meter length of the tear (both halves), kg/(s m)
    R = A.reactions(sol)
    out["Q_in"] = 2 * R[A.nodes_of(tear)].sum()
    out["Q_out"] = -2 * R[A.nodes_of(drain)].sum()
    out["a_eff"] = out["Q_in"] / D.flux_1d()
    return out


def base_flux(model, xs):
    """Flux leaving through the bottom of the clay, kg/(m^2 s)."""
    out = []
    for x in xs:
        e, xi, _ = A.locate(model["bm"], x, 1e-12)
        out.append(D.NDIFF * A.gradient(model["sol"], e, xi, -1.0)[1])
    return np.array(out)


def compare(nref=1):
    """The quantities of the diffusion page from both codes, on one mesh."""
    rows = []
    for a in (D.A_TEAR, 0.002, 2 * D.X1):
        x, y = D.build(2, nref, a=a), build(2, nref, a=a)
        rows += [(f"tear {a * 1e3:g} mm: Q in (g/yr per m)", x["Q_in"] * D.YEAR * 1e3, y["Q_in"] * D.YEAR * 1e3),
                 (f"tear {a * 1e3:g} mm: Q out (g/yr per m)", x["Q_out"] * D.YEAR * 1e3, y["Q_out"] * D.YEAR * 1e3),
                 (f"tear {a * 1e3:g} mm: effective width (mm)", x["a_eff"] * 1e3, y["a_eff"] * 1e3)]
    x, y = D.build(2, nref), build(2, nref)
    rows += [(f"flux at the base, x = {s:g} m (g/(m^2 yr))", fa * D.YEAR * 1e3, fb * D.YEAR * 1e3)
             for s, fa, fb in zip((0.0, 0.5), D.base_flux(x, [0.0, 0.5]), base_flux(y, [0.0, 0.5]))]
    A.report(f"Diffusion through the liner, p = 2, nref = {nref}", rows)


if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
