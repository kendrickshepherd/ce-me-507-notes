"""
Seepage under a concrete gravity dam, computed with the Project 2 code.

This repeats the analyses of c24_seepage.py (the seepage page of Class 24)
with the finite element code of Project 2, through c24_proj2_alt.  The
dimensions, the mesh layout (including the slit along the cutoff), and the
figures come from c24_seepage.py.  The total head h satisfies
-div(K grad h) = 0 in the sand, with h = H1 upstream and h = H2 downstream.

Run this file to compare the two codes.  To draw a figure of the page from
Project 2 models, pass them through view, for example
S.fig_flownets([view(build(cut=c)) for c in (None, "up", "down")]).
"""

import sys

import numpy as np

import c24_proj2_alt as A
import c24_seepage as S

ZERO = lambda x, y: 0.0
view = A.view


def build(p=2, nref=1, cut=None, kappa=None, depth=S.D, side=S.LSIDE, r0=1.2):
    """Solve for the head.  cut is None, 'up', or 'down'."""
    bm = A.basis_mesh(S.make_mesh(1, nref, cut, depth, side, r0), p)
    TOL, B, WIN = S.TOL, S.B, S.WIN
    up = A.sides_where(bm, lambda x, y, n: abs(y) < TOL and x < 0 and n[1] > 0)
    dn = A.sides_where(bm, lambda x, y, n: abs(y) < TOL and x > B and n[1] > 0)
    base = A.sides_where(bm, lambda x, y, n: abs(y) < TOL and 0 < x < B and n[1] > 0)
    kap = S.K if kappa is None else kappa
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(up), lambda x, y: S.H1),
                  A.bnd.DirichletBoundaryCondition(A.nodes_of(dn), lambda x, y: S.H2)]
    sol = A.solve(bm, A.conductivity(kap), ZERO, boundaries)
    out = dict(bm=bm, sol=sol, cut=cut, kappa=kap, up=up, dn=dn, base=base, depth=depth, side=side,
               notes_mesh=lambda: S.make_mesh(p, nref, cut, depth, side, r0))
    R = A.reactions(sol)
    out["q"] = R[A.nodes_of(up)].sum()                   # m^3/s per m, into the sand
    out["q_out"] = -R[A.nodes_of(dn)].sum()
    out["U"] = S.GW * A.side_integral(sol, base)          # kN per m of dam
    win = A.sides_where(bm, lambda x, y, n: abs(y) < TOL and B < x < B + WIN and n[1] > 0)
    kv = kap if np.ndim(kap) == 0 else kap[1]
    out["kv"] = kv
    # average upward gradient over the window, two ways: by integrating the
    # computed gradient, and from the flow that crosses the window (psi)
    out["i_avg_grad"] = A.outflow(sol, win) / (kv * WIN)
    psi = stream_function(out)
    e, xi, _ = A.locate(bm, B + WIN, -1e-9)
    out["psi"] = psi
    out["i_avg"] = A.value(psi, e, xi, 1.0) / (kv * WIN)
    return out


def stream_function(model):
    """The stream function psi (m^3/s per m): psi = 0 on the dam base and the
    cutoff, and psi = q on the rock and the far sides.  For K = diag(k_h, k_v),
    psi satisfies div(diag(1/k_v, 1/k_h) grad psi) = 0."""
    bm, kap, q = model["bm"], model["kappa"], model["q"]
    kh, kv = (kap, kap) if np.ndim(kap) == 0 else (kap[0], kap[1])
    TOL, B = S.TOL, S.B
    xcut = {"up": 0.0, "down": B}.get(model["cut"])
    top = A.sides_where(bm, lambda x, y, n: (abs(y) < TOL and 0 < x < B and n[1] > 0)
                        or (xcut is not None and abs(x - xcut) < TOL))
    outer = A.sides_where(bm, lambda x, y, n: abs(y + model["depth"]) < TOL
                          or abs(x + model["side"]) < TOL or abs(x - B - model["side"]) < TOL)
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(top), ZERO),
                  A.bnd.DirichletBoundaryCondition(A.nodes_of(outer), lambda x, y: q)]
    return A.solve(bm, A.conductivity((1.0 / kv, 1.0 / kh)), ZERO, boundaries)


def surface_gradient(model, xs):
    """Upward exit gradient -dh/dy along the ground downstream of the toe."""
    out = []
    for x in xs:
        e, xi, _ = A.locate(model["bm"], x, -1e-9)
        out.append(-A.gradient(model["sol"], e, xi, 1.0)[1])
    return np.array(out)


def base_head(model, xs):
    out = []
    for x in xs:
        e, xi, _ = A.locate(model["bm"], x, -1e-9)
        out.append(A.value(model["sol"], e, xi, 1.0))
    return np.array(out)


def compare(nref=1):
    """The quantities of the seepage page from both codes, on one mesh."""
    rows = []
    for cut in (None, "up", "down"):
        a, b = S.build(2, nref, cut), build(2, nref, cut)
        name = {None: "no cutoff", "up": "upstream cutoff", "down": "downstream cutoff"}[cut]
        rows += [(f"{name}: q (m^3/day per m)", a["q"] * S.DAY, b["q"] * S.DAY),
                 (f"{name}: q out (m^3/day per m)", a["q_out"] * S.DAY, b["q_out"] * S.DAY),
                 (f"{name}: uplift U (kN/m)", a["U"], b["U"]),
                 (f"{name}: exit gradient from psi", a["i_avg"], b["i_avg"]),
                 (f"{name}: exit gradient from grad h", a["i_avg_grad"], b["i_avg_grad"])]
    xs = np.array([S.B + 0.3, S.B + 1.0])
    rows += [(f"exit gradient at x = B + {x - S.B:.1f} m", ga, gb)
             for x, ga, gb in zip(xs, S.surface_gradient(a, xs), surface_gradient(b, xs))]
    rows += [(f"head on the base at x = {x:.0f} m", ha, hb)
             for x, ha, hb in zip((5.0, 15.0), S.base_head(a, [5.0, 15.0]), base_head(b, [5.0, 15.0]))]
    a = S.build(2, nref, "up", kappa=(4 * S.K, S.K))
    b = build(2, nref, "up", kappa=(4 * S.K, S.K))
    rows += [("k_h = 4 k_v, upstream: q (m^3/day per m)", a["q"] * S.DAY, b["q"] * S.DAY),
             ("k_h = 4 k_v, upstream: exit gradient", a["i_avg"], b["i_avg"])]
    A.report(f"Seepage under the dam, p = 2, nref = {nref}", rows)


if __name__ == "__main__":
    compare(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
