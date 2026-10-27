"""
The convergence demonstration on the main Class 24 page: torsion of a square
bar, -lap(Phi) = 2 with Phi = 0 on the boundary, for which the torsion
constant and the largest shear stress are known exactly.
"""

import math

import numpy as np
import matplotlib.pyplot as plt

import c24_fem as F
from notes_style import INK, LOADCLR


def exact_square(a=1.0):
    """J and the largest |grad Phi| (at the middle of a side) of an a x a square."""
    s = sum(math.tanh(n * math.pi / 2) / n ** 5 for n in range(1, 400, 2))
    J = a ** 4 / 3 * (1 - 192 / math.pi ** 5 * s)
    t = sum(1.0 / (n ** 2 * math.cosh(n * math.pi / 2)) for n in range(1, 400, 2))
    tau = a * (1 - 8 / math.pi ** 2 * t)
    return J, tau


def square(p, n):
    """Quarter of the unit square with n x n elements of degree p."""
    xs = np.linspace(0, 0.5, n + 1)
    mesh = F.Mesh(xs, xs, p=p)
    edge = mesh.sides_where(lambda x, y, nn: abs(x - 0.5) < 1e-12 or abs(y - 0.5) < 1e-12)
    sol = F.solve(mesh, kappa=1.0, f=2.0, dirichlet=[(edge, 0.0)])
    J = 8 * F.integrate(sol)
    e, xi, eta = F.locate(mesh, 0.5, 0.0)
    tau = float(abs(F.gradient(sol, e, 1.0, -1.0)[0, 0]))
    return J, tau, 0.5 / n


def convergence():
    Je, te = exact_square()
    rows = []
    for p in (1, 2):
        for n in (2, 4, 8, 16, 32):
            J, tau, h = square(p, n)
            rows.append((p, n, h, abs(J - Je) / Je, abs(tau - te) / te))
    return rows


def slope(rows, p, col):
    r = [x for x in rows if x[0] == p]
    return math.log(r[-2][col] / r[-1][col]) / math.log(r[-2][2] / r[-1][2])


def fig_convergence(rows):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for p, mk in ((1, "o"), (2, "s")):
        r = [x for x in rows if x[0] == p]
        h = [x[2] for x in r]
        ax.loglog(h, [x[3] for x in r], mk + "-", color=INK, lw=1.6, ms=5,
                  label=f"torsion constant $J$ (an integral), $p$ = {p}")
        ax.loglog(h, [x[4] for x in r], mk + "--", color=LOADCLR, lw=1.6, ms=5,
                  label=f"largest shear stress (a derivative), $p$ = {p}")
    ax.invert_xaxis()
    hs = sorted({x[2] for x in rows}, reverse=True)
    ax.set_xticks(hs)
    ax.set_xticklabels([f"1/{round(1 / h)}" for h in hs])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("element size $h$ (side of the square = 1)")
    ax.set_ylabel("relative error")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="both", color="0.92", lw=0.6)
    fig.tight_layout()
    return fig
