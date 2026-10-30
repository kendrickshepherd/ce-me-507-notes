"""
The convergence demonstration of the main Class 24 page, computed with the
Project 2 code.

This repeats c24_main.py with the finite element code of Project 2, through
c24_proj2_alt: torsion of a square bar, -lap(Phi) = 2 with Phi = 0 on the
boundary, for which the torsion constant and the largest shear stress are
known exactly.  The exact values, the slopes, and the figure come from
c24_main.py, for example CM.fig_convergence(convergence()).

Run this file to compare the two codes.
"""

import c24_proj2_alt as A
import c24_main as CM


def square(p, n):
    """Torsion of the unit square, from a quarter with n x n elements of degree p."""
    bm = A.basis_mesh(CM.square_mesh(1, n), p)
    edge = A.sides_where(bm, lambda x, y, nn: abs(x - 0.5) < 1e-12 or abs(y - 0.5) < 1e-12)
    boundaries = [A.bnd.DirichletBoundaryCondition(A.nodes_of(edge), lambda x, y: 0.0)]
    sol = A.solve(bm, A.conductivity(1.0), lambda x, y: 2.0, boundaries)
    J = 8 * A.integrate(sol)
    e, xi, eta = A.locate(bm, 0.5, 0.0)
    tau = float(abs(A.gradient(sol, e, 1.0, -1.0)[0]))
    return J, tau, 0.5 / n


def convergence():
    Je, te = CM.exact_square()
    rows = []
    for p in (1, 2):
        for n in (2, 4, 8, 16, 32):
            J, tau, h = square(p, n)
            rows.append((p, n, h, abs(J - Je) / Je, abs(tau - te) / te))
    return rows


def compare():
    """The convergence table of the main page from both codes."""
    a, b = CM.convergence(), convergence()
    rows = []
    for ra, rb in zip(a, b):
        rows += [(f"p = {ra[0]}, n = {ra[1]:>2}: error in J", ra[3], rb[3]),
                 (f"p = {ra[0]}, n = {ra[1]:>2}: error in the stress", ra[4], rb[4])]
    for p in (1, 2):
        rows += [(f"slope of the error in J, p = {p}", CM.slope(a, p, 3), CM.slope(b, p, 3)),
                 (f"slope of the error in the stress, p = {p}", CM.slope(a, p, 4), CM.slope(b, p, 4))]
    A.report("Torsion of a square bar (main Class 24 page)", rows)


if __name__ == "__main__":
    compare()
