"""
Steady heat conduction in a rectangle -- the 2D Poisson problem
    -kappa (u_,11 + u_,22) = f   on  ]0, W[ x ]0, H[
-- for the battery-cell pages.  Two solvers:

``series``  the exact solution, as a Fourier series, for the arrangement used
            as the worked example: Dirichlet u = g(x2) on the cooled side
            x1 = 0, insulated sides x1 = W and x2 = 0, and heat flux h = h_tab
            into the cell through tab patches on the top edge x2 = H (h = 0 on
            the rest of the top).  f is uniform and g(x2) = g0 + dg x2 / H.

``grid``    bilinear finite elements on a uniform grid, solved by conjugate
            gradients, for any assignment of whole edges to Gamma_g.  Used for
            the figures that compare boundary arrangements.  heat2d.js is a
            port of this solver for the interactive explorer -- change the two
            together.

Only numpy is used, so the pages render on the CI image.
"""

import math

import numpy as np

EDGES = ("left", "right", "bottom", "top")


# --------------------------------------------------------------------- series
def _chunks(npts, nmodes, budget=2_000_000):
    step = max(1, min(nmodes, budget // max(npts, 1)))
    for s in range(0, nmodes, step):
        yield slice(s, min(nmodes, s + step))


def series(x1, x2, W, H, kappa, f, g0, h_tab, tabs, dg=0.0, M=400, N=200):
    """Exact u and its first and second derivatives at the points (x1, x2).

    Returns a dict with arrays u, u1, u2, u11, u22 of the broadcast shape.
    Three pieces are superposed; each satisfies its own share of the data:
      bulk  (f / kappa)(W x1 - x1^2 / 2)            -- f, insulated at x1 = W
      tabs  sum_m (h_m/kappa) sin(mu_m x1) cosh(mu_m x2) / (mu_m sinh(mu_m H))
      cool  gbar + sum_n g_n cos(nu_n x2) cosh(nu_n (W - x1)) / cosh(nu_n W)
    with mu_m = (m - 1/2) pi / W and nu_n = n pi / H.
    """
    x1, x2 = np.broadcast_arrays(np.asarray(x1, float), np.asarray(x2, float))
    shape = x1.shape
    X1, X2 = x1.ravel(), x2.ravel()
    out = {k: np.zeros(X1.size) for k in ("u", "u1", "u2", "u11", "u22")}

    # bulk generation: the 1D wall solution, rotated to run along x1
    out["u"] += f / kappa * (W * X1 - X1 ** 2 / 2)
    out["u1"] += f / kappa * (W - X1)
    out["u11"] += -f / kappa

    # tab flux on the top edge
    m = np.arange(1, M + 1)
    mu = (m - 0.5) * math.pi / W
    hm = sum(h_tab * (np.cos(mu * a) - np.cos(mu * b)) / mu for a, b in tabs) * 2 / W
    for s in _chunks(X1.size, M):
        lam, c = mu[s], hm[s] / kappa
        e = 1 - np.exp(-2 * lam * H)
        ch = (np.exp(lam * (X2[:, None] - H)) + np.exp(-lam * (X2[:, None] + H))) / e
        sh = (np.exp(lam * (X2[:, None] - H)) - np.exp(-lam * (X2[:, None] + H))) / e
        sn, cs = np.sin(lam * X1[:, None]), np.cos(lam * X1[:, None])
        out["u"] += (c / lam * sn * ch).sum(1)
        out["u1"] += (c * cs * ch).sum(1)
        out["u2"] += (c * sn * sh).sum(1)
        out["u11"] += -(c * lam * sn * ch).sum(1)
        out["u22"] += (c * lam * sn * ch).sum(1)

    # coolant temperature along the cooled side
    out["u"] += g0 + dg / 2
    if dg != 0.0:
        n = np.arange(1, N + 1)
        nu = n * math.pi / H
        gn = np.where(n % 2 == 1, -4 * dg / (n * math.pi) ** 2, 0.0)
        for s in _chunks(X1.size, N):
            lam, c = nu[s], gn[s]
            e = 1 + np.exp(-2 * lam * W)
            C = (np.exp(-lam * X1[:, None]) + np.exp(-lam * (2 * W - X1[:, None]))) / e
            S = (np.exp(-lam * X1[:, None]) - np.exp(-lam * (2 * W - X1[:, None]))) / e
            cs, sn = np.cos(lam * X2[:, None]), np.sin(lam * X2[:, None])
            out["u"] += (c * cs * C).sum(1)
            out["u1"] += -(c * lam * cs * S).sum(1)
            out["u2"] += -(c * lam * sn * C).sum(1)
            out["u11"] += (c * lam ** 2 * cs * C).sum(1)
            out["u22"] += -(c * lam ** 2 * cs * C).sum(1)

    return {k: v.reshape(shape) for k, v in out.items()}


def tab_flux(x1, h_tab, tabs):
    """h on the top edge: h_tab over each tab patch, zero between them."""
    x1 = np.asarray(x1, float)
    h = np.zeros_like(x1)
    for a, b in tabs:
        h[(x1 >= a) & (x1 <= b)] = h_tab
    return h


# ----------------------------------------------------------------------- grid
def grid(W, H, nx, ny, kappa, f, g, dirichlet, h_edge, tol=1e-11, maxit=50000):
    """Bilinear finite elements on an nx-by-ny grid of rectangles.

    f(x1, x2)         heat generated per unit volume (vectorized callable)
    g(x1, x2)         temperature on Gamma_g
    dirichlet         the edges in Gamma_g, a subset of EDGES; a corner that
                      touches any Dirichlet edge is a Dirichlet node
    h_edge(edge, s)   heat flux INTO the body along a Neumann edge, as a
                      function of position s along it (x1 on bottom/top, x2 on
                      left/right)

    Returns a dict: X, Y, U (nodal grids, row index = x2), the Dirichlet mask,
    the heat made, the heat in through Gamma_h, and the heat out through each
    Dirichlet edge -- all per unit depth (W/m) -- plus the CG iteration count.
    """
    hx, hy = W / nx, H / ny
    xs, ys = np.linspace(0, W, nx + 1), np.linspace(0, H, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    a, b = hy / hx, hx / hy
    Ke = kappa / 6 * np.array([
        [2 * a + 2 * b, -2 * a + b, -a - b, a - 2 * b],
        [-2 * a + b, 2 * a + 2 * b, a - 2 * b, -a - b],
        [-a - b, a - 2 * b, 2 * a + 2 * b, -2 * a + b],
        [a - 2 * b, -a - b, -2 * a + b, 2 * a + 2 * b]])
    # the four corners of every element, counter-clockwise from lower left
    sl = [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)),
          (slice(1, None), slice(1, None)), (slice(1, None), slice(0, -1))]

    def Kmul(v):
        out = np.zeros_like(v)
        loc = [v[s] for s in sl]
        for i in range(4):
            out[sl[i]] += sum(Ke[i, j] * loc[j] for j in range(4))
        return out

    # f by 2x2 Gauss quadrature on each element
    F = np.zeros_like(X)
    made = 0.0
    gp = (-1 / math.sqrt(3), 1 / math.sqrt(3))
    xc, yc = X[:-1, :-1] + hx / 2, Y[:-1, :-1] + hy / 2
    for gx in gp:
        for gy in gp:
            fq = f(xc + gx * hx / 2, yc + gy * hy / 2) * hx * hy / 4
            made += fq.sum()
            N = ((1 - gx) * (1 - gy) / 4, (1 + gx) * (1 - gy) / 4,
                 (1 + gx) * (1 + gy) / 4, (1 - gx) * (1 + gy) / 4)
            for i in range(4):
                F[sl[i]] += N[i] * fq

    # Neumann edges, 2-point Gauss on each segment
    D = np.zeros(X.shape, bool)
    idx = {"left": (slice(None), 0), "right": (slice(None), -1),
           "bottom": (0, slice(None)), "top": (-1, slice(None))}
    heat_in = 0.0
    for edge in EDGES:
        if edge in dirichlet:
            D[idx[edge]] = True
            continue
        s, hs = (xs, hx) if edge in ("bottom", "top") else (ys, hy)
        load = np.zeros(s.size)
        for gq in gp:
            hq = h_edge(edge, 0.5 * (s[:-1] + s[1:]) + gq * hs / 2) * hs / 2
            heat_in += hq.sum()
            load[:-1] += (1 - gq) / 2 * hq
            load[1:] += (1 + gq) / 2 * hq
        F[idx[edge]] += load

    U = np.zeros_like(X)
    U[D] = g(X, Y)[D]
    r = F - Kmul(U)
    r[D] = 0.0
    dvec = np.zeros_like(X)
    for i in range(4):
        dvec[sl[i]] += Ke[i, i]
    z = r / dvec
    p = z.copy()
    rz = np.sum(r * z)
    scale = np.sqrt(np.sum(F[~D] ** 2)) + np.sqrt(np.sum(r ** 2)) + 1e-30
    it = 0
    if D.any():
        for it in range(1, maxit + 1):
            Ap = Kmul(p)
            Ap[D] = 0.0
            alpha = rz / np.sum(p * Ap)
            U += alpha * p
            r -= alpha * Ap
            if np.sqrt(np.sum(r ** 2)) < tol * scale:
                break
            z = r / dvec
            rz_new = np.sum(r * z)
            p = z + rz_new / rz * p
            rz = rz_new
    # heat into the body at each Dirichlet node is (K U - F); out is minus that.
    # A corner shared by two Dirichlet edges is counted once, on the edge that
    # comes first in EDGES (as in heat2d.js).
    R = -(Kmul(U) - F)
    seen = np.zeros(X.shape, bool)
    heat_out = {}
    for edge in EDGES:
        if edge not in dirichlet:
            continue
        mask = np.zeros(X.shape, bool)
        mask[idx[edge]] = True
        mask &= ~seen
        heat_out[edge] = R[mask].sum()
        seen |= mask
    return dict(X=X, Y=Y, U=U, D=D, made=made, heat_in=heat_in,
                heat_out=heat_out, iterations=it)
