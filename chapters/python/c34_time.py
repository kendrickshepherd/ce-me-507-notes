"""
Figures and numbers for Class 34: time stepping for the semi-discrete heat
equation and for elastodynamics.

Heat: M d' + K d = F on the unit interval or the unit square, with the
temperature held at zero on the boundary, linear (1D) or bilinear (2D) elements
on uniform meshes, and the consistent mass matrix, M_AB = integral of N_A N_B.
The diffusivity is 1.  The generalized trapezoidal family (alpha method):

    (M + alpha dt K) d_{n+1} = (M - (1 - alpha) dt K) d_n + dt F_{n+alpha}.

Elastodynamics: M d'' + C d' + K d = F for a bar fixed at both ends, wave speed
1, linear elements, solved by the Newmark family.

The widget on the page uses the same meshes and data; change the two together.
Only numpy is used, so the page renders on the CI image.
"""

import math

import numpy as np
import matplotlib.pyplot as plt

import elast_figs as EF
from notes_style import INK

COLORS = {"forward Euler": "#c0392b", "Crank–Nicolson": "#1f6feb", "backward Euler": "#0d9488"}
ALPHAS = {"forward Euler": 0.0, "Crank–Nicolson": 0.5, "backward Euler": 1.0}
N1, N2 = 20, 16                      # elements per side: 1D and 2D
T_END = 0.1


# ---------------------------------------------------------------- matrices
def matrices_1d(n=N1):
    """M and K of the interior nodes of n linear elements on [0, 1]."""
    h = 1.0 / n
    m = n - 1
    K = (np.diag(2 * np.ones(m)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)) / h
    M = (np.diag(4 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) * h / 6
    return M, K, np.linspace(h, 1 - h, m)


def matrices_2d(n=N2):
    """M and K of the interior nodes of n x n bilinear elements on the unit square
    (tensor products of the 1D matrices), nodes numbered along x first."""
    M1, K1, x = matrices_1d(n)
    return np.kron(M1, M1), np.kron(M1, K1) + np.kron(K1, M1), x


def lambda_max_1d(n=N1):
    """The largest eigenvalue of K psi = lambda M psi for linear elements."""
    h = 1.0 / n
    c = math.cos(math.pi * h)
    return 6 / h ** 2 * (1 + c) / (2 - c)


def lambda_max(dim):
    return lambda_max_1d(N1) if dim == 1 else 2 * lambda_max_1d(N2)


# ------------------------------------------------------- exact solutions
def block_edges(n):
    """The hot block covers the nodes from 0.3 to 0.7 and half an element beyond."""
    h = 1.0 / n
    lo = math.ceil(0.3 / h - 1e-9) * h
    hi = math.floor(0.7 / h + 1e-9) * h
    return lo - h / 2, hi + h / 2


def exact_1d(kind, x, t, n, terms=400):
    """The solution of u_t = u_xx on (0, 1), u = 0 at both ends."""
    if kind == "smooth":
        return np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
    a, b = block_edges(n)
    k = np.arange(1, terms + 1)
    bk = 2 / (k * np.pi) * (np.cos(k * np.pi * a) - np.cos(k * np.pi * b))
    return (bk * np.exp(-(k * np.pi) ** 2 * t)) @ np.sin(np.outer(k, x) * np.pi)


def initial(kind, x, n):
    if kind == "smooth":
        return np.sin(np.pi * x)
    a, b = block_edges(n)
    return ((x > a) & (x < b)).astype(float)


def exact(dim, kind, t):
    n = N1 if dim == 1 else N2
    x = np.linspace(1.0 / n, 1 - 1.0 / n, n - 1)
    u = exact_1d(kind, x, t, n)
    return u if dim == 1 else np.kron(u, u)       # separable: u(x, y) = u1(x) u1(y)


def setup(dim, kind):
    if dim == 1:
        M, K, x = matrices_1d(N1)
        d0 = initial(kind, x, N1)
    else:
        M, K, x = matrices_2d(N2)
        d0 = np.kron(initial(kind, x, N2), initial(kind, x, N2))
    return M, K, d0


# ------------------------------------------------------------ the alpha method
def alpha_method(M, K, d0, alpha, dt, nsteps):
    A = M + alpha * dt * K
    B = M - (1 - alpha) * dt * K
    d = [d0.copy()]
    for _ in range(nsteps):
        d.append(np.linalg.solve(A, B @ d[-1]))
    return np.array(d)


def amplification(alpha, z):
    """A(lambda dt) for the alpha method."""
    return (1 - (1 - alpha) * z) / (1 + alpha * z)


def semi_discrete_exact(M, K, d0, t):
    """The solution of M d' + K d = 0 exact in time, by the eigenvectors of K psi = lambda M psi."""
    L = np.linalg.cholesky(M)
    Li = np.linalg.inv(L)
    lam, Q = np.linalg.eigh(Li @ K @ Li.T)
    psi = Li.T @ Q                                 # M-orthonormal eigenvectors
    y0 = psi.T @ M @ d0
    return psi @ (y0 * np.exp(-lam * t))


def final_error(dim, kind, alpha, dt, against="analytical", T=T_END):
    M, K, d0 = setup(dim, kind)
    n = int(round(T / dt))
    d = alpha_method(M, K, d0, alpha, dt, n)[-1]
    ref = exact(dim, kind, n * dt) if against == "analytical" else semi_discrete_exact(M, K, d0, n * dt)
    return np.abs(d - ref).max()


# ---------------------------------------------------------------- figures
def fig_amplification():
    z = np.logspace(-2, 3, 600)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.axhspan(-1, 1, color="#eef3f8", zorder=0)
    for name, a in ALPHAS.items():
        ax.plot(z, amplification(a, z), color=COLORS[name], lw=2.0, label="%s, $\\alpha$ = %g" % (name, a))
    ax.plot(z, np.exp(-z), color=INK, lw=1.4, ls=(0, (5, 3)), label="exact, $e^{-\\lambda\\Delta t}$")
    for v in (-1, 0, 1):
        ax.axhline(v, color="0.6", lw=0.7)
    ax.axvline(2, color="#c0392b", lw=0.8, ls=":")
    ax.text(2.15, -2.6, "$\\lambda\\Delta t = 2$", color="#c0392b", fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(-3, 1.3)
    EF._graph(ax, "$\\lambda\\,\\Delta t$", "amplification factor $A$")
    ax.text(0.012, -0.85, "stable band, $|A| \\leq 1$", fontsize=9, color="0.4")
    ax.legend(fontsize=8.6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    fig.tight_layout()
    return fig


def fig_convergence(dim=1):
    """The error at t = 0.1 against dt, for the smooth initial temperature:
    (a) against the analytical solution, (b) against the exact-in-time solution
    of the semi-discrete equations."""
    lmax = lambda_max(dim)
    dts = T_END / np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    out = {}
    for ax, against in zip(axes, ("analytical", "semi-discrete")):
        for name in ("Crank–Nicolson", "backward Euler", "forward Euler"):       # forward Euler last, on top
            a = ALPHAS[name]
            ok = dts if a >= 0.5 else dts[dts * lmax < 2 / (1 - 2 * a)]
            e = np.array([final_error(dim, "smooth", a, dt, against) for dt in ok])
            out[(against, name)] = (ok, e)
            if name == "forward Euler":
                ax.loglog(ok, e, "o--", color=COLORS[name], lw=1.6, ms=6, mfc="white", mew=1.5, label=name, zorder=5)
            else:
                ax.loglog(ok, e, "o-", color=COLORS[name], lw=1.8, ms=4.5, label=name)
        EF._graph(ax, "time step $\\Delta t$", "largest nodal error at $t$ = 0.1" if against == "analytical" else "")
        ax.minorticks_off()
    axes[0].set_title("(a) Against the analytical solution", fontsize=10.5, loc="left")
    axes[1].set_title("(b) Against the semi-discrete solution, exact in time", fontsize=10.5, loc="left")
    for ax in axes:
        x0 = 2e-3
        ax.plot([x0, x0 * 10], [3e-6, 3e-5], color="0.4", lw=0.9)
        ax.text(x0 * 11, 3e-5, "slope 1", fontsize=8.5, color="0.4", va="center")
        ax.plot([x0, x0 * 10], [1.5e-8, 1.5e-6], color="0.4", lw=0.9)
        ax.text(x0 * 11, 1.5e-6, "slope 2", fontsize=8.5, color="0.4", va="center")
    axes[1].legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig, out


def fig_snapshots(r_fe=(0.9, 1.1), r_big=20.0, steps_fe=15):
    """The hot block in 1D: (a) forward Euler just below and just above its limit,
    (b) Crank-Nicolson and (c) backward Euler with a large step."""
    M, K, d0 = setup(1, "block")
    x = np.linspace(1.0 / N1, 1 - 1.0 / N1, N1 - 1)
    xx = np.linspace(0, 1, 401)
    dtc = 2 / lambda_max(1)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), sharey=True)
    ax = axes[0]
    for r, ls in zip(r_fe, ("-", "--")):
        d = alpha_method(M, K, d0, 0.0, r * dtc, steps_fe)[-1]
        ax.plot(np.r_[0, x, 1], np.r_[0, d, 0], ls, marker="o", ms=3, color=COLORS["forward Euler"], lw=1.6,
                label="$\\Delta t$ = %g $\\Delta t_{cr}$" % r)
    t = steps_fe * r_fe[0] * dtc
    ax.plot(xx, exact_1d("block", xx, t, N1), color=INK, lw=1.2, ls=(0, (5, 3)), label="exact")
    ax.set_title("(a) Forward Euler, %d steps" % steps_fe, fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    out = {}
    for ax, name in ((axes[1], "Crank–Nicolson"), (axes[2], "backward Euler")):
        dt = r_big * dtc
        d = alpha_method(M, K, d0, ALPHAS[name], dt, 1)[-1]
        out[name] = d
        ax.plot(np.r_[0, x, 1], np.r_[0, d, 0], "-o", ms=3, color=COLORS[name], lw=1.6, label="one step of $\\Delta t$ = %g $\\Delta t_{cr}$" % r_big)
        ax.plot(xx, exact_1d("block", xx, dt, N1), color=INK, lw=1.2, ls=(0, (5, 3)), label="exact")
        ax.plot(np.r_[0, x, 1], np.r_[0, d0, 0], color="0.7", lw=0.9, label="initial")
        ax.set_title("(%s) %s" % ("bc"[name == "backward Euler"], name), fontsize=10.5, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    for ax in axes:
        ax.set_ylim(-0.7, 1.5)
        ax.axhline(0, color="0.8", lw=0.7)
        EF._graph(ax, "$x$", "temperature" if ax is axes[0] else "")
    fig.tight_layout()
    return fig, out, dtc


# ------------------------------------------------------------ elastodynamics
NB = 20                              # elements in the bar


def bar_matrices(n=NB):
    """M and K of the interior nodes of a bar fixed at both ends, wave speed 1."""
    return matrices_1d(n)


def newmark(M, C, K, d0, v0, beta, gamma, dt, nsteps):
    """The Newmark family with F = 0; returns the displacements at every step."""
    a = np.linalg.solve(M, -C @ v0 - K @ d0)
    d, v = d0.copy(), v0.copy()
    lhs = M + gamma * dt * C + beta * dt * dt * K
    out = [d.copy()]
    for _ in range(nsteps):
        dp = d + dt * v + dt * dt / 2 * (1 - 2 * beta) * a          # predictors
        vp = v + dt * (1 - gamma) * a
        a = np.linalg.solve(lhs, -C @ vp - K @ dp)
        d = dp + beta * dt * dt * a
        v = vp + gamma * dt * a
        out.append(d.copy())
    return np.array(out)


def plucked(x):
    """A bar pulled at its middle into a triangle of height 1."""
    return 1 - np.abs(2 * x - 1)


def plucked_exact_mid(t, terms=400):
    """The exact midpoint displacement of the plucked bar, by its sine series."""
    k = np.arange(1, terms + 1, 2)
    bk = 8 / (np.pi * k) ** 2 * np.sin(k * np.pi / 2)
    return (bk[None, :] * np.cos(np.pi * np.outer(t, k)) * np.sin(k * np.pi / 2)).sum(axis=1)


def fig_dynamics():
    """(a) Central difference just below and above its limit, (b) average acceleration
    and Newmark with numerical damping at a large step, (c) Rayleigh damping."""
    M, K, x = bar_matrices()
    mid = (NB - 1) // 2
    wmax = math.sqrt(lambda_max_1d(NB))
    dtc = 2 / wmax
    Z = np.zeros_like(K)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.2), sharex=True)
    T = 6.0
    tt = np.linspace(0, T, 1200)
    # (a) the first mode, central difference
    d0, v0 = np.sin(np.pi * x), np.zeros_like(x)
    ax = axes[0]
    ax.plot(tt, np.cos(np.pi * tt), color=INK, lw=1.2, ls=(0, (5, 3)), label="exact")
    for r, ls in ((0.95, "-"), (1.05, "-")):
        dt = r * dtc
        n = int(T / dt)
        d = newmark(M, Z, K, d0, v0, 0.0, 0.5, dt, n)[:, mid]
        lab = "central difference, $\\Delta t$ = %.2f $\\Delta t_{cr}$" % r
        ax.plot(np.arange(n + 1) * dt, np.clip(d, -3, 3), color=COLORS["forward Euler"] if r > 1 else COLORS["Crank–Nicolson"],
                lw=1.4 if r < 1 else 0.8, label=lab)
    ax.set_ylim(-2.2, 2.9)
    ax.set_title("(a) The first mode, central difference: stable below its limit, unstable just above it", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.95, edgecolor="none", loc="upper right", ncol=3)
    # (b) a large step
    ax = axes[1]
    ax.plot(tt, np.cos(np.pi * tt), color=INK, lw=1.2, ls=(0, (5, 3)), label="exact")
    dt = 0.25
    n = int(T / dt)
    for (beta, gamma), col, lab in (((0.25, 0.5), COLORS["Crank–Nicolson"], "average acceleration, $\\beta$ = 1/4, $\\gamma$ = 1/2"),
                                     ((0.3025, 0.6), COLORS["backward Euler"], "$\\beta$ = 0.3025, $\\gamma$ = 0.6")):
        d = newmark(M, Z, K, d0, v0, beta, gamma, dt, n)[:, mid]
        ax.plot(np.arange(n + 1) * dt, d, "-o", ms=3, color=col, lw=1.4, label=lab)
    ax.set_ylim(-1.4, 1.6)
    ax.set_title("(b) The first mode with $\\Delta t$ = 0.25, eight steps per period: period elongation and numerical damping", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=3)
    # (c) the plucked bar, with and without Rayleigh damping
    ax = axes[2]
    d0 = plucked(x)
    ax.plot(tt, plucked_exact_mid(tt), color=INK, lw=1.2, ls=(0, (5, 3)), label="exact, undamped")
    dt = 0.02
    n = int(T / dt)
    for a1, col, lab in ((0.0, COLORS["Crank–Nicolson"], "average acceleration, undamped"),
                         (0.004, COLORS["backward Euler"], "with Rayleigh damping, $a_1$ = 0.004")):
        C = a1 * K
        d = newmark(M, C, K, d0, v0, 0.25, 0.5, dt, n)[:, mid]
        ax.plot(np.arange(n + 1) * dt, d, color=col, lw=1.3, label=lab)
    ax.set_ylim(-1.4, 1.6)
    ax.set_title("(c) A plucked bar: the high modes ripple, and stiffness-proportional damping removes them", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=3)
    for ax in axes:
        ax.axhline(0, color="0.85", lw=0.7)
        EF._graph(ax, "time $t$" if ax is axes[2] else "", "midpoint displacement")
    fig.tight_layout()
    return fig, dtc, wmax


def fig_damping():
    """(a) The damping ratio of Rayleigh damping against frequency; (b) the spectral
    radius of three Newmark methods against omega dt."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6))
    ax = axes[0]
    w = np.linspace(0.5, 70, 400)
    a0, a1 = 0.6, 0.004
    ax.plot(w, a0 / (2 * w), color="#c0392b", lw=1.4, ls="--", label="mass part, $a_0/(2\\omega)$")
    ax.plot(w, a1 * w / 2, color="#1f6feb", lw=1.4, ls="--", label="stiffness part, $a_1\\omega/2$")
    ax.plot(w, a0 / (2 * w) + a1 * w / 2, color=INK, lw=2.0, label="total, $\\zeta$")
    ax.set_ylim(0, 0.2)
    EF._graph(ax, "natural frequency $\\omega$ (rad per unit time)", "damping ratio $\\zeta$")
    ax.legend(fontsize=8.6, frameon=False, loc="upper center")
    ax.set_title("(a) Rayleigh damping, $a_0$ = %g, $a_1$ = %g" % (a0, a1), fontsize=10.5, loc="left")
    ax = axes[1]
    z = np.logspace(-1, 3, 400)
    ax.axhline(1, color="0.75", lw=0.7, zorder=0)
    for (beta, gamma), col, lab, ls in (((0.0, 0.5), COLORS["forward Euler"], "central difference", "-"),
                                         ((0.25, 0.5), COLORS["Crank–Nicolson"], "average acceleration", (0, (6, 3))),
                                         ((0.3025, 0.6), COLORS["backward Euler"], "$\\beta$ = 0.3025, $\\gamma$ = 0.6", "-")):
        rho = [spectral_radius(beta, gamma, zz) for zz in z]
        ax.plot(z, np.minimum(rho, 3), color=col, lw=2.2, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_ylim(0, 2.0)
    EF._graph(ax, "$\\omega\\,\\Delta t$", "spectral radius")
    ax.legend(fontsize=8.6, frameon=False, loc="lower left")
    ax.set_title("(b) Growth per step of an undamped mode", fontsize=10.5, loc="left")
    fig.tight_layout()
    return fig


def spectral_radius(beta, gamma, z):
    """The spectral radius of one Newmark step for an undamped mode with omega dt = z,
    found by applying the step to a unit displacement and to a unit velocity."""
    w2 = z * z                       # omega^2 with dt = 1
    D = 1 + beta * w2
    cols = []
    for d0, v0 in ((1.0, 0.0), (0.0, 1.0)):
        a0 = -w2 * d0
        dp = d0 + v0 + (1 - 2 * beta) * a0 / 2
        vp = v0 + (1 - gamma) * a0
        a1 = -w2 * dp / D
        cols.append([dp + beta * a1, vp + gamma * a1])
    return float(np.abs(np.linalg.eigvals(np.array(cols).T)).max())
