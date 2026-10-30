#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:01:16 2026

@author: kendrickshepherd

The six figures of the seepage page of Class 24 (24_Seepage.qmd), drawn from
either the notes' solver or the Project 2 code.

How the script works
--------------------
1. Solve.  build() solves -div(K grad h) = 0 for the total head h in the sand
   under the dam, and then a second problem for the stream function psi,
   whose contours are the flow lines.  It returns a dictionary ("model") that
   holds the mesh, both solutions, and the quantities of interest:
       q           seepage under the dam, m^3/s per meter of dam, from the
                   reactions at the upstream ground surface (q_out is the
                   same quantity at the downstream surface; they should agree)
       U           uplift force on the base of the dam, kN per meter of dam:
                   gamma_w times the integral of h along the base
       i_avg       average upward exit gradient over the first WIN = 2 m
                   downstream of the toe, from psi (flow through the window
                   divided by K and by the window width)
       i_avg_grad  the same average, by integrating the computed gradient
2. Prepare.  The figure functions draw from the notes' mesh.  A Project 2
   model is converted with view(); a notes model needs nothing.
3. Draw.  Each fig_... function below returns a matplotlib figure, which
   save() writes to a PNG file.

The figure functions
--------------------
The six fig_... functions in this script are copies of the ones in
c24_seepage.py, with comments, so that you can edit them freely.  The
published page uses the originals in c24_seepage.py, so edit these copies,
not the originals.  With the page's data, these copies draw exactly the
page's figures.  Unlike the originals, they also follow changes to the data:
the dimension labels, the position of the cutoff wall, and the axis limits
of Figure 4 are computed from the data rather than typed in.

All six use the same few matplotlib ideas:
  * fig, ax = plt.subplots(...) makes a figure and its axes (one or more
    panels); figsize is in inches.
  * ax.add_patch(Rectangle(...)) or Polygon(...) draws a filled shape, for
    the soil, water, rock, and dam.
  * mtri.Triangulation(x, y, triangles) turns the finite element mesh into
    triangles that matplotlib can contour: every node is a vertex, and every
    element (every quarter of a biquadratic element) is split into two
    triangles by mesh.triangles().
  * ax.tricontourf(tri, values, levels=...) fills between contour levels;
    ax.tricontour(...) draws the contour lines.  The levels= argument decides
    which values are drawn, and so the spacing of the lines.
  * ax.plot(x, y, ...) draws lines and curves; ax.annotate("", xy=..., xytext=...,
    arrowprops=...) draws an arrow from xytext to xy.
  * ax.set_aspect("equal") keeps meters the same length in x and y, and
    ax.axis("off") hides the axes for drawings of the ground.

Changing the problem for class discussion
-----------------------------------------
The data are module constants in c24_seepage.py.  Change them by assignment
here, before the build() calls, and both the solve and these figures follow,
for example
    S.H1 = 15.0         reservoir level above the ground (m)
    S.H2 = 2.0          tailwater level above the ground (m)
    S.DCUT = 12.0       depth of the cutoff wall (m), between 0 and the depth
    S.K = 2.0e-5        hydraulic conductivity (m/s)
    S.WIN = 2.0         width of the window for the exit gradient (m)
    S.IC = 1.0          critical gradient, used for the factor of safety
Two exceptions:
  * The depth of the sand and the extent of the model upstream and
    downstream are default arguments of build(), fixed when the module is
    imported.  Pass them explicitly, build(..., depth=30.0, side=60.0), and
    also set S.D and S.LSIDE to the same values so that the plots show the
    whole layer.
  * The cutoff can only be at the heel (cut="up"), at the toe (cut="down"),
    or absent (cut=None).
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, "/Users/kendrickshepherd/Documents/Courses/ce-me-507-notes/chapters/python")

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # draw to files, not to the screen
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon, Rectangle

import c24_seepage as S                  # Seepage content: data, solver, and evaluation helpers

USE_PROJECT2 = False                     # False: the notes' solver (seconds); True: Project 2 (minutes)
NREF = 2                                 # the page uses 2; 1 is about four times faster
OUT = Path("seepage_figures")            # folder for the figures, inside the folder you run from

if USE_PROJECT2:
    import c24_proj2_alt_seepage as S2   # analyses solved by Project 2
    build = S2.build
    prepare = S2.view                    # converts a Project 2 model for the figure functions
else:
    build = S.build
    prepare = lambda model: model        # the notes' models need no conversion


def save(fig, name):                     # the settings the notes use for page figures
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")


def cutoff_x(model):
    """x of the cutoff wall of a model (0 at the heel, B at the toe), or None."""
    return {"up": 0.0, "down": S.B}.get(model["cut"])


# =============================================================================
# The figure functions (copies of those in c24_seepage.py, with comments)
# =============================================================================

def fig_setting(model):
    """Figure 1: the dam, reservoir, sand, rock, and cutoff, with computed flow lines.

    Only the blue lines in the sand come from the solution: contours of the
    stream function psi of the model passed in.  Everything else is drawn.

    To modify:
      * n_lines: the number of flow lines; they divide the flow q into
        n_lines + 1 equal parts.
      * The dam's shape is fixed: base B, crest 4.5 m wide at 15 m.  Change
        the dam = [...] polygon to draw another shape.
    """
    n_lines = 4
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    xl, xr, yb = -S.LSIDE, S.B + S.LSIDE, -S.D          # left, right, and bottom of the sand
    xc = cutoff_x(model)
    # sand, rock, and water, drawn first (zorder=0) so that lines lie on top
    ax.add_patch(Rectangle((xl, yb), xr - xl, S.D, facecolor=S.SAND, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((xl, yb - 4), xr - xl, 4, facecolor=S.ROCK, edgecolor="none",
                           hatch="xx", zorder=0))
    ax.add_patch(Rectangle((xl, 0), -xl, S.H1, facecolor=S.WATER, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((S.B, 0), S.LSIDE, S.H2, facecolor=S.WATER, edgecolor="none", zorder=0))
    # water surfaces, each with the usual small triangle symbol
    ax.plot([xl, 0], [S.H1, S.H1], color="#2b5fa8", lw=1.4)
    ax.plot([S.B, xr], [S.H2, S.H2], color="#2b5fa8", lw=1.4)
    for xw, yw in ((-20, S.H1), (S.B + 20, S.H2)):
        ax.plot([xw - 1.2, xw + 1.2, xw, xw - 1.2], [yw + 1.4, yw + 1.4, yw + 0.2, yw + 1.4],
                color="#2b5fa8", lw=1.0)
    # the dam (corners listed counterclockwise from the heel) and the cutoff wall
    dam = [(0, 0), (S.B, 0), (4.5, 15.0), (0, 15.0)]
    ax.add_patch(Polygon(dam, closed=True, facecolor=S.CONC, edgecolor=S.INK, lw=1.3, zorder=3))
    if xc is not None:
        ax.plot([xc, xc], [0, -S.DCUT], color=S.INK, lw=3.2, zorder=4, solid_capstyle="butt")
    # computed flow lines: contours of psi at q/5, 2q/5, ... (psi = 0 on the dam
    # base and wall, psi = q on the rock and the far sides)
    psi, q = model["psi"], model["q"]
    m = model["mesh"]
    tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
    ax.tricontour(tri, psi.u, levels=np.linspace(0, q, n_lines + 2)[1:-1], colors="#2b5fa8",
                  linewidths=1.0, linestyles="-", zorder=2)
    ax.annotate("", xy=(S.B + 6, -0.6), xytext=(S.B + 4.5, -3.2),
                arrowprops=dict(arrowstyle="-|>", color="#2b5fa8", lw=1.2), zorder=5)
    ax.plot([xl, xr], [0, 0], color="#8a6d3b", lw=1.2, zorder=1)          # the ground surface
    # labels (positions in meters, in the same coordinates as the drawing)
    t = dict(fontsize=10.5, color=S.INK)
    ax.text(-20, S.H1 / 2, "reservoir", ha="center", va="center", **t)
    ax.text(S.B + 22, S.H2 + 2.2, "tailwater", ha="center", va="bottom", **t)
    ax.text(6.0, 6.0, "concrete\ngravity dam", ha="center", va="center", fontsize=10, color=S.INK, zorder=5)
    ax.text(-26, -12, "sand foundation\nhydraulic conductivity $K$", ha="center", va="center", **t)
    ax.text(S.B + 30, yb - 2, "impermeable rock", ha="center", va="center", fontsize=10.5, color="#4a443a",
            bbox=dict(fc=S.ROCK, ec="none", pad=1.5))
    if xc is not None:
        ax.annotate("cutoff wall", xy=(xc, -6.5 if S.DCUT > 6.5 else -0.8 * S.DCUT),
                    xytext=(xc - 9, -16.5), ha="center", va="center", fontsize=10.5,
                    arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.9))
    ax.text(S.B + 7, -6.0, "seepage", ha="left", va="center", fontsize=10.5, color="#2b5fa8")

    # dimension lines: a two-headed arrow from (x0, y0) to (x1, y1), labeled at its middle
    def dim(x0, y0, x1, y1, lab, off=(0, 0), ha="center", va="center"):
        ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], lab, ha=ha, va=va, fontsize=10, color="0.25")
    dim(-37, 0, -37, S.H1, f"$H_1$ = {S.H1:g} m", off=(1.0, 0), ha="left")
    dim(S.B + 37, 0, S.B + 37, S.H2, f"$H_2$ = {S.H2:g} m", off=(-1.0, 3.2), ha="right")
    dim(0, 17.0, S.B, 17.0, f"$B$ = {S.B:g} m", off=(0, 1.1))
    if xc is not None:
        dim(xc - 3, 0, xc - 3, -S.DCUT, f"$d$ = {S.DCUT:g} m", off=(-0.8, 0), ha="right")
    dim(S.B + 37, 0, S.B + 37, -S.D, f"$D$ = {S.D:g} m", off=(-1.0, 0), ha="right")
    ax.set_xlim(xl - 1, xr + 1)
    ax.set_ylim(yb - 4.5, max(19.5, S.H1 + 3.0))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_domain(model):
    """Figure 2: (a) the domain and its boundary conditions, (b) the mesh.

    Panel (a) is a drawing in the colors of the notes: BDRYCLR (blue) for a
    known head and FLUXCLR (orange) for no flow.  Panel (b) draws every
    element of the model's mesh as a gray rectangle, so it shows exactly the
    mesh that was solved.

    To modify:
      * Pass a model built with another nref or p to show its mesh.
      * The text labels are in panel (a); positions are in meters.
    """
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.4, 6.2))
    xl, xr = -S.LSIDE, S.B + S.LSIDE
    xc = cutoff_x(model)
    ax.add_patch(Rectangle((xl, -S.D), xr - xl, S.D, facecolor="#f4f1ea", edgecolor="none"))
    lw = 3.2
    # known head (Dirichlet): the ground surface upstream and downstream of the dam
    ax.plot([xl, 0], [0, 0], color=S.BDRYCLR, lw=lw, solid_capstyle="butt")
    ax.plot([S.B, xr], [0, 0], color=S.BDRYCLR, lw=lw, solid_capstyle="butt")
    # no flow (zero Neumann): the dam base, the far sides, and the rock
    for seg in (([0, S.B], [0, 0]), ([xl, xl], [-S.D, 0]), ([xr, xr], [-S.D, 0]), ([xl, xr], [-S.D, -S.D])):
        ax.plot(*seg, color=S.FLUXCLR, lw=lw, solid_capstyle="butt")
    if xc is not None:                       # both faces of the cutoff, drawn slightly apart
        ax.plot([xc - 0.25, xc - 0.25], [0, -S.DCUT], color=S.FLUXCLR, lw=2.2)
        ax.plot([xc + 0.25, xc + 0.25], [0, -S.DCUT], color=S.FLUXCLR, lw=2.2)
    t = dict(fontsize=10.5)
    ax.text(-20, 1.0, f"$h = H_1$ = {S.H1:g} m", ha="center", va="bottom", color=S.BDRYCLR, **t)
    ax.text(S.B + 20, 1.0, f"$h = H_2$ = {S.H2:g} m", ha="center", va="bottom", color=S.BDRYCLR, **t)
    ax.text(S.B / 2, 1.0, "dam base: no flow", ha="center", va="bottom", color=S.FLUXCLR, **t)
    if xc is not None:
        ax.text(xc + 1.2, -S.DCUT / 2, "cutoff: no flow\non either face", ha="left", va="center",
                color=S.FLUXCLR, **t)
    ax.text((xl + xr) / 2, -S.D - 1.0, "rock: no flow", ha="center", va="top", color=S.FLUXCLR, **t)
    ax.text(xl - 1.0, -S.D / 2, "no flow", ha="right", va="center", color=S.FLUXCLR, rotation=90, **t)
    ax.text(xr + 1.0, -S.D / 2, "no flow", ha="left", va="center", color=S.FLUXCLR, rotation=90, **t)
    ax.text(-22, -11, r"$-\nabla\cdot(K\,\nabla h) = 0$", ha="center", va="center", fontsize=13, color=S.INK)
    ax.set_title("(a) The sand layer and its boundary conditions", fontsize=11.5, loc="left")
    for a in (ax, bx):
        a.set_xlim(xl - 4, xr + 4)
        a.set_ylim(-S.D - 3.5, 4.5)
        a.set_aspect("equal")
        a.axis("off")
    # (b) every element as a rectangle, from its center and size
    m = model["mesh"]
    for e in range(m.nel):
        (xm, ym), (hx, hy) = m.cent[e], m.size[e]
        bx.add_patch(Rectangle((xm - hx / 2, ym - hy / 2), hx, hy, facecolor="none",
                               edgecolor="0.55", lw=0.35))
    if xc is not None:
        bx.plot([xc, xc], [0, -S.DCUT], color=S.INK, lw=2.0)
    bx.set_title("(b) The biquadratic mesh, graded toward the heel, the toe, and the tip of the cutoff",
                 fontsize=11.5, loc="left")
    fig.tight_layout(h_pad=1.0)
    return fig


def fig_flownets(models, head_step=1, flow_step=1.0, flow_offset=1):
    """Figure 3: a flow net for each model, one panel each.

    Shading: the head h.  Dashed lines: equipotentials every head_step meters.
    Solid lines: flow lines, contours of psi/K every flow_step meters starting
    at flow_offset (psi/K has units of meters, because psi is a discharge per
    meter of dam and K is a velocity).  When head_step equals flow_step, the
    cells of the net are approximately square.

    To modify:
      * head_step, flow_step, flow_offset: the spacing and position of the lines.
        flow_offset=1.0 starts the flow lines at psi/K = 1 m.
      * cmap=S.HEADMAP ("Blues") sets the color map of the shading; alpha its
        transparency.
      * The title shows N_f/N_d, the number of flow channels (of width
        flow_step) for 10 head drops: q = K (H1 - H2) N_f/N_d.
    """
    titles = {None: "(a) No cutoff", "up": "(b) Cutoff at the heel (upstream)",
              "down": "(c) Cutoff at the toe (downstream)"}
    fig, axes = plt.subplots(len(models), 1, figsize=(10.4, 2.75 * len(models) + 0.4))
    for ax, md in zip(np.atleast_1d(axes), models):
        m, sol = md["mesh"], md["sol"]
        tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
        # 20 bands of color between the tailwater and reservoir heads
        ax.tricontourf(tri, sol.u, levels=np.linspace(S.H2, S.H1, 21), cmap=S.HEADMAP, alpha=0.75)
        ax.tricontour(tri, sol.u, levels=np.arange(S.H2 + head_step, S.H1, head_step),
                      colors="#1e3a5f", linewidths=0.8, linestyles="--")
        qK = md["q"] / S.K                       # the total flow, in meters of psi/K
        ax.tricontour(tri, md["psi"].u / S.K, levels=np.arange(flow_offset, qK, flow_step),
                      colors=S.INK, linewidths=1.0)
        ax.add_patch(Rectangle((0, 0), S.B, 1.6, facecolor=S.CONC, edgecolor=S.INK, lw=1.0))   # the dam
        xc = cutoff_x(md)
        if xc is not None:
            ax.plot([xc, xc], [0, -S.DCUT], color=S.INK, lw=2.6)
        nf = qK / (S.H1 - S.H2) * 10
        ax.set_title(titles[md["cut"]] + f":  $q$ = {md['q'] * S.DAY:.2f} m³/day per m,  "
                     f"$N_f/N_d$ = {nf:.2f}/10", fontsize=11, loc="left")
        ax.set_xlim(-S.LSIDE, S.B + S.LSIDE)
        ax.set_ylim(-S.D, 1.8)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout(h_pad=0.6)
    return fig


def fig_uplift(models):
    """Figure 4: (a) uplift pressure on the base, (b) exit gradient downstream of the toe.

    Both panels evaluate the solution along a line of points:
      * S.base_head(model, xs) returns h at the points (x, 0) on the base,
        and the pressure is gamma_w h.
      * S.surface_gradient(model, xs) returns -dh/dy at the points (x, 0) on
        the ground downstream of the toe.
    The numbers in the legends (U and i_avg) were computed by build().

    To modify:
      * The colors and legend labels are in cols and labs.
      * xb and xd are the evaluation points; xd starts 0.01 m past the toe
        because the gradient is singular at the toe itself.
      * The axis limits are set with set_xlim and set_ylim.
    """
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    labs = {None: "no cutoff", "up": "cutoff at the heel", "down": "cutoff at the toe"}
    cols = {None: "0.35", "up": "#2b5fa8", "down": S.LOADCLR}
    # (a) pressure on the base, at 300 points from heel to toe
    xb = np.linspace(0.02, S.B - 0.02, 300)
    for md in models:
        ax.plot(xb, S.GW * S.base_head(md, xb), color=cols[md["cut"]], lw=2.0,
                label=f"{labs[md['cut']]} ($U$ = {md['U']:.0f} kN/m)")
    ax.plot([0, S.B], [S.GW * S.H1, S.GW * S.H2], color="0.6", lw=1.2, ls=":",
            label="straight-line assumption")
    ax.set_xlabel("distance from the heel (m)")
    ax.set_ylabel("uplift pressure on the base (kPa)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_xlim(0, S.B)
    ax.set_ylim(0, max(130, 1.1 * S.GW * S.H1))
    # (b) exit gradient, at 400 points over the first 10 m downstream of the toe
    xd = np.linspace(S.B + 0.01, S.B + 10, 400)
    for md in models:
        bx.plot(xd - S.B, S.surface_gradient(md, xd), color=cols[md["cut"]], lw=2.0,
                label=f"{labs[md['cut']]}: {md['i_avg']:.2f}")
    bx.axhline(S.IC, color=S.LOADCLR, lw=1.0, ls="--")
    bx.text(5.0, S.IC + 0.04, f"critical gradient $i_c$ = {S.IC:.1f}", ha="left", va="bottom", fontsize=9.5,
            color=S.LOADCLR)
    bx.axvspan(0, S.WIN, color="0.92", zorder=0)        # the shaded averaging window
    bx.text(S.WIN / 2, 2.35, "averaging\nwindow", ha="center", va="top", fontsize=9, color="0.35")
    bx.set_xlabel("distance downstream of the toe (m)")
    bx.set_ylabel("upward exit gradient  $-\\partial h/\\partial y$")
    bx.set_ylim(0, 2.5)
    bx.set_xlim(0, 10)
    bx.legend(frameon=False, fontsize=9.5, loc="upper right", title="average over the window",
              title_fontsize=9.5)
    for a in (ax, bx):                       # the notes' plain style: no top or right frame, light grid
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(True, color="0.9", lw=0.7)
        a.set_axisbelow(True)
    ax.set_title("(a) Uplift pressure", fontsize=11.5, loc="left")
    bx.set_title("(b) Exit gradient", fontsize=11.5, loc="left")
    fig.tight_layout(w_pad=2.5)
    return fig


def fig_benchmark(deep):
    """Figure 6: the head under the base, compared with a closed-form solution.

    The closed form is for a dam without a cutoff on sand of unlimited depth
    and extent:
        h(x) = H2 + (H1 - H2)/pi * arccos((2x - B)/B),   0 < x < B.
    It equals H1 at the heel and H2 at the toe, and its slope is infinite at
    both ends, where the gradient is singular.  The finite element model
    approximates unlimited sand with a large but finite layer, so the model
    passed in should have no cutoff and a large depth and side.

    Returns the figure and err, the largest difference along the base
    between the two curves (meters of head).

    To modify:
      * xb sets where the comparison is made.  Points closer to the heel and
        toe show the singularity more strongly; for example,
        np.linspace(0.1 * B, 0.9 * B, 300) measures the middle 80% only.
      * Plot the difference itself, S.base_head(deep, xb) - exact, to see
        where the error is.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xb = np.linspace(0.02, S.B - 0.02, 300)
    exact = S.H2 + (S.H1 - S.H2) / math.pi * np.arccos((2 * xb - S.B) / S.B)
    ax.plot(xb, exact, color=S.INK, lw=2.4, label="closed form, infinitely deep sand")
    ax.plot(xb, S.base_head(deep, xb), color=S.LOADCLR, lw=1.6, ls="--",
            label=f"finite elements, sand {-deep['mesh'].X[:, 1].min():g} m deep")
    ax.plot([0, S.B], [S.H1, S.H2], color="0.6", lw=1.0, ls=":", label="straight line")
    ax.set_xlabel("distance from the heel (m)")
    ax.set_ylabel("head under the base, $h$ (m)")
    ax.legend(frameon=False, fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="0.9", lw=0.7)
    ax.set_xlim(0, S.B)
    fig.tight_layout()
    err = np.max(np.abs(S.base_head(deep, xb) - exact))
    return fig, err


def fig_anisotropic(model, dx=6.0, dy=4.0, arrow=2.4):
    """Figure 5: head contours and flow directions, for anisotropic sand.

    The arrows show the direction of the seepage velocity v = -K grad h at a
    grid of points, with K = diag(k_h, k_v) taken from the model.  At each
    point, S.F.locate finds the element and its parent coordinates, and
    S.F.gradient evaluates grad h there.  Every arrow has the same length
    (arrow, in meters), so only the direction is shown.

    To modify:
      * dx, dy: the spacing of the arrows (m); arrow: their length (m).
      * Pass an isotropic model to see the arrows cross the equipotentials at
        right angles.
    """
    m, sol = model["mesh"], model["sol"]
    fig, ax = plt.subplots(figsize=(10.4, 3.0))
    tri = mtri.Triangulation(m.X[:, 0], m.X[:, 1], m.triangles())
    ax.tricontourf(tri, sol.u, levels=np.linspace(S.H2, S.H1, 21), cmap=S.HEADMAP, alpha=0.75)
    ax.tricontour(tri, sol.u, levels=np.arange(S.H2 + 1, S.H1, 1.0), colors="#1e3a5f", linewidths=0.8,
                  linestyles="--")
    kap = np.diag(model["kappa"]) if np.ndim(model["kappa"]) == 1 else np.eye(2) * model["kappa"]
    xc = cutoff_x(model)
    for x in np.arange(-34, S.B + 36, dx):
        for y in np.arange(-17, -1, dy):
            if xc is not None and abs(x - xc) < 1.5 and y > -S.DCUT:
                continue                         # skip points on the wall itself
            e, xi, eta = S.F.locate(m, x, y)
            g = S.F.gradient(sol, e, xi, eta)[:, 0]
            v = -kap @ g
            v = v / np.linalg.norm(v) * arrow
            ax.annotate("", xy=(x + v[0], y + v[1]), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=0.9, mutation_scale=8))
    ax.add_patch(Rectangle((0, 0), S.B, 1.6, facecolor=S.CONC, edgecolor=S.INK, lw=1.0))
    if xc is not None:
        ax.plot([xc, xc], [0, -S.DCUT], color=S.INK, lw=2.6)
    ax.set_xlim(-S.LSIDE, S.B + S.LSIDE)
    ax.set_ylim(-S.D, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


# =============================================================================
# The analysis
# =============================================================================

# 1. Solve: the same models as the page.  p is the polynomial degree (1 or 2)
#    and NREF the refinement level; each level halves the element sizes.
M = {c: build(p=2, nref=NREF, cut=c) for c in (None, "up", "down")}   # the three cutoff arrangements
aniso = build(p=2, nref=NREF, cut="up", kappa=(4 * S.K, S.K))         # k_h = 4 k_v, cutoff at the heel
deep = build(p=2, nref=NREF, cut=None, depth=200.0, side=200.0, r0=1.25)  # nearly unlimited sand
small = build(p=2, nref=1, cut="up")     # the page draws the mesh at nref = 1

# 2. Prepare each model for the figure functions
V = {c: prepare(m) for c, m in M.items()}

# 3. Draw and save.  For what each figure shows and how to change it, see the
#    docstring of its function above.

# # Figure 1: the setting, with flow lines of the upstream-cutoff model.
# # Discussion: the water dives under the cutoff and rises to the tailwater, so
# # the wall lengthens the path and reduces the flow.
# save(fig_setting(V["up"]), "C24_Seep_Fig1_Setting.png")

# # Figure 2: the domain and the mesh.
# # Discussion: which conditions are essential and which natural; why the
# # cutoff is a slit (both faces carry no flow); why the mesh is graded.
# save(fig_domain(prepare(small)), "C24_Seep_Fig2_Domain.png")

# Figure 3: flow nets for the three arrangements.
# Discussion: count channels and drops and estimate q by hand; compare the
# upstream and downstream cutoffs, which pass almost the same q but have very
# different nets near the toe.
save(fig_flownets([V[None], V["up"], V["down"]]), "C24_Seep_Fig3_FlowNets.png")

# # Figure 4: uplift and exit gradient for the three arrangements.
# # Discussion: the upstream cutoff lowers the uplift, while the downstream
# # cutoff lowers the exit gradient: uplift and piping favor different designs.
# # Without a cutoff the force happens to equal the straight-line value, by
# # symmetry, although the distribution is not straight.
# save(fig_uplift([V[None], V["up"], V["down"]]), "C24_Seep_Fig4_Uplift.png")

# Figure 5: anisotropic sand, k_h = 4 k_v, cutoff at the heel.
# Discussion: the flow is no longer perpendicular to the equipotentials, so a
# hand-drawn net with square cells would be wrong; q rises because water
# travels horizontally more easily.  Try kappa=(10 * S.K, S.K) above.
save(fig_anisotropic(prepare(aniso)), "C24_Seep_Fig5_Anisotropic.png")
save(fig_anisotropic(V["up"]), "C24_Seep_Fig5_Isotropic.png")

# # Figure 6: a check against the closed-form solution for unlimited sand.
# # Discussion: verification of a code against a known answer, and the effect
# # of truncating an unlimited domain.  Change depth, side, or the refinement
# # of "deep" above and watch bench_err.
# fig, bench_err = fig_benchmark(prepare(deep))
# save(fig, "C24_Seep_Fig6_Benchmark.png")

# The numbers behind the figures, for class discussion
print(f"Six figures saved in {OUT.resolve()}")
print(f"{'model':<24}{'q (m3/day/m)':>14}{'balance':>10}{'U (kN/m)':>10}"
      f"{'i_avg':>8}{'from grad':>11}{'FS':>6}")
for name, m in (("no cutoff", M[None]), ("cutoff at the heel", M["up"]),
                ("cutoff at the toe", M["down"]), ("k_h = 4 k_v, heel", aniso)):
    print(f"{name:<24}{m['q'] * S.DAY:>14.2f}{abs(m['q_out'] / m['q'] - 1):>10.0e}{m['U']:>10.0f}"
          f"{m['i_avg']:>8.2f}{m['i_avg_grad']:>11.2f}{S.IC / m['i_avg']:>6.1f}")
# print(f"largest difference from the closed form (Figure 6): {bench_err:.3f} m of head")
