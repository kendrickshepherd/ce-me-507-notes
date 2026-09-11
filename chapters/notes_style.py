"""
Shared drawing helpers for the CE/ME 507 class-notes chapters.

Copied from the CE 321 notes book; the two are presentation layers over the same
matplotlib/structplot stack. If a third course appears, promote this to a shared
module at the repository root rather than copying it a third time.

Imported by chapters/*_Class.qmd so the figures across every class share one
visual language: same ink colours, same arrow and dimension styling, same
isometric projection.

Call ``use_chapter("Class3")`` once per chapter to set (and create) the folder
that ``save()`` writes into.
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

__all__ = [
    "use_chapter", "save", "iso", "iso_face", "iso_slab", "iso_column",
    "beam_line", "arrow_down", "arrow_up", "brace", "dim_x", "tidy",
    "trapezoid_load", "lagrange_basis", "lagrange_basis_deriv",
    "INK", "LOADCLR", "REACTCLR", "TRIBCLR", "DIMCLR",
    "DOMCLR", "BDRYCLR", "FLUXCLR", "MESHCLR", "REFCLR", "BASISCLRS",
]

_IMGDIR = "generated_images/"


def use_chapter(tag):
    """Point save() at generated_images/<tag>/ and make sure it exists."""
    global _IMGDIR
    _IMGDIR = "generated_images/" + tag + "/"
    os.makedirs(_IMGDIR, exist_ok=True)
    return _IMGDIR


# ---------------------------------------------------------------- house style
INK      = "#1a1a1a"   # structure
LOADCLR  = "#c0392b"   # applied loads
REACTCLR = "#1f6feb"   # reactions
TRIBCLR  = "#e67e22"   # tributary shading
DIMCLR   = "0.35"
DOMCLR   = "#cfe3f5"   # domain interior
BDRYCLR  = "#4a7ba7"   # domain boundary
FLUXCLR  = "#0d9488"   # flux / Neumann boundary
MESHCLR  = "#8a5a20"   # mesh / element boundary lines
REFCLR   = "#7c3aed"   # reference (parent) domain, distinguishing it from physical space
BASISCLRS = ["#1f6feb", "#c0392b", "#0d9488", "#e67e22", "#8a5a20", "#7c3aed"]
                       # cycled colors for plotting several basis functions at once

plt.rcParams.update({
    "font.size": 11,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.0,
    "savefig.facecolor": "white",
})


def save(fig, name):
    """Save a figure into the chapter image folder and return its path."""
    path = _IMGDIR + name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def iso(x, y, z, ang=30.0, k=0.55):
    """Cabinet-style projection of a 3D point onto the page."""
    a = math.radians(ang)
    return (x + k * y * math.cos(a), z + k * y * math.sin(a))


def iso_face(ax, pts, fc="#ffffff", ec=INK, lw=1.2, alpha=1.0, zorder=2):
    """Draw one planar face given a list of (x, y, z) corners."""
    xy = [iso(*p) for p in pts]
    ax.add_patch(MplPolygon(xy, closed=True, facecolor=fc, edgecolor=ec,
                            linewidth=lw, alpha=alpha, zorder=zorder))


def iso_slab(ax, x0, x1, y0, y1, z, t, fc="#f4f4f4", zorder=2):
    """A rectangular slab of thickness t whose top surface is at height z."""
    iso_face(ax, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
             fc=fc, zorder=zorder)                                   # top
    iso_face(ax, [(x0, y0, z), (x1, y0, z), (x1, y0, z - t), (x0, y0, z - t)],
             fc="#e2e2e2", zorder=zorder + 1)                        # front
    iso_face(ax, [(x1, y0, z), (x1, y1, z), (x1, y1, z - t), (x1, y0, z - t)],
             fc="#d8d8d8", zorder=zorder + 1)                        # right


def iso_column(ax, x, y, z_top, z_bot, lw=1.6, w=0.16):
    """A square column drawn as a narrow prism."""
    iso_face(ax, [(x - w, y, z_top), (x + w, y, z_top),
                  (x + w, y, z_bot), (x - w, y, z_bot)],
             fc="#ededed", lw=lw, zorder=1)


def beam_line(ax, x0, x1, y=0.0, lw=2.6):
    ax.plot([x0, x1], [y, y], color=INK, lw=lw, solid_capstyle="butt", zorder=3)


def arrow_down(ax, x, y_tip, length, label=None, color=LOADCLR, lw=1.8,
               fs=12, dx=0.0, ha="center"):
    ax.annotate("", xy=(x, y_tip), xytext=(x, y_tip + length),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=13), zorder=5)
    if label:
        ax.text(x + dx, y_tip + length + 0.06, label, color=color, ha=ha,
                va="bottom", fontsize=fs)


def arrow_up(ax, x, y_tip, length, label=None, color=REACTCLR, lw=1.8,
             fs=12, dx=0.0, ha="center"):
    ax.annotate("", xy=(x, y_tip), xytext=(x, y_tip - length),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=13), zorder=5)
    if label:
        ax.text(x + dx, y_tip - length - 0.08, label, color=color, ha=ha,
                va="top", fontsize=fs)


def brace(ax, x0, x1, y, label, color=TRIBCLR, depth=0.16, fs=12, va="top"):
    """A curly-brace-like span marker under a beam, with a label."""
    xm = 0.5 * (x0 + x1)
    xs = np.linspace(x0, x1, 60)
    ys = y - depth * np.sin(np.pi * (xs - x0) / max(x1 - x0, 1e-9)) ** 0.6
    ax.plot(xs, ys, color=color, lw=1.5, zorder=4)
    ax.plot([x0, x0], [y, y - depth * 0.35], color=color, lw=1.5)
    ax.plot([x1, x1], [y, y - depth * 0.35], color=color, lw=1.5)
    ax.text(xm, y - depth - 0.06, label, color=color, ha="center", va=va,
            fontsize=fs)


def dim_x(ax, x0, x1, y, label, color=DIMCLR, fs=12, tick=0.10, va="top"):
    """A horizontal dimension line with end ticks and a centred label."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.1,
                                mutation_scale=10), zorder=4)
    for xv in (x0, x1):
        ax.plot([xv, xv], [y - tick, y + tick], color=color, lw=1.0, zorder=4)
    ax.text(0.5 * (x0 + x1), y - 0.10, label, color=color, ha="center",
            va=va, fontsize=fs)


def tidy(ax):
    """Equal aspect, no axes frame - the default for every schematic here."""
    ax.set_aspect("equal")
    ax.axis("off")


def trapezoid_load(ax, x0, x1, ramp, peak, y_beam=0.0, n=15, color=LOADCLR,
                   label=None, fs=13, label_dy=0.12):
    """Draw a trapezoidal downward line load on the beam between x0 and x1.

    ``ramp`` is the fraction of the span over which the load builds from zero to
    full intensity at each end (0 gives a uniform load, 0.5 a triangle); ``peak``
    is the height of the full-intensity ordinate in data units.

    structplot can now draw this natively -- three DistLoads sharing a `ref_mag`
    meet correctly at the joins since the interpolation fix in plotter.py. This
    helper is kept because it takes an explicit `peak` in data units, where
    structplot derives the ordinate from a fixed fraction of the model extent;
    for these notes the taller profile reads better against the beam.
    """
    L = float(x1 - x0)
    xa, xb = x0 + ramp * L, x1 - ramp * L

    def ordinate(x):
        if x < xa:
            return peak * (x - x0) / (xa - x0) if xa > x0 else peak
        if x > xb:
            return peak * (x1 - x) / (x1 - xb) if x1 > xb else peak
        return peak

    xs = [x0, xa, xb, x1]
    ax.plot(xs, [y_beam + ordinate(x) for x in xs], color=color, lw=1.5,
            zorder=5)
    for xv in (x0, x1):
        ax.plot([xv, xv], [y_beam, y_beam + ordinate(xv)], color=color, lw=1.5,
                zorder=5)

    for xv in np.linspace(x0, x1, n):
        h = ordinate(xv)
        if h > 1e-6:
            ax.annotate("", xy=(xv, y_beam + 0.03), xytext=(xv, y_beam + h),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                        mutation_scale=9), zorder=5)

    if label:
        ax.text(0.5 * (x0 + x1), y_beam + peak + label_dy, label, color=color,
                ha="center", va="bottom", fontsize=fs)


def lagrange_basis(nodes, a, x):
    """The a-th Lagrange basis polynomial for interpolation nodes `nodes`,
    evaluated at `x` (scalar or ndarray). ell_a(nodes[a]) = 1, ell_a(nodes[b]) = 0
    for b != a -- the defining property used throughout Classes 6, 9, and 10.
    """
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    xa = nodes[a]
    for b, xb in enumerate(nodes):
        if b == a:
            continue
        out = out * (x - xb) / (xa - xb)
    return out


def lagrange_basis_deriv(nodes, a, x, h=1e-6):
    """d/dx of the a-th Lagrange basis polynomial, by central difference.
    Good enough for plotting; Class 9's derivation gives the closed form.
    """
    return (lagrange_basis(nodes, a, x + h) - lagrange_basis(nodes, a, x - h)) / (2 * h)
