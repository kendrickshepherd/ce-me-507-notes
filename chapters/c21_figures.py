"""The static figures of Class 21 (CAD, meshing, and spline bases).

Each function returns (fig, info): the figure, and the numbers its caption
quotes, so a caption can never drift away from the geometry that drew it.
The bracket geometry comes from c21_bracket, which mirrors the widget's
JavaScript exactly.
"""
from math import comb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Polygon as MP, PathPatch, Rectangle, Circle, Patch
from matplotlib.lines import Line2D
from matplotlib.path import Path

import c21_bracket as B
import c21_mesh as Mesh
import c21_separatrix as Sep
from notes_style import BASISCLRS

# the bracket widget's palette, so the static views match the rotating one
INK, GREY, FILL = "#18202b", "#6b7684", "#dce6f2"
BLUE, RED, VIO = "#2b5fa8", "#b8322a", "#6d28d9"
GHOST, GHOSTE, OUTF = "#eef0f3", "#c3c9d1", "#f1f2f4"
CUTE = "#e67e22"
TRI, QUAD = "#cfe3f5", "#ffe9a8"
NISO = 7                                  # the widget's isocurve count


def _title(ax, text, pad=6, size=12.5):
    ax.set_title(text, fontsize=size, fontweight="bold", color=INK, pad=pad)


def _plot_axes(ax):
    """Plot axes rather than a schematic: light left and bottom spines only."""
    for k in ("top", "right"): ax.spines[k].set_visible(False)
    for k in ("left", "bottom"): ax.spines[k].set_color("0.6")
    ax.tick_params(colors="0.4", labelsize=9.5)


def _area(X):
    x, y = X[:, 0], X[:, 1]
    return 0.5*abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


# ============================================================================
#  trimming
# ============================================================================
def _region_path(p, xy, n=16):
    """The kept region (outer loop with the hole cut out), in whatever plane xy gives."""
    o = xy(p, B.region_loop_uv(p, n))
    verts = [*o, o[0]]
    codes = [Path.MOVETO] + [Path.LINETO]*(len(o)-1) + [Path.CLOSEPOLY]
    if p.hole:
        h = xy(p, B.hole_loop_uv(p, 96))[::-1]
        verts += [*h, h[0]]
        codes += [Path.MOVETO] + [Path.LINETO]*(len(h)-1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _scr(p, uv): return B.proj(B.to_xyz(p, uv))
def _par(p, uv): return np.asarray(uv, float)


def _is_seam(p, s):
    (u0, _), (u1, _), _ = s
    if u0 == u1 == 0: return True                      # every patch's u = 0 edge is glued
    if u0 == u1 == 1: return p.kind in ('fillet', 'web')
    return False


def _edges(ax, p, xy, lw=2.0, z=6):
    for s in B.region_segs(p):
        c = RED if s[2] == 'trim' else (VIO if _is_seam(p, s) else BLUE)
        ax.plot(*xy(p, B.seg_uv(s, 24)).T, color=c, lw=lw + (.4 if c == VIO else 0),
                solid_capstyle="round", zorder=z)
    if p.hole:
        h = xy(p, B.hole_loop_uv(p, 96))
        ax.plot(*np.vstack([h, h[:1]]).T, color=RED, lw=lw, zorder=z)


def _isocurves(ax, p, xy, dashed_drop=False, z=4):
    """Constant-u and constant-v curves, split where the trim removes them."""
    t = np.linspace(0, 1, 141)
    for i in range(1, NISO):
        c = i/NISO
        for fix_u in (True, False):
            uv = np.c_[np.full_like(t, c), t] if fix_u else np.c_[t, np.full_like(t, c)]
            off = np.array([B.clipped(p, u, v) for u, v in uv])
            pts = xy(p, uv)
            k0 = 0
            for k in range(1, len(t)+1):
                if k == len(t) or off[k] != off[k0]:
                    seg = pts[k0:min(k+1, len(t))]
                    if not off[k0]:
                        ax.plot(*seg.T, color=GREY, lw=0.8, alpha=.75, zorder=z)
                    elif dashed_drop:
                        ax.plot(*seg.T, color=GREY, lw=0.72, alpha=.45, ls=(0, (3, 2.5)), zorder=z)
                    k0 = k


def _uv_arrows(ax, F, xy, L, offs):
    """Arrows along u and v from the (0, 0) corner, labels pushed outside the face."""
    o = xy(F, np.array([[0, 0]]))[0]
    for d, lab, off in (((L, 0), "$u$", offs[0]), ((0, L), "$v$", offs[1])):
        q = xy(F, np.array([d]))[0]
        ax.annotate("", xy=q, xytext=o, zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=13))
        ax.annotate(lab, xy=(o+q)/2, xytext=off, textcoords="offset points", fontsize=13,
                    color=INK, ha="center", va="center", zorder=9)
    ax.plot(*o, "o", ms=6.5, mfc="white", mec=INK, mew=1.6, zorder=10)


def trimming_figure():
    """One flange picked out of the bracket, the untrimmed surface behind it, and
    the same face in its parametric square."""
    M = B.mid_surface()
    F = M[0]                                           # nearest the widget's default camera
    order = sorted(range(len(M)), key=lambda i: -B.depth(M[i].pt(.5, .5))[0])
    sq = np.vstack([np.c_[np.linspace(0, 1, 30), np.zeros(30)], np.c_[np.ones(30), np.linspace(0, 1, 30)],
                    np.c_[np.linspace(1, 0, 30), np.ones(30)], np.c_[np.zeros(30), np.linspace(1, 0, 30)]])

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.82]))
    ax = axs[0]
    for i in order:
        p = M[i]
        if p is F:
            ax.add_patch(PathPatch(_region_path(p, _scr), facecolor=FILL, edgecolor="none", zorder=3))
            _isocurves(ax, p, _scr)
            _edges(ax, p, _scr)
        else:
            ax.add_patch(PathPatch(_region_path(p, _scr), facecolor=GHOST, edgecolor=GHOSTE, lw=0.9, zorder=2))

    ax = axs[1]
    for i in order:
        if M[i] is not F:
            ax.add_patch(PathPatch(_region_path(M[i], _scr), facecolor=GHOST, edgecolor=GHOSTE,
                                   lw=0.9, zorder=1, alpha=.55))
    ax.add_patch(MP(_scr(F, sq), closed=True, facecolor="#f4f6f9", edgecolor=BLUE, lw=1.6,
                    ls=(0, (5, 3)), zorder=2))
    ax.add_patch(PathPatch(_region_path(F, _scr), facecolor=FILL, edgecolor="none", zorder=3))
    _isocurves(ax, F, _scr, dashed_drop=True)
    _edges(ax, F, _scr)

    ax = axs[2]
    ax.add_patch(MP([[0, 0], [1, 0], [1, 1], [0, 1]], closed=True, facecolor="#f4f6f9", edgecolor=BLUE,
                    lw=1.6, ls=(0, (5, 3)), zorder=2))
    ax.add_patch(PathPatch(_region_path(F, _par), facecolor=FILL, edgecolor="none", zorder=3))
    _isocurves(ax, F, _par, dashed_drop=True)
    _edges(ax, F, _par)
    ax.set_xlim(-.08, 1.08); ax.set_ylim(-.08, 1.08)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors="0.4")

    allxy = np.vstack([_scr(p, B.region_loop_uv(p, 8)) for p in M] + [_scr(F, sq)])
    lo, hi = allxy.min(0), allxy.max(0)
    for ax in axs[:2]:
        ax.set_xlim(lo[0]-.12, hi[0]+.12); ax.set_ylim(lo[1]-.12, hi[1]+.12); ax.axis("off")
    for ax in axs: ax.set_aspect("equal")
    for ax, t in zip(axs, ["(a)  one face of the bracket", "(b)  the surface it is cut from",
                           "(c)  the same face, in $(u,v)$"]):
        _title(ax, t)
    _uv_arrows(axs[1], F, _scr, 0.24, [(4, -12), (11, 0)])
    _uv_arrows(axs[2], F, _par, 0.20, [(0, -11), (-11, 0)])
    fig.tight_layout(w_pad=1.0)
    return fig, {}


def _clipped_vec(F, u, v):
    """clipped() for arrays, for the flange (step, chamfers and hole)."""
    st, h = F.step, F.hole
    ta, tb = F.tri['a'], F.tri['b']
    out = (u < st['u0']) & ((v < st['vlo']-1e-9) | (v > st['vhi']+1e-9))
    out |= (tb*(u-1+ta) - ta*v > 0) | (tb*(u-1+ta) + ta*(v-1) > 0)
    out |= ((u-h['uc'])/h['ru'])**2 + ((v-h['vc'])/h['rv'])**2 < 1
    return out


def cut_cell_figure(NC=NISO, zoom=(3, 4)):
    """The flange's parametric cells sorted by the trim, and one cut cell enlarged
    with the Gauss points of the full cell."""
    F = B.mid_surface()[0]
    t = (np.arange(600)+0.5)/600
    frac = np.zeros((NC, NC))
    for i in range(NC):
        for j in range(NC):
            U, V = np.meshgrid((i+t)/NC, (j+t)/NC)
            frac[i, j] = 1 - _clipped_vec(F, U, V).mean()
    ncut = int(((frac > 0) & (frac < 1)).sum())

    def trims(ax, lw):
        for s in B.region_segs(F):
            if s[2] == 'trim':
                ax.plot(*B.seg_uv(s, 24).T, color=RED, lw=lw, zorder=6, solid_capstyle="round")
        h = B.hole_loop_uv(F, 128)
        ax.plot(*np.vstack([h, h[:1]]).T, color=RED, lw=lw, zorder=6)

    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax = axs[0]
    for i in range(NC):
        for j in range(NC):
            ax.add_patch(Rectangle((i/NC, j/NC), 1/NC, 1/NC, facecolor=OUTF, edgecolor="none", zorder=1))
    ax.add_patch(PathPatch(_region_path(F, _par), facecolor=FILL, edgecolor="none", zorder=2))
    for i in range(NC):
        for j in range(NC):
            cut = 0 < frac[i, j] < 1
            ax.add_patch(Rectangle((i/NC, j/NC), 1/NC, 1/NC, facecolor="none",
                                   edgecolor=(CUTE if cut else "#9aa3ae"), lw=(1.7 if cut else 0.8),
                                   zorder=(4 if cut else 3)))
    trims(ax, 2.0)
    zi, zj = zoom
    ax.add_patch(Rectangle((zi/NC, zj/NC), 1/NC, 1/NC, facecolor="none", edgecolor=INK, lw=2.6, zorder=8))
    # the sliver of material in each outer corner cell sits at its inner corner
    for (i, j) in [(NC-1, NC-1), (NC-1, 0)]:
        tip = (i/NC, (j if j else j+1)/NC)
        ax.annotate(f"{100*frac[i, j]:.2f}% material", xy=tip, xytext=(1.08, (j+.5)/NC),
                    fontsize=10, color=INK, ha="left", va="center", zorder=9,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, shrinkA=2, shrinkB=1))
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.tick_params(length=0, labelsize=10, colors="0.4")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, -0.07, "$u$", ha="center", fontsize=12.5)
    ax.text(-0.07, 0.5, "$v$", va="center", fontsize=12.5)
    _title(ax, "(a)  the face's cells", pad=8)

    ax = axs[1]
    u0, v0, w = zi/NC, zj/NC, 1/NC
    ax.add_patch(Rectangle((u0, v0), w, w, facecolor=OUTF, edgecolor="none", zorder=1))
    ax.add_patch(PathPatch(_region_path(F, _par), facecolor=FILL, edgecolor="none", zorder=2))
    trims(ax, 2.6)
    ax.add_patch(Rectangle((u0, v0), w, w, facecolor="none", edgecolor=INK, lw=2.6, zorder=8))
    g = np.array([-np.sqrt(.6), 0, np.sqrt(.6)])
    nin = 0
    for a in g:
        for b in g:
            uq, vq = u0 + w*(1+a)/2, v0 + w*(1+b)/2
            ins = not B.clipped(F, uq, vq)
            nin += ins
            ax.plot(uq, vq, "o", ms=10, mfc=(BLUE if ins else "white"), mec=BLUE, mew=1.8, zorder=9)
    ax.set_xlim(u0-.006, u0+w+.006); ax.set_ylim(v0-.006, v0+w+.006)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(u0+w/2, v0+0.30*w, "hole", ha="center", va="center", fontsize=11, color="0.45", zorder=7)
    _title(ax, f"(b)  one cut cell: {nin} of 9 Gauss points on the part", pad=8)

    fig.legend(handles=[Patch(facecolor=FILL, edgecolor="#9aa3ae", label="material"),
                        Patch(facecolor=OUTF, edgecolor="#9aa3ae", label="trimmed away"),
                        Patch(facecolor="white", edgecolor=CUTE, lw=1.7,
                              label=f"cell cut by a trim curve ({ncut} of {NC*NC})"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=BLUE, mew=1.6,
                               label="Gauss point off the part")],
               loc="lower center", ncol=4, frameon=False, fontsize=9.8, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.5, columnspacing=1.4)
    fig.tight_layout(w_pad=3.0, rect=(0, 0.07, 1, 1))
    return fig, dict(ncells=NC*NC, ncut=ncut, nout=int((frac == 0).sum()),
                     corner_pct=100*frac[NC-1, NC-1], gauss_in=nin)


# ============================================================================
#  meshing the bracket
# ============================================================================
def bracket_mesh_figure():
    faces, fl = Mesh.bracket_faces()
    Lt = np.array([0.35, -0.55, 0.76]); Lt /= np.linalg.norm(Lt)
    vd = B.rot(np.eye(3))[:, 1]                         # direction of increasing depth

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw=dict(width_ratios=[1.12, 1]))
    ax = axs[0]
    for k in sorted(range(len(faces)), key=lambda k: -B.depth(faces[k][1].mean(0))[0]):
        kind, X = faces[k]
        n = np.cross(X[1]-X[0], X[2]-X[0]); n /= np.linalg.norm(n)
        if n @ vd > 0: n = -n                           # light the side we are looking at
        sh = 0.70 + 0.30*max(0.0, float(n @ Lt))
        rgb = np.array(mc.to_rgb(QUAD if kind == 'quad' else TRI))
        ax.add_patch(MP(B.proj(X), closed=True, facecolor=tuple(1-(1-rgb)*sh),
                        edgecolor=INK, lw=0.55, joinstyle="round"))
    xy = np.vstack([B.proj(f[1]) for f in faces]); lo, hi = xy.min(0), xy.max(0)
    ax.set_xlim(lo[0]-.08, hi[0]+.08); ax.set_ylim(lo[1]-.08, hi[1]+.08)
    ax.set_aspect("equal"); ax.axis("off")
    nq = sum(f[0] == 'quad' for f in faces); nt = len(faces) - nq
    _title(ax, f"(a)  mid-surface mesh: {nq} quadrilaterals, {nt} triangles")

    # the front flange laid flat, mirrored so it reads the same way round as in (a)
    ax = axs[1]; M2 = np.array([-1.0, 1.0]); P = fl['P']*M2
    for q in fl['quads']:
        ax.add_patch(MP(P[list(q)], closed=True, facecolor=QUAD, edgecolor=INK, lw=0.7))
    for t in fl['tris']:
        ax.add_patch(MP(P[list(t)], closed=True, facecolor=TRI, edgecolor=INK, lw=0.7))
    F = B.flange(1)
    true_hole = B.to_xyz(F, B.hole_loop_uv(F, 600))[:, [0, 2]]*M2
    ax.plot(*np.vstack([true_hole, true_hole[:1]]).T, color=RED, lw=1.7, zorder=6)
    for s in B.region_segs(F):                          # the trims are gently curved in x-z
        if s[2] == 'trim':
            ax.plot(*(B.to_xyz(F, B.seg_uv(s, 80))[:, [0, 2]]*M2).T, color=RED, lw=1.7, zorder=6)
    o = fl['outer']*M2
    ax.set_xlim(o[:, 0].min()-.10, o[:, 0].max()+.10); ax.set_ylim(o[:, 1].min()-.08, o[:, 1].max()+.08)
    ax.set_aspect("equal"); ax.axis("off")
    _title(ax, "(b)  the front flange, laid flat")
    ax.text(*true_hole.mean(0), f"{fl['nb']-len(fl['outer'])} straight\nedges in place\nof the curve",
            ha="center", va="center", fontsize=9.8, color=INK, linespacing=1.25, zorder=9)
    fig.legend(handles=[Patch(facecolor=QUAD, edgecolor=INK, label="quadrilateral"),
                        Patch(facecolor=TRI, edgecolor=INK, label="triangle"),
                        Line2D([], [], color=RED, lw=1.8, label="the CAD face's true boundary")],
               loc="lower center", ncol=3, frameon=False, fontsize=10.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.05, 1, 1))

    quad_area = sum(_area(fl['P'][list(q)]) for q in fl['quads'])
    tri_area = sum(_area(fl['P'][list(t)]) for t in fl['tris'])
    mesh_hole = fl['P'][len(fl['outer']):fl['nb']]
    return fig, dict(nquad=nq, ntri=nt, hole_sides=len(mesh_hole),
                     hole_loss_pct=100*(1 - _area(mesh_hole)/_area(true_hole)),
                     fillet_facets=Mesh.NU_FIL,
                     fillet_sag_pct=100*(1 - np.cos(np.pi/2/Mesh.NU_FIL/2)),
                     flange_quad_pct=100*quad_area/(quad_area + tri_area))


def _bracket_3d(ax, faces, title):
    Lt = np.array([0.35, -0.55, 0.76]); Lt /= np.linalg.norm(Lt)
    vd = B.rot(np.eye(3))[:, 1]
    for k in sorted(range(len(faces)), key=lambda k: -B.depth(faces[k][1].mean(0))[0]):
        kind, X = faces[k]
        n = np.cross(X[1]-X[0], X[2]-X[0]); n /= np.linalg.norm(n)
        if n @ vd > 0: n = -n
        sh = 0.70 + 0.30*max(0.0, float(n @ Lt))
        rgb = np.array(mc.to_rgb(QUAD if kind == 'quad' else TRI))
        ax.add_patch(MP(B.proj(X), closed=True, facecolor=tuple(1-(1-rgb)*sh),
                        edgecolor=INK, lw=0.55, joinstyle="round"))
    xy = np.vstack([B.proj(f[1]) for f in faces]); lo, hi = xy.min(0), xy.max(0)
    ax.set_xlim(lo[0]-.08, hi[0]+.08); ax.set_ylim(lo[1]-.08, hi[1]+.08)
    ax.set_aspect("equal"); ax.axis("off"); _title(ax, title)


def quad_only_mesh_figure():
    """The bracket meshed with quadrilaterals only, laid out in blocks."""
    faces, r = Mesh.bracket_faces_quad()
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.3), gridspec_kw=dict(width_ratios=[1.12, 1]))
    _bracket_3d(axs[0], faces, f"(a)  quad-only mesh: {len(faces)} quadrilaterals, no triangles")
    ax = axs[1]; M2 = np.array([-1.0, 1.0]); V = r['V']*M2
    for q in r['Q']:
        ax.add_patch(MP(V[list(q)], closed=True, facecolor=QUAD, edgecolor="#5d6672", lw=0.7))
    for v in r['irregular']:
        ax.plot(*V[v], "o", ms=8, mfc="white", mec=RED, mew=2.0, zorder=6)
    o = np.vstack([V[list(q)] for q in r['Q']])
    ax.set_xlim(o[:, 0].min()-.08, o[:, 0].max()+.08); ax.set_ylim(o[:, 1].min()-.08, o[:, 1].max()+.08)
    ax.set_aspect("equal"); ax.axis("off")
    _title(ax, "(b)  the front flange, laid flat")
    fig.legend(handles=[Patch(facecolor=QUAD, edgecolor=INK, label="quadrilateral"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=RED, mew=2,
                               label="irregular vertex")],
               loc="lower center", ncol=2, frameon=False, fontsize=10.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.05, 1, 1))
    val = {}
    import collections
    nb = {}
    for q in r['Q']:
        for a in range(4): nb.setdefault(q[a], set()).update((q[a-1], q[(a+1) % 4]))
    val = collections.Counter(len(nb[v]) for v in r['irregular'])
    return fig, dict(nquad=len(faces), flange_quads=len(r['Q']), valence=dict(val))


# sequential single-hue ramp (light -> dark), used for distortion = 1 - scaled Jacobian
_SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def flange_comparison():
    """Quality numbers for the two flanges, measured identically."""
    import c21_quality as QQ
    f1, fl = Mesh.bracket_faces(); f2, r = Mesh.bracket_faces_quad()
    out = {}
    for key, V, E, bnd, nfaces, ntri in (
            ("qd", fl['P'], list(fl['quads'])+list(fl['tris']), set(range(fl['nb'])), len(f1),
             sum(k == 'tri' for k, _ in f1)),
            ("qo", r['V'], r['Q'], r['fixed'], len(f2), 0)):
        q = QQ.mesh_quality(V, E)
        out[key] = dict(V=V, E=E, bracket=nfaces, tris=ntri, flange=len(E),
                        irregular=len(Mesh._irregular(E, bnd)), amin=q['amin'], amax=q['amax'],
                        sj=q['sj'], sj_min=q['sj_min'], sj_mean=q['sj_mean'],
                        below=100*q['frac_below'](0.85), er_max=q['er_max'], size_ratio=q['size_ratio'])
    return out


def mesh_quality_figure(lo=0.70):
    """The two flanges side by side, each element shaded by its distortion."""
    C = flange_comparison()
    cmap = mc.LinearSegmentedColormap.from_list("distortion", _SEQ)
    norm = mc.Normalize(0.0, 1.0-lo)
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.6))
    M2 = np.array([-1.0, 1.0])
    for ax, key, title in ((axs[0], "qd", "(a)  quad-dominant"),
                           (axs[1], "qo", "(b)  quad-only")):
        c = C[key]; V = c['V']*M2
        for el, sj in zip(c['E'], c['sj']):
            ax.add_patch(MP(V[list(el)], closed=True, facecolor=cmap(norm(1.0-sj)),
                            edgecolor="white", lw=1.0))
        o = np.vstack([V[list(el)] for el in c['E']])
        ax.set_xlim(o[:, 0].min()-.06, o[:, 0].max()+.06); ax.set_ylim(o[:, 1].min()-.06, o[:, 1].max()+.06)
        ax.set_aspect("equal"); ax.axis("off")
        _title(ax, title, pad=24)
        ax.text(0.5, 1.012, f"worst {c['sj_min']:.2f}  ·  mean {c['sj_mean']:.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10.3, color="0.3")
    sm = plt.cm.ScalarMappable(norm=mc.Normalize(lo, 1.0), cmap=mc.LinearSegmentedColormap.from_list(
        "q", _SEQ[::-1]))
    cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.03)
    cb.set_label("scaled Jacobian  (1 = perfect corners)", fontsize=10.5, color=INK)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9.5, colors="0.35")
    return fig, C


# ============================================================================
#  separatrices
# ============================================================================
_PAL = ["#dce6f2", "#fde9c4", "#dff0e4", "#f3dfe9", "#e6e1f5", "#f6f0cf"]


def _patch_colours(Q, patch):
    """Greedy colouring so that neighbouring patches never share a tint."""
    edges = {}
    for qi, q in enumerate(Q):
        for a in range(4):
            edges.setdefault(frozenset((q[a], q[(a+1) % 4])), []).append(qi)
    adj = {}
    for qs in edges.values():
        if len(qs) == 2 and patch[qs[0]] != patch[qs[1]]:
            a, b = patch[qs[0]], patch[qs[1]]
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    col = {}
    for p in sorted(set(patch), key=lambda p: -len(adj.get(p, ()))):
        used = {col[n] for n in adj.get(p, ()) if n in col}
        col[p] = next(c for c in range(len(_PAL)) if c not in used)
    return [_PAL[col[patch[i]]] for i in range(len(Q))]


def separatrix_figure():
    import c21_quality as QQ
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 6.3))
    info = {}
    V1, Q1 = Sep.ogrid()
    V2, Q2, _ = Sep.recombined(h=0.7, seed=3)
    V3, Q3 = Sep.twelve_patch()
    for ax, (V, Q), title, key in ((axs[0], (V1, Q1), "(a)  four blocks around the hole", "a"),
                                   (axs[1], (V2, Q2), "(b)  triangulate, pair, then split", "b"),
                                   (axs[2], (V3, Q3), "(c)  twelve patches", "c")):
        V = np.asarray(V); r = Sep.analyse(V, Q)
        for q, c in zip(Q, _patch_colours(Q, r['patch'])):
            ax.add_patch(MP(V[list(q)], closed=True, facecolor=c, edgecolor="#8b95a2", lw=0.6, zorder=2))
        for e in r['sep']:
            a_, b_ = tuple(e)
            ax.plot(*V[[a_, b_]].T, color=VIO, lw=2.3, solid_capstyle="round", zorder=4)
        for v in r['irregular']:
            ax.plot(*V[v], "o", ms=7.5, mfc="white", mec=RED, mew=2.0, zorder=6)
        ax.add_patch(Circle((0, 0), Sep.R0, facecolor="white", edgecolor=INK, lw=1.6, zorder=3))
        H = Sep.HALF
        ax.plot([-H, H, H, -H, -H], [-H, -H, H, H, -H], color=INK, lw=1.6, zorder=5)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2.15, 2.15); ax.set_ylim(-2.15, 2.15)
        ni = len(r['interior_irregular'])
        _title(ax, title, pad=40)
        ax.text(0.5, 1.012, f"{len(Q)} quads  ·  {r['npatch']} patches\n"
                f"{ni} extraordinary vertices inside",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10.3, color="0.3", linespacing=1.4)
        q = QQ.mesh_quality(V, Q)
        info[key] = dict(nquad=len(Q), n_ev=ni, npatch=r['npatch'],
                         valences=sorted({r['valence'][v] for v in r['interior_irregular']}),
                         amin=q['amin'], amax=q['amax'], sj_min=q['sj_min'], sj_mean=q['sj_mean'],
                         below=100*q['frac_below'](0.8), er_max=q['er_max'], size_ratio=q['size_ratio'])
    fig.legend(handles=[Line2D([], [], color=VIO, lw=2.3, label="separatrix"),
                        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=RED, mew=2,
                               label="irregular vertex"),
                        Patch(facecolor=_PAL[0], edgecolor="#8b95a2", label="one tint per patch")],
               loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=2.4, rect=(0, 0.05, 1, 1))
    return fig, info


# ============================================================================
#  Hermite and Bernstein
# ============================================================================
def _hermite(t):
    return np.array([2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t, -2*t**3 + 3*t**2, t**3 - t**2])


def hermite_figure():
    """The cubic Hermite basis, then a clamped-clamped beam under uniform load
    approximated by two Hermite elements (C1) and two quadratic Lagrange
    elements (C0) through the same nodal values."""
    w = lambda x: x**2*(2-x)**2/24
    th = lambda x: (2*x*(2-x)**2 - 2*x**2*(2-x))/24
    E1, E2 = "#1f6feb", "#e67e22"
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
    t = np.linspace(0, 1, 300)
    labs = [r"$h_{00}$  value, left", r"$h_{10}$  slope, left",
            r"$h_{01}$  value, right", r"$h_{11}$  slope, right"]
    for k, (y, lab) in enumerate(zip(_hermite(t), labs)):
        axs[0].plot(t, y, lw=2.2, color=BASISCLRS[k], ls="-" if k % 2 == 0 else "--", label=lab)
    axs[0].axhline(0, color="0.8", lw=0.8)
    axs[0].set(xlim=(0, 1), xlabel="local coordinate $t$"); axs[0].set_ylim(-0.25, 1.42)
    _title(axs[0], "(a)  cubic Hermite basis", size=12)
    axs[0].legend(frameon=False, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  handlelength=1.8, columnspacing=1.0)
    _plot_axes(axs[0])

    x = np.linspace(0, 2, 400)

    def beam(ax, pieces, title, note):
        ax.plot(x, w(x), color="0.55", lw=1.4, ls=(0, (4, 3)), zorder=2)
        for (xx, yy), c in zip(pieces, (E1, E2)):
            ax.plot(xx, yy, color=c, lw=3, zorder=3)
        ax.set(xlim=(-0.05, 2.05), ylim=(-0.004, 0.058), xlabel="$x$")
        ax.set_yticks([])
        ax.axvline(1, color="0.8", lw=0.8, ls=":")
        _title(ax, title, size=12)
        ax.text(1, 0.0525, note, ha="center", va="top", fontsize=9.6, color="0.3")
        _plot_axes(ax); ax.spines["left"].set_visible(False)

    pcs = []
    for x0, x1 in ((0, 1), (1, 2)):
        tt = np.linspace(0, 1, 200)
        pcs.append((x0+tt, np.array([w(x0), th(x0), w(x1), th(x1)]) @ _hermite(tt)))
    beam(axs[1], pcs, r"(b)  two Hermite elements: $C^1$", "one slope at the shared node")
    axs[1].plot([0.72, 1.28], [w(1)]*2, color=INK, lw=1.8, zorder=5, solid_capstyle="round")

    pcs, sl = [], []
    for x0, x1 in ((0, 1), (1, 2)):
        nodes = [x0, (x0+x1)/2, x1]
        q = np.polyfit(nodes, [w(n) for n in nodes], 2)
        xx = np.linspace(x0, x1, 200); pcs.append((xx, np.polyval(q, xx)))
        sl.append(np.polyval(np.polyder(q), 1))
    beam(axs[2], pcs, r"(c)  two quadratic Lagrange elements: $C^0$", "two slopes at the shared node")
    for s, c in ((sl[0], E1), (sl[1], E2)):              # each element's own tangent, through the node
        axs[2].plot([0.8, 1.2], [w(1)-.2*s, w(1)+.2*s], color=c, lw=1.8, zorder=5, solid_capstyle="round")
    for ax, nodes in ((axs[1], (0, 2)), (axs[2], (0, 0.5, 1.5, 2))):
        ax.plot(1, w(1), "o", color=INK, ms=7, zorder=6)
        for xn in nodes: ax.plot(xn, w(xn), "o", color=INK, ms=5, zorder=6)
    fig.tight_layout(w_pad=1.6)
    return fig, dict(lagrange_slopes=tuple(sl))


def bernstein_figure():
    """One cubic, written in Lagrange coordinates and in Bernstein coordinates."""
    f = np.poly1d([2.2, -3.6, 1.55, 0.18])
    tn = np.arange(4)/3
    d = f(tn)
    Bm = lambda i, s: comb(3, i)*s**i*(1-s)**(3-i)
    c = np.linalg.solve(np.array([[Bm(i, s) for i in range(4)] for s in tn]), d)

    def L(i, s):
        v = np.ones_like(s)
        for j in range(4):
            if j != i: v *= (s-tn[j])/(tn[i]-tn[j])
        return v

    t = np.linspace(0, 1, 300)
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True)
    for i in range(4):
        axs[0, 0].plot(t, L(i, t), lw=2.1, color=BASISCLRS[i], label=f"$L_{i}$")
        axs[0, 1].plot(t, Bm(i, t), lw=2.1, color=BASISCLRS[i], label=f"$B_{i}$")
    axs[0, 0].plot(tn, np.ones(4), "o", color=INK, ms=5, zorder=5)
    _title(axs[0, 0], "(a)  cubic Lagrange basis", size=12)
    _title(axs[0, 1], "(b)  cubic Bernstein basis", size=12)
    for ax in axs[0]:
        ax.axhline(0, color="0.8", lw=0.8); ax.set_ylim(-0.36, 1.32)
        ax.legend(frameon=False, ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                  handlelength=1.6, columnspacing=1.2)
        _plot_axes(ax)
    for ax in axs[1]:
        ax.plot(t, f(t), color=INK, lw=2.8, zorder=3)
        ax.set_ylim(-0.08, 0.84); ax.set_xlabel("local coordinate $t$"); _plot_axes(ax)
    axs[1, 0].plot(tn, d, "o", ms=9, mfc="white", mec=BASISCLRS[0], mew=2.2, zorder=5)
    for i, ((ox, oy), ha) in enumerate([((10, -2), "left"), ((0, 12), "center"),
                                        ((0, 12), "center"), ((-8, 13), "right")]):
        axs[1, 0].annotate(f"$d_{i}={d[i]:.2f}$", (tn[i], d[i]), xytext=(ox, oy), textcoords="offset points",
                           ha=ha, va="center", fontsize=10, color=INK)
    _title(axs[1, 0], r"(c)  the same cubic: $f=\sum d_i L_i$", size=12)
    axs[1, 1].plot(tn, c, color=BASISCLRS[1], lw=1.4, ls=(0, (4, 3)), zorder=2)
    axs[1, 1].plot(tn, c, "s", ms=8.5, mfc="white", mec=BASISCLRS[1], mew=2.2, zorder=5)
    for i, ((ox, oy), ha) in enumerate([((10, -6), "left"), ((0, 13), "center"),
                                        ((0, -14), "center"), ((-8, 13), "right")]):
        axs[1, 1].annotate(f"$c_{i}={c[i]:.2f}$", (tn[i], c[i]), xytext=(ox, oy), textcoords="offset points",
                           ha=ha, va="center", fontsize=10, color=INK)
    _title(axs[1, 1], r"(d)  the same cubic: $f=\sum c_i B_i$", size=12)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    return fig, dict(d=d, c=c)
