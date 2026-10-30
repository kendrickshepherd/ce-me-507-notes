"""Python mirror of the bracket drawn by the Class 21 widget.

The widget's JavaScript (window.C21Bracket in 21_Class.qmd) is the source of
truth.  Every constant and formula here must match it exactly; the static
figures in Class 21 are drawn from this module so that they show the same part
the reader has just been rotating.  Only the mid-surface is mirrored -- the
static figures never need the thickened solid.
"""
import numpy as np

THETA, RF, HW, LF = np.pi/2, 0.40, 0.62, 1.75
BLO, BHI = 0.10, 0.90          # the band kept on the web, fillets and flange roots
ZB0, ZT0 = -0.12, 2.20

# the default camera of the widget
AZ0, EL0 = -2.27, 0.28


def zbF(u): return ZB0 - 0.38*u
def ztF(u): return ZT0 - 0.50*u - 0.30*u*u


class Patch:
    def __init__(self, kind, pt, vlo=None, vhi=None, hole=None, step=None, tri=None):
        self.kind, self._pt = kind, pt
        self.vlo, self.vhi = vlo, vhi
        self.hole, self.step, self.tri = hole, step, tri

    def pt(self, u, v):
        return np.asarray(self._pt(u, v), float)


def web():
    return Patch('web', lambda u, v: (0.0, -HW + 2*HW*u, ZB0 + v*(ZT0-ZB0)),
                 vlo=BLO, vhi=BHI)


def fillet(sgn):
    def pt(u, v):
        cx, cy = RF, sgn*HW
        ph = np.pi - THETA*u
        return (cx + RF*np.cos(ph), cy + sgn*RF*np.sin(ph), ZB0 + v*(ZT0-ZB0))
    return Patch('fillet', pt, vlo=BLO, vhi=BHI)


def flange(sgn):
    def pt(u, v):
        e = (RF*(1-np.cos(THETA)), sgn*(HW + RF*np.sin(THETA)))
        d = (np.sin(THETA), sgn*np.cos(THETA))
        L = LF*u
        return (e[0] + L*d[0], e[1] + L*d[1], zbF(u) + v*(ztF(u) - zbF(u)))
    return Patch('flange', pt,
                 hole=dict(uc=0.50, vc=0.45, ru=0.20, rv=0.18),
                 step=dict(u0=0.28, vlo=BLO, vhi=BHI),
                 tri=dict(a=0.26, b=0.30))


def mid_surface():
    """The five patches in the widget's order: flange, fillet, web, fillet, flange."""
    return [flange(1), fillet(1), web(), fillet(-1), flange(-1)]


# ---- trimming, all in the parametric square ---------------------------------
def VLO(p): return 0.0 if p.vlo is None else p.vlo
def VHI(p): return 1.0 if p.vhi is None else p.vhi


def clipped(p, u, v):
    """True where the trim has removed (u, v) -- identical to clipped() in the JS."""
    if p.step:
        if u < p.step['u0'] and (v < p.step['vlo']-1e-9 or v > p.step['vhi']+1e-9):
            return True
    elif v < VLO(p)-1e-9 or v > VHI(p)+1e-9:
        return True
    if p.tri:
        ta, tb = p.tri['a'], p.tri['b']
        if tb*(u-1+ta) - ta*v > 0: return True
        if tb*(u-1+ta) + ta*(v-1) > 0: return True
    if p.hole:
        h = p.hole
        a, b = (u-h['uc'])/h['ru'], (v-h['vc'])/h['rv']
        if a*a + b*b < 1: return True
    return False


def region_segs(p):
    """The outer boundary of the kept region, as (a, b, kind) with kind 'trim' or 'bnd'."""
    S = []
    if p.step:
        u0, lo, hi = p.step['u0'], p.step['vlo'], p.step['vhi']
        S += [((0, lo), (u0, lo), 'trim'), ((u0, lo), (u0, 0), 'trim')]
        if p.tri:
            ta, tb = p.tri['a'], p.tri['b']
            S += [((u0, 0), (1-ta, 0), 'bnd'), ((1-ta, 0), (1, tb), 'trim'),
                  ((1, tb), (1, 1-tb), 'bnd'), ((1, 1-tb), (1-ta, 1), 'trim'),
                  ((1-ta, 1), (u0, 1), 'bnd')]
        else:
            S += [((u0, 0), (1, 0), 'bnd'), ((1, 0), (1, 1), 'bnd'), ((1, 1), (u0, 1), 'bnd')]
        S += [((u0, 1), (u0, hi), 'trim'), ((u0, hi), (0, hi), 'trim'), ((0, hi), (0, lo), 'bnd')]
    else:
        a, b = VLO(p), VHI(p)
        S += [((0, a), (1, a), 'trim' if a > 0 else 'bnd'), ((1, a), (1, b), 'bnd'),
              ((1, b), (0, b), 'trim' if b < 1 else 'bnd'), ((0, b), (0, a), 'bnd')]
    return S


def seg_uv(seg, n):
    (u0, v0), (u1, v1), _ = seg
    t = np.linspace(0, 1, n+1)
    return np.c_[u0 + (u1-u0)*t, v0 + (v1-v0)*t]


def region_loop_uv(p, n=16):
    """Closed outer loop of the kept region in (u, v), without the duplicated joins."""
    pts = [seg_uv(s, n)[:-1] for s in region_segs(p)]
    return np.vstack(pts)


def hole_loop_uv(p, n=64):
    if not p.hole: return None
    h = p.hole
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[h['uc'] + h['ru']*np.cos(t), h['vc'] + h['rv']*np.sin(t)]


def to_xyz(p, uv):
    return np.array([p.pt(u, v) for u, v in uv])


# ---- the widget's camera -----------------------------------------------------
def rot(q, az=AZ0, el=EL0):
    """Same rotation as rot() in the JS: returns (screen-x, depth, screen-up)."""
    q = np.atleast_2d(q)
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    x = ca*q[:, 0] - sa*q[:, 1]
    y = sa*q[:, 0] + ca*q[:, 1]
    z = q[:, 2]
    return np.c_[x, ce*y - se*z, se*y + ce*z]


def proj(q, az=AZ0, el=EL0):
    r = rot(q, az, el)
    return r[:, [0, 2]]          # matplotlib's y points up, so no sign flip


def depth(q, az=AZ0, el=EL0):
    return rot(q, az, el)[:, 1]  # larger is farther from the viewer
