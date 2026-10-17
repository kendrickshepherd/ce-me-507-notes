"""Element-quality measures for the Class 21 mesh comparisons.

All measures are the usual ones from mesh-quality libraries, chosen so a
triangle and a quadrilateral are judged on the same 0-to-1 scale.
"""
import numpy as np


def _corners(X):
    """Corner angles (degrees) and the sine of each, for a polygon listed CCW."""
    n = len(X); ang = []; sn = []
    for k in range(n):
        a, b = X[k-1]-X[k], X[(k+1) % n]-X[k]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        cr = (b[0]*a[1]-b[1]*a[0])/(la*lb)        # positive for a CCW convex corner
        dt = (a@b)/(la*lb)
        ang.append(np.degrees(np.arctan2(cr, dt)) % 360); sn.append(cr)
    return np.array(ang), np.array(sn)


def element_quality(X):
    """Scaled Jacobian (1 = ideal: square, or equilateral triangle), the largest
    and smallest corner angle, the edge ratio, and the area."""
    X = np.asarray(X, float)
    ang, sn = _corners(X)
    e = np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1)
    x, y = X[:, 0], X[:, 1]
    area = 0.5*(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    sj = sn.min() if len(X) == 4 else sn.min()*2/np.sqrt(3)
    return dict(sj=min(sj, 1.0), amin=ang.min(), amax=ang.max(), edge_ratio=e.max()/e.min(), area=area)


def mesh_quality(V, elems):
    V = np.asarray(V, float)
    Q = [element_quality(V[list(el)]) for el in elems]
    sj = np.array([q['sj'] for q in Q]); er = np.array([q['edge_ratio'] for q in Q])
    ar = np.array([q['area'] for q in Q])
    return dict(n=len(Q), sj=sj, sj_min=sj.min(), sj_mean=sj.mean(),
                amin=min(q['amin'] for q in Q), amax=max(q['amax'] for q in Q),
                er=er, er_max=er.max(), size_ratio=ar.max()/ar.min(), all_valid=bool((ar > 0).all()),
                frac_below=lambda t: float((sj < t).mean()))


def optimize(V, elems, fixed, sweeps=40, h=None):
    """Optimization-based smoothing: move each free node, by pattern search, to
    the nearby position that maximizes the worst element touching it.  An
    element is scored by its scaled Jacobian, or by 2 / (edge ratio) once its
    edges pass 2:1, so a move cannot buy good angles by squashing an element
    flat or shrinking it to nothing.  Elements stay valid because a move that makes any of them worse
    at its worst is refused."""
    V = np.array(V, float); elems = [tuple(e) for e in elems]
    adj = {}
    for k, el in enumerate(elems):
        for v in el: adj.setdefault(v, []).append(k)
    if h is None:
        h = np.median([np.linalg.norm(V[el[0]]-V[el[1]]) for el in elems])
    free = [v for v in adj if v not in fixed]
    def q_el(k):
        e = element_quality(V[list(elems[k])])
        return min(e['sj'], 2.0/e['edge_ratio'])  # angles, unless an element passes 2:1
    def score(v):
        s = [q_el(k) for k in adj[v]]
        return min(s) + 1e-3*np.mean(s)
    dirs = [np.array(d, float)/np.linalg.norm(d) for d in
            [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]]
    for sweep in range(sweeps):
        moved = 0
        for v in free:
            best = score(v); p0 = V[v].copy()
            for step in (0.2*h, 0.08*h, 0.03*h, 0.01*h):
                improved = True
                while improved:
                    improved = False
                    for d in dirs:
                        V[v] = p0 + step*d; s = score(v)
                        if s > best + 1e-9: best, p0, improved = s, V[v].copy(), True; moved += 1
                        else: V[v] = p0
            V[v] = p0
        if moved == 0: break
    return V


_W_TRI_INV = np.linalg.inv(np.array([[1.0, 0.5], [0.0, np.sqrt(3)/2]]))   # equilateral reference corner


def _distortion(X):
    """Sum over an element's corners of (condition number - 1): zero for a
    square (or equilateral triangle), growing without bound as a corner
    flattens, so an element can never be smoothed inside out."""
    n = len(X); tot = 0.0
    for k in range(n):
        A = np.c_[X[(k+1) % n]-X[k], X[k-1]-X[k]]
        if n == 3: A = A @ _W_TRI_INV
        d = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
        if d <= 1e-14: return np.inf
        tot += (A*A).sum()/(2*d) - 1
    return tot


def smooth_global(V, elems, fixed, slide=None, size_weight=0.25, sweeps=60):
    """Minimize the total distortion of the mesh plus a gentle pull of every
    element's area toward the average.  Free nodes move in the plane; nodes in
    `slide` move only along their boundary curve (a dense polyline, closed if
    its ends coincide)."""
    V = np.array(V, float); elems = [tuple(e) for e in elems]; slide = slide or {}
    adj = {}
    for k, el in enumerate(elems):
        for v in el: adj.setdefault(v, []).append(k)
    area = lambda el: 0.5*abs(V[list(el), 0] @ np.roll(V[list(el), 1], -1) - V[list(el), 1] @ np.roll(V[list(el), 0], -1))
    target = np.mean([area(el) for el in elems])
    def local(v):
        s = 0.0
        for k in adj[v]:
            X = V[list(elems[k])]; s += _distortion(X)
            x, y = X[:, 0], X[:, 1]
            a = 0.5*(x @ np.roll(y, -1) - y @ np.roll(x, -1))
            s += size_weight*np.log(max(a, 1e-14)/target)**2
        return s
    h = np.sqrt(target)
    # arclength parametrization for the sliding nodes
    curve = {}
    for v, P in slide.items():
        P = np.asarray(P, float); L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        closed = np.linalg.norm(P[0]-P[-1]) < 1e-12
        s0 = L[np.argmin(np.linalg.norm(P - V[v], axis=1))]
        curve[v] = (P, L, closed, s0)
    def at(v, s):
        P, L, closed, _ = curve[v]
        s = s % L[-1] if closed else min(max(s, 0.0), L[-1])
        i = min(np.searchsorted(L, s) , len(L)-1); i = max(i, 1)
        t = (s - L[i-1])/(L[i]-L[i-1]); return P[i-1] + t*(P[i]-P[i-1]), s
    free = [v for v in adj if v not in fixed]
    dirs = [np.array(d, float)/np.linalg.norm(d) for d in [(1, 0), (-1, 0), (0, 1), (0, -1),
                                                            (1, 1), (1, -1), (-1, 1), (-1, -1)]]
    for sweep in range(sweeps):
        moved = 0
        for v in free:
            best = local(v)
            if v in curve:
                s_cur = curve[v][3]
                for step in (0.15*h, 0.05*h, 0.015*h):
                    improved = True
                    while improved:
                        improved = False
                        for sg in (1, -1):
                            old = V[v].copy(); V[v], s_new = at(v, s_cur + sg*step); f = local(v)
                            if f < best - 1e-12: best, s_cur, improved = f, s_new, True; moved += 1
                            else: V[v] = old
                P_, L_, c_, _ = curve[v]; curve[v] = (P_, L_, c_, s_cur)
            else:
                p0 = V[v].copy()
                for step in (0.15*h, 0.05*h, 0.015*h):
                    improved = True
                    while improved:
                        improved = False
                        for d in dirs:
                            V[v] = p0 + step*d; f = local(v)
                            if f < best - 1e-12: best, p0, improved = f, V[v].copy(), True; moved += 1
                            else: V[v] = p0
                V[v] = p0
        if moved == 0: break
    return V
