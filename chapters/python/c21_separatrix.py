"""Quad meshes of a square plate with a circular hole, and their separatrices.

Separatrices are traced, not drawn: from every irregular vertex, follow each
interior edge and continue "straight through" every regular (valence-4)
interior vertex until the path reaches the boundary or another irregular
vertex.  Cutting the mesh along all separatrices leaves the patches a
tensor-product spline would need.
"""
import numpy as np
import matplotlib.tri as mtri

HALF, R0 = 2.0, 0.8


def ogrid(ns=6, nt=5):
    """Four structured blocks between the square and the hole, one per side."""
    corners = np.array([[HALF, -HALF], [HALF, HALF], [-HALF, HALF], [-HALF, -HALF]])
    nk = 4*ns
    X = np.zeros((nk, nt+1, 2))
    for k in range(nk):
        side, t = divmod(k, ns)
        S = corners[side] + (corners[(side+1) % 4]-corners[side])*t/ns
        th = -np.pi/4 + (np.pi/2)*k/ns
        C = R0*np.array([np.cos(th), np.sin(th)])
        for r in range(nt+1):
            X[k, r] = (1-r/nt)*C + (r/nt)*S
    idx = lambda k, r: (k % nk)*(nt+1) + r
    V = X.reshape(-1, 2)
    Q = [(idx(k, r), idx(k, r+1), idx(k+1, r+1), idx(k+1, r)) for k in range(nk) for r in range(nt)]
    return V, Q


def trisplit(h=0.62, seed=3, smooth=10):
    """Triangulate the plate, then split every triangle into three quads."""
    rng = np.random.default_rng(seed)
    nsq = int(round(2*HALF/h))
    sq = []
    for c0, c1 in [((-HALF,-HALF),(HALF,-HALF)),((HALF,-HALF),(HALF,HALF)),
                   ((HALF,HALF),(-HALF,HALF)),((-HALF,HALF),(-HALF,-HALF))]:
        c0, c1 = np.array(c0), np.array(c1)
        for i in range(nsq): sq.append(c0 + (c1-c0)*i/nsq)
    nci = int(round(2*np.pi*R0/(0.8*h)))
    ci = [R0*np.array([np.cos(a), np.sin(a)]) for a in 2*np.pi*np.arange(nci)/nci]
    bnd = np.array(sq + ci); nb = len(bnd)
    pts = []
    dy = h*np.sqrt(3)/2
    for r, y in enumerate(np.arange(-HALF, HALF+dy, dy)):
        for x in np.arange(-HALF + (h/2 if r % 2 else 0), HALF+h, h):
            q = np.array([x, y]) + rng.normal(0, 0.08*h, 2)
            if np.max(np.abs(q)) < HALF-0.45*h and np.linalg.norm(q) > R0+0.45*h \
               and np.min(np.linalg.norm(bnd-q, axis=1)) > 0.6*h:
                pts.append(q)
    P = np.vstack([bnd, pts])

    def tris(P):
        T = mtri.Triangulation(P[:, 0], P[:, 1]).triangles
        return T[np.linalg.norm(P[T].mean(1), axis=1) > R0]

    T = tris(P)
    for _ in range(smooth):
        nbr = [set() for _ in range(len(P))]
        for t in T:
            for a in range(3): nbr[t[a]].update((t[(a+1) % 3], t[(a+2) % 3]))
        Q = P.copy()
        for k in range(nb, len(P)): Q[k] = P[list(nbr[k])].mean(0)
        P = Q; T = tris(P)
    # orient CCW
    T = [t if np.cross(np.r_[P[t[1]]-P[t[0]],0], np.r_[P[t[2]]-P[t[0]],0])[2] > 0 else t[[0,2,1]] for t in T]
    V = [tuple(p) for p in P]; key = {}
    def node(xy):
        k = (round(xy[0], 9), round(xy[1], 9))
        if k not in key: key[k] = len(V); V.append(tuple(xy))
        return key[k]
    def mid(a, b):
        m = (P[a]+P[b])/2
        if abs(np.linalg.norm(P[a])-R0) < 1e-9 and abs(np.linalg.norm(P[b])-R0) < 1e-9:
            m = R0*m/np.linalg.norm(m)          # keep the hole round
        return node(m)
    Qd = []
    for a, b, c in T:
        g = node(P[[a, b, c]].mean(0))
        mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
        Qd += [(a, mab, g, mca), (b, mbc, g, mab), (c, mca, g, mbc)]
    return np.array(V), Qd, len(T)


def recombined(h=0.62, seed=3, smooth=10):
    """The usual route to an all-quad mesh: triangulate, pair triangles into
    quads, then split every element (quad into 4, leftover triangle into 3)."""
    import c21_mesh
    rng = np.random.default_rng(seed)
    nsq = int(round(2*HALF/h))
    sq = []
    for c0, c1 in [((-HALF,-HALF),(HALF,-HALF)),((HALF,-HALF),(HALF,HALF)),
                   ((HALF,HALF),(-HALF,HALF)),((-HALF,HALF),(-HALF,-HALF))]:
        c0, c1 = np.array(c0), np.array(c1)
        for i in range(nsq): sq.append(c0 + (c1-c0)*i/nsq)
    nci = int(round(2*np.pi*R0/(0.8*h)))
    ci = [R0*np.array([np.cos(a), np.sin(a)]) for a in 2*np.pi*np.arange(nci)/nci]
    bnd = np.array(sq + ci); nb = len(bnd)
    pts = []; dy = h*np.sqrt(3)/2
    for r, y in enumerate(np.arange(-HALF, HALF+dy, dy)):
        for x in np.arange(-HALF + (h/2 if r % 2 else 0), HALF+h, h):
            q = np.array([x, y]) + rng.normal(0, 0.08*h, 2)
            if np.max(np.abs(q)) < HALF-0.45*h and np.linalg.norm(q) > R0+0.45*h \
               and np.min(np.linalg.norm(bnd-q, axis=1)) > 0.6*h:
                pts.append(q)
    P = np.vstack([bnd, pts])
    def tris(P):
        T = mtri.Triangulation(P[:, 0], P[:, 1]).triangles
        return T[np.linalg.norm(P[T].mean(1), axis=1) > R0]
    T = tris(P)
    for _ in range(smooth):
        nbr = [set() for _ in range(len(P))]
        for t in T:
            for a in range(3): nbr[t[a]].update((t[(a+1) % 3], t[(a+2) % 3]))
        Q = P.copy()
        for k in range(nb, len(P)): Q[k] = P[list(nbr[k])].mean(0)
        P = Q; T = tris(P)
    quads, tri_left = c21_mesh.pair_into_quads(P, T)
    V = [tuple(p) for p in P]; key = {}
    def node(xy):
        k = (round(xy[0], 9), round(xy[1], 9))
        if k not in key: key[k] = len(V); V.append(tuple(xy))
        return key[k]
    def mid(a, b):
        m = (P[a]+P[b])/2
        if abs(np.linalg.norm(P[a])-R0) < 1e-9 and abs(np.linalg.norm(P[b])-R0) < 1e-9:
            m = R0*m/np.linalg.norm(m)
        return node(m)
    out = []
    for q in quads:
        g = node(P[list(q)].mean(0)); m = [mid(q[i], q[(i+1) % 4]) for i in range(4)]
        out += [(q[i], m[i], g, m[i-1]) for i in range(4)]
    for t in tri_left:
        g = node(P[list(t)].mean(0)); m = [mid(t[i], t[(i+1) % 3]) for i in range(3)]
        out += [(t[i], m[i], g, m[i-1]) for i in range(3)]
    return np.array(V), out, (len(quads), len(tri_left))


def laplace(V, Q, fixed, iters=80):
    """Plain Laplacian smoothing of every node not in `fixed`."""
    V = np.array(V, float); nbr = {}
    for q in Q:
        for a in range(4):
            nbr.setdefault(q[a], set()).update((q[a-1], q[(a+1) % 4]))
    free = [v for v in nbr if v not in fixed]
    for _ in range(iters):
        for v in free:
            V[v] = V[list(nbr[v])].mean(0)
    return V


def tfi_grid(bot, rig, top, lef):
    """Coons (transfinite) patch from four node arrays.  bot and top run in the
    first direction, lef and rig in the second; corners are shared."""
    nx, ny = len(bot)-1, len(lef)-1
    G = np.zeros((nx+1, ny+1, 2))
    for i in range(nx+1):
        for j in range(ny+1):
            u, v = i/nx, j/ny
            G[i, j] = ((1-v)*bot[i] + v*top[i] + (1-u)*lef[j] + u*rig[j]
                       - ((1-u)*(1-v)*bot[0] + u*(1-v)*bot[-1] + (1-u)*v*top[0] + u*v*top[-1]))
    return G


class _Assembler:
    """Collects patches into one mesh, merging the nodes patches share."""
    def __init__(self): self.V, self.key, self.Q, self.patch_of = [], {}, [], []
    def node(self, p):
        k = (round(float(p[0]), 9), round(float(p[1]), 9))
        if k not in self.key: self.key[k] = len(self.V); self.V.append((float(p[0]), float(p[1])))
        return self.key[k]
    def patch(self, G):
        ids = [[self.node(G[i, j]) for j in range(G.shape[1])] for i in range(G.shape[0])]
        pid = (max(self.patch_of)+1) if self.patch_of else 0
        for i in range(G.shape[0]-1):
            for j in range(G.shape[1]-1):
                self.Q.append((ids[i][j], ids[i+1][j], ids[i+1][j+1], ids[i][j+1]))
                self.patch_of.append(pid)
    def done(self):
        V = np.array(self.V); Q = []
        for q in self.Q:                                     # orient every quad CCW
            X = V[list(q)]; a = 0.5*(X[:, 0] @ np.roll(X[:, 1], -1) - X[:, 1] @ np.roll(X[:, 0], -1))
            Q.append(q if a > 0 else q[::-1])
        return V, Q


def _line(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([a + (b-a)*t for t in np.linspace(0, 1, n+1)])


def twelve_patch(nb=6, na=2, m=2, s=1.2, smooth=80):
    """A ring of four patches round the hole, four side patches and four corner
    patches.  The ring's four outer corners are valence-5 vertices, and their
    separatrices close up into a loop round the hole."""
    A = _Assembler()
    for k in range(4):                                       # build one quarter, rotate it four times
        c, sn_ = np.cos(k*np.pi/2), np.sin(k*np.pi/2)
        R = lambda P: np.atleast_2d(P) @ np.array([[c, sn_], [-sn_, c]])
        th = np.linspace(-np.pi/4, np.pi/4, nb+1)
        hole = R(np.c_[R0*np.cos(th), R0*np.sin(th)])
        box = R(_line((s, -s), (s, s), nb))
        A.patch(tfi_grid(hole, R(_line(hole[-1] @ np.array([[c, -sn_], [sn_, c]]), (s, s), m)),
                         box, R(_line(hole[0] @ np.array([[c, -sn_], [sn_, c]]), (s, -s), m))))   # ring
        A.patch(tfi_grid(box, R(_line((s, s), (HALF, s), na)),
                         R(_line((HALF, -s), (HALF, s), nb)), R(_line((s, -s), (HALF, -s), na))))  # side
        A.patch(tfi_grid(R(_line((s, s), (HALF, s), na)), R(_line((HALF, s), (HALF, HALF), na)),
                         R(_line((s, HALF), (HALF, HALF), na)), R(_line((s, s), (s, HALF), na))))  # corner
    V, Q = A.done()
    fixed = {i for i, p in enumerate(V)
             if abs(np.hypot(*p)-R0) < 1e-9 or max(abs(p[0]), abs(p[1])) > HALF-1e-9}
    return laplace(V, Q, fixed, smooth), Q


def analyse(V, Q):
    """Irregular vertices, separatrix edges and the patches they cut out."""
    V = np.asarray(V)
    edges = {}
    for qi, q in enumerate(Q):
        for a in range(4):
            e = frozenset((q[a], q[(a+1) % 4]))
            edges.setdefault(e, []).append(qi)
    bnd_edge = {e for e, qs in edges.items() if len(qs) == 1}
    nbr = {i: set() for i in range(len(V))}
    nq = {i: 0 for i in range(len(V))}
    for e in edges:
        a, b = tuple(e); nbr[a].add(b); nbr[b].add(a)
    for q in Q:
        for v in q: nq[v] += 1
    on_bnd = {v for e in bnd_edge for v in e}

    def corner(v):                               # the plate's four corners
        return abs(abs(V[v, 0])-HALF) < 1e-9 and abs(abs(V[v, 1])-HALF) < 1e-9

    def regular(v):
        if v not in on_bnd: return len(nbr[v]) == 4
        return nq[v] == (1 if corner(v) else 2)

    def ordered(v):                              # neighbours by angle
        n = list(nbr[v]); d = V[n]-V[v]
        return [n[i] for i in np.argsort(np.arctan2(d[:, 1], d[:, 0]))]

    irregular = [v for v in range(len(V)) if v in nbr and nbr[v] and not regular(v)]
    sep = set()
    for v in irregular:
        for w in nbr[v]:
            e = frozenset((v, w))
            if e in bnd_edge: continue
            prev, cur = v, w; sep.add(e)
            while cur not in on_bnd and regular(cur):
                ring = ordered(cur)
                nxt = ring[(ring.index(prev)+2) % 4]
                sep.add(frozenset((cur, nxt))); prev, cur = cur, nxt
    # patches: quads joined across every edge that is not a separatrix
    parent = list(range(len(Q)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for e, qs in edges.items():
        if len(qs) == 2 and e not in sep:
            parent[find(qs[0])] = find(qs[1])
    roots = sorted({find(i) for i in range(len(Q))})
    patch = [roots.index(find(i)) for i in range(len(Q))]
    interior_irr = [v for v in irregular if v not in on_bnd]
    return dict(irregular=irregular, interior_irregular=interior_irr, sep=sep,
                patch=patch, npatch=len(roots), valence={v: len(nbr[v]) for v in irregular})
