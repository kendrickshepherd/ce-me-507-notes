"""Quad-dominant surface mesh of the bracket's mid-surface, for the Class 21 figure.

Fillets and web are bands of their parametric squares, so they get structured
quads.  Each flange is planar (y = const), so it is meshed in its own (x, z)
plane: Delaunay on boundary + interior points, smoothing, then greedy pairing of
triangles into quads -- the same recipe production quad-dominant meshers use.
"""
import numpy as np
import matplotlib.tri as mtri
from matplotlib.path import Path
import c21_bracket as B

NV = 8          # divisions across the band height, shared by every seam
NU_WEB, NU_FIL = 5, 3


def _seg_points(p, seg, h, n_force=None):
    xyz = B.to_xyz(p, B.seg_uv(seg, 64))
    L = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
    n = n_force or max(1, int(round(L/h)))
    return B.to_xyz(p, B.seg_uv(seg, n))[:-1]


def flange_mesh(h=0.22, h_hole=0.20, smooth=12):
    F = B.flange(1)
    segs = B.region_segs(F)
    outer = []
    for s in segs:
        seam = (s[0][0] == 0 and s[1][0] == 0)
        outer.append(_seg_points(F, s, h, NV if seam else None))
    outer = np.vstack(outer)[:, [0, 2]]
    hl = B.to_xyz(F, B.hole_loop_uv(F, 400))[:, [0, 2]]
    per = np.sum(np.linalg.norm(np.diff(np.vstack([hl, hl[:1]]), axis=0), axis=1))
    nh = int(round(per/h_hole))
    hole = B.to_xyz(F, B.hole_loop_uv(F, nh))[:, [0, 2]]

    po, ph = Path(np.vstack([outer, outer[:1]])), Path(np.vstack([hole, hole[:1]]))
    bnd = np.vstack([outer, hole])
    lo, hi = outer.min(0), outer.max(0)
    pts = []
    dy = h*np.sqrt(3)/2
    for r, z in enumerate(np.arange(lo[1], hi[1]+dy, dy)):
        for x in np.arange(lo[0] + (h/2 if r % 2 else 0), hi[0]+h, h):
            q = np.array([x, z])
            if po.contains_point(q) and not ph.contains_point(q) \
               and np.min(np.linalg.norm(bnd - q, axis=1)) > 0.6*h:
                pts.append(q)
    P = np.vstack([bnd, np.array(pts)])
    nb = len(bnd)

    def keep_tris(P):
        T = mtri.Triangulation(P[:, 0], P[:, 1]).triangles
        c = P[T].mean(1)
        ok = po.contains_points(c) & ~ph.contains_points(c)
        return T[ok]

    T = keep_tris(P)
    # Laplacian smoothing of the interior points, re-triangulating each pass
    for _ in range(smooth):
        nbr = [set() for _ in range(len(P))]
        for t in T:
            for a in range(3):
                nbr[t[a]].update((t[(a+1) % 3], t[(a+2) % 3]))
        Q = P.copy()
        for k in range(nb, len(P)):
            if nbr[k]: Q[k] = P[list(nbr[k])].mean(0)
        P = Q
        T = keep_tris(P)
    return P, T, nb, outer, hole


def _tri_area(P, t):
    a, b, c = P[t[0]], P[t[1]], P[t[2]]
    return 0.5*((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))


def pair_into_quads(P, T, max_dev=45.0):
    """Greedy pairing of neighbouring triangles into quads whose corners all
    stay within max_dev degrees of 90."""
    T = [tuple(t) if _tri_area(P, t) > 0 else (t[0], t[2], t[1]) for t in T]
    edge = {}
    for k, t in enumerate(T):
        for a in range(3):
            e = frozenset((t[a], t[(a+1) % 3]))
            edge.setdefault(e, []).append(k)
    cand = []
    for e, ks in edge.items():
        if len(ks) != 2: continue
        k1, k2 = ks
        p, q = tuple(e)
        r = [v for v in T[k1] if v not in e][0]
        s = [v for v in T[k2] if v not in e][0]
        # walk the quad around in the orientation of triangle k1
        t1 = T[k1]; i = t1.index(r)
        a, b = t1[(i+1) % 3], t1[(i+2) % 3]       # r -> a -> b is CCW
        quad = (r, a, s, b)
        X = P[list(quad)]
        ang, convex = [], True
        for m in range(4):
            u, v = X[m-1]-X[m], X[(m+1) % 4]-X[m]
            f, g = X[(m+1) % 4]-X[m], X[m-1]-X[m]
            cr = f[0]*g[1] - f[1]*g[0]
            if cr <= 0: convex = False
            ang.append(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v), -1, 1))))
        if not convex: continue
        dev = max(abs(x-90) for x in ang)
        if dev <= max_dev: cand.append((dev, k1, k2, quad))
    cand.sort()
    used, quads = set(), []
    for dev, k1, k2, quad in cand:
        if k1 in used or k2 in used: continue
        used.update((k1, k2)); quads.append(quad)
    tris = [T[k] for k in range(len(T)) if k not in used]
    return quads, tris


def band_mesh(p, nu):
    u = np.linspace(0, 1, nu+1); v = np.linspace(B.BLO, B.BHI, NV+1)
    X = np.array([[p.pt(a, b) for b in v] for a in u])        # (nu+1, NV+1, 3)
    return [np.array([X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]]) for i in range(nu) for j in range(NV)]


def _lift(faces_xz, P):
    out = []
    for sgn in (1, -1):
        y = sgn*(B.HW + B.RF)
        out += [(kind, np.array([[P[k, 0], y, P[k, 1]] for k in ids])) for kind, ids in faces_xz]
    return out


def _bands():
    faces = []
    for p in (B.fillet(1), B.fillet(-1)):
        faces += [('quad', f) for f in band_mesh(p, NU_FIL)]
    faces += [('quad', f) for f in band_mesh(B.web(), NU_WEB)]
    return faces


from functools import lru_cache


@lru_cache(maxsize=None)
def bracket_faces():
    """All faces of the quad-dominant mid-surface mesh as 3D polygons, tagged
    'quad' or 'tri'.  The flange is smoothed by the same optimizer as the
    quad-only mesh, so the two can be compared fairly."""
    P, T, nb, outer, hole = flange_mesh()
    quads, tris = pair_into_quads(P, T)
    P = smooth_flange(P, list(quads)+list(tris), set(range(nb)))
    faces = _lift([('quad', q) for q in quads] + [('tri', t) for t in tris], P) + _bands()
    return faces, dict(P=P, T=T, quads=quads, tris=tris, outer=outer, hole=hole, nb=nb)


@lru_cache(maxsize=None)
def bracket_faces_quad():
    """The same bracket, quad-only: the block-layout flanges and the same bands."""
    r = flange_blocks()
    return _lift([('quad', q) for q in r['Q']], r['V']) + _bands(), r


# =============================================================================
#  A quad-only, boundary-aligned flange, laid out in blocks
# =============================================================================
# The flange is essentially a ring about 2.4 elements wide round the hole, so
# the layout follows the twelve-patch plate: a ring of quads hugs the hole and
# its four outer corners (valence 5) send spokes to the flange's corners.  Rows
# of blocks run along every edge, and each chamfered corner is filled by the
# standard three-block triangle, whose centre is a valence-3 vertex.
import c21_separatrix as _S
import c21_quality as _Q

RING_D = 0.20                              # thickness of the ring round the hole
V_THETA = (240.0, 330.0, 47.5, 127.5)      # where the ring's corners sit (parametric angle)
SEG = dict(left=5, bottom=4, right=3, top=4)    # hole segments on each side of the ring


def _flange_xz(uv):
    return B.to_xyz(B.flange(1), np.atleast_2d(uv))[:, [0, 2]]


def _uv_line(a, b, n):
    """n segments along a straight line in (u, v), mapped to the flange."""
    return _flange_xz(_S._line(a, b, n))


def _hole_and_box(thetas):
    """Points on the hole and on the ring's outer curve (the hole offset outward
    by RING_D along its normal) at the given parametric angles."""
    h = B.flange(1).hole
    t = np.radians(np.asarray(thetas, float))
    P = _flange_xz(np.c_[h['uc'] + h['ru']*np.cos(t), h['vc'] + h['rv']*np.sin(t)])
    e = 1e-6
    Pp = _flange_xz(np.c_[h['uc'] + h['ru']*np.cos(t+e), h['vc'] + h['rv']*np.sin(t+e)])
    T = (Pp - P)/e; T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.c_[T[:, 1], -T[:, 0]]                       # outward for a counter-clockwise loop
    return P, P + RING_D*N


def _arc(a, b, n):
    """n equal steps of parametric angle from a to b (degrees), both ends included."""
    return np.linspace(a, b, n+1)


def _triangle(A, side01, side12, side20):
    """Fill a triangle with three quad blocks meeting at its centre.  side01 runs
    corner0 -> corner1, side12 corner1 -> corner2, side20 corner2 -> corner0."""
    n01, n12, n20 = len(side01)-1, len(side12)-1, len(side20)-1
    s01, s12, s20 = (n12+n20-n01)//2, (n01+n20-n12)//2, (n01+n12-n20)//2
    assert min(s01, s12, s20) >= 1 and (n01+n12+n20) % 2 == 0
    O = (side01[0] + side12[0] + side20[0])/3
    M01, M12, M20 = side01[s20], side12[s01], side20[s12]
    sp01, sp12, sp20 = _S._line(O, M01, s01), _S._line(O, M12, s12), _S._line(O, M20, s20)
    A.patch(_S.tfi_grid(side01[:s20+1], sp01[::-1], sp20[::-1], side20[s12:][::-1]))   # at corner 0
    A.patch(_S.tfi_grid(side12[:s01+1], sp12[::-1], sp01[::-1], side01[s20:][::-1]))   # at corner 1
    A.patch(_S.tfi_grid(side20[:s12+1], sp20[::-1], sp12[::-1], side12[s01:][::-1]))   # at corner 2
    return O


def flange_blocks(smooth=60):
    """The front flange, quad-only, in its own (x, z) plane.  Returns the node
    coordinates, the quads, the block-boundary edges and the irregular vertices."""
    P = {k: _flange_xz(uv)[0] for k, uv in dict(P1=(0, .1), P2=(.28, .1), P3=(.28, 0), P4=(.74, 0),
                                                  P5=(1, .3), P6=(1, .7), P7=(.74, 1), P8=(.28, 1),
                                                  P9=(.28, .9), P10=(0, .9)).items()}
    hs = (P['P10'][1] - P['P1'][1])/NV                  # the seam's own spacing, shared with the fillet
    S1 = np.array([P['P1'][0], P['P1'][1] + 1*hs]); S2 = np.array([P['P1'][0], P['P1'][1] + 6*hs])
    t1, t2, t3, t4 = V_THETA
    t3 += 360
    hole_b, box_b = _hole_and_box(_arc(t1, t2, SEG['bottom']))
    hole_r, box_r = _hole_and_box(_arc(t2, t3, SEG['right']))
    hole_t, box_t = _hole_and_box(_arc(t3, t4+360, SEG['top']))
    hole_l, box_l = _hole_and_box(_arc(t4+360, t1+360, SEG['left']))
    V1, V2, V3, V4 = box_b[0], box_r[0], box_t[0], box_l[0]
    L = _S._line
    A = _S._Assembler()
    # the ring round the hole
    for hh, bb in ((hole_b, box_b), (hole_r, box_r), (hole_t, box_t), (hole_l, box_l)):
        A.patch(_S.tfi_grid(hh, L(hh[-1], bb[-1], 1), bb, L(hh[0], bb[0], 1)))
    seam = lambda a, b: L(a, b, int(round(abs(b[1]-a[1])/hs)))
    # left side: three blocks against the seam
    A.patch(_S.tfi_grid(_uv_line((0, .1), (.28, .1), 2), L(P['P2'], V1, 1), L(S1, V1, 2), seam(P['P1'], S1)))
    A.patch(_S.tfi_grid(L(S1, V1, 2), box_l[::-1], L(S2, V4, 2), seam(S1, S2)))
    A.patch(_S.tfi_grid(L(S2, V4, 2), L(V4, P['P9'], 2), _uv_line((0, .9), (.28, .9), 2), seam(S2, P['P10'])))
    # top: the block under the top edge, its left side running V4 -> P9 -> P8
    left_top = np.vstack([L(V4, P['P9'], 2), _uv_line((.28, .9), (.28, 1), 1)[1:]])
    A.patch(_S.tfi_grid(box_t[::-1], L(V3, P['P7'], 3), _uv_line((.28, 1), (.74, 1), 4), left_top))
    # bottom: the block over the bottom edge, its left side running P3 -> P2 -> V1
    left_bot = np.vstack([_uv_line((.28, 0), (.28, .1), 1), L(P['P2'], V1, 1)[1:]])
    A.patch(_S.tfi_grid(_uv_line((.28, 0), (.74, 0), 4), L(P['P4'], V2, 2), box_b, left_bot))
    # right: the block against the end edge, between the two chamfer triangles
    A.patch(_S.tfi_grid(L(V2, P['P5'], 3), _uv_line((1, .3), (1, .7), 3), L(V3, P['P6'], 3), box_r))
    O_br = _triangle(A, L(V2, P['P4'], 2), _uv_line((.74, 0), (1, .3), 3), L(P['P5'], V2, 3))
    O_tr = _triangle(A, L(V3, P['P6'], 3), _uv_line((1, .7), (.74, 1), 4), L(P['P7'], V3, 3))
    V, Q = A.done()
    # boundary nodes stay put; everything else is smoothed
    outer = B.to_xyz(B.flange(1), B.region_loop_uv(B.flange(1), 400))[:, [0, 2]]
    from matplotlib.path import Path as _Path
    hole_all = np.vstack([hole_b, hole_r, hole_t, hole_l])
    def on_boundary(p):
        d_out = np.min(np.linalg.norm(outer - p, axis=1))
        d_h = np.min(np.linalg.norm(hole_all - p, axis=1))
        return d_out < 2e-3 or d_h < 1e-9
    fixed = {i for i, p in enumerate(V) if on_boundary(p)}
    V = _S.laplace(V, Q, fixed, smooth)
    V = smooth_flange(V, Q, fixed)
    # block borders (edges whose two quads come from different blocks) and the
    # vertices where other than four quads meet, for drawing
    owner = {}
    for qi, q in enumerate(Q):
        for a in range(4):
            owner.setdefault(frozenset((q[a], q[(a+1) % 4])), []).append(A.patch_of[qi])
    borders = [tuple(e) for e, ps in owner.items() if len(ps) == 2 and ps[0] != ps[1]]
    return dict(V=V, Q=Q, fixed=fixed, borders=borders, irregular=_irregular(Q, fixed),
                hole=hole_all)


def _irregular(elems, boundary):
    """Interior vertices that do not have exactly four edges."""
    nbr = {}
    for el in elems:
        for a in range(len(el)):
            nbr.setdefault(el[a], set()).update((el[a-1], el[(a+1) % len(el)]))
    return [v for v, n in nbr.items() if v not in boundary and len(n) != 4]



def flange_sliders(V, boundary):
    """Sort a flange mesh's boundary nodes: the corners and the seam stay fixed
    (the seam must keep matching the fillet), every other boundary node may
    slide along its own edge or round the hole."""
    F = B.flange(1)
    pieces = []
    for sg in B.region_segs(F):
        seam = sg[0][0] == 0 and sg[1][0] == 0
        pieces.append((B.to_xyz(F, B.seg_uv(sg, 400))[:, [0, 2]], seam))
    hole = B.to_xyz(F, B.hole_loop_uv(F, 2000))[:, [0, 2]]
    hole = np.vstack([hole, hole[:1]])
    corners = np.array([P[0] for P, _ in pieces])
    fixed, slide = set(), {}
    for v in boundary:
        p = V[v]
        if np.min(np.linalg.norm(corners - p, axis=1)) < 1e-6: fixed.add(v); continue
        dh = np.min(np.linalg.norm(hole - p, axis=1))
        best = min(((np.min(np.linalg.norm(P - p, axis=1)), P, seam) for P, seam in pieces), key=lambda t: t[0])
        if dh < best[0]: slide[v] = hole
        elif best[2]: fixed.add(v)
        else: slide[v] = best[1]
    return fixed, slide



def smooth_flange(V, elems, boundary):
    """The smoothing both flange meshes get, so they can be compared fairly:
    first minimize the total distortion with boundary nodes free to slide along
    their edges and round the hole (corners and the seam stay put), then lift
    the worst elements with the boundary held where it ended up."""
    fx, sl = flange_sliders(V, boundary)
    V = _Q.smooth_global(V, elems, fx, sl, size_weight=0.25)
    return _Q.optimize(V, elems, set(boundary))
