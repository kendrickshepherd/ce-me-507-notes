// Two-dimensional Galerkin finite elements for steady heat conduction,
//     -div(kappa grad u) = f   on the rectangle ]0, W[ x ]0, H[,
// in the browser, for the explorer on the 2D Galerkin page.  A port of
// fem2d.py, which draws the static figures on that page -- change the two
// together.  Hughes' conventions throughout: global nodes A = 1..nnp, local
// nodes a = 1..nen, elements e = 1..nel, 1-based ID / IEN / LM arrays with
// ID = 0 marking a Dirichlet node.  (JavaScript arrays are 0-based, so
// IEN[e-1][a-1] holds the global number A of local node a of element e.)
// Local nodes: corners counter-clockwise from (xi, eta) = (-1, -1), then for
// Q9 the mid-side nodes counter-clockwise from the bottom, then the centre.

export const EDGES = ["left", "right", "bottom", "top"];
export const LOCAL_IJ = {
  1: [[0, 0], [1, 0], [1, 1], [0, 1]],
  2: [[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1], [1, 1]]
};
const SIDES = {bottom: t => [t, -1], right: t => [1, t], top: t => [t, 1], left: t => [-1, t]};

export function gauss(n) {
  // Gauss-Legendre points and weights on [-1, 1], by Newton's method on P_n
  const x = new Array(n), w = new Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), dp = 1;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0;
      for (let j = 1; j <= n; j++) {
        const p3 = p2; p2 = p1;
        p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
      }
      dp = n * (z * p1 - p2) / (z * z - 1);
      const z0 = z;
      z = z0 - p1 / dp;
      if (Math.abs(z - z0) < 1e-15) break;
    }
    x[n - 1 - i] = z;
    w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  return {x, w};
}

export function lagrange1d(p, s) {
  // N_i(s) and N_i,s(s), i = 0..p, on equally spaced nodes in [-1, 1]
  const nd = Array.from({length: p + 1}, (_, i) => -1 + 2 * i / p), N = [], dN = [];
  for (let i = 0; i <= p; i++) {
    let v = 1, d = 0;
    for (let k = 0; k <= p; k++) if (k !== i) v *= (s - nd[k]) / (nd[i] - nd[k]);
    for (let m = 0; m <= p; m++) {
      if (m === i) continue;
      let t = 1 / (nd[i] - nd[m]);
      for (let k = 0; k <= p; k++) if (k !== i && k !== m) t *= (s - nd[k]) / (nd[i] - nd[k]);
      d += t;
    }
    N.push(v); dN.push(d);
  }
  return {N, dN};
}

export function shape(p, xi, eta) {
  // N_a, N_a,xi, N_a,eta for a = 1..nen (index a-1): tensor products N_i(xi) N_j(eta)
  const X = lagrange1d(p, xi), Y = lagrange1d(p, eta);
  const N = [], Nxi = [], Neta = [];
  for (const [i, j] of LOCAL_IJ[p]) {
    N.push(X.N[i] * Y.N[j]); Nxi.push(X.dN[i] * Y.N[j]); Neta.push(X.N[i] * Y.dN[j]);
  }
  return {N, Nxi, Neta};
}

export function parentNodes(p) {
  return LOCAL_IJ[p].map(([i, j]) => [-1 + 2 * i / p, -1 + 2 * j / p]);
}

export function makeMesh({W, H, n1, n2, p = 1, distort = 0, order = "x1"}) {
  const N1 = p * n1 + 1, N2 = p * n2 + 1, nnp = N1 * N2, nel = n1 * n2, nen = (p + 1) ** 2;
  const nodeNumber = (I, J) => order === "x1" ? 1 + I + N1 * J : 1 + J + N2 * I;
  const elementNumber = (e1, e2) => order === "x1" ? 1 + e1 + n1 * e2 : 1 + e2 + n2 * e1;
  const h1 = W / n1, h2 = H / n2;
  const X = new Array(nnp), IJ = new Array(nnp);
  for (let I = 0; I < N1; I++) for (let J = 0; J < N2; J++) {
    const A = nodeNumber(I, J), x1 = W * I / (N1 - 1), x2 = H * J / (N2 - 1);
    const d1 = distort * h1 * Math.sin(2 * Math.PI * x1 / W) * Math.sin(Math.PI * x2 / H);
    const d2 = distort * h2 * Math.sin(Math.PI * x1 / W) * Math.sin(2 * Math.PI * x2 / H);
    X[A - 1] = [x1 + d1, x2 + d2];
    IJ[A - 1] = [I, J];
  }
  const IEN = new Array(nel), EIJ = new Array(nel);
  for (let e1 = 0; e1 < n1; e1++) for (let e2 = 0; e2 < n2; e2++) {
    const e = elementNumber(e1, e2);
    EIJ[e - 1] = [e1, e2];
    IEN[e - 1] = LOCAL_IJ[p].map(([i, j]) => nodeNumber(p * e1 + i, p * e2 + j));
  }
  const edgeNodes = {
    left: Array.from({length: N2}, (_, J) => nodeNumber(0, J)),
    right: Array.from({length: N2}, (_, J) => nodeNumber(N1 - 1, J)),
    bottom: Array.from({length: N1}, (_, I) => nodeNumber(I, 0)),
    top: Array.from({length: N1}, (_, I) => nodeNumber(I, N2 - 1))
  };
  const boundarySides = [];
  for (let e = 1; e <= nel; e++) {
    const [e1, e2] = EIJ[e - 1];
    if (e2 === 0) boundarySides.push([e, "bottom", "bottom"]);
    if (e1 === n1 - 1) boundarySides.push([e, "right", "right"]);
    if (e2 === n2 - 1) boundarySides.push([e, "top", "top"]);
    if (e1 === 0) boundarySides.push([e, "left", "left"]);
  }
  const mesh = {W, H, n1, n2, p, distort, order, N1, N2, nnp, nel, nen, X, IJ, IEN, EIJ,
                edgeNodes, boundarySides, nodeNumber, elementNumber};
  mesh.map = (e, xi, eta) => mapPoint(mesh, e, xi, eta);
  return mesh;
}

export function mapPoint(mesh, e, xi, eta) {
  // x(xi, eta), DF = [[x1,xi, x1,eta], [x2,xi, x2,eta]], J = det DF, and the shape data
  const sh = shape(mesh.p, xi, eta), ien = mesh.IEN[e - 1];
  let x1 = 0, x2 = 0, a11 = 0, a12 = 0, a21 = 0, a22 = 0;
  for (let a = 0; a < ien.length; a++) {
    const [X1, X2] = mesh.X[ien[a] - 1];
    x1 += sh.N[a] * X1; x2 += sh.N[a] * X2;
    a11 += sh.Nxi[a] * X1; a12 += sh.Neta[a] * X1;
    a21 += sh.Nxi[a] * X2; a22 += sh.Neta[a] * X2;
  }
  return {x: [x1, x2], DF: [[a11, a12], [a21, a22]], J: a11 * a22 - a12 * a21, sh};
}

export function gradX(DF, J, dxi, deta) {
  // DF^{-T} (dxi, deta): the chain rule
  return [(DF[1][1] * dxi - DF[1][0] * deta) / J, (-DF[0][1] * dxi + DF[0][0] * deta) / J];
}

export function tabFlux(hTab, tabs) {
  return {h: s => tabs.some(([a, b]) => s >= a && s <= b) ? hTab : 0, breaks: tabs.flat()};
}

export function dirichlet(mesh, g, overrides = new Map()) {
  // Map A -> g_A.  g: {edge: (x1, x2) => value} for the Dirichlet edges; a corner on
  // two Dirichlet edges takes the first edge in EDGES.  overrides: A -> value or null.
  const gd = new Map();
  for (const edge of EDGES) {
    if (!(edge in g)) continue;
    for (const A of mesh.edgeNodes[edge]) {
      if (gd.has(A)) continue;
      const [x1, x2] = mesh.X[A - 1];
      gd.set(A, g[edge](x1, x2));
    }
  }
  for (const [A, v] of overrides) {
    if (A < 1 || A > mesh.nnp) continue;
    if (v === null) gd.delete(A); else gd.set(A, v);
  }
  return gd;
}

export function elementArrays(mesh, e, pb, gd) {
  const p = mesh.p, nen = mesh.nen, G = gauss(p + 1);
  const ke = Array.from({length: nen}, () => new Array(nen).fill(0));
  const body = new Array(nen).fill(0), fh = new Array(nen).fill(0), gp = [];
  for (let qi = 0; qi < G.x.length; qi++) for (let qj = 0; qj < G.x.length; qj++) {
    const xi = G.x[qi], eta = G.x[qj], w = G.w[qi] * G.w[qj];
    const m = mapPoint(mesh, e, xi, eta);
    const grads = m.sh.Nxi.map((d, a) => gradX(m.DF, m.J, d, m.sh.Neta[a]));
    const fq = pb.f(m.x[0], m.x[1]);
    for (let a = 0; a < nen; a++) {
      body[a] += m.sh.N[a] * fq * m.J * w;
      for (let b = 0; b < nen; b++)
        ke[a][b] += pb.kappa * (grads[a][0] * grads[b][0] + grads[a][1] * grads[b][1]) * m.J * w;
    }
    gp.push({xi, eta, w, x: m.x, DF: m.DF, J: m.J, N: m.sh.N, Nxi: m.sh.Nxi, Neta: m.sh.Neta, grads});
  }
  // Neumann sides of this element
  const T = gauss(p + 2);
  for (const [ee, side, edge] of mesh.boundarySides) {
    if (ee !== e || !pb.h[edge]) continue;
    const {h, breaks} = pb.h[edge], axis = (edge === "left" || edge === "right") ? 1 : 0;
    const s0 = mapPoint(mesh, e, ...SIDES[side](-1)).x[axis], s1 = mapPoint(mesh, e, ...SIDES[side](1)).x[axis];
    const cuts = [-1, 1, ...breaks.filter(b => b > Math.min(s0, s1) && b < Math.max(s0, s1))
                                   .map(b => 2 * (b - s0) / (s1 - s0) - 1)].sort((a, b) => a - b);
    for (let c = 0; c + 1 < cuts.length; c++) {
      const t0 = cuts[c], t1 = cuts[c + 1];
      for (let q = 0; q < T.x.length; q++) {
        const t = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * T.x[q], wt = 0.5 * (t1 - t0) * T.w[q];
        const m = mapPoint(mesh, e, ...SIDES[side](t));
        const along = (side === "bottom" || side === "top") ? 1 : 2;   // column of DF along the side
        const jl = along === 1 ? Math.hypot(m.DF[0][0], m.DF[1][0]) : Math.hypot(m.DF[0][1], m.DF[1][1]);
        const hv = h(m.x[axis]);
        if (hv === 0) continue;
        for (let a = 0; a < nen; a++) fh[a] += m.sh.N[a] * hv * jl * wt;
      }
    }
  }
  const ge = mesh.IEN[e - 1].map(A => gd.has(A) ? gd.get(A) : 0);
  const fg = ke.map(r => r.reduce((s, v, b) => s + v * ge[b], 0));
  const total = body.map((v, a) => v + fh[a] - fg[a]);
  return {ke, body, h: fh, g: fg, ge, total, gp};
}

// ---------------------------------------------------------------- linear algebra
function toCSR(n, trip) {
  // trip: Map key -> value with key = i * n + j; returns CSR arrays (0-based)
  const keys = [...trip.keys()].sort((a, b) => a - b);
  const rowPtr = new Int32Array(n + 1), col = new Int32Array(keys.length), val = new Float64Array(keys.length);
  keys.forEach((k, t) => {
    const i = Math.floor(k / n);
    col[t] = k - i * n; val[t] = trip.get(k); rowPtr[i + 1]++;
  });
  for (let i = 0; i < n; i++) rowPtr[i + 1] += rowPtr[i];
  return {n, rowPtr, col, val};
}

export function csrMul(A, x, y = new Float64Array(A.n)) {
  for (let i = 0; i < A.n; i++) {
    let s = 0;
    for (let t = A.rowPtr[i]; t < A.rowPtr[i + 1]; t++) s += A.val[t] * x[A.col[t]];
    y[i] = s;
  }
  return y;
}

export function csrDense(A) {
  const M = Array.from({length: A.n}, () => new Array(A.n).fill(0));
  for (let i = 0; i < A.n; i++) for (let t = A.rowPtr[i]; t < A.rowPtr[i + 1]; t++) M[i][A.col[t]] = A.val[t];
  return M;
}

function denseSolve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    if (Math.abs(M[c][c]) < 1e-300) return null;
    for (let r = c + 1; r < n; r++) {
      const t = M[r][c] / M[c][c];
      if (t !== 0) for (let k = c; k <= n; k++) M[r][k] -= t * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

function pcg(A, b, tol = 1e-12, maxit = 20000) {
  const n = A.n, x = new Float64Array(n), r = Float64Array.from(b), z = new Float64Array(n);
  const dg = new Float64Array(n), Ap = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let t = A.rowPtr[i]; t < A.rowPtr[i + 1]; t++) if (A.col[t] === i) dg[i] = A.val[t];
  const bn = Math.hypot(...b) || 1;
  for (let i = 0; i < n; i++) z[i] = r[i] / dg[i];
  const p = Float64Array.from(z);
  let rz = r.reduce((s, v, i) => s + v * z[i], 0), it = 0;
  for (it = 1; it <= maxit; it++) {
    csrMul(A, p, Ap);
    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    const al = rz / pAp;
    let rr = 0;
    for (let i = 0; i < n; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; rr += r[i] * r[i]; }
    if (Math.sqrt(rr) < tol * bn) break;
    let rz2 = 0;
    for (let i = 0; i < n; i++) { z[i] = r[i] / dg[i]; rz2 += r[i] * z[i]; }
    const be = rz2 / rz;
    for (let i = 0; i < n; i++) p[i] = z[i] + be * p[i];
    rz = rz2;
  }
  return {x: Array.from(x), it};
}

export function solve(mesh, pb, gd) {
  // Assemble with the LM array (Hughes) and solve K d = F; also assemble the
  // full, all-node matrix Kfull and vector Ffull, before any Dirichlet row or
  // column is removed, to show what is cut out.  R = Kfull U - Ffull gives the
  // heat flowing INTO the body at each Dirichlet node (per unit depth).
  const nnp = mesh.nnp, nen = mesh.nen, ID = new Array(nnp);
  let neq = 0;
  for (let A = 1; A <= nnp; A++) ID[A - 1] = gd.has(A) ? 0 : ++neq;
  const LM = mesh.IEN.map(row => row.map(A => ID[A - 1]));
  const tripK = new Map(), tripKf = new Map();
  const F = new Array(neq).fill(0), Ffull = new Array(nnp).fill(0), els = [];
  let minJ = Infinity, maxJ = -Infinity;
  for (let e = 1; e <= mesh.nel; e++) {
    const ea = elementArrays(mesh, e, pb, gd);
    els.push(ea);
    for (const g of ea.gp) { minJ = Math.min(minJ, g.J); maxJ = Math.max(maxJ, g.J); }
    const ien = mesh.IEN[e - 1], lm = LM[e - 1];
    for (let a = 0; a < nen; a++) {
      Ffull[ien[a] - 1] += ea.body[a] + ea.h[a];
      for (let b = 0; b < nen; b++) {
        const k = (ien[a] - 1) * nnp + (ien[b] - 1);
        tripKf.set(k, (tripKf.get(k) || 0) + ea.ke[a][b]);
      }
      const P = lm[a];
      if (!P) continue;
      F[P - 1] += ea.total[a];
      for (let b = 0; b < nen; b++) {
        const Q = lm[b];
        if (!Q) continue;
        const k = (P - 1) * neq + (Q - 1);
        tripK.set(k, (tripK.get(k) || 0) + ea.ke[a][b]);
      }
    }
  }
  const K = toCSR(neq, tripK), Kfull = toCSR(nnp, tripKf);
  let d = [], it = 0, ok = true;
  if (neq && minJ > 0) {
    if (neq <= 300) { d = denseSolve(csrDense(K), F); ok = d !== null; if (!ok) d = []; }
    else { const r = pcg(K, F); d = r.x; it = r.it; }
  } else if (neq) ok = false;
  const U = Array.from({length: nnp}, (_, i) => ID[i] ? (d[ID[i] - 1] ?? NaN) : gd.get(i + 1));
  const KU = ok ? Array.from(csrMul(Kfull, U)) : new Array(nnp).fill(NaN);
  const R = KU.map((v, i) => ID[i] ? 0 : v - Ffull[i]);
  return {mesh, ID, LM, IEN: mesh.IEN, els, K, F, Kfull, Ffull, d, U, R, gd, neq, nnp,
          minJ, maxJ, ok, iterations: it};
}

// ------------------------------------------------------------ evaluation
export function sampleElement(mesh, U, e, k) {
  // u^h, grad u^h and x on a k-by-k grid of the parent square of element e
  const ien = mesh.IEN[e - 1], pts = [];
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    const xi = -1 + 2 * i / (k - 1), eta = -1 + 2 * j / (k - 1), m = mapPoint(mesh, e, xi, eta);
    let u = 0, gxi = 0, geta = 0;
    for (let a = 0; a < ien.length; a++) {
      const Ua = U[ien[a] - 1];
      u += m.sh.N[a] * Ua; gxi += m.sh.Nxi[a] * Ua; geta += m.sh.Neta[a] * Ua;
    }
    pts.push({i, j, x: m.x, u, grad: gradX(m.DF, m.J, gxi, geta), J: m.J});
  }
  return pts;
}

export function errors(sol, exact, nq) {
  const mesh = sol.mesh, G = gauss(nq || mesh.p + 3);
  let l2 = 0, h1 = 0;
  for (let e = 1; e <= mesh.nel; e++) {
    const ien = mesh.IEN[e - 1];
    for (let qi = 0; qi < G.x.length; qi++) for (let qj = 0; qj < G.x.length; qj++) {
      const m = mapPoint(mesh, e, G.x[qi], G.x[qj]), w = G.w[qi] * G.w[qj] * m.J;
      let u = 0, gxi = 0, geta = 0;
      for (let a = 0; a < ien.length; a++) {
        const Ua = sol.U[ien[a] - 1];
        u += m.sh.N[a] * Ua; gxi += m.sh.Nxi[a] * Ua; geta += m.sh.Neta[a] * Ua;
      }
      const g = gradX(m.DF, m.J, gxi, geta), [ue, u1, u2] = exact(m.x[0], m.x[1]);
      l2 += w * (ue - u) ** 2;
      h1 += w * ((u1 - g[0]) ** 2 + (u2 - g[1]) ** 2);
    }
  }
  return {l2: Math.sqrt(l2), h1: Math.sqrt(h1)};
}

export function exactCell({W, H, kappa, f, gLeft, hTab, tabs, M = 400, Ng = 40}) {
  // The exact solution when Gamma_g is the left edge (u = gLeft(x2)), the tabs
  // carry hTab, and the rest of the boundary is insulated: a port of
  // heat2d.series, with gLeft expanded in cos(n pi x2 / H) numerically.
  const mu = Array.from({length: M}, (_, m) => (m + 0.5) * Math.PI / W);
  const hm = mu.map(l => (2 / W) * tabs.reduce((s, [a, b]) => s + hTab * (Math.cos(l * a) - Math.cos(l * b)) / l, 0));
  const G = gauss(64), gn = new Array(Ng + 1).fill(0);
  for (let q = 0; q < 64; q++) {
    const y = H * (G.x[q] + 1) / 2, wq = H / 2 * G.w[q], gv = gLeft(y);
    for (let n = 0; n <= Ng; n++) gn[n] += (n === 0 ? 1 / H : 2 / H) * gv * Math.cos(n * Math.PI * y / H) * wq;
  }
  return (x, y) => {
    let u = gn[0] + f / kappa * (W * x - x * x / 2), u1 = f / kappa * (W - x), u2 = 0;
    for (let m = 0; m < M; m++) {
      const l = mu[m], c = hm[m] / kappa, e1 = 1 - Math.exp(-2 * l * H);
      const ch = (Math.exp(l * (y - H)) + Math.exp(-l * (y + H))) / e1;
      const sh = (Math.exp(l * (y - H)) - Math.exp(-l * (y + H))) / e1;
      const sn = Math.sin(l * x), cs = Math.cos(l * x);
      u += c / l * sn * ch; u1 += c * cs * ch; u2 += c * sn * sh;
    }
    for (let n = 1; n <= Ng; n++) {
      const l = n * Math.PI / H, e2 = 1 + Math.exp(-2 * l * W);
      const C = (Math.exp(-l * x) + Math.exp(-l * (2 * W - x))) / e2;
      const S = (Math.exp(-l * x) - Math.exp(-l * (2 * W - x))) / e2;
      const cy = Math.cos(l * y), sy = Math.sin(l * y);
      u += gn[n] * cy * C; u1 += -gn[n] * l * cy * S; u2 += -gn[n] * l * sy * C;
    }
    return [u, u1, u2];
  };
}

// ------------------------------------------------------------ drawing (canvas)
export function makeCanvas(width, x0, x1, y0, y1, {margin = [14, 16, 38, 50], axes = true, xlabel = "x₁ (m)", ylabel = "x₂ (m)", aspect = true, height = null} = {}) {
  // A canvas with equal-aspect world coordinates [x0, x1] x [y0, y1] (y up)
  const [mt, mr, mb, ml] = axes ? margin : [8, 8, 8, 8];
  const pw = width - ml - mr;
  const ph = aspect ? pw * (y1 - y0) / (x1 - x0) : (height - mt - mb);
  const h = Math.round(ph + mt + mb), dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.width = width + "px"; canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const sx = x => ml + (x - x0) / (x1 - x0) * pw, sy = y => mt + ph - (y - y0) / (y1 - y0) * ph;
  const out = {canvas, ctx, sx, sy, width, height: h, pw, ph, ml, mt};
  if (axes) {
    ctx.save();
    ctx.strokeStyle = "#555"; ctx.fillStyle = "#333"; ctx.lineWidth = 1;
    ctx.font = "11px sans-serif";
    const ticks = (a, b) => { const t = [], st = niceStep((b - a) / 4); for (let v = Math.ceil(a / st - 1e-9) * st; v <= b + 1e-9; v += st) t.push(+v.toFixed(10)); return t; };
    for (const v of ticks(x0, x1)) {
      ctx.beginPath(); ctx.moveTo(sx(v), mt + ph + 2); ctx.lineTo(sx(v), mt + ph + 6); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(String(v), sx(v), mt + ph + 8);
    }
    for (const v of ticks(y0, y1)) {
      ctx.beginPath(); ctx.moveTo(ml - 6, sy(v)); ctx.lineTo(ml - 2, sy(v)); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(v), ml - 8, sy(v));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(xlabel, ml + pw / 2, h - 2);
    ctx.translate(12, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0);
    ctx.restore();
  }
  return out;
}

function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw * 0.999);
}

export function drawField(cv, mesh, valueAt, color, k) {
  // Fill every element with valueAt(e, xi, eta), sampled on a k-by-k grid of sub-cells
  const {ctx, sx, sy} = cv;
  for (let e = 1; e <= mesh.nel; e++) {
    const P = [];
    for (let i = 0; i <= k; i++) {
      const row = [];
      for (let j = 0; j <= k; j++) row.push(mapPoint(mesh, e, -1 + 2 * i / k, -1 + 2 * j / k).x);
      P.push(row);
    }
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const v = valueAt(e, -1 + 2 * (i + 0.5) / k, -1 + 2 * (j + 0.5) / k);
      ctx.fillStyle = ctx.strokeStyle = color(v);
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(sx(P[i][j][0]), sy(P[i][j][1])); ctx.lineTo(sx(P[i + 1][j][0]), sy(P[i + 1][j][1]));
      ctx.lineTo(sx(P[i + 1][j + 1][0]), sy(P[i + 1][j + 1][1])); ctx.lineTo(sx(P[i][j + 1][0]), sy(P[i][j + 1][1]));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }
}

export function elementOutline(mesh, e, m = 12) {
  // the element's boundary, traced counter-clockwise, as world points
  const pts = [], sides = [t => [t, -1], t => [1, t], t => [-t, 1], t => [-1, -t]];
  for (const s of sides) for (let i = 0; i < m; i++) pts.push(mapPoint(mesh, e, ...s(-1 + 2 * i / m)).x);
  return pts;
}

export function drawMesh(cv, mesh, {color = "#8a5a20", width = 1, highlight = null, hiColor = "#e67e22"} = {}) {
  const {ctx, sx, sy} = cv, m = mesh.p === 1 ? 1 : 12;
  const path = pts => { ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))); ctx.closePath(); };
  ctx.strokeStyle = color; ctx.lineWidth = width;
  for (let e = 1; e <= mesh.nel; e++) { path(elementOutline(mesh, e, m)); ctx.stroke(); }
  if (highlight) {
    ctx.strokeStyle = hiColor; ctx.lineWidth = 3;
    path(elementOutline(mesh, highlight, m)); ctx.stroke();
  }
}

export function drawBoundary(cv, mesh, dirEdges, tabs = []) {
  const {ctx, sx, sy} = cv, {W, H} = mesh;
  const seg = {left: [[0, 0], [0, H]], right: [[W, 0], [W, H]], bottom: [[0, 0], [W, 0]], top: [[0, H], [W, H]]};
  const line = (a, b, c, w) => { ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(sx(a[0]), sy(a[1])); ctx.lineTo(sx(b[0]), sy(b[1])); ctx.stroke(); };
  for (const e of EDGES) if (!dirEdges.includes(e)) line(...seg[e], "#0d9488", 3.5);
  if (!dirEdges.includes("top")) for (const [a, b] of tabs) line([a, H], [b, H], "#e67e22", 5.5);
  for (const e of EDGES) if (dirEdges.includes(e)) line(...seg[e], "#4a7ba7", 5);
}

export function drawNodes(cv, mesh, gd, {r = 3.5, labels = false, idLabels = null, hiNodes = null, hiNode = null, localLabels = null} = {}) {
  const {ctx, sx, sy} = cv;
  ctx.font = "10px sans-serif";
  for (let A = 1; A <= mesh.nnp; A++) {
    const [x, y] = mesh.X[A - 1], dir = gd.has(A);
    ctx.beginPath(); ctx.arc(sx(x), sy(y), r, 0, 2 * Math.PI);
    ctx.fillStyle = dir ? "#4a7ba7" : "white"; ctx.fill();
    ctx.strokeStyle = hiNodes && hiNodes.has(A) ? "#e67e22" : "#1a1a1a"; ctx.lineWidth = hiNodes && hiNodes.has(A) ? 2 : 1;
    ctx.stroke();
    if (A === hiNode) { ctx.beginPath(); ctx.arc(sx(x), sy(y), r + 4, 0, 2 * Math.PI); ctx.strokeStyle = "#c0392b"; ctx.lineWidth = 2; ctx.stroke(); }
    if (labels) {
      ctx.fillStyle = "#1a1a1a"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(String(A), sx(x) + 3, sy(y) - 2);
    }
    if (idLabels) {
      ctx.fillStyle = dir ? "#4a7ba7" : "#7c3aed"; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(idLabels(A), sx(x) + 3, sy(y) + 2);
    }
  }
  if (localLabels) {
    ctx.font = "bold 11px sans-serif"; ctx.fillStyle = "#e67e22"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    for (const [A, a] of localLabels) { const [x, y] = mesh.X[A - 1]; ctx.fillText(a, sx(x) - 4, sy(y) - 2); }
  }
}

export function drawElementNumbers(cv, mesh) {
  const {ctx, sx, sy} = cv;
  ctx.font = "italic 11px sans-serif"; ctx.fillStyle = "#8a5a20"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let e = 1; e <= mesh.nel; e++) { const [x, y] = mapPoint(mesh, e, 0, 0).x; ctx.fillText(String(e), sx(x), sy(y)); }
}

export function drawSpy(size, n, entries, {shadeRows = null, highlight = null, labelStep = null, title = ""} = {}) {
  // A sparsity picture of an n-by-n matrix.  entries: iterable of [i, j] (0-based)
  const pad = 26, cell = (size - pad - 6) / Math.max(n, 1);
  const cv = document.createElement("canvas"), dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
  const hgt = size + (title ? 18 : 0);
  cv.width = Math.round(size * dpr); cv.height = Math.round(hgt * dpr);
  cv.style.width = size + "px"; cv.style.height = hgt + "px";
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const oy = title ? 18 : 0;
  if (title) { ctx.font = "12px sans-serif"; ctx.fillStyle = "#1a1a1a"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(title, 0, 0); }
  ctx.fillStyle = "white"; ctx.fillRect(pad, oy + pad, n * cell, n * cell);
  if (shadeRows) {
    ctx.fillStyle = "rgba(74,123,167,0.18)";
    for (const i of shadeRows) { ctx.fillRect(pad, oy + pad + i * cell, n * cell, cell); ctx.fillRect(pad + i * cell, oy + pad, cell, n * cell); }
  }
  const s = Math.max(cell * 0.86, 1);
  ctx.fillStyle = "#555";
  for (const [i, j] of entries) ctx.fillRect(pad + j * cell + (cell - s) / 2, oy + pad + i * cell + (cell - s) / 2, s, s);
  if (highlight) {
    ctx.fillStyle = "#c0392b";
    for (const [i, j] of highlight) ctx.fillRect(pad + j * cell + (cell - s) / 2, oy + pad + i * cell + (cell - s) / 2, s, s);
  }
  ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 1; ctx.strokeRect(pad, oy + pad, n * cell, n * cell);
  const step = labelStep || Math.max(1, Math.ceil(n / 12));
  ctx.font = "9px sans-serif"; ctx.fillStyle = "#666";
  for (let i = 0; i < n; i += step) {
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(String(i + 1), pad + (i + 0.5) * cell, oy + pad - 2);
    ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(i + 1), pad - 3, oy + pad + (i + 0.5) * cell);
  }
  return cv;
}

export function csrEntries(A) {
  const out = [];
  for (let i = 0; i < A.n; i++) for (let t = A.rowPtr[i]; t < A.rowPtr[i + 1]; t++) if (Math.abs(A.val[t]) > 1e-12) out.push([i, A.col[t]]);
  return out;
}

export function parseOverrides(text) {
  // "8 = 30, 13 = free" -> Map {8 -> 30, 13 -> null}; returns {map, errors}
  const map = new Map(), errs = [];
  for (const item of text.split(/[,;\n]+/).map(s => s.trim()).filter(Boolean)) {
    const m = item.match(/^(\d+)\s*[=:]\s*(.+)$/);
    if (!m) { errs.push(item); continue; }
    const A = parseInt(m[1]), v = m[2].trim().toLowerCase();
    if (v === "free") map.set(A, null);
    else if (Number.isFinite(Number(v))) map.set(A, Number(v));
    else errs.push(item);
  }
  return {map, errs};
}

export function parseExpr(text) {
  // "25 + 3*y/0.2" -> (x, y) => value, with Math's functions available; null if invalid
  try {
    const fn = new Function("x", "y", "const {sin, cos, tan, exp, log, sqrt, abs, pow, min, max, PI, E} = Math; return (" + text + ");");
    const v = fn(0.05, 0.1);
    return Number.isFinite(v) ? fn : null;
  } catch (err) { return null; }
}

// ------------------------------------------------------------ parent-element pictures
export function diverging(v) {
  // blue for positive, white at zero, red for negative (values in [-1, 1])
  const t = Math.max(0, Math.min(1, (1 + v) / 2));
  const c = t >= 0.5 ? [255 - (t - 0.5) * 2 * (255 - 31), 255 - (t - 0.5) * 2 * (255 - 111), 255 - (t - 0.5) * 2 * (255 - 180)]
                     : [255 - (0.5 - t) * 2 * (255 - 192), 255 - (0.5 - t) * 2 * (255 - 57), 255 - (0.5 - t) * 2 * (255 - 43)];
  return `rgb(${c.map(Math.round).join(",")})`;
}

export function drawParentFunction(p, a, size, label) {
  // N_a(xi, eta) on the parent square, with its node black and the others white
  const cv = makeCanvas(size, -1.15, 1.15, -1.15, 1.15, {axes: false});
  const {ctx, sx, sy} = cv, k = 40;
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
    const xi = -1 + 2 * (i + 0.5) / k, eta = -1 + 2 * (j + 0.5) / k;
    ctx.fillStyle = diverging(shape(p, xi, eta).N[a]);
    ctx.fillRect(sx(-1 + 2 * i / k), sy(-1 + 2 * (j + 1) / k), sx(-1 + 2 * (i + 1) / k) - sx(-1 + 2 * i / k) + 0.6,
                 sy(-1 + 2 * j / k) - sy(-1 + 2 * (j + 1) / k) + 0.6);
  }
  ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 1.2;
  ctx.strokeRect(sx(-1), sy(1), sx(1) - sx(-1), sy(-1) - sy(1));
  parentNodes(p).forEach(([x, y], b) => {
    ctx.beginPath(); ctx.arc(sx(x), sy(y), 3.2, 0, 2 * Math.PI);
    ctx.fillStyle = b === a ? "#1a1a1a" : "white"; ctx.fill(); ctx.strokeStyle = "#1a1a1a"; ctx.lineWidth = 1; ctx.stroke();
  });
  if (label) { ctx.font = "11px sans-serif"; ctx.fillStyle = "#1a1a1a"; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(label, 3, 1); }
  return cv.canvas;
}

export function drawParentSquare(p, qSel, size) {
  // grid lines, numbered local nodes, and the Gauss points, one of them selected
  const cv = makeCanvas(size, -1.35, 1.35, -1.35, 1.35, {axes: false});
  const {ctx, sx, sy} = cv, G = gauss(p + 1);
  ctx.fillStyle = "#f1ecfb"; ctx.fillRect(sx(-1), sy(1), sx(1) - sx(-1), sy(-1) - sy(1));
  ctx.strokeStyle = "#7c3aed"; ctx.lineWidth = 0.7;
  for (let v = -1; v <= 1.0001; v += 0.5) {
    ctx.beginPath(); ctx.moveTo(sx(v), sy(-1)); ctx.lineTo(sx(v), sy(1)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx(-1), sy(v)); ctx.lineTo(sx(1), sy(v)); ctx.stroke();
  }
  ctx.lineWidth = 1.6; ctx.strokeRect(sx(-1), sy(1), sx(1) - sx(-1), sy(-1) - sy(1));
  ctx.font = "11px sans-serif";
  parentNodes(p).forEach(([x, y], a) => {
    ctx.beginPath(); ctx.arc(sx(x), sy(y), 3.5, 0, 2 * Math.PI); ctx.fillStyle = "#7c3aed"; ctx.fill();
    ctx.fillStyle = "#1a1a1a"; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(String(a + 1), sx(x) + 4, sy(y) - 2);
  });
  let q = 0;
  for (let qi = 0; qi < G.x.length; qi++) for (let qj = 0; qj < G.x.length; qj++) {
    q++;
    const X = sx(G.x[qi]), Y = sy(G.x[qj]), sel = q === qSel;
    ctx.strokeStyle = sel ? "#c0392b" : "#c0392b"; ctx.lineWidth = sel ? 3 : 1.6;
    const r = sel ? 6 : 4;
    ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X - r, Y + r); ctx.lineTo(X + r, Y - r); ctx.stroke();
  }
  ctx.fillStyle = "#555"; ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("ξ →", sx(0), sy(-1) + 8);
  ctx.save(); ctx.translate(sx(-1) - 12, sy(0)); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "bottom"; ctx.fillText("η →", 0, 0); ctx.restore();
  return cv.canvas;
}

export function drawElementDetail(mesh, e, qSel, size) {
  // the physical element: mapped grid lines, nodes, Gauss points, and at the
  // selected Gauss point the two columns of DF
  const out = [], sides = 48;
  for (let i = 0; i <= sides; i++) for (const s of [[t => [t, -1]], [t => [t, 1]], [t => [-1, t]], [t => [1, t]]]) out.push(mapPoint(mesh, e, ...s[0](-1 + 2 * i / sides)).x);
  const xs = out.map(p => p[0]), ys = out.map(p => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = 0.34 * Math.max(x1 - x0, y1 - y0);          // room for the node labels
  x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
  const span = Math.max(x1 - x0, y1 - y0), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const cv = makeCanvas(size, cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2, {axes: false});
  const {ctx, sx, sy} = cv, m = 30;
  const curve = f => { ctx.beginPath(); for (let i = 0; i <= m; i++) { const [x, y] = f(-1 + 2 * i / m); i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)); } ctx.stroke(); };
  ctx.fillStyle = "#cfe3f5"; ctx.beginPath();
  elementOutline(mesh, e, 24).forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)));
  ctx.closePath(); ctx.fill();
  for (let v = -1; v <= 1.0001; v += 0.5) {
    ctx.strokeStyle = Math.abs(Math.abs(v) - 1) < 1e-9 ? "#1a1a1a" : "rgba(26,26,26,0.45)";
    ctx.lineWidth = Math.abs(Math.abs(v) - 1) < 1e-9 ? 1.6 : 0.7;
    curve(t => mapPoint(mesh, e, v, t).x); curve(t => mapPoint(mesh, e, t, v).x);
  }
  const G = gauss(mesh.p + 1);
  let q = 0, sel = null;
  for (let qi = 0; qi < G.x.length; qi++) for (let qj = 0; qj < G.x.length; qj++) {
    q++;
    const mp = mapPoint(mesh, e, G.x[qi], G.x[qj]), [X, Y] = [sx(mp.x[0]), sy(mp.x[1])], r = q === qSel ? 6 : 4;
    ctx.strokeStyle = "#c0392b"; ctx.lineWidth = q === qSel ? 3 : 1.6;
    ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X - r, Y + r); ctx.lineTo(X + r, Y - r); ctx.stroke();
    if (q === qSel) sel = mp;
  }
  // label each node outward from the element's centre, so neighbours do not collide
  ctx.font = "11px sans-serif";
  const cen = mapPoint(mesh, e, 0, 0).x;
  mesh.IEN[e - 1].forEach((A, a) => {
    const [x, y] = mesh.X[A - 1];
    ctx.beginPath(); ctx.arc(sx(x), sy(y), 3.5, 0, 2 * Math.PI); ctx.fillStyle = "#1a1a1a"; ctx.fill();
    let dx = sx(x) - sx(cen[0]), dy = sy(y) - sy(cen[1]);
    const r = Math.hypot(dx, dy);
    if (r < 1) { dx = 0.7; dy = -0.7; } else { dx /= r; dy /= r; }
    ctx.textAlign = dx > 0.35 ? "left" : dx < -0.35 ? "right" : "center";
    ctx.textBaseline = dy > 0.35 ? "top" : dy < -0.35 ? "bottom" : "middle";
    ctx.fillText(`${a + 1} (A=${A})`, sx(x) + 7 * dx, sy(y) + 7 * dy);
  });
  if (sel) {
    const L = 0.35 * span / Math.max(Math.hypot(sel.DF[0][0], sel.DF[1][0]), Math.hypot(sel.DF[0][1], sel.DF[1][1]));
    const arrowTo = (dx, dy, lab) => {
      const X0 = sx(sel.x[0]), Y0 = sy(sel.x[1]), X1 = sx(sel.x[0] + L * dx), Y1 = sy(sel.x[1] + L * dy);
      ctx.strokeStyle = ctx.fillStyle = "#1a1a1a"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
      const an = Math.atan2(Y1 - Y0, X1 - X0);
      ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X1 - 9 * Math.cos(an - 0.35), Y1 - 9 * Math.sin(an - 0.35));
      ctx.lineTo(X1 - 9 * Math.cos(an + 0.35), Y1 - 9 * Math.sin(an + 0.35)); ctx.closePath(); ctx.fill();
      ctx.font = "italic 12px serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(lab, X1 + 4, Y1);
    };
    arrowTo(sel.DF[0][0], sel.DF[1][0], "∂x/∂ξ");
    arrowTo(sel.DF[0][1], sel.DF[1][1], "∂x/∂η");
  }
  return cv.canvas;
}
