// Two-dimensional linear elasticity by Galerkin's method, in the browser, for
// the explorer on the second elasticity page.  A port of python/elast2d.py,
// which draws the static figures on that page -- change the two together.
//
// Hughes' conventions: nodes A = 1..nnp, local nodes a = 1..nen, elements
// e = 1..nel, directions i = 1, 2, ID(i, A) with 0 marking a prescribed
// displacement component, and LM(i, a, e) = ID(i, IEN(a, e)).  JavaScript
// arrays are 0-based, so IEN[e-1][a-1] holds the global number A of local
// node a of element e, ID[i-1][A-1] holds ID(i, A), and LM[e-1][2(a-1) + i-1]
// holds LM(i, a, e).  Local nodes: corners counter-clockwise from
// (xi, eta) = (-1, -1), then for Q9 the mid-side nodes counter-clockwise
// from the bottom, then the centre (the same as fem2d.js).
//
// Units: millimetres, newtons, megapascals (N/mm^2).

import {gauss, shape, makeCanvas, drawSpy} from "./fem2d.js";

export const CANTILEVER = {L: 400, D: 100, t: 10, P: 8000};
export const PLATE = {L: 200, W: 100, a: 12.5, t: 10, s: 50};
export const WSHAPE = {name: "W250×73 (W10×49)", d: 253, bf: 254, tf: 14.2, tw: 8.6, q: 1.0};

const LOCAL_IJ = {1: [[0, 0], [1, 0], [1, 1], [0, 1]],
                  2: [[0, 0], [2, 0], [2, 2], [0, 2], [1, 0], [2, 1], [1, 2], [0, 1], [1, 1]]};
const SIDE_PARAM = {bottom: t => [t, -1], right: t => [1, t], top: t => [t, 1], left: t => [-1, t]};
const SIDE_LOCAL = {1: {bottom: [0, 1], right: [1, 2], top: [2, 3], left: [3, 0]},
                    2: {bottom: [0, 4, 1], right: [1, 5, 2], top: [2, 6, 3], left: [3, 7, 0]}};

// ------------------------------------------------------------- material
export function dmatrix(E, nu, plane = "stress") {
  // D in Voigt form: sigma = D eps, sigma = (s11, s22, s12), eps = (e11, e22, 2 e12)
  if (plane === "stress") {
    const c = E / (1 - nu * nu);
    return [[c, c * nu, 0], [c * nu, c, 0], [0, 0, c * (1 - nu) / 2]];
  }
  const c = E / ((1 + nu) * (1 - 2 * nu));
  return [[c * (1 - nu), c * nu, 0], [c * nu, c * (1 - nu), 0], [0, 0, c * (1 - 2 * nu) / 2]];
}

// ---------------------------------------------------------------- meshing
export function graded(n, ratio = 1) {
  if (Math.abs(ratio - 1) < 1e-12) return Array.from({length: n + 1}, (_, i) => i / n);
  const w = Array.from({length: n}, (_, k) => Math.pow(ratio, k)), tot = w.reduce((a, b) => a + b, 0);
  const out = [0];
  let s = 0;
  for (const v of w) { s += v; out.push(s / tot); }
  return out;
}

function ratioFor(first, total, n) {
  if (n * first >= total) return 1;
  let lo = 1 + 1e-12, hi = 50;
  for (let it = 0; it < 200; it++) {
    const r = 0.5 * (lo + hi), s = first * (Math.pow(r, n) - 1) / (r - 1);
    if (s < total) lo = r; else hi = r;
  }
  return 0.5 * (lo + hi);
}

const line = (P, Q) => t => [(1 - t) * P[0] + t * Q[0], (1 - t) * P[1] + t * Q[1]];
const arc = (r, a0, a1) => t => [r * Math.cos(a0 + t * (a1 - a0)), r * Math.sin(a0 + t * (a1 - a0))];

function patch(P00, P10, P11, P01, sx, sy, curves = {}, names = {}) {
  // transfinite interpolation from [0, 1]^2; corners counter-clockwise
  const c = Object.assign({bottom: line(P00, P10), right: line(P10, P11), top: line(P01, P11), left: line(P00, P01)}, curves);
  const point = (s, t) => {
    const B = c.bottom(s), T = c.top(s), L = c.left(t), R = c.right(t);
    return [0, 1].map(k => (1 - t) * B[k] + t * T[k] + (1 - s) * L[k] + s * R[k]
      - ((1 - s) * (1 - t) * P00[k] + s * (1 - t) * P10[k] + s * t * P11[k] + (1 - s) * t * P01[k]));
  };
  return {P: [P00, P10, P11, P01], sx, sy, point,
          names: Object.assign({bottom: null, right: null, top: null, left: null}, names)};
}

function params(v, p) {
  if (p === 1) return v.slice();
  const out = [v[0]];
  for (let k = 0; k < v.length - 1; k++) out.push(0.5 * (v[k] + v[k + 1]), v[k + 1]);
  return out;
}

function buildMesh(patches, p, info, key) {
  const scale = Math.max(...patches.flatMap(pt => pt.P.flatMap(Q => Q.map(Math.abs))));
  const tol = 1e-7 * scale, coords = [], buckets = new Map(), ien = [], sides = {};
  const node = x => {
    const kx = Math.round(x[0] / tol), ky = Math.round(x[1] / tol);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      for (const A of buckets.get(`${kx + dx},${ky + dy}`) || []) {
        if (Math.abs(coords[A][0] - x[0]) <= tol && Math.abs(coords[A][1] - x[1]) <= tol) return A;
      }
    }
    coords.push([x[0], x[1]]);
    const k = `${kx},${ky}`;
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(coords.length - 1);
    return coords.length - 1;
  };
  for (const pt of patches) {
    const S = params(pt.sx, p), T = params(pt.sy, p);
    const grid = S.map(s => T.map(t => node(pt.point(s, t))));
    const nx = pt.sx.length - 1, ny = pt.sy.length - 1;
    for (let ey = 0; ey < ny; ey++) for (let ex = 0; ex < nx; ex++) {
      const e = ien.length;
      ien.push(LOCAL_IJ[p].map(([i, j]) => grid[p * ex + i][p * ey + j]));
      for (const [side, edge] of [["bottom", ey === 0], ["right", ex === nx - 1], ["top", ey === ny - 1], ["left", ex === 0]]) {
        if (edge && pt.names[side]) (sides[pt.names[side]] ||= []).push([e + 1, side]);
      }
    }
  }
  const perm = rcm(coords.length, ien), inv = new Array(perm.length);
  perm.forEach((old, k) => { inv[old] = k; });
  const X = perm.map(old => coords[old]);
  const IEN = ien.map(row => row.map(A => inv[A] + 1));
  return {X, IEN, p, nen: (p + 1) * (p + 1), nnp: X.length, nel: IEN.length, sides, info, key};
}

function rcm(nnp, ien) {
  // reverse Cuthill-McKee; ties broken by node number, as in elast2d.py
  const nb = Array.from({length: nnp}, () => new Set());
  for (const row of ien) for (const A of row) for (const B of row) nb[A].add(B);
  const adj = nb.map((s, A) => [...s].filter(B => B !== A).sort((x, y) => x - y));
  const deg = adj.map(a => a.length), seen = new Uint8Array(nnp), order = [];
  const byDeg = (x, y) => deg[x] - deg[y] || x - y;
  while (order.length < nnp) {
    let start = -1;
    for (let A = 0; A < nnp; A++) if (!seen[A] && (start < 0 || byDeg(A, start) < 0)) start = A;
    seen[start] = 1;
    const queue = [start];
    for (let h = 0; h < queue.length; h++) {
      const nxt = adj[queue[h]].filter(B => !seen[B]).sort(byDeg);
      for (const B of nxt) { seen[B] = 1; queue.push(B); }
    }
    order.push(...queue);
  }
  return order.reverse();
}

export function sideNodes(mesh, name) {
  const out = new Set();
  for (const [e, sd] of mesh.sides[name] || []) for (const a of SIDE_LOCAL[mesh.p][sd]) out.add(mesh.IEN[e - 1][a]);
  return [...out].sort((a, b) => a - b);
}

// ------------------------------------------------------------- geometries
export function cantileverMesh(p = 1, ny = 2, {L = CANTILEVER.L, D = CANTILEVER.D} = {}) {
  const nx = 4 * ny;
  return buildMesh([patch([0, -D / 2], [L, -D / 2], [L, D / 2], [0, D / 2], graded(nx), graded(ny), {},
                          {left: "left", right: "right", bottom: "bottom", top: "top"})],
                   p, {kind: "cantilever", L, D}, `cantilever|${p}|${ny}`);
}

export const PLATE_LEVELS = {1: [2, 4], 2: [3, 6], 3: [4, 8], 4: [6, 12]};

export function plateMesh(p = 1, level = 2, quarter = false, {L = PLATE.L, W = PLATE.W, a = PLATE.a} = {}) {
  const [m, nr] = PLATE_LEVELS[level], Lh = L / 2, Wh = W / 2, tc = Math.atan2(Wh, Lh);
  const sx = graded(nr, ratioFor(a * tc / m, Lh - a, nr));
  const outer = th => {
    const c = Math.cos(th), s = Math.sin(th);
    const k = Math.min(Math.abs(c) > 1e-12 ? Lh / Math.abs(c) : 1e30, Math.abs(s) > 1e-12 ? Wh / Math.abs(s) : 1e30);
    return [k * c, k * s];
  };
  const sideOf = (t0, t1) => {
    const P = outer(0.5 * (t0 + t1));
    if (Math.abs(P[0] - Lh) < 1e-9) return "right";
    if (Math.abs(P[0] + Lh) < 1e-9) return "left";
    return P[1] > 0 ? "top" : "bottom";
  };
  let cuts = [0, tc, Math.PI / 2, Math.PI - tc, Math.PI, Math.PI + tc, 1.5 * Math.PI, 2 * Math.PI - tc, 2 * Math.PI];
  let nts = [m, 2 * m, 2 * m, m, m, 2 * m, 2 * m, m];
  if (quarter) { cuts = cuts.slice(0, 3); nts = nts.slice(0, 2); }
  const patches = [];
  for (let k = 0; k < cuts.length - 1; k++) {
    const a0 = cuts[k], a1 = cuts[k + 1], names = {left: "hole", right: sideOf(a0, a1)};
    if (quarter && k === 0) names.bottom = "bottom";
    if (quarter && k === cuts.length - 2) names.top = "left";
    patches.push(patch([a * Math.cos(a0), a * Math.sin(a0)], outer(a0), outer(a1), [a * Math.cos(a1), a * Math.sin(a1)],
                       sx, graded(nts[k]), {left: arc(a, a0, a1)}, names));
  }
  return buildMesh(patches, p, {kind: "plate", quarter, L, W, a}, `plate|${quarter}|${p}|${level}`);
}

export function wshapeMesh(p = 1, level = 2, {d = WSHAPE.d, bf = WSHAPE.bf, tf = WSHAPE.tf, tw = WSHAPE.tw} = {}) {
  const h = tf / level, ntf = level, ntw = Math.max(1, Math.round(tw / h));
  const no = Math.max(2, Math.round((bf - tw) / 2 / h)), nh = Math.max(2, Math.round((d - 2 * tf) / h));
  const xs = [-bf / 2, -tw / 2, tw / 2, bf / 2], nxs = [no, ntw, no], patches = [];
  for (const [y0, y1, which] of [[0, tf, "bot"], [d - tf, d, "top"]]) {
    for (let k = 0; k < 3; k++) {
      const names = {};
      if (which === "bot") {
        names.bottom = "bottom face";
        if (k !== 1) names.top = k === 0 ? "bottom flange, inside left" : "bottom flange, inside right";
      } else {
        names.top = "top face";
        if (k !== 1) names.bottom = k === 0 ? "top flange, inside left" : "top flange, inside right";
      }
      if (k === 0) names.left = which === "bot" ? "bottom-left tip" : "top-left tip";
      if (k === 2) names.right = which === "bot" ? "bottom-right tip" : "top-right tip";
      patches.push(patch([xs[k], y0], [xs[k + 1], y0], [xs[k + 1], y1], [xs[k], y1], graded(nxs[k]), graded(ntf), {}, names));
    }
  }
  patches.splice(3, 0, patch([-tw / 2, tf], [tw / 2, tf], [tw / 2, d - tf], [-tw / 2, d - tf], graded(ntw), graded(nh), {},
                             {left: "web, left face", right: "web, right face"}));
  return buildMesh(patches, p, {kind: "wshape", d, bf, tf, tw}, `wshape|${p}|${level}`);
}

// ------------------------------------------------------- element arrays
function elementGeometry(mesh, e, xi, eta) {
  const sh = shape(mesh.p, xi, eta), ien = mesh.IEN[e - 1];
  let x1 = 0, x2 = 0, a11 = 0, a12 = 0, a21 = 0, a22 = 0;
  for (let a = 0; a < ien.length; a++) {
    const [X1, X2] = mesh.X[ien[a] - 1];
    x1 += sh.N[a] * X1; x2 += sh.N[a] * X2;
    a11 += sh.Nxi[a] * X1; a12 += sh.Neta[a] * X1; a21 += sh.Nxi[a] * X2; a22 += sh.Neta[a] * X2;
  }
  const J = a11 * a22 - a12 * a21;
  const dNx = sh.N.map((_, a) => [(a22 * sh.Nxi[a] - a21 * sh.Neta[a]) / J, (-a12 * sh.Nxi[a] + a11 * sh.Neta[a]) / J]);
  return {x: [x1, x2], DF: [[a11, a12], [a21, a22]], J, N: sh.N, dNx};
}

export function bmatrix(dNx) {
  // B (3 x 2 nen): columns 2a, 2a+1 hold [[N,1, 0], [0, N,2], [N,2, N,1]]
  const n = dNx.length, B = [new Array(2 * n).fill(0), new Array(2 * n).fill(0), new Array(2 * n).fill(0)];
  dNx.forEach(([d1, d2], a) => { B[0][2 * a] = d1; B[1][2 * a + 1] = d2; B[2][2 * a] = d2; B[2][2 * a + 1] = d1; });
  return B;
}

export function elementArrays(mesh, e, D, t = 1) {
  // k^e = sum over Gauss points of w J t B^T D B, and the Gauss-point data
  const nq = mesh.p + 1, {x: g, w} = gauss(nq), nd = 2 * mesh.nen;
  const ke = Array.from({length: nd}, () => new Array(nd).fill(0)), gp = [];
  for (let i = 0; i < nq; i++) for (let j = 0; j < nq; j++) {
    const geo = elementGeometry(mesh, e, g[i], g[j]), B = bmatrix(geo.dNx), wt = w[i] * w[j] * geo.J * t;
    const DB = D.map(row => B[0].map((_, c) => row[0] * B[0][c] + row[1] * B[1][c] + row[2] * B[2][c]));
    for (let q = 0; q < nd; q++) {
      const b0 = B[0][q], b1 = B[1][q], b2 = B[2][q];
      if (!b0 && !b1 && !b2) continue;
      const kq = ke[q];
      for (let r = 0; r < nd; r++) kq[r] += wt * (b0 * DB[0][r] + b1 * DB[1][r] + b2 * DB[2][r]);
    }
    gp.push({xi: g[i], eta: g[j], w: w[i] * w[j], J: geo.J, x: geo.x, B});
  }
  return {ke, gp};
}

function sideIntegral(mesh, e, side, h, t = 1) {
  // f_a,i = integral over the side of N_a h_i t dGamma; h(x, y) -> [h1, h2]
  const {x: g, w} = gauss(mesh.p + 2), fe = Array.from({length: mesh.nen}, () => [0, 0]);
  for (let k = 0; k < g.length; k++) {
    const [xi, eta] = SIDE_PARAM[side](g[k]), geo = elementGeometry(mesh, e, xi, eta);
    const tang = side === "bottom" || side === "top" ? [geo.DF[0][0], geo.DF[1][0]] : [geo.DF[0][1], geo.DF[1][1]];
    const ds = w[k] * Math.hypot(...tang) * t, hv = h(geo.x[0], geo.x[1]);
    geo.N.forEach((Na, a) => { fe[a][0] += Na * hv[0] * ds; fe[a][1] += Na * hv[1] * ds; });
  }
  return fe;
}

// ------------------------------------------------- skyline Cholesky
function skyline(neq, LM) {
  // row P holds the columns first[P]..P of the lower triangle
  const first = new Int32Array(neq).fill(neq);
  for (const lm of LM) {
    let mn = Infinity;
    for (const P of lm) if (P > 0 && P - 1 < mn) mn = P - 1;
    for (const P of lm) if (P > 0 && mn < first[P - 1]) first[P - 1] = mn;
  }
  const ptr = new Int32Array(neq + 1);
  for (let i = 0; i < neq; i++) ptr[i + 1] = ptr[i] + (i - first[i] + 1);
  return {neq, first, ptr, a: new Float64Array(ptr[neq])};
}

function factor(sk) {
  // L L^T in place; returns the first row whose pivot vanishes, or -1
  const {neq, first, ptr, a} = sk;
  for (let i = 0; i < neq; i++) {
    const fi = first[i], pi = ptr[i], dia = a[pi + i - fi];
    for (let j = fi; j < i; j++) {
      const fj = first[j], pj = ptr[j], k0 = Math.max(fi, fj);
      let s = a[pi + j - fi];
      for (let k = k0; k < j; k++) s -= a[pi + k - fi] * a[pj + k - fj];
      a[pi + j - fi] = s / a[pj + j - fj];
    }
    let s = dia;
    for (let k = fi; k < i; k++) s -= a[pi + k - fi] * a[pi + k - fi];
    if (!(s > 1e-11 * dia)) return i;
    a[pi + i - fi] = Math.sqrt(s);
  }
  return -1;
}

function backsolve(sk, b) {
  const {neq, first, ptr, a} = sk, y = Float64Array.from(b);
  for (let i = 0; i < neq; i++) {
    const fi = first[i], pi = ptr[i];
    let s = y[i];
    for (let k = fi; k < i; k++) s -= a[pi + k - fi] * y[k];
    y[i] = s / a[pi + i - fi];
  }
  for (let i = neq - 1; i >= 0; i--) {
    const fi = first[i], pi = ptr[i];
    y[i] /= a[pi + i - fi];
    const xi = y[i];
    for (let k = fi; k < i; k++) y[k] -= a[pi + k - fi] * xi;
  }
  return y;
}

// ------------------------------------------------------------- solving
export function createSolver() {
  // K = E K1 for a material of modulus E, so the arrays are built and
  // factored for E = 1 and reused while only E, the loads, or the prescribed
  // values change.  A new nu, plane mode, mesh, or set of prescribed
  // components rebuilds them.
  const cache = {elemKey: null, sysKey: null};
  return function solve(mesh, {E, nu, plane = "stress", t = 1, dirichlet = {}, traction = {}}) {
    const t0 = performance.now(), th = plane === "stress" ? t : 1;
    const nnp = mesh.nnp, nen = mesh.nen, nd = 2 * nen;
    const elemKey = `${mesh.key}|${nu}|${plane}|${th}`;
    let rebuilt = false;
    if (cache.elemKey !== elemKey) {
      const D1 = dmatrix(1, nu, plane);
      cache.els = [];
      cache.minJ = Infinity;
      for (let e = 1; e <= mesh.nel; e++) {
        const ea = elementArrays(mesh, e, D1, th);
        cache.els.push(ea);
        for (const g of ea.gp) cache.minJ = Math.min(cache.minJ, g.J);
      }
      cache.unit = {};
      cache.elemKey = elemKey; cache.sysKey = null;
    }
    // prescribed components and values; a node on two Dirichlet sides gets the average
    const gsum = [new Float64Array(nnp), new Float64Array(nnp)], gcnt = [new Int32Array(nnp), new Int32Array(nnp)];
    for (const [name, comps] of Object.entries(dirichlet)) {
      for (const A of sideNodes(mesh, name)) {
        const [x, y] = mesh.X[A - 1];
        for (const [i, g] of Object.entries(comps)) {
          gsum[i][A - 1] += typeof g === "function" ? g(x, y) : g;
          gcnt[i][A - 1] += 1;
        }
      }
    }
    const fixedKey = [0, 1].map(i => Array.from(gcnt[i], c => c > 0 ? 1 : 0).join("")).join("/");
    const sysKey = `${elemKey}|${fixedKey}`;
    if (cache.sysKey !== sysKey) {
      const ID = [new Array(nnp).fill(0), new Array(nnp).fill(0)];
      let neq = 0;
      for (let A = 0; A < nnp; A++) for (let i = 0; i < 2; i++) if (!gcnt[i][A]) ID[i][A] = ++neq;
      const LM = mesh.IEN.map(row => row.flatMap(A => [ID[0][A - 1], ID[1][A - 1]]));
      const sk = skyline(neq, LM);
      LM.forEach((lm, e) => {
        const ke = cache.els[e].ke;
        for (let q = 0; q < nd; q++) {
          const P = lm[q];
          if (!P) continue;
          for (let r = 0; r < nd; r++) {
            const Q = lm[r];
            if (!Q || Q > P) continue;
            sk.a[sk.ptr[P - 1] + (Q - 1 - sk.first[P - 1])] += ke[q][r];
          }
        }
      });
      const bad = neq ? factor(sk) : -1;
      Object.assign(cache, {ID, LM, sk, neq, bad, sysKey, profile: sk.ptr[neq]});
      rebuilt = true;
    }
    const {ID, LM, sk, neq, bad} = cache;
    const g = [new Float64Array(nnp), new Float64Array(nnp)];
    for (let i = 0; i < 2; i++) for (let A = 0; A < nnp; A++) if (gcnt[i][A]) g[i][A] = gsum[i][A] / gcnt[i][A];
    // the all-node load vector: tractions on Gamma_h (constant tractions use cached unit loads)
    const Ffull = [new Float64Array(nnp), new Float64Array(nnp)];
    for (const [name, h] of Object.entries(traction)) {
      if (typeof h === "function") {
        for (const [e, sd] of mesh.sides[name] || []) {
          const fe = sideIntegral(mesh, e, sd, h, th);
          mesh.IEN[e - 1].forEach((A, a) => { Ffull[0][A - 1] += fe[a][0]; Ffull[1][A - 1] += fe[a][1]; });
        }
      } else {
        if (!cache.unit[name]) {
          const u = new Float64Array(nnp);
          for (const [e, sd] of mesh.sides[name] || []) {
            const fe = sideIntegral(mesh, e, sd, () => [1, 0], th);
            mesh.IEN[e - 1].forEach((A, a) => { u[A - 1] += fe[a][0]; });
          }
          cache.unit[name] = u;
        }
        const u = cache.unit[name];
        for (let A = 0; A < nnp; A++) { Ffull[0][A] += h[0] * u[A]; Ffull[1][A] += h[1] * u[A]; }
      }
    }
    // the right-hand side for K1: F/E - K1_fg g
    const rhs = new Float64Array(neq);
    for (let A = 0; A < nnp; A++) for (let i = 0; i < 2; i++) if (ID[i][A]) rhs[ID[i][A] - 1] += Ffull[i][A] / E;
    LM.forEach((lm, e) => {
      const ien = mesh.IEN[e], ge = ien.flatMap(A => [g[0][A - 1], g[1][A - 1]]);
      if (!ge.some(v => v !== 0)) return;
      const ke = cache.els[e].ke;
      for (let q = 0; q < nd; q++) {
        const P = lm[q];
        if (!P) continue;
        let s = 0;
        for (let r = 0; r < nd; r++) if (!lm[r]) s += ke[q][r] * ge[r];
        rhs[P - 1] -= s;
      }
    });
    const ok = neq > 0 && bad < 0 && cache.minJ > 0;
    const d = ok ? backsolve(sk, rhs) : new Float64Array(neq).fill(NaN);
    const U = Array.from({length: nnp}, (_, A) => [0, 1].map(i => ID[i][A] ? d[ID[i][A] - 1] : g[i][A]));
    // reactions: E K1 U - F at the prescribed components
    const R = Array.from({length: nnp}, (_, A) => [-Ffull[0][A], -Ffull[1][A]]);
    cache.els.forEach((ea, e) => {
      const ien = mesh.IEN[e], ue = ien.flatMap(A => U[A - 1]);
      for (let q = 0; q < nd; q++) {
        let s = 0;
        for (let r = 0; r < nd; r++) s += ea.ke[q][r] * ue[r];
        R[ien[q >> 1] - 1][q & 1] += E * s;
      }
    });
    for (let A = 0; A < nnp; A++) for (let i = 0; i < 2; i++) if (ID[i][A]) R[A][i] = 0;
    const D = dmatrix(E, nu, plane);
    const reason = !neq ? "every displacement component is prescribed"
      : cache.minJ <= 0 ? "an element is folded" : bad >= 0 ? "rigid" : "";
    return {mesh, E, nu, plane, t: th, D, ID, LM, neq, U, R, Ffull, g, ok, reason, bad,
            fixed: gcnt.map(c => Array.from(c, v => v > 0)), minJ: cache.minJ, rebuilt,
            profile: cache.profile, ms: performance.now() - t0,
            ke: e => cache.els[e - 1].ke.map(row => row.map(v => E * v)), gp: e => cache.els[e - 1].gp};
  };
}

// ------------------------------------------------------------- evaluation
export function evaluate(sol, e, xi, eta) {
  // position, displacement, strain (Voigt), stress, s33, and von Mises at a parent point
  const mesh = sol.mesh, geo = elementGeometry(mesh, e, xi, eta), ien = mesh.IEN[e - 1];
  let u1 = 0, u2 = 0, e11 = 0, e22 = 0, g12 = 0;
  ien.forEach((A, a) => {
    const [U1, U2] = sol.U[A - 1], [d1, d2] = geo.dNx[a];
    u1 += geo.N[a] * U1; u2 += geo.N[a] * U2;
    e11 += d1 * U1; e22 += d2 * U2; g12 += d2 * U1 + d1 * U2;
  });
  const D = sol.D, s = [0, 1, 2].map(r => D[r][0] * e11 + D[r][1] * e22 + D[r][2] * g12);
  const s33 = sol.plane === "strain" ? sol.nu * (s[0] + s[1]) : 0;
  return {x: geo.x, u: [u1, u2], eps: [e11, e22, g12], sig: s, s33, vm: vonMises(s[0], s[1], s[2], s33), J: geo.J, N: geo.N};
}

export function vonMises(s11, s22, s12, s33 = 0) {
  return Math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * s12 * s12);
}

export function nodalStress(sol) {
  // sigma at each node, averaged over the elements that share it
  const mesh = sol.mesh, p = mesh.p, S = Array.from({length: mesh.nnp}, () => [0, 0, 0]), cnt = new Array(mesh.nnp).fill(0);
  for (let e = 1; e <= mesh.nel; e++) {
    LOCAL_IJ[p].forEach(([i, j], a) => {
      const r = evaluate(sol, e, -1 + 2 * i / p, -1 + 2 * j / p), A = mesh.IEN[e - 1][a] - 1;
      for (let k = 0; k < 3; k++) S[A][k] += r.sig[k];
      cnt[A]++;
    });
  }
  return S.map((v, A) => v.map(c => c / cnt[A]));
}

export function fieldValue(sol, kind, e, xi, eta, nodal = null) {
  // the quantity drawn on the map: displacement or stress, element-wise or averaged at the nodes
  const r = evaluate(sol, e, xi, eta);
  let s = r.sig;
  if (nodal) {
    s = [0, 0, 0];
    mesh_ien(sol, e).forEach((A, a) => { for (let k = 0; k < 3; k++) s[k] += r.N[a] * nodal[A - 1][k]; });
  }
  const s33 = sol.plane === "strain" ? sol.nu * (s[0] + s[1]) : 0;
  switch (kind) {
    case "|u|": return Math.hypot(...r.u);
    case "u1": return r.u[0];
    case "u2": return r.u[1];
    case "s11": return s[0];
    case "s22": return s[1];
    case "s12": return s[2];
    case "s33": return s33;
    default: return vonMises(s[0], s[1], s[2], s33);
  }
}

const mesh_ien = (sol, e) => sol.mesh.IEN[e - 1];

export function locate(mesh, x, y) {
  // the element containing (x, y) and its parent coordinates, by Newton's method
  let best = null;
  for (let e = 1; e <= mesh.nel; e++) {
    const pts = mesh.IEN[e - 1].map(A => mesh.X[A - 1]);
    const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
    if (x < Math.min(...xs) - 1e-9 || x > Math.max(...xs) + 1e-9 || y < Math.min(...ys) - 1e-9 || y > Math.max(...ys) + 1e-9) continue;
    let z = [0, 0];
    for (let it = 0; it < 25; it++) {
      const geo = elementGeometry(mesh, e, z[0], z[1]), r0 = geo.x[0] - x, r1 = geo.x[1] - y, [[a, b], [c, d]] = geo.DF, J = a * d - b * c;
      z = [z[0] - (d * r0 - b * r1) / J, z[1] - (-c * r0 + a * r1) / J];
    }
    const m = Math.max(Math.abs(z[0]), Math.abs(z[1]));
    if (m <= 1 + 1e-7) return [e, z[0], z[1]];
    if (!best || m < best[3]) best = [e, z[0], z[1], m];
  }
  return best ? best.slice(0, 3) : null;
}

// ------------------------------------------------- closed-form solutions
export function cantileverExact(x, y, E, nu, plane = "stress", {L = CANTILEVER.L, D = CANTILEVER.D, t = CANTILEVER.t, P = CANTILEVER.P} = {}) {
  // Timoshenko and Goodier: fixed at x = 0, a downward force P at x = L, spread
  // over the end as a parabolic shear traction; y from mid-depth
  if (plane === "strain") { E = E / (1 - nu * nu); nu = nu / (1 - nu); t = 1; }
  const I = t * D ** 3 / 12;
  return {u1: P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y * y - D * D / 4)),
          u2: -P / (6 * E * I) * (3 * nu * y * y * (L - x) + (4 + 5 * nu) * D * D * x / 4 + (3 * L - x) * x * x),
          s11: P * (L - x) * y / I, s22: 0, s12: -P / (2 * I) * (D * D / 4 - y * y)};
}

export function heywood(d, W, s) {
  const kt = 2 + (1 - d / W) ** 3;
  return {kt, peak: kt * s * W / (W - d)};
}

// ------------------------------------------------- the geometries, for the UI
export const GEOMETRIES = {
  plate: {label: "Plate with a hole", plane: "stress", t: PLATE.t, tRange: [-100, 100], uRange: [-0.1, 0.1],
          make: (p, lev) => plateMesh(p, lev, false),
          sides: () => {
            const Lh = PLATE.L / 2, Wh = PLATE.W / 2, a = PLATE.a;
            return [{name: "left", label: "left end", pts: [[-Lh, Wh], [-Lh, -Wh]]},
                    {name: "right", label: "right end", pts: [[Lh, -Wh], [Lh, Wh]]},
                    {name: "top", label: "top edge", pts: [[Lh, Wh], [-Lh, Wh]]},
                    {name: "bottom", label: "bottom edge", pts: [[-Lh, -Wh], [Lh, -Wh]]},
                    {name: "hole", label: "the hole", pts: d3range(65).map(k => [a * Math.cos(-2 * Math.PI * k / 64), a * Math.sin(-2 * Math.PI * k / 64)])}];
          },
          outline: () => ({outer: [[-PLATE.L / 2, -PLATE.W / 2], [PLATE.L / 2, -PLATE.W / 2], [PLATE.L / 2, PLATE.W / 2], [-PLATE.L / 2, PLATE.W / 2]],
                           holes: [d3range(64).map(k => [PLATE.a * Math.cos(-2 * Math.PI * k / 64), PLATE.a * Math.sin(-2 * Math.PI * k / 64)])]}),
          defaults: () => ({left: {x: ["disp", 0], y: ["disp", 0]}, right: {x: ["traction", PLATE.s], y: ["free", 0]}})},
  quarter: {label: "Plate with a hole: a quarter, by symmetry", plane: "stress", t: PLATE.t, tRange: [-100, 100], uRange: [-0.1, 0.1],
            make: (p, lev) => plateMesh(p, lev, true),
            sides: () => {
              const Lh = PLATE.L / 2, Wh = PLATE.W / 2, a = PLATE.a;
              return [{name: "left", label: "left edge, x₁ = 0 (a line of symmetry)", pts: [[0, Wh], [0, a]]},
                      {name: "bottom", label: "bottom edge, x₂ = 0 (a line of symmetry)", pts: [[a, 0], [Lh, 0]]},
                      {name: "right", label: "right end", pts: [[Lh, 0], [Lh, Wh]]},
                      {name: "top", label: "top edge", pts: [[Lh, Wh], [0, Wh]]},
                      {name: "hole", label: "the hole", pts: d3range(17).map(k => [a * Math.cos(Math.PI / 2 * (16 - k) / 16), a * Math.sin(Math.PI / 2 * (16 - k) / 16)])}];
            },
            outline: () => {
              const Lh = PLATE.L / 2, Wh = PLATE.W / 2, a = PLATE.a;
              return {outer: [[a, 0], [Lh, 0], [Lh, Wh], [0, Wh], [0, a],
                              ...d3range(1, 16).map(k => [a * Math.cos(Math.PI / 2 * (16 - k) / 16), a * Math.sin(Math.PI / 2 * (16 - k) / 16)])], holes: []};
            },
            defaults: () => ({left: {x: ["disp", 0], y: ["free", 0]}, bottom: {x: ["free", 0], y: ["disp", 0]},
                              right: {x: ["traction", PLATE.s], y: ["free", 0]}})},
  wshape: {label: `W-section, ${WSHAPE.name}`, plane: "strain", t: 1, tRange: [-1.5, 1.5], uRange: [-0.25, 0.25],
           make: (p, lev) => wshapeMesh(p, lev),
           sides: () => {
             const {d, bf, tf, tw} = WSHAPE, b = bf / 2, w = tw / 2;
             return [{name: "top face", label: "top face of the top flange", pts: [[b, d], [-b, d]]},
                     {name: "top-left tip", label: "left tip of the top flange", pts: [[-b, d], [-b, d - tf]]},
                     {name: "top flange, inside left", label: "underside of the top flange, left", pts: [[-b, d - tf], [-w, d - tf]]},
                     {name: "web, left face", label: "left face of the web", pts: [[-w, d - tf], [-w, tf]]},
                     {name: "bottom flange, inside left", label: "top of the bottom flange, left", pts: [[-w, tf], [-b, tf]]},
                     {name: "bottom-left tip", label: "left tip of the bottom flange", pts: [[-b, tf], [-b, 0]]},
                     {name: "bottom face", label: "underside of the bottom flange", pts: [[-b, 0], [b, 0]]},
                     {name: "bottom-right tip", label: "right tip of the bottom flange", pts: [[b, 0], [b, tf]]},
                     {name: "bottom flange, inside right", label: "top of the bottom flange, right", pts: [[b, tf], [w, tf]]},
                     {name: "web, right face", label: "right face of the web", pts: [[w, tf], [w, d - tf]]},
                     {name: "top flange, inside right", label: "underside of the top flange, right", pts: [[w, d - tf], [b, d - tf]]},
                     {name: "top-right tip", label: "right tip of the top flange", pts: [[b, d - tf], [b, d]]}];
           },
           outline: () => {
             const {d, bf, tf, tw} = WSHAPE, b = bf / 2, w = tw / 2;
             return {outer: [[-b, 0], [b, 0], [b, tf], [w, tf], [w, d - tf], [b, d - tf], [b, d], [-b, d], [-b, d - tf], [-w, d - tf], [-w, tf], [-b, tf]], holes: []};
           },
           defaults: () => ({"bottom face": {x: ["disp", 0], y: ["disp", 0]}, "top face": {x: ["free", 0], y: ["traction", -WSHAPE.q]}})},
  cantilever: {label: "Cantilever (an exact solution is known)", plane: "stress", t: CANTILEVER.t, tRange: [-20, 20], uRange: [-1, 1],
               make: (p, lev) => cantileverMesh(p, [1, 2, 4, 8][lev - 1]),
               sides: () => {
                 const {L, D} = CANTILEVER;
                 return [{name: "left", label: "fixed end, x₁ = 0", pts: [[0, D / 2], [0, -D / 2]]},
                         {name: "right", label: "loaded end, x₁ = L", pts: [[L, -D / 2], [L, D / 2]]},
                         {name: "top", label: "top edge", pts: [[L, D / 2], [0, D / 2]]},
                         {name: "bottom", label: "bottom edge", pts: [[0, -D / 2], [L, -D / 2]]}];
               },
               outline: () => ({outer: [[0, -CANTILEVER.D / 2], [CANTILEVER.L, -CANTILEVER.D / 2], [CANTILEVER.L, CANTILEVER.D / 2], [0, CANTILEVER.D / 2]], holes: []}),
               defaults: () => ({})}
};

function d3range(a, b) {
  if (b === undefined) { b = a; a = 0; }
  return Array.from({length: Math.max(0, b - a)}, (_, k) => a + k);
}

export function bcToData(state, geomKey) {
  // the editor's state -> {dirichlet, traction} for solve()
  const dirichlet = {}, traction = {};
  for (const [name, s] of Object.entries(state)) {
    const comps = {}, h = [0, 0];
    let anyH = false;
    ["x", "y"].forEach((dir, i) => {
      const [type, v] = s[dir] || ["free", 0];
      if (type === "disp") comps[i] = v;
      else if (type === "traction" && v !== 0) { h[i] = v; anyH = true; }
    });
    if (Object.keys(comps).length) dirichlet[name] = comps;
    if (anyH) traction[name] = h;
  }
  return {dirichlet, traction};
}

// ------------------------------------------------------------- drawing
const NAVY = "#1e3a5f", AMBER = "#d97706", RED = "#c0392b", INK = "#1a1a1a", ORANGE = "#e67e22";

export function deformedCanvas(width, mesh, sol, scale, pad = 0.06) {
  // a canvas whose world box holds both the body and its exaggerated deformed shape
  const xs = [], ys = [];
  mesh.X.forEach(([x, y], A) => {
    xs.push(x); ys.push(y);
    if (sol && sol.ok) { xs.push(x + scale * sol.U[A][0]); ys.push(y + scale * sol.U[A][1]); }
  });
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const m = pad * Math.max(x1 - x0, y1 - y0);
  return makeCanvas(width, x0 - m, x1 + m, y0 - m, y1 + m, {xlabel: "x₁ (mm)", ylabel: "x₂ (mm)"});
}

export function drawDeformed(cv, mesh, sol, valueAt, color, scale, k = 3) {
  // fill every element, at its deformed position, with valueAt(e, xi, eta) on a k-by-k grid of sub-cells
  const {ctx, sx, sy} = cv;
  const pos = (e, xi, eta) => {
    const r = evaluate(sol, e, xi, eta);
    return [r.x[0] + scale * r.u[0], r.x[1] + scale * r.u[1]];
  };
  for (let e = 1; e <= mesh.nel; e++) {
    const P = [];
    for (let i = 0; i <= k; i++) { const row = []; for (let j = 0; j <= k; j++) row.push(pos(e, -1 + 2 * i / k, -1 + 2 * j / k)); P.push(row); }
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      ctx.fillStyle = ctx.strokeStyle = color(valueAt(e, -1 + 2 * (i + 0.5) / k, -1 + 2 * (j + 0.5) / k));
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      ctx.moveTo(sx(P[i][j][0]), sy(P[i][j][1])); ctx.lineTo(sx(P[i + 1][j][0]), sy(P[i + 1][j][1]));
      ctx.lineTo(sx(P[i + 1][j + 1][0]), sy(P[i + 1][j + 1][1])); ctx.lineTo(sx(P[i][j + 1][0]), sy(P[i][j + 1][1]));
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }
}

export function drawDeformedMesh(cv, mesh, sol, scale, {color = "rgba(26,26,26,0.45)", width = 0.7, highlight = null} = {}) {
  const {ctx, sx, sy} = cv, m = mesh.p === 1 ? 1 : 6;
  const sides = [t => [t, -1], t => [1, t], t => [-t, 1], t => [-1, -t]];
  const outline = e => {
    const pts = [];
    for (const s of sides) for (let i = 0; i < m; i++) {
      const [xi, eta] = s(-1 + 2 * i / m), r = sol ? evaluate(sol, e, xi, eta) : null;
      if (r) pts.push([r.x[0] + scale * r.u[0], r.x[1] + scale * r.u[1]]);
      else {
        const sh = shape(mesh.p, xi, eta);
        pts.push([0, 1].map(k => mesh.IEN[e - 1].reduce((acc, A, a) => acc + sh.N[a] * mesh.X[A - 1][k], 0)));
      }
    }
    return pts;
  };
  const path = pts => { ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))); ctx.closePath(); };
  ctx.strokeStyle = color; ctx.lineWidth = width;
  for (let e = 1; e <= mesh.nel; e++) { path(outline(e)); ctx.stroke(); }
  if (highlight) { ctx.strokeStyle = ORANGE; ctx.lineWidth = 2.5; path(outline(highlight)); ctx.stroke(); }
}

export function drawOutline(cv, outline, {color = "#555", dash = [5, 4], width = 1} = {}) {
  const {ctx, sx, sy} = cv;
  ctx.save(); ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = width;
  for (const poly of [outline.outer, ...outline.holes]) {
    ctx.beginPath(); poly.forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))); ctx.closePath(); ctx.stroke();
  }
  ctx.restore();
}

function glyphs(sides, state, {tMax = 1, uMax = 1, size = 1, maxArrow = 1}) {
  // supports and loads for each side, as world-coordinate shapes: a side held in
  // both directions is hatched, {hatch: {line, ticks}}; a side held in one
  // direction gets triangles (rollers), {tri: [p, q, r], filled}; prescribed
  // nonzero displacements and tractions are arrows, {arrow: [from, to]}
  const out = [];
  for (const sd of sides) {
    const s = state[sd.name];
    if (!s) continue;
    const pts = sd.pts, segs = [];
    let len = 0;
    for (let k = 0; k < pts.length - 1; k++) {
      const l = Math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]);
      segs.push([pts[k], pts[k + 1], len, l]); len += l;
    }
    const at = f => {
      const L = f * len;
      for (const [P, Q, l0, l] of segs) if (L <= l0 + l + 1e-12) {
        const u = l > 0 ? (L - l0) / l : 0;
        return {x: [P[0] + u * (Q[0] - P[0]), P[1] + u * (Q[1] - P[1])], tang: [(Q[0] - P[0]) / (l || 1), (Q[1] - P[1]) / (l || 1)]};
      }
      const [P, Q, , l] = segs[segs.length - 1];
      return {x: Q, tang: [(Q[0] - P[0]) / (l || 1), (Q[1] - P[1]) / (l || 1)]};
    };
    const n = Math.max(2, Math.min(7, Math.round(len / (size * 0.9)) + 1));
    // the outward normal of a counter-clockwise boundary is the tangent turned clockwise
    const normal = tang => [tang[1], -tang[0]];
    const held = ["x", "y"].every(dir => (s[dir] || ["free"])[0] === "disp");
    if (held) {
      // held in both directions: hatching on the outside, slanted at 45 degrees
      const nt = Math.max(3, Math.min(40, Math.round(len / (0.3 * size)))), L = 0.5 * size, ticks = [];
      for (let k = 0; k < nt; k++) {
        const {x, tang} = at((k + 0.5) / nt), nrm = normal(tang);
        const d = [(nrm[0] - tang[0]) / Math.SQRT2, (nrm[1] - tang[1]) / Math.SQRT2];
        ticks.push([x, [x[0] + L * d[0], x[1] + L * d[1]]]);
      }
      out.push({hatch: {line: pts, ticks}, color: NAVY});
      // a prescribed nonzero displacement is an arrow beyond the hatching
      const {x, tang} = at(0.5), nrm = normal(tang);
      ["x", "y"].forEach((dir, i) => {
        const v = s[dir][1];
        if (v === 0) return;
        const e = i === 0 ? [1, 0] : [0, 1], Lu = size * (0.6 + 0.6 * Math.min(1, Math.abs(v) / uMax)), sg = Math.sign(v);
        const b = [x[0] + 0.75 * size * nrm[0], x[1] + 0.75 * size * nrm[1]];
        out.push({arrow: [b, [b[0] + sg * e[0] * Lu, b[1] + sg * e[1] * Lu]], color: NAVY});
      });
      continue;
    }
    ["x", "y"].forEach((dir, i) => {
      const [type, v] = s[dir] || ["free", 0];
      for (let k = 0; k < n; k++) {
        const f = n === 1 ? 0.5 : (k + 0.5) / n, {x, tang} = at(f);
        const nrm = normal(tang);
        if (type === "disp") {
          const e = i === 0 ? [1, 0] : [0, 1], ne = nrm[0] * e[0] + nrm[1] * e[1], sgn = ne >= 0 ? 1 : -1, h = 0.42 * size;
          // a support acting along the side sits just outside it
          const off = Math.abs(ne) < 0.5 ? 0.55 * h : 0, apex = [x[0] + off * nrm[0], x[1] + off * nrm[1]];
          const base = [apex[0] + sgn * e[0] * h, apex[1] + sgn * e[1] * h], perp = [-e[1], e[0]];
          out.push({tri: [apex, [base[0] + perp[0] * h * 0.55, base[1] + perp[1] * h * 0.55], [base[0] - perp[0] * h * 0.55, base[1] - perp[1] * h * 0.55]], color: NAVY, filled: v === 0});
          if (v !== 0 && k === Math.floor(n / 2)) {
            const L = size * (0.6 + 0.6 * Math.min(1, Math.abs(v) / uMax)), sg = Math.sign(v);
            out.push({arrow: [[x[0] + sgn * e[0] * 1.3 * h, x[1] + sgn * e[1] * 1.3 * h], [x[0] + sgn * e[0] * 1.3 * h + sg * e[0] * L, x[1] + sgn * e[1] * 1.3 * h + sg * e[1] * L]], color: NAVY});
          }
        } else if (type === "traction" && v !== 0) {
          const e = i === 0 ? [1, 0] : [0, 1], L = maxArrow * size * (0.35 + 0.65 * Math.min(1, Math.abs(v) / tMax)), sg = Math.sign(v);
          const ne = nrm[0] * e[0] + nrm[1] * e[1], outward = ne * sg >= 0;
          // a traction along the side is drawn just outside it
          const off = Math.abs(ne) < 0.5 ? 0.3 * size : 0, xb = [x[0] + off * nrm[0], x[1] + off * nrm[1]];
          const from = outward ? xb : [xb[0] - sg * e[0] * L, xb[1] - sg * e[1] * L], to = outward ? [xb[0] + sg * e[0] * L, xb[1] + sg * e[1] * L] : xb;
          out.push({arrow: [from, to], color: RED});
        }
      }
    });
  }
  return out;
}

export function drawGlyphs(cv, sides, state, opts) {
  const {ctx, sx, sy} = cv;
  for (const g of glyphs(sides, state, opts)) {
    if (g.hatch) {
      ctx.strokeStyle = g.color; ctx.lineCap = "round";
      ctx.lineWidth = 3; ctx.beginPath();
      g.hatch.line.forEach(([x, y], k) => k ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))); ctx.stroke();
      ctx.lineWidth = 1.2; ctx.beginPath();
      for (const [[x0, y0], [x1, y1]] of g.hatch.ticks) { ctx.moveTo(sx(x0), sy(y0)); ctx.lineTo(sx(x1), sy(y1)); }
      ctx.stroke(); ctx.lineCap = "butt";
    } else if (g.tri) {
      ctx.beginPath(); g.tri.forEach(([x, y], k) => k ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))); ctx.closePath();
      ctx.fillStyle = g.filled ? g.color : "white"; ctx.strokeStyle = g.color; ctx.lineWidth = 1.2; ctx.fill(); ctx.stroke();
    } else {
      const [[x0, y0], [x1, y1]] = g.arrow, X0 = sx(x0), Y0 = sy(y0), X1 = sx(x1), Y1 = sy(y1), a = Math.atan2(Y1 - Y0, X1 - X0);
      ctx.strokeStyle = ctx.fillStyle = g.color; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X1, Y1); ctx.lineTo(X1 - 8 * Math.cos(a - 0.4), Y1 - 8 * Math.sin(a - 0.4));
      ctx.lineTo(X1 - 8 * Math.cos(a + 0.4), Y1 - 8 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
    }
  }
}

// ------------------------------------------------- the boundary-condition editor
export function bcEditor(geomKey, {width = 520, value = null} = {}) {
  // An input for OJS: click a side, then choose for each direction whether
  // that side is free, carries a traction h_i, or has a prescribed displacement
  // g_i.  Its value is {side name: {x: [type, value], y: [type, value]}}.
  const G = GEOMETRIES[geomKey], sides = G.sides(), outline = G.outline();
  let state = JSON.parse(JSON.stringify(value || G.defaults()));
  let sel = sides.find(s => state[s.name] && Object.values(state[s.name]).some(v => v[0] !== "free"))?.name || sides[0].name;
  const root = document.createElement("div");
  root.style.cssText = "display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;";
  const all = [...outline.outer, ...outline.holes.flat()];
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const m = 0.2 * span, x0 = Math.min(...xs) - m, x1 = Math.max(...xs) + m, y0 = Math.min(...ys) - m, y1 = Math.max(...ys) + m;
  const W = Math.min(width, 360), H = W * (y1 - y0) / (x1 - x0);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", W); svg.setAttribute("height", H); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.cssText = "background:white; max-width:100%; height:auto;";
  const X = x => (x - x0) / (x1 - x0) * W, Y = y => H - (y - y0) / (y1 - y0) * H;
  const left = document.createElement("div"), hint = document.createElement("div");
  hint.style.cssText = `font-size:0.85rem; color:#555; min-height:1.4em; max-width:${W}px`;
  left.append(svg, hint);
  const panel = document.createElement("div");
  panel.style.cssText = "flex:1; min-width:16rem; font-size:0.92rem; line-height:1.5;";
  root.append(left, panel);
  // The radios and boxes in the panel fire their own input events, and a radio
  // fires its input event before its change handler updates the state.  If
  // those reached OJS, it would read the editor's value from before the change.
  // Only emit() speaks for the editor.
  for (const type of ["input", "change"]) panel.addEventListener(type, ev => ev.stopPropagation());
  const el = (tag, attrs = {}, parent = svg) => {
    const n = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    parent.appendChild(n); return n;
  };
  const poly = pts => pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
  const emit = () => { root.value = JSON.parse(JSON.stringify(state)); root.dispatchEvent(new Event("input", {bubbles: true})); };
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const d = [outline.outer, ...outline.holes].map(pl => "M" + pl.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("L") + "Z").join(" ");
    el("path", {d, fill: "#e3eef8", stroke: "none", "fill-rule": "evenodd"});
    for (const sd of sides) {
      const s = state[sd.name] || {}, types = ["x", "y"].map(k => (s[k] || ["free"])[0]);
      const col = types.includes("disp") ? NAVY : types.includes("traction") ? AMBER : "#7a8a99";
      el("polyline", {points: poly(sd.pts), fill: "none", stroke: col, "stroke-width": sd.name === sel ? 6 : 3, "stroke-linecap": "round"});
      if (sd.name === sel) el("polyline", {points: poly(sd.pts), fill: "none", stroke: ORANGE, "stroke-width": 2, "stroke-dasharray": "4 3"});
    }
    const size = 0.07 * span, tMax = Math.max(...Object.values(state).flatMap(s => ["x", "y"].map(k => s[k] && s[k][0] === "traction" ? Math.abs(s[k][1]) : 0)), 1e-9);
    for (const g of glyphs(sides, state, {tMax, uMax: Math.abs(G.uRange[1]), size, maxArrow: 1.6})) {
      // the side itself is already drawn in navy, so a hatched side only needs its ticks
      if (g.hatch) for (const [p, q] of g.hatch.ticks) el("line", {x1: X(p[0]), y1: Y(p[1]), x2: X(q[0]), y2: Y(q[1]), stroke: g.color, "stroke-width": 1.3});
      else if (g.tri) el("polygon", {points: poly(g.tri), fill: g.filled ? g.color : "white", stroke: g.color, "stroke-width": 1.2});
      else {
        const [[ax, ay], [bx, by]] = g.arrow, A = [X(ax), Y(ay)], B = [X(bx), Y(by)], a = Math.atan2(B[1] - A[1], B[0] - A[0]);
        el("line", {x1: A[0], y1: A[1], x2: B[0], y2: B[1], stroke: g.color, "stroke-width": 1.8});
        el("polygon", {points: [B, [B[0] - 7 * Math.cos(a - 0.4), B[1] - 7 * Math.sin(a - 0.4)], [B[0] - 7 * Math.cos(a + 0.4), B[1] - 7 * Math.sin(a + 0.4)]].map(q => q.join(",")).join(" "), fill: g.color});
      }
    }
    hoverEl = el("polyline", {points: hover ? poly(hover.pts) : "", fill: "none", stroke: ORANGE, "stroke-width": 8,
                              "stroke-opacity": 0.45, "stroke-linecap": "round", "pointer-events": "none"});
  }
  // The side nearest the pointer, within a few pixels.  Thin parts, such as the
  // web of the W-shape, put several sides within a finger's width of each other,
  // so the nearest one is chosen rather than whichever was drawn last.
  let hover = null, hoverEl = null;
  const nearest = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * H / r.height;
    let best = null, bd = 14;
    for (const sd of sides) for (let k = 0; k < sd.pts.length - 1; k++) {
      const ax = X(sd.pts[k][0]), ay = Y(sd.pts[k][1]), bx = X(sd.pts[k + 1][0]), by = Y(sd.pts[k + 1][1]);
      const L2 = (bx - ax) ** 2 + (by - ay) ** 2;
      const u = L2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2)) : 0;
      const dd = Math.hypot(px - ax - u * (bx - ax), py - ay - u * (by - ay));
      if (dd < bd) { bd = dd; best = sd; }
    }
    return best;
  };
  const hoverTo = sd => {
    hover = sd;
    hoverEl.setAttribute("points", sd ? poly(sd.pts) : "");
    svg.style.cursor = sd ? "pointer" : "default";
    hint.textContent = sd ? `Click to select: ${sd.label}` : "Point at a side of the body, then click to select it.";
  };
  svg.addEventListener("mousemove", ev => { const sd = nearest(ev); if (sd !== hover) hoverTo(sd); });
  svg.addEventListener("mouseleave", () => hoverTo(null));
  svg.addEventListener("click", ev => {
    const sd = nearest(ev);
    if (sd) { sel = sd.name; draw(); drawPanel(); }
  });
  function drawPanel() {
    const sd = sides.find(s => s.name === sel), s = state[sel] || (state[sel] = {x: ["free", 0], y: ["free", 0]});
    panel.innerHTML = "";
    const h = document.createElement("div");
    h.innerHTML = `<b>Selected: ${sd.label}</b><br><span style="color:#555">Click any side of the body to select it.</span>`;
    panel.appendChild(h);
    ["x", "y"].forEach((dir, i) => {
      const box = document.createElement("div");
      box.style.cssText = "margin-top:0.55rem; padding:0.35rem 0.6rem; background:#f7f7f7; border-left:3px solid " + (s[dir][0] === "disp" ? NAVY : s[dir][0] === "traction" ? AMBER : "#bbb");
      const name = `bc-${Math.random().toString(36).slice(2)}`;
      const opts = [["free", `free: h<sub>${i + 1}</sub> = 0`], ["traction", `traction h<sub>${i + 1}</sub> (Neumann)`], ["disp", `displacement g<sub>${i + 1}</sub> (Dirichlet)`]];
      box.innerHTML = `<div><b>Direction x<sub>${i + 1}</sub></b></div>` + opts.map(([v, lab]) =>
        `<label style="margin-right:0.8rem; white-space:nowrap"><input type="radio" name="${name}" value="${v}" ${s[dir][0] === v ? "checked" : ""}> ${lab}</label>`).join("");
      if (s[dir][0] !== "free") {
        const [lo, hi] = s[dir][0] === "traction" ? G.tRange : G.uRange, step = (hi - lo) / 400;
        const unit = s[dir][0] === "traction" ? "MPa" : "mm";
        const row = document.createElement("div");
        row.style.cssText = "display:flex; align-items:center; gap:0.5rem; margin-top:0.25rem";
        row.innerHTML = `<input type="range" min="${lo}" max="${hi}" step="${step}" value="${s[dir][1]}" style="flex:1">
          <input type="number" min="${lo}" max="${hi}" step="${step}" value="${s[dir][1]}" style="width:5.5rem"> <span>${unit}</span>`;
        const [rng, num] = row.querySelectorAll("input");
        const set = v => { s[dir][1] = Math.max(lo, Math.min(hi, +v)); rng.value = num.value = s[dir][1]; draw(); emit(); };
        rng.addEventListener("input", () => set(rng.value));
        num.addEventListener("change", () => set(num.value));
        box.appendChild(row);
      }
      box.querySelectorAll(`input[name="${name}"]`).forEach(r => r.addEventListener("change", () => {
        s[dir] = [r.value, r.value === "traction" ? (G.tRange[1] / 4) : 0];
        draw(); drawPanel(); emit();
      }));
      panel.appendChild(box);
    });
    const reset = document.createElement("button");
    reset.textContent = "Restore the starting conditions";
    reset.style.cssText = "margin-top:0.7rem";
    reset.addEventListener("click", () => { state = JSON.parse(JSON.stringify(G.defaults())); draw(); drawPanel(); emit(); });
    panel.appendChild(reset);
  }
  draw(); drawPanel(); hoverTo(null);
  root.value = JSON.parse(JSON.stringify(state));
  return root;
}

// ------------------------------------------- plane stress and plane strain, in 3D
export function planeGraphic(geomKey, plane, {width = 340} = {}) {
  // the 2D body extruded along x3: a thin plate (plane stress) or a long body
  // held at both ends (plane strain), in an oblique projection
  const G = GEOMETRIES[geomKey], ol = G.outline();
  const all = [...ol.outer, ...ol.holes.flat()];
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const depth = plane === "stress" ? 0.07 * span : 1.6 * span;
  const ang = Math.PI / 6, k = 0.5;
  const P = (x, y, z) => [x - cx + k * z * Math.cos(ang), y - cy + k * z * Math.sin(ang)];
  const proj = [...all.map(([x, y]) => P(x, y, 0)), ...all.map(([x, y]) => P(x, y, depth))];
  const px = proj.map(p => p[0]), py = proj.map(p => p[1]);
  const m = 0.12 * span, X0 = Math.min(...px) - m, X1 = Math.max(...px) + m, Y0 = Math.min(...py) - m, Y1 = Math.max(...py) + m;
  const W = width, H = Math.min(W * (Y1 - Y0) / (X1 - X0), 1.1 * W);
  const sc = Math.min(W / (X1 - X0), H / (Y1 - Y0));
  const SX = q => (q[0] - X0) * sc, SY = q => H - (q[1] - Y0) * sc;
  const pt = q => `${SX(q).toFixed(1)},${SY(q).toFixed(1)}`;
  const parts = [];
  const face = (pts, fill, stroke = INK, extra = "") => parts.push(`<polygon points="${pts.map(pt).join(" ")}" fill="${fill}" stroke="${stroke}" stroke-width="0.8" ${extra}/>`);
  const dir = [Math.cos(ang), Math.sin(ang)];
  // back face, then the walls that face the viewer, farthest first, then the front face
  const backD = [ol.outer, ...ol.holes].map(pl => "M" + pl.map(([x, y]) => pt(P(x, y, depth))).join("L") + "Z").join(" ");
  parts.push(`<path d="${backD}" fill="${plane === "strain" ? "#cfd8e3" : "#dfe6ee"}" stroke="${INK}" stroke-width="0.8" fill-rule="evenodd"/>`);
  if (plane === "strain") parts.push(`<path d="${backD}" fill="url(#hatch)" stroke="none" fill-rule="evenodd"/>`);
  const walls = [];
  for (const pl of [ol.outer, ...ol.holes]) for (let i = 0; i < pl.length; i++) {
    const a = pl[i], b = pl[(i + 1) % pl.length], nrm = [b[1] - a[1], -(b[0] - a[0])];
    if (nrm[0] * dir[0] + nrm[1] * dir[1] <= 0) continue;
    walls.push({a, b, key: (a[0] + b[0]) * dir[0] + (a[1] + b[1]) * dir[1]});
  }
  walls.sort((u, v) => v.key - u.key);
  for (const {a, b} of walls) face([P(...a, 0), P(...b, 0), P(...b, depth), P(...a, depth)], "#b9c6d4", "#6b7b8c");
  const frontD = [ol.outer, ...ol.holes].map(pl => "M" + pl.map(([x, y]) => pt(P(x, y, 0))).join("L") + "Z").join(" ");
  parts.push(`<path d="${frontD}" fill="#e3eef8" stroke="${INK}" stroke-width="1.2" fill-rule="evenodd"/>`);
  if (plane === "strain") parts.push(`<path d="${frontD}" fill="url(#hatch)" stroke="none" fill-rule="evenodd"/>`);
  // the x3 axis
  const o = P(Math.min(...xs) - 0.06 * span, Math.min(...ys) - 0.06 * span, 0);
  const ax = (q, lab) => parts.push(`<line x1="${SX(o)}" y1="${SY(o)}" x2="${SX(q)}" y2="${SY(q)}" stroke="${RED}" stroke-width="1.3" marker-end="url(#ah)"/><text x="${SX(q) + 4}" y="${SY(q) - 3}" fill="${RED}" font-size="12">${lab}</text>`);
  ax([o[0] + 0.18 * span, o[1]], "x₁"); ax([o[0], o[1] + 0.18 * span], "x₂");
  ax([o[0] + 0.25 * span * k * dir[0] * 2, o[1] + 0.25 * span * k * dir[1] * 2], "x₃");
  const defs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="${RED}"/></marker>
    <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#5b6b7c" stroke-width="1"/></pattern></defs>`;
  const svg = `<svg width="${W}" height="${H}" style="background:white">${defs}${parts.join("")}</svg>`;
  const text = plane === "stress"
    ? `<b>Plane stress: a thin plate.</b> The faces x₃ = ±t/2 carry no load, so σ₃₃ = σ₁₃ = σ₂₃ = 0 through the thin thickness. The plate is free to thin or thicken: ε₃₃ = −ν(σ₁₁ + σ₂₂)/E ≠ 0.`
    : `<b>Plane strain: a long body.</b> Far from its ends, or with its ends held (hatched), no slice can stretch along x₃: ε₃₃ = ε₁₃ = ε₂₃ = 0. Holding it takes a stress σ₃₃ = ν(σ₁₁ + σ₂₂) ≠ 0.`;
  const div = document.createElement("div");
  div.style.cssText = "max-width:" + W + "px; font-size:0.9rem; line-height:1.45";
  div.innerHTML = svg + `<div style="margin-top:0.3rem">${text}</div>`;
  return div;
}

// ------------------------------------------------- the global matrix, for the spy plot
export function kEntries(sol) {
  const seen = new Set(), out = [];
  for (const lm of sol.LM) for (const P of lm) if (P) for (const Q of lm) if (Q) {
    const k = (P - 1) * sol.neq + (Q - 1);
    if (!seen.has(k)) { seen.add(k); out.push([P - 1, Q - 1]); }
  }
  return out;
}

export {drawSpy};
