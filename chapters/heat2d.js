// Steady heat conduction in a rectangle -- the 2D Poisson problem
//     -kappa (u_,11 + u_,22) = f   on ]0, W[ x ]0, H[
// in the browser, for the explorer on the battery-cell page.  A port of
// grid() in heat2d.py (bilinear finite elements on a uniform grid, solved by
// Jacobi-preconditioned conjugate gradients) -- change the two together.
//
// Whole edges are assigned to Gamma_g ("left", "right", "bottom", "top"); a
// corner node touching any Dirichlet edge is a Dirichlet node.  Every other
// edge is part of Gamma_h and carries h(edge, s), the heat flux INTO the body
// at position s along it (s = x1 on bottom/top, s = x2 on left/right).
// Nodes are numbered k = j (nx + 1) + i, with i along x1 and j along x2.

export const EDGES = ["left", "right", "bottom", "top"];

export function solveGrid({W, H, nx, ny, kappa, f, g, dirichlet, h,
                           tol = 1e-10, maxit = 20000}) {
  const hx = W / nx, hy = H / ny, n1 = nx + 1, nn = (nx + 1) * (ny + 1);
  const a = hy / hx, b = hx / hy, c = kappa / 6;
  const Ke = [
    [2 * a + 2 * b, -2 * a + b, -a - b, a - 2 * b],
    [-2 * a + b, 2 * a + 2 * b, a - 2 * b, -a - b],
    [-a - b, a - 2 * b, 2 * a + 2 * b, -2 * a + b],
    [a - 2 * b, -a - b, -2 * a + b, 2 * a + 2 * b]
  ].map(r => r.map(v => c * v));
  const fx = typeof f === "function" ? f : () => f;
  const dset = new Set(dirichlet);

  // corners of element (i, j), counter-clockwise from lower left
  const nodes = (i, j) => [j * n1 + i, j * n1 + i + 1, (j + 1) * n1 + i + 1, (j + 1) * n1 + i];

  function Kmul(v, out) {
    out.fill(0);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const e = nodes(i, j);
      const v0 = v[e[0]], v1 = v[e[1]], v2 = v[e[2]], v3 = v[e[3]];
      for (let p = 0; p < 4; p++) {
        const k = Ke[p];
        out[e[p]] += k[0] * v0 + k[1] * v1 + k[2] * v2 + k[3] * v3;
      }
    }
    return out;
  }

  // f by 2x2 Gauss quadrature on each element
  const F = new Float64Array(nn);
  const gp = [-1 / Math.sqrt(3), 1 / Math.sqrt(3)];
  let made = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const e = nodes(i, j), xc = (i + 0.5) * hx, yc = (j + 0.5) * hy;
    for (const gx of gp) for (const gy of gp) {
      const fq = fx(xc + gx * hx / 2, yc + gy * hy / 2) * hx * hy / 4;
      made += fq;
      const N = [(1 - gx) * (1 - gy) / 4, (1 + gx) * (1 - gy) / 4,
                 (1 + gx) * (1 + gy) / 4, (1 - gx) * (1 + gy) / 4];
      for (let p = 0; p < 4; p++) F[e[p]] += N[p] * fq;
    }
  }

  // the nodes along each edge, in order of increasing s
  const edgeNodes = {
    left:   Array.from({length: ny + 1}, (_, j) => j * n1),
    right:  Array.from({length: ny + 1}, (_, j) => j * n1 + nx),
    bottom: Array.from({length: nx + 1}, (_, i) => i),
    top:    Array.from({length: nx + 1}, (_, i) => ny * n1 + i)
  };
  const D = new Uint8Array(nn);
  let heatIn = 0;
  for (const edge of EDGES) {
    const ids = edgeNodes[edge];
    if (dset.has(edge)) { for (const k of ids) D[k] = 1; continue; }
    const ds = edge === "bottom" || edge === "top" ? hx : hy;
    for (let s = 0; s + 1 < ids.length; s++) {
      for (const gq of gp) {
        const hq = h(edge, (s + 0.5) * ds + gq * ds / 2) * ds / 2;
        heatIn += hq;
        F[ids[s]] += (1 - gq) / 2 * hq;
        F[ids[s + 1]] += (1 + gq) / 2 * hq;
      }
    }
  }

  const U = new Float64Array(nn);
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
    const k = j * n1 + i;
    if (D[k]) U[k] = g(i * hx, j * hy);
  }
  const r = new Float64Array(nn), z = new Float64Array(nn), p = new Float64Array(nn);
  const Ap = new Float64Array(nn), dg = new Float64Array(nn);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const e = nodes(i, j);
    for (let q = 0; q < 4; q++) dg[e[q]] += Ke[q][q];
  }
  Kmul(U, Ap);
  let rz = 0, fnorm = 0, rnorm = 0;
  for (let k = 0; k < nn; k++) {
    r[k] = D[k] ? 0 : F[k] - Ap[k];
    z[k] = r[k] / dg[k];
    p[k] = z[k];
    rz += r[k] * z[k];
    if (!D[k]) fnorm += F[k] * F[k];
    rnorm += r[k] * r[k];
  }
  const scale = Math.sqrt(fnorm) + Math.sqrt(rnorm) + 1e-30;
  const hasDirichlet = D.some(v => v);
  let it = 0, converged = !hasDirichlet ? false : Math.sqrt(rnorm) < tol * scale;
  if (hasDirichlet) {
    for (it = 1; it <= maxit && !converged; it++) {
      Kmul(p, Ap);
      let pAp = 0;
      for (let k = 0; k < nn; k++) { if (D[k]) Ap[k] = 0; pAp += p[k] * Ap[k]; }
      const alpha = rz / pAp;
      let rr = 0;
      for (let k = 0; k < nn; k++) {
        U[k] += alpha * p[k];
        r[k] -= alpha * Ap[k];
        rr += r[k] * r[k];
      }
      if (Math.sqrt(rr) < tol * scale) { converged = true; break; }
      let rzNew = 0;
      for (let k = 0; k < nn; k++) { z[k] = r[k] / dg[k]; rzNew += r[k] * z[k]; }
      const beta = rzNew / rz;
      for (let k = 0; k < nn; k++) p[k] = z[k] + beta * p[k];
      rz = rzNew;
    }
  }

  // heat out through each Dirichlet edge: -(K U - F) at its nodes, with a
  // corner shared by two Dirichlet edges counted once
  Kmul(U, Ap);
  const heatOut = {}, seen = new Uint8Array(nn);
  for (const edge of EDGES) {
    if (!dset.has(edge)) continue;
    let s = 0;
    for (const k of edgeNodes[edge]) if (!seen[k]) { s -= Ap[k] - F[k]; seen[k] = 1; }
    heatOut[edge] = s;
  }

  // bilinear interpolation of u and its gradient inside the grid
  function locate(x, y) {
    const i = Math.min(nx - 1, Math.max(0, Math.floor(x / hx)));
    const j = Math.min(ny - 1, Math.max(0, Math.floor(y / hy)));
    const s = Math.min(1, Math.max(0, x / hx - i)), t = Math.min(1, Math.max(0, y / hy - j));
    const e = nodes(i, j);
    return {s, t, u0: U[e[0]], u1: U[e[1]], u2: U[e[2]], u3: U[e[3]]};
  }
  function at(x, y) {
    const {s, t, u0, u1, u2, u3} = locate(x, y);
    return (1 - s) * (1 - t) * u0 + s * (1 - t) * u1 + s * t * u2 + (1 - s) * t * u3;
  }
  function grad(x, y) {
    const {s, t, u0, u1, u2, u3} = locate(x, y);
    return [((1 - t) * (u1 - u0) + t * (u2 - u3)) / hx,
            ((1 - s) * (u3 - u0) + s * (u2 - u1)) / hy];
  }
  let umax = -Infinity, kmax = 0, umin = Infinity;
  for (let k = 0; k < nn; k++) {
    if (U[k] > umax) { umax = U[k]; kmax = k; }
    if (U[k] < umin) umin = U[k];
  }
  return {W, H, nx, ny, hx, hy, U, D, made, heatIn, heatOut, iterations: it,
          converged, hasDirichlet, at, grad, umin, umax,
          xmax: (kmax % n1) * hx, ymax: Math.floor(kmax / n1) * hy};
}

// h on the top edge of the cell: h_tab over each tab patch, zero between
export function tabFlux(s, hTab, tabs) {
  for (const [a, b] of tabs) if (s >= a && s <= b) return hTab;
  return 0;
}

// round, evenly spaced isotherm levels covering [lo, hi]
export function niceLevels(lo, hi, target = 12) {
  const span = Math.max(hi - lo, 1e-9), raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return {levels: out, step};
}
