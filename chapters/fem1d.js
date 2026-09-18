// One-dimensional Galerkin finite elements for Hughes' model problem
//     -(K u_,x)_,x = f  on ]0, L[,
// in the browser.  A port of fem1d.py, which draws the static figures on the
// same pages -- change the two together.  Hughes' conventions throughout:
// global nodes A = 1..nnp, local nodes a = 1..p+1, and 1-based ID / IEN / LM
// arrays with ID = 0 marking a Dirichlet node.  (JavaScript arrays are
// 0-based, so IEN[e][a-1] holds the global number A of local node a.)
//
// Boundary-condition arrangements, as on the strong-form pages:
//   "a"  Neumann h at x = 0,   Dirichlet g at x = L   (Hughes)
//   "b"  Dirichlet g at x = 0, Neumann h at x = L
//   "c"  Dirichlet g0 at x = 0 and g at x = L

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

export function parentNodes(p) {
  return Array.from({length: p + 1}, (_, a) => -1 + 2 * a / p);
}

export function shape(p, xi) {
  // N_a(xi) and N_a,xi(xi), a = 1..p+1 stored at index a-1
  const nd = parentNodes(p), N = [], dN = [];
  for (let a = 0; a <= p; a++) {
    let v = 1, d = 0;
    for (let b = 0; b <= p; b++) if (b !== a) v *= (xi - nd[b]) / (nd[a] - nd[b]);
    for (let m = 0; m <= p; m++) {
      if (m === a) continue;
      let t = 1 / (nd[a] - nd[m]);
      for (let b = 0; b <= p; b++) if (b !== a && b !== m) t *= (xi - nd[b]) / (nd[a] - nd[b]);
      d += t;
    }
    N.push(v); dN.push(d);
  }
  return {N, dN};
}

export function makeMesh(L, nel, kind = "uniform", custom = [], grade = 2) {
  const s = Array.from({length: nel + 1}, (_, i) => i / nel);
  if (kind === "left") return s.map(t => L * t ** grade);
  if (kind === "right") return s.map(t => L * (1 - (1 - t) ** grade));
  if (kind === "custom") {
    const inner = custom.filter(c => c > 0 && c < L).sort((a, b) => a - b);
    const uniq = inner.filter((c, i) => i === 0 || c - inner[i - 1] > 1e-12 * L);
    return [0, ...uniq, L];
  }
  return s.map(t => L * t);
}

export function exact(pb, nsub = 64, nq = 8) {
  // exact u, u_,x for constant K from F1 = int_0^x f and F2 = int_0^x (x-s) f
  const G = gauss(nq);
  const F = x => {
    if (x <= 0) return [0, 0];
    const m = Math.max(1, Math.ceil(nsub * x / pb.L));
    let f1 = 0, f2 = 0;
    for (let k = 0; k < m; k++) {
      const a = x * k / m, b = x * (k + 1) / m;
      for (let q = 0; q < nq; q++) {
        const s = 0.5 * (a + b) + 0.5 * (b - a) * G.x[q];
        const ws = 0.5 * (b - a) * G.w[q] * pb.f(s);
        f1 += ws; f2 += ws * (x - s);
      }
    }
    return [f1, f2];
  };
  const {K, L, h, g, g0} = pb;
  const [F1L, F2L] = F(L);
  let C1, C2;
  if (pb.bc === "a") { C2 = -h / K; C1 = g - C2 * L + F2L / K; }
  else if (pb.bc === "b") { C1 = g; C2 = (h + F1L) / K; }
  else { C1 = g0; C2 = (g - g0 + F2L / K) / L; }
  return {u: x => C1 + C2 * x - F(x)[1] / K, du: x => C2 - F(x)[0] / K};
}

export function dirichletNodes(pb, nnp) {
  if (pb.bc === "a") return new Map([[nnp, pb.g]]);
  if (pb.bc === "b") return new Map([[1, pb.g]]);
  return new Map([[1, pb.g0], [nnp, pb.g]]);
}

function linsolve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
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

export function solve(pb, p, verts, nint) {
  const nel = verts.length - 1, nen = p + 1, nnp = p * nel + 1;
  const X = new Array(nnp), IEN = [];
  for (let e = 0; e < nel; e++) {
    const row = [];
    for (let a = 0; a < nen; a++) {
      const A = p * e + a + 1;
      row.push(A);
      X[A - 1] = verts[e] + (verts[e + 1] - verts[e]) * a / p;
    }
    IEN.push(row);
  }
  const gnode = dirichletNodes(pb, nnp);
  const ID = new Array(nnp);
  let neq = 0;
  for (let A = 1; A <= nnp; A++) ID[A - 1] = gnode.has(A) ? 0 : ++neq;
  const LM = IEN.map(row => row.map(A => ID[A - 1]));

  const G = gauss(nint || p + 3);
  const sh = G.x.map(xi => shape(p, xi));
  const Kg = Array.from({length: neq}, () => new Array(neq).fill(0));
  const F = new Array(neq).fill(0);
  const ke = [], fe = [];
  for (let e = 0; e < nel; e++) {
    const he = verts[e + 1] - verts[e], jac = he / 2;
    const k = Array.from({length: nen}, () => new Array(nen).fill(0));
    const body = new Array(nen).fill(0), fh = new Array(nen).fill(0);
    G.x.forEach((xi, q) => {
      const {N, dN} = sh[q], xq = verts[e] + (xi + 1) * jac, fq = pb.f(xq);
      for (let a = 0; a < nen; a++) {
        body[a] += N[a] * fq * jac * G.w[q];
        for (let b = 0; b < nen; b++) k[a][b] += pb.K * dN[a] * dN[b] / jac * G.w[q];
      }
    });
    if (pb.bc === "a" && e === 0) fh[0] = pb.h;
    if (pb.bc === "b" && e === nel - 1) fh[nen - 1] = pb.h;
    const ge = IEN[e].map(A => gnode.has(A) ? gnode.get(A) : 0);
    const fg = k.map(r => r.reduce((s, v, b) => s + v * ge[b], 0));
    const total = body.map((v, a) => v + fh[a] - fg[a]);
    ke.push(k); fe.push({body, h: fh, g: fg, total});
    for (let a = 0; a < nen; a++) {
      const P = LM[e][a];
      if (!P) continue;
      F[P - 1] += total[a];
      for (let b = 0; b < nen; b++) {
        const Q = LM[e][b];
        if (Q) Kg[P - 1][Q - 1] += k[a][b];
      }
    }
  }
  const d = neq ? linsolve(Kg, F) : [];
  const U = Array.from({length: nnp}, (_, i) => ID[i] ? d[ID[i] - 1] : gnode.get(i + 1));
  return {p, verts, nel, nnp, neq, X, IEN, ID, LM, ke, fe, K: Kg, F, d, U, gnode};
}

export function evaluate(sol, m = 30) {
  // u^h, u^h_,x sampled inside each element: [{e, x:[], uh:[], duh:[]}]
  const {p, verts, IEN, U} = sol, out = [];
  for (let e = 0; e < sol.nel; e++) {
    const jac = (verts[e + 1] - verts[e]) / 2, Ue = IEN[e].map(A => U[A - 1]);
    const r = {e: e + 1, x: [], uh: [], duh: []};
    for (let i = 0; i < m; i++) {
      const xi = -1 + 2 * i / (m - 1), {N, dN} = shape(p, xi);
      r.x.push(verts[e] + (xi + 1) * jac);
      r.uh.push(N.reduce((s, v, a) => s + v * Ue[a], 0));
      r.duh.push(dN.reduce((s, v, a) => s + v * Ue[a], 0) / jac);
    }
    out.push(r);
  }
  return out;
}

export function errors(sol, ex, nq = 10) {
  const {p, verts, IEN, U} = sol, G = gauss(nq);
  const sh = G.x.map(xi => shape(p, xi));
  let l2 = 0, h1 = 0;
  for (let e = 0; e < sol.nel; e++) {
    const jac = (verts[e + 1] - verts[e]) / 2, Ue = IEN[e].map(A => U[A - 1]);
    G.x.forEach((xi, q) => {
      const {N, dN} = sh[q], x = verts[e] + (xi + 1) * jac;
      const uh = N.reduce((s, v, a) => s + v * Ue[a], 0);
      const duh = dN.reduce((s, v, a) => s + v * Ue[a], 0) / jac;
      l2 += G.w[q] * jac * (ex.u(x) - uh) ** 2;
      h1 += G.w[q] * jac * (ex.du(x) - duh) ** 2;
    });
  }
  return {l2: Math.sqrt(l2), h1: Math.sqrt(h1)};
}

// ------------------------------------------------------------ TeX output
export function num(v, sig = 4) {
  if (Math.abs(v) < 1e-12) return "0";
  const s = Number(v.toPrecision(sig)).toString();
  if (s.includes("e")) {
    const [m, ex] = s.split("e");
    return `${m}\\times10^{${parseInt(ex)}}`;
  }
  return s;
}

function indices(n, maxn, show) {
  if (n <= maxn) return Array.from({length: n}, (_, i) => i);
  return [...Array.from({length: show}, (_, i) => i), null, n - 1];
}

// A bmatrix.  Past `maxn` rows it keeps the first `show` rows and columns
// and the last one, with dots between.  `color(i, j)` may return a colour
// to highlight an entry (0-based indices); zeros are drawn light grey.
export function texMatrix(M, {sig = 4, maxn = 14, show = 5, color = null, scale = 1} = {}) {
  const n = M.length, m = M[0] ? M[0].length : 0;
  const ri = indices(n, maxn, show), ci = indices(m, maxn, show);
  const rows = ri.map(i => {
    if (i === null) return ci.map(j => j === null ? "\\ddots" : "\\vdots").join(" & ");
    return ci.map(j => {
      if (j === null) return "\\cdots";
      const v = M[i][j] / scale;
      const c = color ? color(i, j) : null;
      if (Math.abs(M[i][j]) < 1e-12 && !c) return "\\color{#bbbbbb}{0}";
      return c ? `\\color{${c}}{\\mathbf{${num(v, sig)}}}` : num(v, sig);
    }).join(" & ");
  });
  return "\\begin{bmatrix}" + rows.join(" \\\\ ") + "\\end{bmatrix}";
}

export function texVector(v, opts = {}) {
  const color = opts.color ? (i => opts.color(i)) : null;
  return texMatrix(v.map(x => [x]), {...opts, color: color && ((i, j) => color(i))});
}

export function texSymbols(n, sym, {maxn = 14, show = 5} = {}) {
  // a column of symbols d_1 ... d_n, truncated like texMatrix
  const rows = indices(n, maxn, show).map(i => i === null ? "\\vdots" : `${sym}_{${i + 1}}`);
  return "\\begin{bmatrix}" + rows.join(" \\\\ ") + "\\end{bmatrix}";
}

export function niceScale(M) {
  // a power of ten to pull out of a matrix so its entries read as plain numbers
  const flat = M.flat().map(Math.abs).filter(v => v > 1e-12);
  if (!flat.length) return 1;
  const k = Math.floor(Math.log10(Math.max(...flat)));
  return (k >= -1 && k <= 3) ? 1 : 10 ** k;
}
