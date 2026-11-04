// Drawings for the widgets on the first elasticity page: stress on a small
// block in 2D and 3D, the split of the displacement gradient into strain and
// rotation, and the elasticity matrix of isotropic and orthotropic materials.
// Every function returns an HTML element built from an SVG string.
// Units: MPa for stress, GPa for moduli.

export const COMP = {           // one color per stress component, used in the matrices too
  s11: "#c0392b", s22: "#1f6feb", s33: "#7c3aed",
  s12: "#0d9488", s13: "#e67e22", s23: "#8a5a20"
};
const INK = "#1a1a1a";
const SUB = ["₀", "₁", "₂", "₃"];

function arrowSVG(x1, y1, x2, y2, color, w = 2.2, head = 9) {
  const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1);
  if (L < 2) return "";
  const h = Math.min(head, 0.6 * L);
  const hx = x2 - h * Math.cos(a), hy = y2 - h * Math.sin(a);
  return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${hx.toFixed(1)}" y2="${hy.toFixed(1)}" stroke="${color}" stroke-width="${w}"/>`
    + `<polygon points="${x2.toFixed(1)},${y2.toFixed(1)} ${(x2 - h * Math.cos(a - 0.38)).toFixed(1)},${(y2 - h * Math.sin(a - 0.38)).toFixed(1)} ${(x2 - h * Math.cos(a + 0.38)).toFixed(1)},${(y2 - h * Math.sin(a + 0.38)).toFixed(1)}" fill="${color}"/>`;
}

const label = (x, y, txt, color, anchor = "middle", size = 13) =>
  `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${color}" font-size="${size}" text-anchor="${anchor}" dominant-baseline="middle">${txt}</text>`;

function wrap(svg, w, h, extra = "") {
  const div = document.createElement("div");
  div.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="background:white; max-width:100%">${svg}</svg>${extra}`;
  return div;
}

// ------------------------------------------------------- stress on a 2D block
export function compliance2D(s11, s22, s12, E, nu) {
  // plane stress: strains (and engineering shear) from the stresses; E in MPa
  return {e11: (s11 - nu * s22) / E, e22: (s22 - nu * s11) / E, g12: 2 * (1 + nu) * s12 / E, e33: -nu * (s11 + s22) / E};
}

export function block2D({s11 = 0, s22 = 0, s12 = 0, E = 200000, nu = 0.3, mag = 100, smax = 300, size = 380} = {}) {
  // the block's faces carry the tractions t = sigma n: normal arrows from the
  // middle of each face, shear arrows along each face just outside it
  const W = size, H = size, c = W / 2, h = 0.2 * W, X = x => c + x, Y = y => c - y;
  const nlen = v => (0.06 + 0.16 * Math.min(1, Math.abs(v) / smax)) * W;      // normal arrows
  const slen = v => (0.25 + 0.95 * Math.min(1, Math.abs(v) / smax)) * h;      // shear arrows, along the face
  const st = compliance2D(s11, s22, s12, E, nu);
  const corners = [[-h, -h], [h, -h], [h, h], [-h, h]];
  const def = corners.map(([x, y]) => [x + mag * (st.e11 * x + 0.5 * st.g12 * y), y + mag * (0.5 * st.g12 * x + st.e22 * y)]);
  let s = `<polygon points="${def.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}" fill="#fde9d2" stroke="#e67e22" stroke-width="1.4"/>`;
  s += `<polygon points="${corners.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")}" fill="none" stroke="${INK}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  const gap = 10, sgap = 9;
  for (const [n, v, col, nm] of [[[1, 0], s11, COMP.s11, "σ₀₀"], [[-1, 0], s11, COMP.s11, "σ₀₀"], [[0, 1], s22, COMP.s22, "σ₁₁"], [[0, -1], s22, COMP.s22, "σ₁₁"]]) {
    if (Math.abs(v) < 1e-9) continue;
    const L = nlen(v), base = [n[0] * (h + gap), n[1] * (h + gap)], tip = [base[0] + n[0] * L, base[1] + n[1] * L];
    s += v > 0 ? arrowSVG(X(base[0]), Y(base[1]), X(tip[0]), Y(tip[1]), col, 2.8, 11) : arrowSVG(X(tip[0]), Y(tip[1]), X(base[0]), Y(base[1]), col, 2.8, 11);
    if (n[0] + n[1] > 0) {
      const lp = [tip[0] + n[0] * 8, tip[1] + n[1] * 8];
      s += label(X(lp[0]) + (n[0] ? 4 : 0), Y(lp[1]) - (n[1] ? 10 : 0), nm, col, n[0] ? "start" : "middle", 16);
    }
  }
  if (Math.abs(s12) > 1e-9) {
    const L = slen(s12), sg = Math.sign(s12);
    // +x1 face: t2 = s21 along +x2 (for s12 > 0); -x1 face: along -x2; +x2 face: t1 = s12 along +x1; -x2 face: along -x1
    for (const [face, dir, nm, lab] of [[[1, 0], [0, 1], "σ₁₀", true], [[-1, 0], [0, -1], "σ₁₀", false], [[0, 1], [1, 0], "σ₀₁", true], [[0, -1], [-1, 0], "σ₀₁", false]]) {
      const mid = [face[0] * (h + sgap), face[1] * (h + sgap)], d = [dir[0] * sg * L / 2, dir[1] * sg * L / 2];
      s += arrowSVG(X(mid[0] - d[0]), Y(mid[1] - d[1]), X(mid[0] + d[0]), Y(mid[1] + d[1]), COMP.s12, 2.8, 11);
      if (lab) {
        const end = [mid[0] + d[0], mid[1] + d[1]];
        s += face[0] ? label(X(end[0]) + 9, Y(end[1]) + (sg > 0 ? -8 : 8), nm, COMP.s12, "start", 16)
                     : label(X(end[0]) + (sg > 0 ? 10 : -10), Y(end[1]) - 13, nm, COMP.s12, sg > 0 ? "start" : "end", 16);
      }
    }
  }
  s += arrowSVG(16, H - 16, 56, H - 16, INK, 1.2, 7) + arrowSVG(16, H - 16, 16, H - 56, INK, 1.2, 7);
  s += label(62, H - 16, "x₀", INK, "start", 13) + label(16, H - 66, "x₁", INK, "middle", 13);
  s += label(W - 8, H - 12, `deformation × ${mag}`, "#e67e22", "end", 11);
  const div = wrap(s, W, H);
  div.strain = st;
  return div;
}

// ------------------------------------------------------- stress on a 3D block
function rot(yaw, pitch) {
  // world (x1, x2, x3) -> screen (right, up, toward the viewer), x3 is up
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return v => {
    const a = cy * v[0] - sy * v[1], b = sy * v[0] + cy * v[1], z = v[2];   // about x3; b points away from the viewer
    // tilt the camera down by `pitch`: farther points rise on the page, and the top faces the viewer
    return [a, cp * z + sp * b, sp * z - cp * b];
  };
}

export function block3D({S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], yaw = -2.3, pitch = 0.5, E = 200000, nu = 0.3, mag = 100,
                         smax = 300, size = 420, hidden = false} = {}) {
  const W = size, H = size, R = rot(yaw, pitch), c = [W / 2, H / 2], k = 0.17 * W;
  const P = v => { const r = R(v); return [c[0] + k * r[0], c[1] - k * r[1], r[2]]; };
  const names = [["s11", "s12", "s13"], ["s12", "s22", "s23"], ["s13", "s23", "s33"]];
  const tr = S[0][0] + S[1][1] + S[2][2];
  const eps = S.map((row, i) => row.map((v, j) => (1 + nu) / E * v - (i === j ? nu / E * tr : 0)));
  const V = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) V.push([x, y, z]);
  const faces = [];
  for (let j = 0; j < 3; j++) for (const sg of [-1, 1]) {
    const n = [0, 0, 0]; n[j] = sg;
    const vs = V.filter(v => v[j] === sg);
    const u = [0, 1, 2].filter(m => m !== j);
    vs.sort((p, q) => Math.atan2(p[u[1]], p[u[0]]) - Math.atan2(q[u[1]], q[u[0]]));
    const facing = R(n)[2] > 0;
    faces.push({j, sg, n, vs, facing});
  }
  const defP = v => {
    const d = [0, 1, 2].map(i => v[i] + mag * (eps[i][0] * v[0] + eps[i][1] * v[1] + eps[i][2] * v[2]));
    return P(d);
  };
  const ptsStr = vs => vs.map(v => P(v)).map(q => `${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(" ");
  let s = "";
  const arrowsOn = (f, faint) => {
    let a = "";
    const base = f.n.map(v => 1.0 * v);
    for (let i = 0; i < 3; i++) {
      const sij = S[i][f.j] * f.sg;           // t_i = s_ij n_j
      if (Math.abs(sij) < 1e-9) continue;
      const L = 0.3 + 0.75 * Math.min(1, Math.abs(S[i][f.j]) / smax), dir = [0, 0, 0];
      dir[i] = Math.sign(sij);
      let p0, p1;
      if (i === f.j) {                       // normal: outward for tension, inward from outside for compression
        const out = f.n.map(v => v * 0.08);
        if (S[i][f.j] < 0) { p0 = base.map((v, m) => v + out[m] + f.n[m] * L); p1 = base.map((v, m) => v + out[m]); }
        else { p0 = base.map((v, m) => v + out[m]); p1 = p0.map((v, m) => v + f.n[m] * L); }
      } else {                               // shear: along the face, starting at its centre
        const off = f.n.map(v => v * 0.03);
        p0 = base.map((v, m) => v + off[m]);
        p1 = base.map((v, m) => v + off[m] + dir[m] * L);
      }
      const A = P(p0), B = P(p1), col = COMP[names[i][f.j]];
      a += `<g opacity="${faint ? 0.3 : 1}">${arrowSVG(A[0], A[1], B[0], B[1], col, 2.4)}</g>`;
      if (!faint) {
        const lab = `σ${SUB[i + 1]}${SUB[f.j + 1]}`, ext = p1.map((v, m) => v + (p1[m] - p0[m]) / Math.max(L, 1e-9) * 0.2);
        const Tp = P(ext);
        a += label(Tp[0], Tp[1], lab, col, "middle", 14);
      }
    }
    return a;
  };
  const back = faces.filter(f => !f.facing), front = faces.filter(f => f.facing);
  if (hidden) for (const f of back) s += arrowsOn(f, true);
  for (const f of back) s += `<polygon points="${ptsStr(f.vs)}" fill="#eef2f6" stroke="#9aa5b1" stroke-width="0.8" stroke-dasharray="3 3"/>`;
  // the deformed block, as a wire frame
  const edges = [];
  for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) {
    const dif = [0, 1, 2].filter(m => V[a][m] !== V[b][m]);
    if (dif.length === 1) edges.push([a, b]);
  }
  for (const [a, b] of edges) {
    const A = defP(V[a]), B = defP(V[b]);
    s += `<line x1="${A[0].toFixed(1)}" y1="${A[1].toFixed(1)}" x2="${B[0].toFixed(1)}" y2="${B[1].toFixed(1)}" stroke="#e67e22" stroke-width="1.5"/>`;
  }
  front.sort((f, g) => R(f.n)[2] - R(g.n)[2]);
  for (const f of front) s += `<polygon points="${ptsStr(f.vs)}" fill="#dfe8f1" fill-opacity="0.82" stroke="${INK}" stroke-width="1.3"/>`;
  for (const f of front) s += arrowsOn(f, false);
  // axes
  const O = [W * 0.12, H * 0.86];
  for (const [i, nm] of [[0, "x₀"], [1, "x₁"], [2, "x₂"]]) {
    const e = [0, 0, 0]; e[i] = 1;
    const r = R(e), B = [O[0] + 30 * r[0], O[1] - 30 * r[1]];
    s += arrowSVG(O[0], O[1], B[0], B[1], INK, 1.2, 7) + label(B[0] + 8 * r[0], B[1] - 8 * r[1], nm, INK, "middle", 12);
  }
  s += label(W - 8, H - 12, `deformation × ${mag}`, "#e67e22", "end", 11);
  const div = wrap(s, W, H);
  div.strain = eps;
  return div;
}

// --------------------------------------- the displacement gradient: strain + rotation
export function gradientSplit({H = [[0, 0], [0, 0]], size = 210} = {}) {
  // three squares: x + H x, x + eps x, x + omega x (the linearized rotation)
  const eps = [[H[0][0], 0.5 * (H[0][1] + H[1][0])], [0.5 * (H[0][1] + H[1][0]), H[1][1]]];
  const om = [[0, 0.5 * (H[0][1] - H[1][0])], [-0.5 * (H[0][1] - H[1][0]), 0]];
  const panel = (M, title, col, ox) => {
    const h = 0.26 * size, c = [ox + size / 2, size / 2 + 6], X = x => c[0] + x, Y = y => c[1] - y;
    const sq = [[-h, -h], [h, -h], [h, h], [-h, h]];
    const d = sq.map(([x, y]) => [x + M[0][0] * x + M[0][1] * y, y + M[1][0] * x + M[1][1] * y]);
    let s = `<polygon points="${d.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}" fill="${col}" fill-opacity="0.18" stroke="${col}" stroke-width="1.8"/>`;
    s += `<polygon points="${sq.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")}" fill="none" stroke="${INK}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
    // the material line along x1 and along x2, and the angle each turns through
    for (const [v, nm] of [[[h, 0], "dx₀"], [[0, h], "dx₁"]]) {
      const w = [v[0] + M[0][0] * v[0] + M[0][1] * v[1], v[1] + M[1][0] * v[0] + M[1][1] * v[1]];
      s += arrowSVG(X(0), Y(0), X(w[0]), Y(w[1]), col, 1.8, 7);
    }
    s += label(ox + size / 2, 14, title, INK, "middle", 12.5);
    return s;
  };
  const W = 3 * size;
  const s = panel(H, "displacement gradient ∂uᵢ/∂xⱼ", "#e67e22", 0) + panel(eps, "strain εᵢⱼ (symmetric part)", "#c0392b", size)
          + panel(om, "rotation ωᵢⱼ (skew part)", "#1f6feb", 2 * size);
  const div = wrap(s, W, size + 14);
  div.eps = eps; div.om = om;
  return div;
}

// --------------------------------------------------- the elasticity matrix
export function lamina({E1, E2, G12, nu12, theta}) {
  // plane-stress stiffness of an orthotropic sheet whose fibers lie at angle
  // theta from x1: the 3 x 3 matrix C (Voigt, engineering shear) in x1-x2
  // axes, from the material-axis compliance by rotating the fourth-order tensor
  const nu21 = nu12 * E2 / E1, den = 1 - nu12 * nu21;
  const Q = [[E1 / den, nu12 * E2 / den, 0], [nu12 * E2 / den, E2 / den, 0], [0, 0, G12]];
  const vo = [[0, 0], [1, 1], [0, 1]];          // Voigt index -> (i, j)
  const c = {};                                  // c_ijkl in material axes
  const idx = (i, j) => (i === j ? i : 2);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 2; k++) for (let l = 0; l < 2; l++)
    c[`${i}${j}${k}${l}`] = Q[idx(i, j)][idx(k, l)];
  const ct = Math.cos(theta), st = Math.sin(theta), Rm = [[ct, -st], [st, ct]];
  const cr = (i, j, k, l) => {
    let s = 0;
    for (let p = 0; p < 2; p++) for (let q = 0; q < 2; q++) for (let r = 0; r < 2; r++) for (let w = 0; w < 2; w++)
      s += Rm[i][p] * Rm[j][q] * Rm[k][r] * Rm[l][w] * c[`${p}${q}${r}${w}`];
    return s;
  };
  return vo.map(([i, j]) => vo.map(([k, l]) => cr(i, j, k, l)));
}

export function inv3(M) {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
          [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}

export function polarModulus(C, {theta = 0, size = 280, Eref = null, isotropic = false} = {}) {
  // E(phi) = 1 / (m^T S m), m = (cos^2, sin^2, cos sin): the stiffness of a
  // strip pulled along the direction phi
  const S = inv3(C), n = 180, pts = [];
  for (let k = 0; k <= n; k++) {
    const ph = 2 * Math.PI * k / n, cc = Math.cos(ph), ss = Math.sin(ph), m = [cc * cc, ss * ss, cc * ss];
    const v = m.reduce((acc, mi, r) => acc + mi * (S[r][0] * m[0] + S[r][1] * m[1] + S[r][2] * m[2]), 0);
    pts.push([ph, 1 / v]);
  }
  const Emax = Eref || Math.max(...pts.map(p => p[1])), c = size / 2, R = 0.42 * size;
  const X = (ph, E) => c + R * E / Emax * Math.cos(ph), Y = (ph, E) => c - R * E / Emax * Math.sin(ph);
  let s = "";
  for (const f of [0.25, 0.5, 0.75, 1]) s += `<circle cx="${c}" cy="${c}" r="${R * f}" fill="none" stroke="#e5e5e5"/>`;
  s += `<line x1="${c - R - 8}" y1="${c}" x2="${c + R + 8}" y2="${c}" stroke="#ccc"/><line x1="${c}" y1="${c - R - 8}" x2="${c}" y2="${c + R + 8}" stroke="#ccc"/>`;
  s += `<polygon points="${pts.map(([ph, E]) => `${X(ph, E).toFixed(1)},${Y(ph, E).toFixed(1)}`).join(" ")}" fill="#c0392b" fill-opacity="0.12" stroke="#c0392b" stroke-width="2"/>`;
  const fx = Math.cos(theta), fy = Math.sin(theta);
  if (!isotropic) s += `<line x1="${c - R * fx}" y1="${c + R * fy}" x2="${c + R * fx}" y2="${c - R * fy}" stroke="#555" stroke-width="1" stroke-dasharray="5 4"/>`;
  if (!isotropic) s += label(c + (R + 10) * fx, c - (R + 10) * fy, "fibers", "#555", fx >= 0 ? "start" : "end", 11);
  s += label(c + R + 4, c + 12, "x₀", INK, "end", 12) + label(c + 10, c - R - 2, "x₁", INK, "start", 12);
  s += label(c, size - 8, `largest: ${(Emax).toFixed(Emax < 10 ? 2 : 0)} GPa (outer circle)`, "#555", "middle", 11);
  return wrap(s, size, size);
}

export function response({C, sig = [0, 0, 0], size = 260, mag = 1} = {}) {
  // the square under a uniform stress: eps = C^-1 sigma, drawn without rotation
  const S = inv3(C), e = [0, 1, 2].map(r => S[r][0] * sig[0] + S[r][1] * sig[1] + S[r][2] * sig[2]);
  const h = 0.25 * size, c = size / 2, X = x => c + x, Y = y => c - y;
  const sq = [[-h, -h], [h, -h], [h, h], [-h, h]];
  const d = sq.map(([x, y]) => [x + mag * (e[0] * x + 0.5 * e[2] * y), y + mag * (0.5 * e[2] * x + e[1] * y)]);
  let s = `<polygon points="${d.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}" fill="#fde9d2" stroke="#e67e22" stroke-width="1.6"/>`;
  s += `<polygon points="${sq.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")}" fill="none" stroke="${INK}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
  const L = 0.17 * size, sm = Math.max(Math.abs(sig[0]), Math.abs(sig[1]), Math.abs(sig[2]), 1e-12);
  for (const [n, v, col] of [[[1, 0], sig[0], COMP.s11], [[-1, 0], sig[0], COMP.s11], [[0, 1], sig[1], COMP.s22], [[0, -1], sig[1], COMP.s22]]) {
    if (Math.abs(v) < 1e-12) continue;
    const l = L * Math.abs(v) / sm, b = [n[0] * (h + 6), n[1] * (h + 6)], t = [b[0] + n[0] * l, b[1] + n[1] * l];
    s += v > 0 ? arrowSVG(X(b[0]), Y(b[1]), X(t[0]), Y(t[1]), col, 2.2) : arrowSVG(X(t[0]), Y(t[1]), X(b[0]), Y(b[1]), col, 2.2);
  }
  if (Math.abs(sig[2]) > 1e-12) {
    const l = L * Math.abs(sig[2]) / sm, sg = Math.sign(sig[2]), o = h + 12;
    for (const [m, dvec] of [[[o, 0], [0, 1]], [[-o, 0], [0, -1]], [[0, o], [1, 0]], [[0, -o], [-1, 0]]])
      s += arrowSVG(X(m[0] - sg * dvec[0] * l / 2), Y(m[1] - sg * dvec[1] * l / 2), X(m[0] + sg * dvec[0] * l / 2), Y(m[1] + sg * dvec[1] * l / 2), COMP.s12, 2.2);
  }
  const div = wrap(s, size, size);
  div.strain = e;
  return div;
}
