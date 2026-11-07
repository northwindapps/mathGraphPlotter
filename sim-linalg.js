/* Linear algebra: matrices as transformations, eigenvectors, and PCA.
 * A = [[a, b], [c, d]] maps î = (1,0) → (a, c) and ĵ = (0,1) → (b, d).
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const TAU = Math.PI * 2;
  const COL = { i: '#e5484d', j: '#2fa36b', v: '#f5c451', Av: '#e09b14', grid: 'rgba(91,155,255,0.55)', eig: '#d6409f', pos: '61,139,253', neg: '224,155,20' };
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const fmt = (v, d = 2) => (Math.abs(v) < 0.005 ? '0' : (+v.toFixed(d)).toString());
  function rng(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); };

  // eigen-decomposition of a real 2×2 matrix
  function eig2(a, b, c, d) {
    const tr = a + d, det = a * d - b * c, disc = (tr * tr) / 4 - det;
    if (disc < -1e-12) return { complex: true, re: tr / 2, im: Math.sqrt(-disc), det, tr };
    const s = Math.sqrt(Math.max(0, disc)), l1 = tr / 2 + s, l2 = tr / 2 - s;
    const unit = (v) => { const n = Math.hypot(...v) || 1; return [v[0] / n, v[1] / n]; };
    // (A − λI)v = 0: use whichever row of A − λI is non-zero
    const vec = (l) => (Math.abs(b) > 1e-9 ? unit([b, l - a]) : Math.abs(c) > 1e-9 ? unit([l - d, c]) : null);
    let v1 = vec(l1), v2 = vec(l2);
    if (!v1) { v1 = a >= d ? [1, 0] : [0, 1]; v2 = a >= d ? [0, 1] : [1, 0]; } // diagonal matrix: the axes
    return { complex: false, l1, l2, v1, v2, det, tr, repeated: s < 1e-9 };
  }

  const PRESETS = {
    identity: { label: 'Identity', A: [1, 0, 0, 1] },
    rotate: { label: 'Rotate 45°', A: [Math.SQRT1_2, -Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2] },
    scale: { label: 'Stretch', A: [2, 0, 0, 0.5] },
    shear: { label: 'Shear', A: [1, 1, 0, 1] },
    reflect: { label: 'Reflect', A: [-1, 0, 0, 1] },
    project: { label: 'Squash (det 0)', A: [1, 1, 0.5, 0.5] },
    symmetric: { label: 'Symmetric', A: [2, 1, 1, 2] },
  };
  const F_SHAPE = [[0.3, 0.2], [0.55, 0.2], [0.55, 0.9], [1.1, 0.9], [1.1, 1.15], [0.55, 1.15], [0.55, 1.5], [1.3, 1.5], [1.3, 1.75], [0.3, 1.75]];

  const MODES = [
    { value: 'transform', label: 'Matrix = transformation', hint: 'Drag the red î and green ĵ arrow tips to change the matrix' },
    { value: 'eigen', label: 'Eigenvectors', hint: 'Drag the yellow vector v · find where Av points the same way' },
    { value: 'pca', label: 'PCA', hint: 'Drag the end of the projection line to rotate it · click empty space to add points' },
  ];

  const sim = {
    id: 'linalg',
    get hint() { return MODES.find((m) => m.value === this.o.mode).hint; },
    o: { mode: 'transform', A: [1, 1, 0, 1], t: 1, animating: false, v: [1, 0.3], power: null, pca: { sx: 2.2, sy: 0.7, rot: 30, theta: 0, collapse: 0, collapsing: false } },
    W: 0, H: 0, hudTick: 0,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.pts) this.makeCloud();
      this.buildPanel();
    },
    unmount() { this.drag = null; },
    resize(W, H) { this.W = W; this.H = H; },

    // ---------------------------------------------------------------- matrix helpers
    M() { // matrix currently shown (animated from the identity)
      const [a, b, c, d] = this.o.A, t = this.o.mode === 'transform' ? this.o.t : 1;
      return [1 + (a - 1) * t, b * t, c * t, 1 + (d - 1) * t];
    },
    apply(M, [x, y]) { return [M[0] * x + M[1] * y, M[2] * x + M[3] * y]; },
    setA(A) { this.o.A = A.map((v) => +v.toFixed(3)); this.o.power = null; this.syncSliders(); this.refreshTex(); },

    // ---------------------------------------------------------------- PCA data
    makeCloud(seed = 4) {
      const p = this.o.pca, r = rng(seed), a = (p.rot * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
      this.pts = Array.from({ length: 160 }, () => { const u = gauss(r) * p.sx * 0.5, w = gauss(r) * p.sy * 0.5; return [ca * u - sa * w + 0.3, sa * u + ca * w - 0.2]; });
      p.collapse = 0; p.collapsing = false;
    },
    pcaStats() {
      const P = this.pts, n = P.length;
      let mx = 0, my = 0;
      for (const [x, y] of P) { mx += x; my += y; }
      mx /= n; my /= n;
      let sxx = 0, sxy = 0, syy = 0;
      for (const [x, y] of P) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); syy += (y - my) ** 2; }
      const C = [sxx / (n - 1), sxy / (n - 1), sxy / (n - 1), syy / (n - 1)];
      return { mean: [mx, my], C, e: eig2(...C) };
    },
    varAlong(st, th) { const u = [Math.cos(th), Math.sin(th)], C = st.C; return u[0] * (C[0] * u[0] + C[1] * u[1]) + u[1] * (C[2] * u[0] + C[3] * u[1]); },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({ options: MODES, value: o.mode, onChange: (v) => { o.mode = v; o.power = null; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; } }));
      p.append(s0);
      if (o.mode === 'pca') { this.panelPCA(p); return; }

      const s1 = UI.section('The matrix');
      this.texEl = el('div', 'eq');
      s1.append(this.texEl);
      s1.append(UI.segmented({ options: Object.entries(PRESETS).map(([value, q]) => ({ value, label: q.label })), value: null, onChange: (v) => { this.setA(PRESETS[v].A); if (o.mode === 'transform') { o.t = 0; o.animating = true; } } }));
      const LABELS = ['a: where î lands, x', 'b: where ĵ lands, x', 'c: where î lands, y', 'd: where ĵ lands, y'];
      this.sliders = LABELS.map((label, k) => UI.slider({
        label, min: -3, max: 3, step: 0.01, value: o.A[k], format: (v) => v.toFixed(2), onInput: (v) => { o.A[k] = v; o.power = null; this.refreshTex(); },
      }));
      // order sliders as a, c (column 1) then b, d (column 2)
      s1.append(this.sliders[0], this.sliders[2], this.sliders[1], this.sliders[3]);
      this.refreshTex();
      p.append(s1);

      if (o.mode === 'transform') {
        const s2 = UI.section('Watch it move');
        const row = el('div', 'controls-row');
        row.append(UI.button('▶ Animate from the identity', () => { o.t = 0; o.animating = true; }));
        s2.append(row, UI.slider({ label: 'Blend I → A', min: 0, max: 1, step: 0.01, value: o.t, format: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.t = v; o.animating = false; } }));
        s2.append(UI.termCard({ color: COL.i, tex: String.raw`A = \begin{pmatrix} | & | \\ \hat\imath' & \hat\jmath' \\ | & | \end{pmatrix}`, title: 'The columns tell the whole story',
          body: 'A matrix is fully described by where it sends the two basis arrows: column 1 is where <b style="color:#e5484d">î</b> lands, column 2 where <b style="color:#2fa36b">ĵ</b> lands. ' +
            'Every other point follows, because Av = x·î′ + y·ĵ′. Grid lines stay straight, parallel and evenly spaced: that is what <i>linear</i> means.' }));
        s2.append(UI.termCard({ color: '#3d8bfd', tex: String.raw`\det A = ad - bc`, title: 'The determinant is an area',
          body: 'The unit square becomes a parallelogram whose area is |det A|. det = 0 squashes the plane onto a line (information is lost, so no inverse exists). ' +
            'A <b>negative</b> det means the plane was flipped over: watch the "F" become mirror-imaged and the square turn orange.' }));
        s2.append(UI.termCard({ color: '#9aa3b2', tex: String.raw`(AB)v = A(Bv)`, title: 'Why matrices multiply this way',
          body: 'Multiplying matrices = doing one transformation after the other. That is why AB ≠ BA in general: rotating then shearing is not the same as shearing then rotating. ' +
            'Every layer of a neural network starts with exactly this kind of linear map.' }));
        p.append(s2);
      } else {
        const s2 = UI.section('Eigenvectors');
        const row = el('div', 'controls-row');
        row.append(UI.button('▶ Apply A again and again', () => { const n = Math.hypot(...o.v) || 1; o.power = { pts: [[o.v[0] / n, o.v[1] / n]], k: 0, t: 0 }; }));
        s2.append(row);
        s2.append(UI.termCard({ color: COL.eig, tex: String.raw`A\mathbf{v} = \lambda\mathbf{v}`, title: 'Directions that only stretch',
          body: 'Most vectors get turned by A. An <b>eigenvector</b> keeps its direction; it is only scaled by its <b>eigenvalue</b> λ (negative λ flips it around). ' +
            'Drag the yellow v around the circle: when Av lines up with v, you have found one. The pink dashed lines are the eigen-directions.' }));
        s2.append(UI.termCard({ color: '#3d8bfd', tex: String.raw`\det(A - \lambda I) = 0`, title: 'How to compute them',
          body: 'Av = λv means (A − λI)v = 0 has a non-zero solution, so A − λI must squash the plane: its determinant is 0. For 2×2 that is a quadratic, λ² − (a+d)λ + (ad − bc) = 0. ' +
            'If its roots are complex (try <b>Rotate 45°</b>), no real direction survives: the matrix is a rotation, and the complex λ = r·e<sup>iθ</sup> encodes the angle, just like the Fourier helix.' }));
        s2.append(UI.termCard({ color: COL.v, tex: String.raw`v_{k+1} = \frac{A v_k}{\|A v_k\|}`, title: 'Power iteration (how Google ranked the web)',
          body: 'Keep applying A and re-normalizing: the component along the eigenvector with the largest |λ| grows fastest and takes over. ' +
            'PageRank is this exact loop on a matrix with one row per web page; the dominant eigenvector is the importance score of every page.' }));
        p.append(s2);
      }
    },

    syncSliders() {
      if (!this.sliders) return;
      this.sliders.forEach((s, k) => s.set(this.o.A[k]));
    },
    refreshTex() {
      if (!this.texEl) return;
      const [a, b, c, d] = this.o.A, e = eig2(a, b, c, d);
      let eigTex = e.complex
        ? String.raw`\lambda = ${fmt(e.re)} \pm ${fmt(e.im)}\,i \;\;(\text{rotation})`
        : String.raw`\lambda_1 = ${fmt(e.l1)},\ \lambda_2 = ${fmt(e.l2)}`;
      UI.tex(this.texEl, String.raw`\begin{gathered} A = \begin{pmatrix} \textcolor{${COL.i}}{${fmt(a)}} & \textcolor{${COL.j}}{${fmt(b)}} \\ \textcolor{${COL.i}}{${fmt(c)}} & \textcolor{${COL.j}}{${fmt(d)}} \end{pmatrix}
        \qquad \det A = ${fmt(e.det)} \\[4pt] ${eigTex} \end{gathered}`, true);
    },

    panelPCA(p) {
      const q = this.o.pca;
      const s1 = UI.section('Principal component analysis');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\max_{\|w\|=1} \operatorname{Var}(w^\top x) = \max_{\|w\|=1} w^\top \Sigma\, w \;\;\Longrightarrow\;\; \Sigma w = \lambda w`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Which single direction keeps the most information about this cloud? Project every point onto a line (the small dots) and measure how spread out the shadows are. ' +
        'Rotate the line: the spread peaks exactly along the <b>top eigenvector of the covariance matrix Σ</b>, and the peak value <i>is</i> its eigenvalue. ' +
        'It is a Lagrange problem (maximize wᵀΣw subject to ‖w‖ = 1), and ∇f = λ∇g becomes Σw = λw.';
      s1.append(eq, n1);
      this.thetaSlider = UI.slider({ label: 'Projection angle θ', min: 0, max: 180, step: 0.5, value: q.theta, format: (v) => `${v.toFixed(1)}°`, onInput: (v) => { q.theta = v; } });
      const row = el('div', 'controls-row');
      row.append(UI.button('Snap to PC1', () => { const st = this.pcaStats(); q.theta = (((Math.atan2(st.e.v1[1], st.e.v1[0]) * 180) / Math.PI) + 360) % 180; this.thetaSlider.set(q.theta); }),
        UI.button('⇣ Collapse 2D → 1D', () => { q.collapsing = !q.collapsing; }));
      s1.append(this.thetaSlider, row);
      const s2 = UI.section('The data');
      const remake = () => this.makeCloud(4);
      s2.append(UI.slider({ label: 'Spread along main axis', min: 0.3, max: 3, step: 0.05, value: q.sx, format: (v) => v.toFixed(2), onInput: (v) => { q.sx = v; remake(); } }));
      s2.append(UI.slider({ label: 'Spread across', min: 0.1, max: 3, step: 0.05, value: q.sy, format: (v) => v.toFixed(2), onInput: (v) => { q.sy = v; remake(); } }));
      s2.append(UI.slider({ label: 'Tilt', min: -90, max: 90, step: 1, value: q.rot, format: (v) => `${v}°`, onInput: (v) => { q.rot = v; remake(); } }));
      s2.append(UI.button('🎲 New random sample', () => this.makeCloud(Math.floor(Math.random() * 1e6))));
      s2.append(UI.termCard({ color: COL.v, tex: String.raw`\frac{\lambda_1}{\lambda_1 + \lambda_2}`, title: 'Variance explained',
        body: 'How much of the total spread survives if you keep only PC1. Make the cloud long and thin and it approaches 100%: the data was really one-dimensional. ' +
          'Make it round and it drops to 50%: no direction is special. Real uses: compressing face images (eigenfaces), visualizing high-dimensional data, de-noising.' }));
      p.append(s1, s2);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o;
      if (o.mode === 'transform' && o.animating) { o.t = Math.min(1, o.t + 1 / 90); if (o.t >= 1) o.animating = false; }
      if (o.mode === 'eigen' && o.power) {
        const pw = o.power;
        if (++pw.t % 24 === 0 && pw.k < 25) {
          const last = pw.pts[pw.pts.length - 1], Av = this.apply(o.A, last), n = Math.hypot(...Av);
          if (n > 1e-9) pw.pts.push([Av[0] / n, Av[1] / n]);
          pw.k++;
        }
      }
      if (o.mode === 'pca') {
        const q = o.pca;
        q.collapse = clamp(q.collapse + (q.collapsing ? 1 : -1) / 50, 0, 1);
        this.drawPCA(ctx, W, H);
      } else this.drawPlane(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    view(W, H, R = 4.2) {
      const side = Math.min(W - 40, H - 120), S = side / (2 * R), cx = W / 2, cy = 100 + side / 2;
      return { S, cx, cy, R, sx: (x) => cx + x * S, sy: (y) => cy - y * S, wx: (px) => (px - cx) / S, wy: (py) => (cy - py) / S };
    },

    arrow(ctx, x0, y0, x1, y1, color, width = 3) {
      const a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0), h = Math.min(12, L * 0.4);
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - h * 0.6 * Math.cos(a), y1 - h * 0.6 * Math.sin(a)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4)); ctx.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
    },

    drawPlane(ctx, W, H) {
      const o = this.o, V = this.view(W, H), M = this.M(), R = V.R;
      this.V = V;
      const P = (pt) => [V.sx(pt[0]), V.sy(pt[1])];
      ctx.save(); ctx.beginPath(); ctx.rect(V.cx - R * V.S, V.cy - R * V.S, 2 * R * V.S, 2 * R * V.S); ctx.clip();
      // original grid (faint)
      ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1; ctx.beginPath();
      for (let k = -R; k <= R; k++) { ctx.moveTo(V.sx(k), V.sy(-R)); ctx.lineTo(V.sx(k), V.sy(R)); ctx.moveTo(V.sx(-R), V.sy(k)); ctx.lineTo(V.sx(R), V.sy(k)); }
      ctx.stroke();
      if (o.mode === 'transform') {
        // transformed grid
        const ext = 12;
        ctx.strokeStyle = COL.grid; ctx.lineWidth = 1; ctx.beginPath();
        for (let k = -ext; k <= ext; k++) {
          let p0 = P(this.apply(M, [k, -ext])), p1 = P(this.apply(M, [k, ext])); ctx.moveTo(...p0); ctx.lineTo(...p1);
          p0 = P(this.apply(M, [-ext, k])); p1 = P(this.apply(M, [ext, k])); ctx.moveTo(...p0); ctx.lineTo(...p1);
        }
        ctx.stroke();
        // unit square → parallelogram
        const det = M[0] * M[3] - M[1] * M[2], sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map((q) => P(this.apply(M, q)));
        ctx.fillStyle = `rgba(${det >= 0 ? COL.pos : COL.neg},0.35)`;
        ctx.beginPath(); sq.forEach((q, k) => (k ? ctx.lineTo(...q) : ctx.moveTo(...q))); ctx.closePath(); ctx.fill();
        const c = P(this.apply(M, [0.5, 0.5]));
        ctx.fillStyle = '#e8ebf1'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`area ${fmt(Math.abs(det))}`, c[0], c[1]);
        // the F shape (shows flips)
        ctx.fillStyle = 'rgba(245,196,81,0.85)';
        ctx.beginPath(); F_SHAPE.map((q) => P(this.apply(M, [q[0] - 2.2, q[1] - 0.2]))).forEach((q, k) => (k ? ctx.lineTo(...q) : ctx.moveTo(...q))); ctx.closePath(); ctx.fill();
      } else {
        this.drawEigen(ctx, V, M, P);
      }
      ctx.restore();
      // axes and basis vectors
      ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(V.sx(-R), V.sy(0)); ctx.lineTo(V.sx(R), V.sy(0)); ctx.moveTo(V.sx(0), V.sy(-R)); ctx.lineTo(V.sx(0), V.sy(R)); ctx.stroke();
      const ip = P([M[0], M[2]]), jp = P([M[1], M[3]]), o0 = P([0, 0]);
      this.arrow(ctx, o0[0], o0[1], ip[0], ip[1], COL.i, 3.5);
      this.arrow(ctx, o0[0], o0[1], jp[0], jp[1], COL.j, 3.5);
      ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillStyle = COL.i; ctx.fillText('î′', ip[0] + 6, ip[1] - 4);
      ctx.fillStyle = COL.j; ctx.fillText('ĵ′', jp[0] + 6, jp[1] - 4);
      for (const q of [ip, jp]) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, TAU); ctx.stroke(); }
      this.handles = { i: ip, j: jp };
    },

    drawEigen(ctx, V, M, P) {
      const o = this.o, e = eig2(...M);
      // a ring of unit vectors and where A sends them
      for (let k = 0; k < 32; k++) {
        const a = (k / 32) * TAU, u = [Math.cos(a), Math.sin(a)], Au = this.apply(M, u), p0 = P(u), p1 = P(Au);
        ctx.strokeStyle = `hsla(${(k / 32) * 360},70%,62%,0.35)`; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(...P([0, 0])); ctx.lineTo(...p0); ctx.stroke();
        ctx.strokeStyle = `hsla(${(k / 32) * 360},70%,62%,0.85)`; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
        ctx.fillStyle = `hsla(${(k / 32) * 360},70%,62%,1)`; ctx.beginPath(); ctx.arc(p1[0], p1[1], 2.5, 0, TAU); ctx.fill();
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(...P([0, 0]), V.S, 0, TAU); ctx.stroke();
      // eigen-directions
      if (!e.complex) {
        for (const [v, l] of [[e.v1, e.l1], [e.v2, e.l2]]) {
          ctx.strokeStyle = COL.eig; ctx.lineWidth = 2; ctx.setLineDash([7, 6]);
          ctx.beginPath(); ctx.moveTo(...P([-v[0] * 9, -v[1] * 9])); ctx.lineTo(...P([v[0] * 9, v[1] * 9])); ctx.stroke(); ctx.setLineDash([]);
          const lp = P([v[0] * 3.4, v[1] * 3.4]);
          ctx.fillStyle = COL.eig; ctx.font = '600 13px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(`λ = ${fmt(l)}`, lp[0], lp[1] - 12);
          if (e.repeated) break;
        }
      }
      // the user's vector v and Av
      const v = o.v, Av = this.apply(M, v), o0 = P([0, 0]), pv = P(v), pAv = P(Av);
      const cross = v[0] * Av[1] - v[1] * Av[0], aligned = Math.hypot(...Av) > 1e-6 && Math.abs(cross) / (Math.hypot(...v) * Math.hypot(...Av)) < 0.035;
      if (aligned) { ctx.save(); ctx.shadowColor = COL.v; ctx.shadowBlur = 18; }
      this.arrow(ctx, o0[0], o0[1], pAv[0], pAv[1], COL.Av, 3.5);
      this.arrow(ctx, o0[0], o0[1], pv[0], pv[1], COL.v, 3.5);
      if (aligned) ctx.restore();
      ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillStyle = COL.v; ctx.fillText('v', pv[0] + 8, pv[1] - 4);
      ctx.fillStyle = COL.Av; ctx.fillText('Av', pAv[0] + 8, pAv[1] - 4);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pv[0], pv[1], 7, 0, TAU); ctx.stroke();
      this.vHandle = pv; this.aligned = aligned;
      // power iteration trail
      if (o.power) {
        o.power.pts.forEach((q, k) => {
          const pq = P([q[0] * 2.2, q[1] * 2.2]);
          ctx.fillStyle = `rgba(245,196,81,${0.25 + 0.75 * (k / o.power.pts.length)})`;
          ctx.beginPath(); ctx.arc(pq[0], pq[1], 4, 0, TAU); ctx.fill();
          if (k) { const pr = P([o.power.pts[k - 1][0] * 2.2, o.power.pts[k - 1][1] * 2.2]); ctx.strokeStyle = 'rgba(245,196,81,0.4)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(...pr); ctx.lineTo(...pq); ctx.stroke(); }
        });
      }
    },

    drawPCA(ctx, W, H) {
      const q = this.o.pca, st = this.pcaStats(), wide = W > 820;
      const side = wide ? Math.min(W * 0.58, H - 130) : Math.min(W - 40, H * 0.55), R = 4;
      const S = side / (2 * R), cx = 30 + side / 2, cy = 100 + side / 2;
      const sx = (x) => cx + x * S, sy = (y) => cy - y * S;
      this.pv = { S, cx, cy, side, wx: (px) => (px - cx) / S, wy: (py) => (cy - py) / S };
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(cx - side / 2, cy - side / 2, side, side);
      ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.strokeRect(cx - side / 2 + 0.5, cy - side / 2 + 0.5, side - 1, side - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(cx - side / 2, cy - side / 2, side, side); ctx.clip();
      const [mx, my] = st.mean, th = (q.theta * Math.PI) / 180, u = [Math.cos(th), Math.sin(th)];
      // covariance ellipse (2σ) and principal axes
      if (!st.e.complex) {
        const a1 = Math.atan2(st.e.v1[1], st.e.v1[0]), r1 = 2 * Math.sqrt(Math.max(st.e.l1, 0)), r2 = 2 * Math.sqrt(Math.max(st.e.l2, 0));
        ctx.strokeStyle = 'rgba(214,64,159,0.7)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.ellipse(sx(mx), sy(my), r1 * S, r2 * S, -a1, 0, TAU); ctx.stroke();
        for (const [v, r, lab] of [[st.e.v1, r1, 'PC1'], [st.e.v2, r2, 'PC2']]) {
          this.arrow(ctx, sx(mx), sy(my), sx(mx + v[0] * r), sy(my + v[1] * r), COL.eig, 2.2);
          ctx.fillStyle = COL.eig; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
          ctx.fillText(lab, sx(mx + v[0] * r) + 6, sy(my + v[1] * r) - 4);
        }
      }
      // projection line through the mean
      ctx.strokeStyle = COL.v; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx(mx - u[0] * 8), sy(my - u[1] * 8)); ctx.lineTo(sx(mx + u[0] * 8), sy(my + u[1] * 8)); ctx.stroke();
      // points, their shadows on the line, and the collapse animation
      const k = q.collapse;
      for (const [x, y] of this.pts) {
        const t = (x - mx) * u[0] + (y - my) * u[1], px = mx + t * u[0], py = my + t * u[1];
        if (k < 0.02) { ctx.strokeStyle = 'rgba(245,196,81,0.18)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(sx(x), sy(y)); ctx.lineTo(sx(px), sy(py)); ctx.stroke(); }
        const dx = x + (px - x) * k, dy = y + (py - y) * k;
        ctx.fillStyle = '#c9d6ea'; ctx.beginPath(); ctx.arc(sx(dx), sy(dy), 3, 0, TAU); ctx.fill();
        ctx.fillStyle = COL.v; ctx.beginPath(); ctx.arc(sx(px), sy(py), 1.8, 0, TAU); ctx.fill();
      }
      ctx.restore();
      const hp = [sx(mx + u[0] * 3.3), sy(my + u[1] * 3.3)];
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(hp[0], hp[1], 8, 0, TAU); ctx.stroke();
      this.lineHandle = hp;
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('Data, its covariance ellipse, and the projection line (drag ○ to rotate)', cx - side / 2, cy - side / 2 - 6);

      // variance of the projection vs. angle
      const bx = wide ? cx + side / 2 + 40 : 30, by = wide ? 100 : cy + side / 2 + 50, bw = wide ? W - bx - 30 : W - 60, bh = wide ? Math.min(300, H * 0.42) : H - by - 40;
      if (bh > 60) {
        ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(bx, by, bw, bh);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
        ctx.fillStyle = '#c9cfda'; ctx.fillText('Variance of the projected points vs. angle θ', bx, by - 6);
        const vmax = Math.max(st.e.l1 || 1, 1e-6) * 1.15, X = (deg) => bx + 10 + (deg / 180) * (bw - 20), Y = (v) => by + bh - 18 - (v / vmax) * (bh - 30);
        const pts = []; for (let d = 0; d <= 180; d += 1) pts.push([X(d), Y(this.varAlong(st, (d * Math.PI) / 180))]);
        ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 2; ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
        if (!st.e.complex) {
          for (const [l, lab] of [[st.e.l1, 'λ₁'], [st.e.l2, 'λ₂']]) {
            ctx.strokeStyle = 'rgba(214,64,159,0.6)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(bx, Y(l)); ctx.lineTo(bx + bw, Y(l)); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = COL.eig; ctx.font = '11px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(`${lab} = ${fmt(l)}`, bx + bw - 6, Y(l) - 2);
          }
        }
        ctx.fillStyle = COL.v; ctx.beginPath(); ctx.arc(X(q.theta), Y(this.varAlong(st, th)), 6, 0, TAU); ctx.fill();
        ctx.fillStyle = '#7d8696'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        for (let d = 0; d <= 180; d += 45) ctx.fillText(`${d}°`, X(d), by + bh - 2);
      }
    },

    updateHud() {
      const o = this.o;
      let html;
      if (o.mode === 'pca') {
        const st = this.pcaStats(), th = (o.pca.theta * Math.PI) / 180, v = this.varAlong(st, th);
        const e = st.e, ex = e.complex ? 0 : e.l1 / (e.l1 + e.l2);
        const pc1 = e.complex ? 0 : ((((Math.atan2(e.v1[1], e.v1[0]) * 180) / Math.PI) + 360) % 180);
        html = `variance along your line <b>${fmt(v, 3)}</b> · max is λ₁ = <b>${fmt(e.l1, 3)}</b> at θ = ${pc1.toFixed(1)}° · PC1 explains <b>${(ex * 100).toFixed(1)}%</b>`;
      } else {
        const M = this.M(), e = eig2(...M);
        if (o.mode === 'transform') {
          html = `det = <b>${fmt(e.det)}</b> ${Math.abs(e.det) < 0.01 ? '· <span class="warn">the plane is squashed flat: not invertible</span>' : e.det < 0 ? '· <span class="warn">negative: the plane is flipped (see the F)</span>' : '· area scale factor'}`;
        } else {
          const v = o.v, Av = this.apply(M, v);
          const ratio = Math.hypot(...Av) / (Math.hypot(...v) || 1) * Math.sign(v[0] * Av[0] + v[1] * Av[1]);
          html = (e.complex ? `complex eigenvalues ${fmt(e.re)} ± ${fmt(e.im)}i: <b>no real direction survives</b> (rotation by ${fmt((Math.atan2(e.im, e.re) * 180) / Math.PI, 1)}°, scale ${fmt(Math.hypot(e.re, e.im))})`
            : `λ₁ = <b>${fmt(e.l1)}</b>, λ₂ = <b>${fmt(e.l2)}</b>`) +
            `<div>${this.aligned ? `<b>v is an eigenvector!</b> Av = ${fmt(ratio)}·v` : `v and Av point in different directions`}</div>`;
          if (o.power) html += `<div class="dim">power iteration step ${o.power.k}: ${e.complex ? 'keeps circling, since there is no dominant direction' : 'converging to the eigenvector with the largest |λ|'}</div>`;
        }
      }
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      const o = this.o;
      if (o.mode === 'pca') {
        const pv = this.pv, q = o.pca;
        if (!pv) return;
        if (type === 'down') {
          if (this.lineHandle && Math.hypot(x - this.lineHandle[0], y - this.lineHandle[1]) < 16) this.drag = 'line';
          else if (Math.abs(x - pv.cx) < pv.side / 2 && Math.abs(y - pv.cy) < pv.side / 2) { this.pts.push([pv.wx(x), pv.wy(y)]); }
        } else if (type === 'move' && this.drag === 'line') {
          const st = this.pcaStats(), a = Math.atan2(pv.wy(y) - st.mean[1], pv.wx(x) - st.mean[0]);
          q.theta = ((((a * 180) / Math.PI) % 180) + 180) % 180;
          this.thetaSlider && this.thetaSlider.set(+q.theta.toFixed(1));
        } else if (type === 'up' || type === 'leave') this.drag = null;
        return;
      }
      const V = this.V;
      if (!V) return;
      if (type === 'down') {
        const hit = (p) => p && Math.hypot(x - p[0], y - p[1]) < 16;
        if (o.mode === 'eigen' && hit(this.vHandle)) this.drag = 'v';
        else if (this.handles && hit(this.handles.i)) this.drag = 'i';
        else if (this.handles && hit(this.handles.j)) this.drag = 'j';
        if (this.drag && this.drag !== 'v') { o.t = 1; o.animating = false; }
      } else if (type === 'move' && this.drag) {
        const wx = +V.wx(x).toFixed(2), wy = +V.wy(y).toFixed(2);
        if (this.drag === 'v') { const n = Math.hypot(wx, wy) || 1; o.v = [wx / n * clamp(n, 0.4, 3.5), wy / n * clamp(n, 0.4, 3.5)]; o.power = null; }
        else if (this.drag === 'i') this.setA([wx, o.A[1], wy, o.A[3]]);
        else this.setA([o.A[0], wx, o.A[2], wy]);
      } else if (type === 'up' || type === 'leave') this.drag = null;
    },
  };

  sim._internals = { eig2 };
  (window.Sims = window.Sims || {}).linalg = sim;
})();
