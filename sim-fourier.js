/* Fourier explorer: series (epicycles), transform (winding machine), and 2D drawing with circles. */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { sig: '#f5c451', target: '#7f8899', vec: '#e8ebf1', com: '#e5484d', spec: '#2fa36b', wind: '#3d8bfd' };
  const TAU = Math.PI * 2;
  const NS = 512, MAXK = 40;          // series: samples per period, max harmonics
  const T_WIN = 3, M_WIN = 1500, FMAX = 8; // transform: window length (s), samples, max winding freq
  const NC = 256;                     // drawing: points on the path = number of circles available

  const WAVES = {
    square: { label: 'Square', f: (p) => (p < 0.5 ? 1 : -1) },
    saw: { label: 'Sawtooth', f: (p) => 1 - 2 * p },
    triangle: { label: 'Triangle', f: (p) => (p < 0.5 ? 4 * p - 1 : 3 - 4 * p) },
    pulse: { label: 'Pulse', f: (p) => (p < 0.2 ? 1.2 : -0.3) },
    draw: { label: '✎ Draw your own', f: null },
  };

  const SHAPES = {
    heart: {
      label: 'Heart',
      pt: (t) => [16 * Math.sin(t) ** 3, 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)],
    },
    star: {
      label: 'Star',
      pt: (t) => {
        const k = (t / TAU) * 10, i = Math.floor(k) % 10, f = k - Math.floor(k);
        const v = (j) => { const r = j % 2 ? 0.42 : 1, a = Math.PI / 2 + (j * TAU) / 10; return [r * Math.cos(a), r * Math.sin(a)]; };
        const a = v(i), b = v(i + 1);
        return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      },
    },
    infinity: {
      label: 'Infinity',
      pt: (t) => { const d = 1 + Math.sin(t) ** 2; return [Math.cos(t) / d, (Math.sin(t) * Math.cos(t)) / d]; },
    },
  };

  const MODES = [
    { value: 'series', label: 'Build from sines', hint: 'Pick a waveform, or choose “Draw your own” and paint on the right-hand graph' },
    { value: 'transform', label: 'Winding machine', hint: 'Drag on the spectrum plot to choose the winding frequency' },
    { value: 'draw', label: 'Draw with circles', hint: 'Draw any closed shape on the canvas with your mouse or finger' },
    { value: 'spec3d', label: '3D: time ↔ frequency', hint: 'Drag to rotate · use the view buttons to fly between the time and frequency views' },
    { value: 'helix', label: '3D: complex helix', hint: 'Drag to rotate · use the view buttons to look at each shadow head-on' },
  ];

  // ------------------------------------------------------------------ math helpers
  function seriesCoeffs(x) {
    const amp = new Float64Array(MAXK + 1), ph = new Float64Array(MAXK + 1);
    for (let k = 0; k <= MAXK; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < NS; n++) { const a = (TAU * k * n) / NS; re += x[n] * Math.cos(a); im -= x[n] * Math.sin(a); }
      re /= NS; im /= NS;
      amp[k] = k === 0 ? re : 2 * Math.hypot(re, im);
      ph[k] = Math.atan2(im, re);
    }
    return { amp, ph };
  }
  const partialSum = (c, N, p) => {
    let v = c.amp[0];
    for (let k = 1; k <= N; k++) v += c.amp[k] * Math.cos(TAU * k * p + c.ph[k]);
    return v;
  };

  function resample(points, n) { // closed path, equal arc-length spacing
    const pts = [...points, points[0]], seg = [0];
    for (let i = 1; i < pts.length; i++) seg.push(seg[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const L = seg[seg.length - 1], out = [];
    let j = 1;
    for (let i = 0; i < n; i++) {
      const d = (i / n) * L;
      while (j < seg.length - 1 && seg[j] < d) j++;
      const f = (d - seg[j - 1]) / Math.max(1e-9, seg[j] - seg[j - 1]);
      out.push([pts[j - 1][0] + (pts[j][0] - pts[j - 1][0]) * f, pts[j - 1][1] + (pts[j][1] - pts[j - 1][1]) * f]);
    }
    return out;
  }

  function circleCoeffs(points) { // complex DFT of a closed path, sorted by size (k = 0 first)
    let cx = 0, cy = 0;
    for (const [x, y] of points) { cx += x; cy += y; }
    cx /= points.length; cy /= points.length;
    let rmax = 1e-9;
    for (const [x, y] of points) rmax = Math.max(rmax, Math.hypot(x - cx, y - cy));
    const z = points.map(([x, y]) => [(x - cx) / rmax, (y - cy) / rmax]);
    const out = [];
    for (let k = -NC / 2; k < NC / 2; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < NC; n++) {
        const a = (-TAU * k * n) / NC, c = Math.cos(a), s = Math.sin(a);
        re += z[n][0] * c - z[n][1] * s;
        im += z[n][0] * s + z[n][1] * c;
      }
      out.push({ k, re: re / NC, im: im / NC, r: Math.hypot(re, im) / NC });
    }
    out.sort((a, b) => (a.k === 0 ? -1 : b.k === 0 ? 1 : b.r - a.r));
    return out;
  }

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'fourier',
    get hint() { return MODES.find((m) => m.value === this.o.mode).hint; },
    o: {
      mode: 'series', running: true, speed: 0.2,
      wave: 'square', N: 5, autoN: false,
      comps: [{ f: 2, a: 1 }, { f: 5, a: 0.7 }, { f: 6.5, a: 0 }], offset: 1, wf: 0.5, sweep: true,
      shape: 'heart', circles: 40,
      N3: 7, hw: 2, hs: 0, helixPos: true, helixNeg: false, helixAvg: false, autoRot: true,
      cam: { spec3d: { yaw: -0.65, pitch: 0.42 }, helix: { yaw: -0.75, pitch: 0.38 } },
    },
    p: 0, t: 0, W: 0, H: 0, hudTick: 0,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.coef) this.setWave(this.o.wave);
      if (!this.spectrum) this.computeTransform();
      if (!this.circ) this.setShape(this.o.shape);
      this.buildPanel();
    },
    unmount() { this.drawing = null; this.painting = null; },
    resize(W, H) { this.W = W; this.H = H; },

    // ---------------------------------------------------------------- series
    setWave(key) {
      this.o.wave = key;
      if (key === 'draw') {
        if (!this.custom) this.custom = Float32Array.from(this.samples || new Float32Array(NS));
        this.samples = this.custom;
      } else {
        const f = WAVES[key].f;
        this.samples = Float32Array.from({ length: NS }, (_, n) => f(n / NS));
      }
      this.coef = seriesCoeffs(this.samples);
    },

    // ---------------------------------------------------------------- transform
    signal(t) {
      let v = this.o.offset;
      for (const c of this.o.comps) if (c.a) v += c.a * Math.cos(TAU * c.f * t);
      return v;
    },
    computeTransform() {
      this.xs = Float64Array.from({ length: M_WIN }, (_, i) => this.signal(((i + 0.5) / M_WIN) * T_WIN));
      const n = 480;
      this.spectrum = Float64Array.from({ length: n + 1 }, (_, j) => Math.hypot(...this.centerOfMass((j / n) * FMAX)));
    },
    centerOfMass(f) {
      let re = 0, im = 0;
      for (let i = 0; i < M_WIN; i++) {
        const a = -TAU * f * ((i + 0.5) / M_WIN) * T_WIN;
        re += this.xs[i] * Math.cos(a); im += this.xs[i] * Math.sin(a);
      }
      return [re / M_WIN, im / M_WIN];
    },

    // ---------------------------------------------------------------- drawing
    setShape(key, points) {
      this.o.shape = key;
      const pts = points || Array.from({ length: NC }, (_, i) => SHAPES[key].pt((i / NC) * TAU));
      this.circ = circleCoeffs(resample(pts, NC));
      this.pathCache = null;
    },
    circlePath(M) {
      if (this.pathCache && this.pathCache.M === M) return this.pathCache.pts;
      const n = 600, pts = [];
      for (let i = 0; i <= n; i++) {
        const p = i / n;
        let x = 0, y = 0;
        for (let j = 0; j < M; j++) {
          const c = this.circ[j], a = TAU * c.k * p, co = Math.cos(a), si = Math.sin(a);
          x += c.re * co - c.im * si; y += c.re * si + c.im * co;
        }
        pts.push([x, y]);
      }
      this.pathCache = { M, pts };
      return pts;
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({
        options: MODES, value: o.mode,
        onChange: (v) => { o.mode = v; this.p = 0; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; },
      }));
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => { o.running = !o.running; play.textContent = o.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play);
      s0.append(row, UI.slider({ label: 'Animation speed', min: 0.05, max: 0.6, step: 0.01, value: o.speed, format: (v) => `${v.toFixed(2)}×`, onInput: (v) => { o.speed = v; } }));
      p.append(s0);
      if (o.mode === 'series') this.panelSeries(p);
      else if (o.mode === 'transform') this.panelTransform(p);
      else if (o.mode === 'spec3d') this.panelSpec3D(p);
      else if (o.mode === 'helix') this.panelHelix(p);
      else this.panelDraw(p);
    },

    // ---------------------------------------------------------------- 3D helpers
    // Orthographic camera: yaw spins around the vertical axis, pitch tilts to look from above.
    camera(W, H, scale) {
      const c = this.o.cam[this.o.mode], cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const S = Math.min(W, H) * scale, ox = W / 2, oy = H * 0.56;
      return (X, Y, Z) => {
        const x = X * cy - Y * sy, d = X * sy + Y * cy;
        return [ox + x * S, oy - (Z * cp + d * sp) * S, d * cp - Z * sp];
      };
    },
    flyTo(yaw, pitch) { const c = this.o.cam[this.o.mode]; c.target = { yaw, pitch }; this.o.autoRot = false; if (this.rotBox) this.rotBox.querySelector('input').checked = false; },
    updateCamera() {
      const c = this.o.cam[this.o.mode];
      if (!c) return;
      if (c.target) {
        c.yaw += (c.target.yaw - c.yaw) * 0.1; c.pitch += (c.target.pitch - c.pitch) * 0.1;
        if (Math.abs(c.target.yaw - c.yaw) + Math.abs(c.target.pitch - c.pitch) < 1e-3) { c.yaw = c.target.yaw; c.pitch = c.target.pitch; c.target = null; }
      } else if (this.o.autoRot && !this.rot) c.yaw = -0.65 + 0.45 * Math.sin(this.t * 0.25);
    },
    viewButtons(views) {
      const row = el('div', 'controls-row');
      for (const [label, yaw, pitch] of views) row.append(UI.button(label, () => this.flyTo(yaw, pitch)));
      this.rotBox = UI.check({ label: 'Gently sway the 3D view', checked: this.o.autoRot, onChange: (v) => { this.o.autoRot = v; } });
      return [row, this.rotBox];
    },
    line3(ctx, P, pts, color, width, dash) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.lineJoin = 'round';
      ctx.beginPath();
      pts.forEach((q, k) => { const [x, y] = P(...q); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    },
    text3(ctx, P, q, txt, color, align = 'center') {
      const [x, y] = P(...q);
      ctx.fillStyle = color; ctx.font = '11.5px system-ui, sans-serif'; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(txt, x, y);
    },

    panelSpec3D(p) {
      const o = this.o;
      const s1 = UI.section('One signal, two views');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\underbrace{x(t)}_{\text{time domain}} = \sum_{k=1}^{N} \underbrace{A_k}_{\text{spectrum}}\cos(2\pi k f_0 t + \phi_k)`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Each sine is laid out at its own depth along a <b>frequency axis</b>. Look from the <b>front</b>: depth disappears and you see the waves stacked on top of each other, ' +
        'with their sum (yellow, front row) being the signal. Look from the <b>side</b>: time disappears, each wave collapses to a vertical line reaching up to its amplitude (the green bars), and that row of bars <i>is</i> the spectrum. ' +
        'Same object, two shadows. The Fourier transform is just turning your head.';
      s1.append(eq, n1, ...this.viewButtons([['⟵ Time view (front)', 0, 0], ['Frequency view (side) ⟶', -Math.PI / 2, 0], ['3D', -0.65, 0.42]]));
      const s2 = UI.section('Signal');
      s2.append(UI.segmented({ options: Object.entries(WAVES).filter(([k]) => k !== 'draw').map(([value, w]) => ({ value, label: w.label })), value: o.wave === 'draw' ? 'square' : o.wave, onChange: (v) => this.setWave(v) }));
      s2.append(UI.slider({ label: 'Number of sines N', min: 1, max: 15, step: 1, value: o.N3, format: (v) => `${v}`, onInput: (v) => { o.N3 = v; } }));
      const n2 = el('p', 'note');
      n2.innerHTML = 'The moving slice marks one instant: the yellow dot\'s height equals the <b>sum</b> of the dots on every wave behind it. ' +
        'Try the square wave from the side: every other bar is missing (only odd harmonics).';
      s2.append(n2);
      p.append(s1, s2);
    },

    panelHelix(p) {
      const o = this.o;
      const s1 = UI.section("Euler's formula in 3D");
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered} e^{i\omega t} = \textcolor{#5b9bff}{\cos\omega t} + i\,\textcolor{#e09b14}{\sin\omega t} \\[3pt]
        \cos\omega t = \tfrac{1}{2}\big(e^{i\omega t} + e^{-i\omega t}\big) \end{gathered}`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Plot the complex number e<sup>iωt</sup> against time and it traces a <b>corkscrew</b>. Its shadow on the floor is <b style="color:#5b9bff">cos</b>, ' +
        'on the back wall <b style="color:#e09b14">sin</b>, and looking straight down the time axis you see the <b style="color:#2fa36b">unit circle</b>: the winding from the winding machine. ' +
        'Fourier uses complex exponentials because one helix carries both cos and sin, i.e. amplitude <i>and</i> phase.';
      s1.append(eq, n1, ...this.viewButtons([['3D', -0.75, 0.38], ['Down the time axis', -Math.PI / 2, 0], ['Side: sin', 0, 0], ['Top: cos', 0, Math.PI / 2 - 0.001]]));
      const s2 = UI.section('Explore');
      s2.append(UI.slider({ label: 'Frequency ω (turns in view)', min: 0.5, max: 5, step: 0.05, value: o.hw, format: (v) => v.toFixed(2), onInput: (v) => { o.hw = v; } }));
      s2.append(UI.termCard({
        color: '#d6409f', tex: String.raw`e^{-i\omega t}`, title: 'Negative frequency',
        body: 'Spins the other way. Add the two helices and their imaginary parts cancel, leaving a flat, purely real cosine. ' +
          'That is why every real signal\'s spectrum is mirror-symmetric (±f): each real wave is a pair of opposite spins.',
        controls: [
          UI.check({ label: 'Show e^{+iωt} (yellow)', checked: o.helixPos, onChange: (v) => { o.helixPos = v; } }),
          UI.check({ label: 'Show e^{−iωt} (pink)', checked: o.helixNeg, onChange: (v) => { o.helixNeg = v; } }),
          UI.check({ label: 'Show their average = cos ωt (white)', checked: o.helixAvg, onChange: (v) => { o.helixAvg = v; } }),
        ],
      }));
      s2.append(UI.termCard({
        color: '#e09b14', tex: String.raw`e^{(\sigma + i\omega)t} = e^{st}`, title: 'Add decay → Laplace',
        body: 'Give the exponent a real part σ and the helix becomes a spiral that shrinks (σ < 0) or grows (σ > 0). These spirals e<sup>st</sup> are exactly the building blocks of the ' +
          '<b>Laplace transform</b>: every point on its s-plane is one of them. σ = 0 brings you back to Fourier.',
        controls: [UI.slider({ label: 'Decay σ', min: -3, max: 1.5, step: 0.05, value: o.hs, format: (v) => v.toFixed(2), onInput: (v) => { o.hs = v; } })],
      }));
      p.append(s1, s2);
    },

    drawSpec3D(ctx, W, H) {
      const o = this.o, c = this.coef, N = o.N3, P = this.camera(W, H, 0.3);
      let amax = 1e-6;
      for (let k = 1; k <= N; k++) amax = Math.max(amax, c.amp[k]);
      const zs = 0.42 / Math.max(amax, 1), X = (t) => -1.2 + 1.2 * t, Y = (k) => -0.9 + (k / N) * 1.8;
      // floor grid
      for (let k = 0; k <= N; k++) this.line3(ctx, P, [[X(0), Y(k), -0.62], [X(2), Y(k), -0.62]], 'rgba(255,255,255,0.06)', 1);
      for (let t = 0; t <= 2; t += 0.5) this.line3(ctx, P, [[X(t), Y(0), -0.62], [X(t), Y(N), -0.62]], 'rgba(255,255,255,0.06)', 1);
      this.line3(ctx, P, [[X(0), Y(0), -0.62], [X(2.15), Y(0), -0.62]], 'rgba(255,255,255,0.4)', 1.5);
      this.line3(ctx, P, [[X(2.12), Y(0), -0.62], [X(2.12), Y(N) + 0.15, -0.62]], 'rgba(255,255,255,0.4)', 1.5);
      this.text3(ctx, P, [X(2.15) + 0.06, Y(0) - 0.08, -0.62], 'time →', '#c9cfda', 'left');
      this.text3(ctx, P, [X(2.12), Y(N) + 0.28, -0.62], 'frequency →', '#c9cfda');
      // draw rows back-to-front for the current camera
      const rows = [];
      for (let k = 0; k <= N; k++) rows.push({ k, depth: P(0, Y(k), 0)[2] });
      rows.sort((a, b) => b.depth - a.depth);
      const tc = (this.p % 2), cursor = [];
      for (const { k } of rows) {
        const fn = k === 0 ? (t) => partialSum(c, N, t % 1) : (t) => c.amp[k] * Math.cos(TAU * k * t + c.ph[k]);
        const pts = [];
        for (let i = 0; i <= 320; i++) { const t = (i / 320) * 2; pts.push([X(t), Y(k), fn(t) * zs]); }
        const color = k === 0 ? COL.sig : `hsla(${170 + (k / N) * 160}, 75%, 62%, ${c.amp[k] < 1e-4 ? 0.15 : 0.9})`;
        this.line3(ctx, P, [[X(0), Y(k), 0], [X(2), Y(k), 0]], 'rgba(255,255,255,0.08)', 1);
        this.line3(ctx, P, pts, color, k === 0 ? 3 : 1.8);
        // spectrum bar on the side wall
        if (k > 0) {
          this.line3(ctx, P, [[X(2.12), Y(k), 0], [X(2.12), Y(k), c.amp[k] * zs]], COL.spec, 5); // same height as the wave's peak
          if (N <= 10 || k % 2 === 1) this.text3(ctx, P, [X(2.12), Y(k), -0.7], `${k}f₀`, '#7d8696');
        }
        cursor.push([k, fn(tc) * zs]);
      }
      // the moving time slice: each wave's value now, and their sum
      this.line3(ctx, P, [[X(tc), Y(0), -0.62], [X(tc), Y(N), -0.62]], 'rgba(255,255,255,0.25)', 1, [3, 4]);
      for (const [k, z] of cursor) {
        const [x, y] = P(X(tc), Y(k), z);
        ctx.fillStyle = k === 0 ? COL.sig : '#e8ebf1';
        ctx.beginPath(); ctx.arc(x, y, k === 0 ? 5.5 : 3, 0, TAU); ctx.fill();
      }
      this.text3(ctx, P, [X(0) - 0.08, Y(0), 0], 'sum = signal', COL.sig, 'right');
    },

    drawHelix(ctx, W, H) {
      const o = this.o, P = this.camera(W, H, 0.32), w = TAU * o.hw, s = o.hs;
      const R = 0.62 / Math.max(1, Math.exp(s)), X = (t) => -1.3 + 2.6 * t, F = -0.78, B = 0.78, E = -1.42; // floor z, back-wall y, end-wall x
      const pt = (t, sign = 1) => { const a = Math.exp(s * t) * R; return [X(t), a * Math.cos(w * t), sign * a * Math.sin(w * t)]; };
      // walls
      const wall = (corners, label, color) => {
        ctx.fillStyle = 'rgba(255,255,255,0.035)'; ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
        ctx.beginPath(); corners.forEach((q, k) => { const [x, y] = P(...q); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.closePath(); ctx.fill(); ctx.stroke();
        this.text3(ctx, P, label[0], label[1], color);
      };
      wall([[X(0), -B, F], [X(1), -B, F], [X(1), B, F], [X(0), B, F]], [[X(1) + 0.12, -B, F], 'floor: Re = cos'], '#5b9bff');
      wall([[X(0), B, F], [X(1), B, F], [X(1), B, -F], [X(0), B, -F]], [[X(0.5), B, -F + 0.1], 'back wall: Im = sin'], '#e09b14');
      wall([[E, -B, F], [E, B, F], [E, B, -F], [E, -B, -F]], [[E, 0, -F + 0.1], 'end: the circle'], '#2fa36b');
      // time axis
      this.line3(ctx, P, [[E, 0, 0], [X(1) + 0.15, 0, 0]], 'rgba(255,255,255,0.35)', 1.2);
      this.text3(ctx, P, [X(1) + 0.24, 0, 0], 'time →', '#c9cfda', 'left');
      const n = 500, ts = Array.from({ length: n + 1 }, (_, i) => i / n);
      const curves = [];
      if (o.helixPos) curves.push([1, COL.sig]);
      if (o.helixNeg) curves.push([-1, '#d6409f']);
      for (const [sign, color] of curves) {
        const h = ts.map((t) => pt(t, sign));
        this.line3(ctx, P, h.map(([x, y]) => [x, y, F]), sign > 0 ? 'rgba(91,155,255,0.85)' : 'rgba(91,155,255,0.4)', 1.8);  // cos shadow
        this.line3(ctx, P, h.map(([x, , z]) => [x, B, z]), sign > 0 ? 'rgba(224,155,20,0.85)' : 'rgba(214,64,159,0.6)', 1.8); // sin shadow
        this.line3(ctx, P, h.map(([, y, z]) => [E, y, z]), 'rgba(47,163,107,0.8)', 1.6);                                   // circle / spiral
        this.line3(ctx, P, h, color, 3);
      }
      if (o.helixAvg) this.line3(ctx, P, ts.map((t) => { const [x, y] = pt(t); return [x, y, 0]; }), '#ffffff', 2.6);
      // moving point with its three projections
      const tc = this.p % 1, sign = o.helixPos ? 1 : -1;
      if (o.helixPos || o.helixNeg) {
        const q = pt(tc, sign), proj = [[q[0], q[1], F], [q[0], B, q[2]], [E, q[1], q[2]]];
        for (const pr of proj) this.line3(ctx, P, [q, pr], 'rgba(255,255,255,0.35)', 1, [3, 3]);
        const dotAt = (pp, col, r) => { const [x, y] = P(...pp); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); };
        dotAt(proj[0], '#5b9bff', 4); dotAt(proj[1], '#e09b14', 4); dotAt(proj[2], '#2fa36b', 4); dotAt(q, '#ffffff', 5.5);
      }
    },

    panelSeries(p) {
      const o = this.o;
      const s1 = UI.section('Fourier series');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`x(t) = a_0 + \sum_{k=1}^{\textcolor{${COL.sig}}{N}} \textcolor{${COL.spec}}{A_k}\cos\!\big(2\pi k f_0 t + \phi_k\big)`, true);
      s1.append(eq);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Any repeating signal is a sum of sine waves at <b>whole-number multiples</b> of its base frequency f<sub>0</sub>. ' +
        'Each circle is one of those sines: its <b>size</b> is the amplitude A<sub>k</sub>, its <b>speed</b> is k·f<sub>0</sub>, and its starting angle is the phase φ<sub>k</sub>. ' +
        'Stack them tip to tail, and the height of the last tip traces the signal.';
      s1.append(n1);
      s1.append(UI.segmented({ options: Object.entries(WAVES).map(([value, w]) => ({ value, label: w.label })), value: o.wave, onChange: (v) => { this.setWave(v); } }));
      this.nSlider = UI.slider({ label: 'Number of sines N', min: 1, max: MAXK, step: 1, value: o.N, format: (v) => `${v}`, onInput: (v) => { o.N = v; } });
      s1.append(this.nSlider);
      this.autoCheck = UI.check({ label: 'Add sines one by one', checked: o.autoN, onChange: (v) => { o.autoN = v; this.nextN = this.t + 0.8; } });
      s1.append(this.autoCheck);

      const s2 = UI.section('Things to notice');
      s2.append(UI.termCard({
        color: COL.spec, tex: String.raw`A_k`, title: 'The spectrum',
        body: 'The bars at the bottom are the recipe: how much of each harmonic. A square wave uses only <b>odd</b> harmonics, shrinking like 1/k. ' +
          'A triangle shrinks like 1/k², so it converges much faster: it is smoother.',
      }));
      s2.append(UI.termCard({
        color: COL.com, tex: String.raw`\approx 9\%`, title: 'Gibbs phenomenon',
        body: 'At a sudden jump the sum always overshoots by about 9%. More sines make the overshoot narrower but never smaller. ' +
          'This is why sharp edges “ring” in audio and image compression (JPEG artifacts!).',
      }));
      s2.append(UI.termCard({
        color: COL.sig, tex: String.raw`c_k = \tfrac{1}{T}\!\int_0^T\! x(t)\,e^{-i2\pi k f_0 t}dt`, title: 'Where the recipe comes from',
        body: 'Each amplitude and phase is found by multiplying the signal by a sine at that frequency and averaging. ' +
          'The <b>Winding machine</b> mode shows what that integral really does.',
      }));
      p.append(s1, s2);
    },

    panelTransform(p) {
      const o = this.o;
      const s1 = UI.section('Fourier transform');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`X(\textcolor{${COL.wind}}{f}) = \int x(t)\,\textcolor{${COL.wind}}{e^{-i2\pi f t}}\,dt`, true);
      s1.append(eq);
      const n1 = el('p', 'note');
      n1.innerHTML = 'The factor <b>e<sup>−i2πft</sup></b> is a point going around a circle f times per second. Multiplying by it ' +
        '<b>wraps the signal around a circle</b> (bottom-left). The integral is the wrapped curve\'s <b style="color:#e5484d">center of mass</b>. ' +
        'At most frequencies the curve is balanced and the center sits near 0. When f matches a frequency inside the signal, ' +
        'the peaks all land on the same side, the center jumps away, and you get a <b>peak in X(f)</b>.';
      s1.append(n1);
      this.wfSlider = UI.slider({ label: 'Winding frequency f', min: 0, max: FMAX, step: 0.01, value: o.wf, format: (v) => `${v.toFixed(2)} Hz`, onInput: (v) => { o.wf = v; o.sweep = false; this.sweepBox.querySelector('input').checked = false; } });
      this.sweepBox = UI.check({ label: 'Sweep f automatically', checked: o.sweep, onChange: (v) => { o.sweep = v; } });
      s1.append(this.wfSlider, this.sweepBox);

      const s2 = UI.section('The signal');
      const recompute = () => this.computeTransform();
      o.comps.forEach((c, i) => {
        const card = UI.termCard({
          color: ['#f5c451', '#e09b14', '#d6409f'][i], tex: String.raw`a_${i + 1}\cos(2\pi f_${i + 1} t)`, title: `Component ${i + 1}`,
          body: i === 0 ? 'Set an amplitude to 0 to remove a component.' : '',
          controls: [
            UI.slider({ label: 'frequency', min: 0.5, max: 7.5, step: 0.1, value: c.f, format: (v) => `${v.toFixed(1)} Hz`, onInput: (v) => { c.f = v; recompute(); } }),
            UI.slider({ label: 'amplitude', min: 0, max: 1.5, step: 0.05, value: c.a, format: (v) => v.toFixed(2), onInput: (v) => { c.a = v; recompute(); } }),
          ],
        });
        s2.append(card);
      });
      s2.append(UI.slider({ label: 'Constant offset', min: 0, max: 1.5, step: 0.05, value: o.offset, format: (v) => v.toFixed(2), onInput: (v) => { o.offset = v; recompute(); } }));
      const n2 = el('p', 'note');
      n2.innerHTML = 'The offset shows up as the peak at <b>f = 0</b> (a constant never wobbles). The peaks have width because we only watch ' +
        `${T_WIN} seconds: a longer window gives sharper peaks. That is the time–frequency uncertainty trade-off.`;
      s2.append(n2);
      p.append(s1, s2);
    },

    panelDraw(p) {
      const o = this.o;
      const s1 = UI.section('Drawing with circles');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`z(t) = \sum_{k} c_k\, e^{i 2\pi k t}`, true);
      s1.append(eq);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Treat each point of a closed drawing as a complex number z = x + iy. The Fourier transform turns the drawing into circles: ' +
        'circle k spins k times per lap (negative k spins backwards), with radius |c<sub>k</sub>|. Biggest circles first: a few give the rough ' +
        'shape, many more add the fine detail. <b>Draw your own shape on the canvas!</b>';
      s1.append(n1);
      s1.append(UI.segmented({ options: Object.entries(SHAPES).map(([value, s]) => ({ value, label: s.label })), value: o.shape, onChange: (v) => this.setShape(v) }));
      s1.append(UI.slider({ label: 'Number of circles', min: 1, max: NC, step: 1, value: o.circles, format: (v) => `${v}`, onInput: (v) => { o.circles = v; } }));
      const n2 = el('p', 'note');
      n2.innerHTML = 'This is the same idea behind JPEG and MP3 compression: keep the biggest coefficients, drop the tiny ones, and the result still looks (or sounds) almost the same.';
      s1.append(n2);
      p.append(s1);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o, dt = 1 / 60;
      if (o.running) {
        this.t += dt;
        this.p = (this.p + dt * o.speed) % 2;
        if (o.mode === 'series' && o.autoN && this.t >= (this.nextN || 0)) {
          o.N = o.N >= MAXK ? 1 : o.N + (o.N < 10 ? 1 : 2);
          o.N = Math.min(o.N, MAXK);
          this.nSlider && this.nSlider.set(o.N);
          this.nextN = this.t + (o.N < 10 ? 0.9 : 0.5);
        }
        if (o.mode === 'transform' && o.sweep) {
          o.wf += dt * 0.35;
          if (o.wf > FMAX) o.wf = 0;
          this.wfSlider && this.wfSlider.set(+o.wf.toFixed(2));
        }
      }
      this.updateCamera();
      if (o.mode === 'series') this.drawSeries(ctx, W, H);
      else if (o.mode === 'transform') this.drawTransform(ctx, W, H);
      else if (o.mode === 'spec3d') this.drawSpec3D(ctx, W, H);
      else if (o.mode === 'helix') this.drawHelix(ctx, W, H);
      else this.drawCircles(ctx, W, H);
      if (++this.hudTick % 10 === 0) this.updateHud();
    },

    updateHud() {
      const o = this.o;
      let html = '';
      if (o.mode === 'series') {
        let err = 0;
        for (let n = 0; n < NS; n += 2) err += (partialSum(this.coef, o.N, n / NS) - this.samples[n]) ** 2;
        html = `<b>${o.N}</b> sine${o.N > 1 ? 's' : ''} · RMS error ${Math.sqrt(err / (NS / 2)).toFixed(3)}`;
      } else if (o.mode === 'transform') {
        const [re, im] = this.centerOfMass(o.wf);
        html = `Winding at <b>f = ${o.wf.toFixed(2)} Hz</b> · center of mass |X(f)|/T = ${Math.hypot(re, im).toFixed(3)}`;
      } else if (o.mode === 'spec3d') {
        const c = o.cam.spec3d, side = Math.abs(c.yaw + Math.PI / 2) < 0.05 && c.pitch < 0.05, front = Math.abs(c.yaw) < 0.05 && c.pitch < 0.05;
        html = front ? '<b>Time view</b>: the sum of the sines is the signal' : side ? '<b>Frequency view</b>: each sine collapsed to a bar, i.e. the spectrum' : `<b>${o.N3}</b> sines laid out along a frequency axis · drag to rotate`;
      } else if (o.mode === 'helix') {
        html = `e<sup>(${o.hs.toFixed(2)} + i·${o.hw.toFixed(2)}·2π)t</sup> · ${o.hs === 0 ? 'pure rotation (Fourier)' : o.hs < 0 ? 'shrinking spiral (decaying, as in Laplace)' : 'growing spiral (unstable, as in Laplace)'}`;
      } else {
        html = `<b>${o.circles}</b> of ${NC} circles · shape: ${o.shape === 'custom' ? 'your drawing' : SHAPES[o.shape].label}`;
      }
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    plotFrame(ctx, x, y, w, h, title) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      if (title) {
        ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui, sans-serif';
        ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(title, x + 8, y + 7);
      }
    },

    drawSeries(ctx, W, H) {
      const o = this.o, c = this.coef, N = o.N;
      const topH = H * 0.64, cy = topH * 0.55, scale = Math.min(topH * 0.26, W * 0.11);
      const cx = Math.max(scale * 1.6, W * 0.17);
      const gx = Math.min(W * 0.4, cx + scale * 1.9), gw = W - gx - 20;
      this.seriesGeom = { gx, gw, cy, scale };

      // graph area (two periods)
      this.plotFrame(ctx, gx, cy - scale * 1.7, gw, scale * 3.4, '');
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.beginPath(); ctx.moveTo(gx, cy); ctx.lineTo(gx + gw, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(gx + gw / 2, cy - scale * 1.7); ctx.lineTo(gx + gw / 2, cy + scale * 1.7); ctx.stroke();
      const curve = (fn, color, width, dash) => {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
        ctx.beginPath();
        for (let i = 0; i <= 600; i++) {
          const q = (i / 600) * 2, px = gx + (i / 600) * gw, py = cy - fn(q % 1) * scale;
          if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        }
        ctx.stroke(); ctx.setLineDash([]);
      };
      curve((q) => this.samples[Math.min(NS - 1, Math.floor(q * NS))], COL.target, 1.5, [5, 4]);
      curve((q) => partialSum(c, N, q), COL.sig, 2.4);

      // epicycles
      const q = this.p % 1;
      let x = cx, y = cy - c.amp[0] * scale;
      ctx.lineWidth = 1;
      for (let k = 1; k <= N; k++) {
        const r = c.amp[k] * scale;
        if (r < 0.3) continue;
        const a = TAU * k * q + c.ph[k];
        const nx = x + Math.sin(a) * r, ny = y - Math.cos(a) * r;
        ctx.strokeStyle = `rgba(255,255,255,${Math.max(0.08, 0.28 - k * 0.006)})`;
        ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.stroke();
        ctx.strokeStyle = COL.vec; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(nx, ny); ctx.stroke(); ctx.lineWidth = 1;
        x = nx; y = ny;
      }
      // connector from tip to the graph
      const gxCur = gx + (this.p / 2) * gw, gyCur = cy - partialSum(c, N, q) * scale;
      ctx.strokeStyle = 'rgba(245,196,81,0.55)'; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gxCur, gyCur); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = COL.sig;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.arc(gxCur, gyCur, 4.5, 0, TAU); ctx.fill();
      ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(o.wave === 'draw' ? 'drag here to paint the target (dashed)' : 'dashed: target · yellow: sum of N sines', gx + 8, cy - scale * 1.7 - 4);

      // spectrum
      const sy = topH + 14, sh = H - sy - 34, sw = W - 40, sx0 = 20;
      this.plotFrame(ctx, sx0, sy, sw, sh, 'Spectrum: amplitude of each harmonic k');
      let amax = 1e-6;
      for (let k = 1; k <= MAXK; k++) amax = Math.max(amax, c.amp[k]);
      const bw = (sw - 20) / MAXK;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.font = '10px system-ui, sans-serif';
      for (let k = 1; k <= MAXK; k++) {
        const bh = (c.amp[k] / amax) * (sh - 44), bx = sx0 + 10 + (k - 1) * bw;
        ctx.fillStyle = k <= N ? COL.spec : 'rgba(255,255,255,0.13)';
        ctx.fillRect(bx + bw * 0.18, sy + sh - 16 - bh, bw * 0.64, bh);
        if (k === 1 || k % 5 === 0) { ctx.fillStyle = '#7d8696'; ctx.fillText(String(k), bx + bw / 2, sy + sh - 14); }
      }
    },

    drawTransform(ctx, W, H) {
      const o = this.o, wide = W > 720;
      const pad = 14;
      const tBox = wide ? { x: pad, y: pad + 40, w: W * 0.5 - pad * 1.5, h: H * 0.26 } : { x: pad, y: pad + 40, w: W - pad * 2, h: H * 0.22 };
      const wBox = wide ? { x: pad, y: tBox.y + tBox.h + pad, w: tBox.w, h: H - tBox.y - tBox.h - pad * 2 }
        : { x: pad, y: tBox.y + tBox.h + pad, w: (W - pad * 3) / 2, h: H - tBox.y - tBox.h - pad * 2 };
      const fBox = wide ? { x: W * 0.5 + pad / 2, y: pad + 40, w: W * 0.5 - pad * 1.5, h: H - pad * 2 - 40 }
        : { x: wBox.x + wBox.w + pad, y: wBox.y, w: wBox.w, h: wBox.h };
      this.fBox = fBox;

      // time plot
      let amax = Math.abs(o.offset);
      for (const c of o.comps) amax += c.a;
      amax = Math.max(0.5, amax);
      this.plotFrame(ctx, tBox.x, tBox.y, tBox.w, tBox.h, `Signal x(t), first ${T_WIN} s`);
      const ty = (v) => tBox.y + tBox.h * 0.55 - (v / amax) * tBox.h * 0.38;
      ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.beginPath(); ctx.moveTo(tBox.x, ty(0)); ctx.lineTo(tBox.x + tBox.w, ty(0)); ctx.stroke();
      ctx.strokeStyle = 'rgba(61,139,253,0.45)'; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let i = 0; i <= 400; i++) {
        const t = (i / 400) * T_WIN, px = tBox.x + (i / 400) * tBox.w, py = ty(Math.cos(TAU * o.wf * t) * amax * 0.9);
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.lineWidth = 2;
      for (let i = 0; i < 400; i++) {
        const t0 = (i / 400) * T_WIN, t1 = ((i + 1) / 400) * T_WIN;
        ctx.strokeStyle = `hsl(${(i / 400) * 300}, 80%, 62%)`;
        ctx.beginPath(); ctx.moveTo(tBox.x + (i / 400) * tBox.w, ty(this.signal(t0))); ctx.lineTo(tBox.x + ((i + 1) / 400) * tBox.w, ty(this.signal(t1))); ctx.stroke();
      }
      ctx.fillStyle = '#7d8696'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      ctx.fillText('blue: a wave at the winding frequency f', tBox.x + tBox.w - 8, tBox.y + 7);

      // winding plot
      this.plotFrame(ctx, wBox.x, wBox.y, wBox.w, wBox.h, 'Wrapped around a circle: x(t)·e^(−i2πft)');
      const wcx = wBox.x + wBox.w / 2, wcy = wBox.y + wBox.h / 2 + 8, R = Math.min(wBox.w, wBox.h - 30) * 0.45 / amax;
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(wcx - amax * R, wcy); ctx.lineTo(wcx + amax * R, wcy); ctx.moveTo(wcx, wcy - amax * R); ctx.lineTo(wcx, wcy + amax * R); ctx.stroke();
      ctx.lineWidth = 1.6;
      const steps = 900;
      let prev = null;
      for (let i = 0; i <= steps; i++) {
        const t = (i / steps) * T_WIN, v = this.signal(t), a = -TAU * o.wf * t;
        const pt = [wcx + v * Math.cos(a) * R, wcy - v * Math.sin(a) * R];
        if (prev) { ctx.strokeStyle = `hsla(${(i / steps) * 300}, 80%, 62%, 0.85)`; ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke(); }
        prev = pt;
      }
      const [cr, ci] = this.centerOfMass(o.wf);
      const comX = wcx + cr * R, comY = wcy - ci * R;
      ctx.strokeStyle = COL.com; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(wcx, wcy); ctx.lineTo(comX, comY); ctx.stroke();
      ctx.fillStyle = COL.com; ctx.beginPath(); ctx.arc(comX, comY, 6, 0, TAU); ctx.fill();
      ctx.fillStyle = '#e5484d'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('● center of mass', wBox.x + 8, wBox.y + wBox.h - 6);

      // spectrum |X(f)|
      this.plotFrame(ctx, fBox.x, fBox.y, fBox.w, fBox.h, 'Fourier transform |X(f)| (size of the center of mass)');
      const sp = this.spectrum, n = sp.length - 1;
      let smax = 1e-6;
      for (const v of sp) smax = Math.max(smax, v);
      const ix = fBox.x + 34, iw = fBox.w - 50, iy = fBox.y + 34, ih = fBox.h - 64;
      const fx = (f) => ix + (f / FMAX) * iw, fy = (v) => iy + ih - (v / smax) * ih * 0.92;
      ctx.font = '10.5px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let f = 0; f <= FMAX; f++) {
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.beginPath(); ctx.moveTo(fx(f), iy); ctx.lineTo(fx(f), iy + ih); ctx.stroke();
        ctx.fillStyle = '#7d8696'; ctx.fillText(`${f}`, fx(f), iy + ih + 4);
      }
      ctx.fillText('frequency f (Hz)', ix + iw / 2, iy + ih + 17);
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let j = 0; j <= n; j++) { const px = fx((j / n) * FMAX), py = fy(sp[j]); if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke();
      ctx.strokeStyle = COL.com; ctx.lineWidth = 2.4; ctx.beginPath();
      const jmax = Math.round((o.wf / FMAX) * n);
      for (let j = 0; j <= jmax; j++) { const px = fx((j / n) * FMAX), py = fy(sp[j]); if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke();
      ctx.strokeStyle = 'rgba(61,139,253,0.8)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(fx(o.wf), iy); ctx.lineTo(fx(o.wf), iy + ih); ctx.stroke();
      ctx.fillStyle = COL.com; ctx.beginPath(); ctx.arc(fx(o.wf), fy(Math.hypot(cr, ci)), 5, 0, TAU); ctx.fill();
      // component markers
      o.comps.forEach((c, i) => {
        if (!c.a) return;
        ctx.fillStyle = ['#f5c451', '#e09b14', '#d6409f'][i];
        ctx.beginPath(); ctx.moveTo(fx(c.f), iy + ih); ctx.lineTo(fx(c.f) - 5, iy + ih + 0); ctx.lineTo(fx(c.f), iy + ih - 8); ctx.lineTo(fx(c.f) + 5, iy + ih); ctx.fill();
      });
    },

    drawCircles(ctx, W, H) {
      const o = this.o, cx = W / 2, cy = H / 2 + 10, S = Math.min(W, H) * 0.36;
      this.circGeom = { cx, cy, S };
      if (this.drawing) {
        ctx.strokeStyle = COL.sig; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
        ctx.beginPath();
        this.drawing.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        return;
      }
      const M = Math.min(o.circles, NC), path = this.circlePath(M), q = this.p % 1;
      const toS = ([x, y]) => [cx + x * S, cy - y * S];
      // faint full path, bright trail
      ctx.lineJoin = 'round';
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1.5; ctx.beginPath();
      path.forEach((pt, i) => { const [x, y] = toS(pt); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke();
      const upto = Math.floor(q * (path.length - 1));
      ctx.strokeStyle = COL.sig; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= upto; i++) { const [x, y] = toS(path[i]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
      // circles
      let x = 0, y = 0;
      for (let j = 0; j < M; j++) {
        const c = this.circ[j], a = TAU * c.k * q, co = Math.cos(a), si = Math.sin(a);
        const nx = x + c.re * co - c.im * si, ny = y + c.re * si + c.im * co;
        if (j > 0 && c.r * S > 0.6) {
          const [sx, sy] = toS([x, y]), [ex, ey] = toS([nx, ny]);
          ctx.strokeStyle = `rgba(255,255,255,${j < 8 ? 0.22 : 0.1})`; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(sx, sy, c.r * S, 0, TAU); ctx.stroke();
          ctx.strokeStyle = 'rgba(232,235,241,0.75)'; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
        }
        x = nx; y = ny;
      }
      const [tx, ty] = toS([x, y]);
      ctx.fillStyle = COL.sig; ctx.beginPath(); ctx.arc(tx, ty, 4.5, 0, TAU); ctx.fill();
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      const o = this.o;
      if (o.mode === 'spec3d' || o.mode === 'helix') { // drag to rotate
        const c = o.cam[o.mode];
        if (type === 'down') { this.rot = { x, y, yaw: c.yaw, pitch: c.pitch }; c.target = null; o.autoRot = false; if (this.rotBox) this.rotBox.querySelector('input').checked = false; }
        else if (type === 'move' && this.rot) { c.yaw = this.rot.yaw - (x - this.rot.x) * 0.008; c.pitch = clamp(this.rot.pitch + (y - this.rot.y) * 0.008, -0.3, Math.PI / 2 - 0.001); }
        else if (type === 'up' || type === 'leave') this.rot = null;
        return;
      }
      if (o.mode === 'series') {
        const g = this.seriesGeom;
        if (o.wave !== 'draw' || !g) return;
        if (type === 'down' && x >= g.gx && x <= g.gx + g.gw) this.painting = { last: null };
        if (type === 'up' || type === 'leave') { this.painting = null; return; }
        if (!this.painting || (type !== 'down' && type !== 'move')) return;
        const idx = Math.floor((((x - g.gx) / g.gw) * 2 % 1 + 1) % 1 * NS);
        const val = clamp((g.cy - y) / g.scale, -1.5, 1.5);
        const last = this.painting.last;
        if (last && Math.abs(idx - last.idx) < NS / 2) {
          const a = Math.min(idx, last.idx), b = Math.max(idx, last.idx);
          for (let n = a; n <= b; n++) this.custom[n] = last.val + ((val - last.val) * (n - last.idx)) / ((idx - last.idx) || 1);
        } else this.custom[idx] = val;
        this.painting.last = { idx, val };
        this.coef = seriesCoeffs(this.custom);
      } else if (o.mode === 'transform') {
        const b = this.fBox;
        if (type === 'down' && b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h) this.dragF = true;
        if (type === 'up' || type === 'leave') { this.dragF = false; return; }
        if (!this.dragF) return;
        o.wf = clamp(((x - (b.x + 34)) / (b.w - 50)) * FMAX, 0, FMAX);
        o.sweep = false;
        this.sweepBox && (this.sweepBox.querySelector('input').checked = false);
        this.wfSlider && this.wfSlider.set(+o.wf.toFixed(2));
      } else {
        if (type === 'down') this.drawing = [[x, y]];
        else if (type === 'move' && this.drawing) {
          const l = this.drawing[this.drawing.length - 1];
          if (Math.hypot(x - l[0], y - l[1]) > 2) this.drawing.push([x, y]);
        } else if ((type === 'up' || type === 'leave') && this.drawing) {
          const pts = this.drawing;
          this.drawing = null;
          if (pts.length >= 12) { this.setShape('custom', pts.map(([px, py]) => [px, -py])); this.p = 0; }
        }
      }
    },
  };
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  (window.Sims = window.Sims || {}).fourier = sim;
})();
