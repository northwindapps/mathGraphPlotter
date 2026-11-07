/* Convolution: slide & sum, the convolution theorem, and 2D image kernels. */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const TAU = Math.PI * 2;
  const COL = { f: '#3d8bfd', g: '#e09b14', prod: '#2fa36b', y: '#f5c451' };
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const T0 = -5, T1 = 5, DT = 0.02, NT = Math.round((T1 - T0) / DT) + 1; // signal grid
  const tAt = (i) => T0 + i * DT;

  function rng(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------------------------------------------------------------------ signals and kernels
  const SIGNALS = {
    rect: { label: 'Rectangle', f: (t) => (t > -2 && t < 0.5 ? 1 : 0) },
    step: { label: 'Step', f: (t) => (t > -1 ? 1 : 0) },
    tri: { label: 'Triangle', f: (t) => Math.max(0, 1 - Math.abs(t + 0.5) / 1.5) },
    noisy: { label: 'Noisy wave', f: (() => { const r = rng(9), noise = Array.from({ length: NT }, () => r() - 0.5); return (t) => Math.sin(1.4 * t) * 0.8 + 0.6 * noise[clamp(Math.round((t - T0) / DT), 0, NT - 1)]; })() },
    spikes: { label: 'Three spikes', impulses: [[-3, 1], [-0.5, 0.6], [1.5, 1]] },
  };
  const KERNELS = {
    box: { label: 'Box (moving average)', g: (t, w) => (Math.abs(t) <= w / 2 ? 1 / w : 0), param: 'width',
      about: 'Every output point is the plain average of the input over a window of width w. It smooths noise and blunts edges.' },
    gauss: { label: 'Gaussian blur', g: (t, w) => Math.exp(-(t * t) / (2 * (w / 3) ** 2)) / ((w / 3) * Math.sqrt(TAU)), param: 'width',
      about: 'A weighted average where close neighbours count more. Smoother than the box, with no sharp window edges, so no ripples.' },
    rc: { label: 'RC circuit response', g: (t, w) => (t >= 0 ? Math.exp(-t / (w / 3)) / (w / 3) : 0), param: 'time constant',
      about: 'The impulse response of the RC low-pass filter from the Circuits tab, h(t) = e<sup>−t/τ</sup>/τ. It is <b>one-sided</b>: the circuit only remembers the past. ' +
        'Convolving any input with it gives exactly the capacitor voltage, so the output lags and rounds off.' },
    diff: { label: 'Edge detector', impulses: (w) => [[0, 1 / Math.max(w, 0.1)], [w, -1 / Math.max(w, 0.1)]], param: 'spacing',
      about: 'Two spikes, +1 and −1: the output is f(t) − f(t − d), scaled, i.e. how fast the input is changing. Flat regions give 0; edges give spikes. This is a derivative.' },
    echo: { label: 'Echo', impulses: (w) => [[0, 1], [w, 0.6]], param: 'delay',
      about: 'A spike now plus a quieter spike later: the output is the input plus a delayed, softer copy. Rooms, canyons and audio reverb are convolutions with an impulse response like this.' },
  };

  function sample(fn) { return Float64Array.from({ length: NT }, (_, i) => fn(tAt(i))); }

  // ------------------------------------------------------------------ FFT (for the convolution theorem)
  function fft(re, im, inv) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = ((inv ? 2 : -2) * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < h; k++) {
          const a = i + k, b = a + h, br = re[b] * cr - im[b] * ci, bi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - br; im[b] = im[a] - bi; re[a] += br; im[a] += bi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
    if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  // ------------------------------------------------------------------ 2D kernels
  const K2 = {
    identity: { label: 'Identity', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
    box: { label: 'Box blur', k: [1, 1, 1, 1, 1, 1, 1, 1, 1].map((v) => v / 9) },
    gauss: { label: 'Gaussian', k: [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16) },
    sharpen: { label: 'Sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
    edges: { label: 'Edges (Laplacian)', k: [-1, -1, -1, -1, 8, -1, -1, -1, -1] },
    sobelx: { label: 'Sobel ↔ (vertical edges)', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
    sobely: { label: 'Sobel ↕ (horizontal edges)', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
    emboss: { label: 'Emboss', k: [-2, -1, 0, -1, 1, 1, 0, 1, 2] },
  };
  const IMG = 128;
  function testImage() {
    if (window.Sims && Sims.image && Sims.image._internals) { // reuse the Images tab's test card, downsampled
      const big = Sims.image._internals.scene('card'), out = new Float32Array(IMG * IMG);
      for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
        const k = 2 * y * 256 + 2 * x; out[y * IMG + x] = (big[k] + big[k + 1] + big[k + 256] + big[k + 257]) / 4;
      }
      return out;
    }
    return Float32Array.from({ length: IMG * IMG }, (_, k) => ((k % IMG) + Math.floor(k / IMG)) / (2 * IMG));
  }
  function conv2(src, k) { // true convolution: the kernel is flipped (CNNs skip the flip; it is the same up to mirroring the kernel)
    const out = new Float32Array(IMG * IMG), at = (x, y) => src[clamp(y, 0, IMG - 1) * IMG + clamp(x, 0, IMG - 1)];
    for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
      let s = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) s += k[(1 - j) * 3 + (1 - i)] * at(x + i, y + j);
      out[y * IMG + x] = s;
    }
    return out;
  }

  const MODES = [
    { value: 'slide', label: 'Slide & sum', hint: 'Drag across either plot to move the kernel by hand' },
    { value: 'theorem', label: 'Convolution theorem', hint: 'Same result two ways: slide & sum, or multiply the spectra' },
    { value: 'image', label: 'Image kernels (CNN)', hint: 'Hover over either image to see the 3×3 multiply-and-add at that pixel' },
  ];

  const sim = {
    id: 'conv',
    get hint() { return MODES.find((m) => m.value === this.o.mode).hint; },
    o: { mode: 'slide', signal: 'rect', kernel: 'box', width: 1.5, t: -5, running: true, speed: 1, k2: 'sobelx', kv: K2.sobelx.k.slice(), hover: [64, 64] },
    W: 0, H: 0, hudTick: 0,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      this.recompute();
      if (!this.img) { this.img = testImage(); this.imgOut = conv2(this.img, this.o.kv); }
      this.buildPanel();
    },
    unmount() { this.drag = null; },
    resize(W, H) { this.W = W; this.H = H; },

    // ---------------------------------------------------------------- 1D computation
    sig() { return SIGNALS[this.o.signal]; },
    ker() { return KERNELS[this.o.kernel]; },
    imps(K) { return typeof K.impulses === 'function' ? K.impulses(this.o.width) : K.impulses; },
    recompute() {
      const S = this.sig(), K = this.ker(), w = this.o.width;
      this.fArr = S.impulses ? null : sample(S.f);
      // y(t) = ∫ f(τ) g(t − τ) dτ, with spikes handled exactly
      const y = new Float64Array(NT);
      for (let i = 0; i < NT; i++) {
        const t = tAt(i);
        let s = 0;
        if (S.impulses && K.impulses) { for (const [p, a] of S.impulses) for (const [q, b] of this.imps(K)) if (Math.abs(t - p - q) < DT / 2) s += (a * b) / DT; }
        else if (S.impulses) { for (const [p, a] of S.impulses) s += a * K.g(t - p, w); }
        else if (K.impulses) { for (const [q, b] of this.imps(K)) s += b * S.f(t - q); }
        else { for (let j = 0; j < NT; j++) s += this.fArr[j] * K.g(t - tAt(j), w) * DT; }
        y[i] = s;
      }
      this.y = y;
      let lo = 0, hi = 0;
      for (const v of y) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      this.yRange = [lo - 0.1 * (hi - lo + 0.1), hi + 0.1 * (hi - lo + 0.1)];
      this.spec = null;
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({ options: MODES, value: o.mode, onChange: (v) => { o.mode = v; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; } }));
      p.append(s0);
      if (o.mode === 'image') { this.panelImage(p); return; }

      const s1 = UI.section(o.mode === 'slide' ? 'Convolution: flip, slide, multiply, add' : 'The convolution theorem');
      const eq = el('div', 'eq');
      UI.tex(eq, o.mode === 'slide'
        ? String.raw`(\textcolor{${COL.f}}{f} * \textcolor{${COL.g}}{g})(t) = \int \textcolor{${COL.f}}{f(\tau)}\,\textcolor{${COL.g}}{g(t-\tau)}\,d\tau`
        : String.raw`\mathcal{F}\{f * g\} = \mathcal{F}\{f\}\cdot\mathcal{F}\{g\}`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = o.mode === 'slide'
        ? 'To get the output at time t: take the kernel g, <b>flip</b> it (that is the minus sign in g(t − τ)), <b>slide</b> it to position t, <b>multiply</b> it point by point with f, and <b>add up</b> the result (the green area). ' +
          'Do that for every t and the output curve draws itself.'
        : 'Sliding and summing for every output point costs N² operations. Instead: take the spectra of f and g, <b>multiply</b> them frequency by frequency, and transform back. Same answer, ' +
          'about N·log N work with the FFT. That is why audio effects, image filters and big CNN layers often run in the frequency domain. A blur is a low-pass because its spectrum is small at high frequencies.';
      s1.append(eq, n1);
      s1.append(UI.segmented({ options: Object.entries(SIGNALS).map(([value, s]) => ({ value, label: `f: ${s.label}` })), value: o.signal, onChange: (v) => { o.signal = v; this.recompute(); } }));
      s1.append(UI.segmented({ options: Object.entries(KERNELS).map(([value, k]) => ({ value, label: `g: ${k.label}` })), value: o.kernel, onChange: (v) => { o.kernel = v; this.recompute(); this.buildPanel(); } }));
      const K = this.ker();
      s1.append(UI.slider({ label: `Kernel ${K.param}`, min: 0.2, max: 3, step: 0.05, value: o.width, format: (v) => v.toFixed(2), onInput: (v) => { o.width = v; this.recompute(); } }));
      const kn = el('p', 'note'); kn.innerHTML = K.about; s1.append(kn);
      if (o.mode === 'slide') {
        const row = el('div', 'controls-row');
        const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => { o.running = !o.running; play.textContent = o.running ? '⏸ Pause' : '▶ Play'; });
        row.append(play, UI.button('↺ Restart', () => { o.t = T0; }));
        s1.append(row, UI.slider({ label: 'Slide speed', min: 0.2, max: 3, step: 0.1, value: o.speed, format: (v) => `${v.toFixed(1)}×`, onInput: (v) => { o.speed = v; } }));
        const s2 = UI.section('Where you meet it');
        s2.append(UI.termCard({ color: COL.g, tex: String.raw`y = h * x`, title: 'Every linear, time-invariant system',
          body: 'Feed a system a single spike and record what comes out: its <b>impulse response</b> h. The response to <i>any</i> input is then the input convolved with h. ' +
            'Circuits, speakers, room acoustics, camera blur and the RLC equations are all this.' }));
        s2.append(UI.termCard({ color: COL.prod, tex: String.raw`P(X+Y) = p_X * p_Y`, title: 'Probability',
          body: 'The distribution of a sum of two independent random numbers is the convolution of their distributions. Convolve a box with itself a few times and you get a bell curve: ' +
            'the central limit theorem from the Probability tab is repeated convolution.' }));
        p.append(s1, s2);
      } else p.append(s1);
    },

    panelImage(p) {
      const o = this.o;
      const s1 = UI.section('2D convolution');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`y[i,j] = \sum_{m=-1}^{1}\sum_{n=-1}^{1} k[m,n]\; x[i-m,\, j-n]`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'The same slide-multiply-add, in 2D: a small grid of weights (the <b>kernel</b>) slides over every pixel. Blur, sharpen and edge detection are all just different 9-number kernels. ' +
        'Hover the image to see the arithmetic for one pixel.';
      s1.append(eq, n1);
      s1.append(UI.segmented({ options: Object.entries(K2).map(([value, k]) => ({ value, label: k.label })), value: o.k2, onChange: (v) => { o.k2 = v; o.kv = K2[v].k.slice(); this.imgOut = conv2(this.img, o.kv); this.buildPanel(); } }));
      const grid = el('div', 'kgrid');
      o.kv.forEach((v, k) => {
        const inp = el('input');
        Object.assign(inp, { type: 'number', step: 'any', value: +v.toFixed(4) });
        inp.addEventListener('input', () => { const x = parseFloat(inp.value); if (Number.isFinite(x)) { o.kv[k] = x; this.imgOut = conv2(this.img, o.kv); } });
        grid.append(inp);
      });
      const gl = el('p', 'note', 'Edit the kernel (3 × 3):');
      s1.append(gl, grid);
      const s2 = UI.section('Convolutional neural networks');
      s2.append(UI.termCard({ color: COL.y, tex: String.raw`k \in \mathbb{R}^{3\times3}\ \text{learned}`, title: 'A CNN learns its kernels',
        body: 'A convolutional layer is exactly this operation, but the 9 numbers (times many channels) are <b>weights learned by gradient descent</b>, as in the Learning tab. ' +
          'Trained networks really do discover edge detectors like Sobel in their first layer, then combine them into textures, parts and objects in deeper layers.' }));
      s2.append(UI.termCard({ color: COL.g, tex: String.raw`\sum k = 1 \;\text{vs.}\; \sum k = 0`, title: 'Reading a kernel',
        body: 'Weights that add up to 1 (blur, sharpen) keep overall brightness. Weights that add up to 0 (edges, Sobel) output 0 on flat areas and respond only to <b>change</b>; ' +
          'those outputs are shown as |value|. Sobel ↔ responds to left-right changes, i.e. vertical edges.' }));
      s2.append(UI.termCard({ color: '#9aa3b2', tex: String.raw`\text{conv} \;\text{vs.}\; \text{corr}`, title: 'A small print detail',
        body: 'True convolution flips the kernel; deep-learning libraries skip the flip (technically cross-correlation). Because the kernel is learned anyway, the difference does not matter there. This page does the real flip.' }));
      p.append(s1, s2);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o;
      if (o.mode === 'slide' && o.running && !this.drag) { o.t += (o.speed * 2) / 60; if (o.t > T1) o.t = T0; }
      if (o.mode === 'slide') this.drawSlide(ctx, W, H);
      else if (o.mode === 'theorem') this.drawTheorem(ctx, W, H);
      else this.drawImage(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    plotBox(ctx, b, title) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      if (title) { ctx.fillStyle = '#c9cfda'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(title, b.x, b.y - 6); }
    },
    curve(ctx, pts, color, width, fill, baseY) {
      if (fill) {
        ctx.fillStyle = fill; ctx.beginPath(); ctx.moveTo(pts[0][0], baseY);
        for (const [x, y] of pts) ctx.lineTo(x, y);
        ctx.lineTo(pts[pts.length - 1][0], baseY); ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    },
    spike(ctx, x, y0, y1, color) {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
      const d = y1 < y0 ? 1 : -1; ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x - 6, y1 + 9 * d); ctx.lineTo(x + 6, y1 + 9 * d); ctx.closePath(); ctx.fill();
    },

    drawSlide(ctx, W, H) {
      const o = this.o, S = this.sig(), K = this.ker(), w = o.width, t0 = o.t;
      const x0 = 50, bw = W - 100, sx = (t) => x0 + ((t - T0) / (T1 - T0)) * bw;
      const top = { x: x0, y: 100, w: bw, h: (H - 200) * 0.5 }, bot = { x: x0, y: top.y + top.h + 60, w: bw, h: (H - 200) * 0.5 };
      this.boxes = { top, bot, sx };
      // top: f(τ), the flipped & shifted kernel g(t − τ), and their product
      let fmax = 1;
      if (this.fArr) for (const v of this.fArr) fmax = Math.max(fmax, Math.abs(v));
      let gmax = 0;
      if (!K.impulses) for (let i = 0; i < NT; i++) gmax = Math.max(gmax, Math.abs(K.g(tAt(i), w)));
      const ymax = Math.max(fmax, gmax, ...(K.impulses ? this.imps(K).map(([, b]) => Math.abs(b)) : [0]), 1) * 1.1;
      const ty = (v) => top.y + top.h / 2 - (v / ymax) * (top.h / 2 - 8);
      this.plotBox(ctx, top, `f(τ) (blue), the flipped kernel g(t − τ) slid to t = ${t0.toFixed(2)} (orange), and their product (green area)`);
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.beginPath(); ctx.moveTo(top.x, ty(0)); ctx.lineTo(top.x + top.w, ty(0)); ctx.stroke();
      ctx.save(); ctx.beginPath(); ctx.rect(top.x, top.y, top.w, top.h); ctx.clip();
      const gAt = (tau) => K.g(t0 - tau, w);
      if (this.fArr) {
        // product area
        if (!K.impulses) {
          const prod = []; for (let i = 0; i < NT; i++) prod.push([sx(tAt(i)), ty(this.fArr[i] * gAt(tAt(i)))]);
          this.curve(ctx, prod, 'rgba(47,163,107,0.9)', 1.5, 'rgba(47,163,107,0.45)', ty(0));
        }
        const fp = []; for (let i = 0; i < NT; i++) fp.push([sx(tAt(i)), ty(this.fArr[i])]);
        this.curve(ctx, fp, COL.f, 2.4);
      } else for (const [p, a] of S.impulses) this.spike(ctx, sx(p), ty(0), ty(a), COL.f);
      if (K.impulses) {
        for (const [q, b] of this.imps(K)) {
          const pos = t0 - q; // the flipped kernel puts each spike at τ = t − q
          this.spike(ctx, sx(pos), ty(0), ty(b), COL.g);
          if (this.fArr) { const v = S.f(pos) * b; ctx.fillStyle = COL.prod; ctx.beginPath(); ctx.arc(sx(pos), ty(v), 5, 0, TAU); ctx.fill(); }
        }
      } else {
        const gp = []; for (let i = 0; i < NT; i++) gp.push([sx(tAt(i)), ty(gAt(tAt(i)))]);
        this.curve(ctx, gp, COL.g, 2.2);
        if (S.impulses) for (const [p, a] of S.impulses) { const v = a * gAt(p); ctx.fillStyle = COL.prod; ctx.beginPath(); ctx.arc(sx(p), ty(v), 5, 0, TAU); ctx.fill(); }
      }
      ctx.restore();
      // bottom: the output, built up to t
      const [ylo, yhi] = this.yRange, by = (v) => bot.y + bot.h - ((v - ylo) / (yhi - ylo)) * bot.h;
      this.plotBox(ctx, bot, '(f * g)(t): each point is the green area above');
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.beginPath(); ctx.moveTo(bot.x, by(0)); ctx.lineTo(bot.x + bot.w, by(0)); ctx.stroke();
      const iEnd = clamp(Math.round((t0 - T0) / DT), 0, NT - 1), yp = [];
      for (let i = 0; i <= iEnd; i++) yp.push([sx(tAt(i)), by(this.y[i])]);
      if (yp.length > 1) this.curve(ctx, yp, COL.y, 2.6);
      ctx.fillStyle = COL.y; ctx.beginPath(); ctx.arc(sx(t0), by(this.y[iEnd]), 6, 0, TAU); ctx.fill();
      for (const b of [top, bot]) { ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(sx(t0), b.y); ctx.lineTo(sx(t0), b.y + b.h); ctx.stroke(); ctx.setLineDash([]); }
      ctx.fillStyle = '#7d8696'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let t = T0; t <= T1; t++) ctx.fillText(String(t), sx(t), bot.y + bot.h + 4);
    },

    // circular versions on a 256-point grid for the theorem
    theorem() {
      if (this.spec) return this.spec;
      const N = 256, S = this.sig(), K = this.ker(), w = this.o.width, dt = (T1 - T0) / N;
      const x = new Float64Array(N), h = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const t = T0 + i * dt;
        x[i] = S.impulses ? 0 : S.f(t);
        const tau = i < N / 2 ? i * dt : (i - N) * dt; // kernel stored with its origin at index 0 (wrap-around)
        h[i] = K.impulses ? 0 : K.g(tau, w) * dt;
      }
      if (S.impulses) for (const [p, a] of S.impulses) x[Math.round((p - T0) / dt)] += a / dt;
      if (K.impulses) for (const [q, b] of this.imps(K)) h[((Math.round(q / dt) % N) + N) % N] += b;
      // direct circular convolution: N² multiply-adds
      const yd = new Float64Array(N);
      for (let n = 0; n < N; n++) { let s = 0; for (let m = 0; m < N; m++) s += x[m] * h[(n - m + N) % N]; yd[n] = s; }
      // FFT route: multiply spectra
      const xr = Float64Array.from(x), xi = new Float64Array(N), hr = Float64Array.from(h), hi = new Float64Array(N);
      fft(xr, xi, false); fft(hr, hi, false);
      const yr = new Float64Array(N), yi = new Float64Array(N), X = [], Hm = [], Y = [];
      for (let k = 0; k < N; k++) {
        yr[k] = xr[k] * hr[k] - xi[k] * hi[k]; yi[k] = xr[k] * hi[k] + xi[k] * hr[k];
        X.push(Math.hypot(xr[k], xi[k])); Hm.push(Math.hypot(hr[k], hi[k])); Y.push(Math.hypot(yr[k], yi[k]));
      }
      fft(yr, yi, true);
      let err = 0, scale = 1e-12; for (let n = 0; n < N; n++) { err = Math.max(err, Math.abs(yr[n] - yd[n])); scale = Math.max(scale, Math.abs(yd[n])); }
      this.spec = { N, x, h, yd, yf: yr, X, H: Hm, Y, err: err / scale };
      return this.spec;
    },

    drawTheorem(ctx, W, H) {
      const s = this.theorem(), N = s.N, pad = 30, gw = (W - pad * 4) / 3, gh = (H - 260) / 2; // rows start below the HUD
      const boxes = [0, 1, 2].map((c) => [{ x: pad + c * (gw + pad), y: 130, w: gw, h: gh }, { x: pad + c * (gw + pad), y: 130 + gh + 70, w: gw, h: gh }]);
      const line = (b, arr, color, center = false) => {
        let lo = Math.min(0, ...arr), hi = Math.max(1e-9, ...arr); if (hi - lo < 1e-9) hi = lo + 1;
        const pts = Array.from(arr).map((v, k) => { // Array.from: typed arrays would map to numbers
          const kk = center ? (k + N / 2) % N : k;
          return [b.x + (kk / (N - 1)) * b.w, b.y + b.h - 6 - ((v - lo) / (hi - lo)) * (b.h - 14)];
        });
        if (center) pts.sort((a, c) => a[0] - c[0]);
        this.curve(ctx, pts, color, 2);
      };
      const titles = [['f(t)', 'g(t)  (kernel, origin at the left edge, wraps around)', '(f * g)(t): direct sum (yellow) and via FFT (dashed)'], ['|F(ω)|', '|G(ω)|', '|F(ω)| · |G(ω)|']];
      const data = [[s.x, s.h, s.yd], [s.X, s.H, s.Y]], colors = [COL.f, COL.g, COL.y];
      for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) {
        const b = boxes[c][r];
        this.plotBox(ctx, b, titles[r][c]);
        ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
        line(b, data[r][c], colors[c], r === 1);
        if (r === 0 && c === 2) { ctx.setLineDash([6, 5]); line(b, Array.from(s.yf), '#ffffff'); ctx.setLineDash([]); }
        ctx.restore();
      }
      // operators between the panels
      ctx.fillStyle = '#e8ebf1'; ctx.font = '600 22px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('*', boxes[1][0].x - pad / 2, boxes[1][0].y + gh / 2);
      ctx.fillText('×', boxes[1][1].x - pad / 2, boxes[1][1].y + gh / 2);
      ctx.fillStyle = '#9aa3b2'; ctx.font = '12px system-ui';
      ctx.fillText('FFT ↓                       FFT ↓                       ↑ inverse FFT', W / 2, 130 + gh + 35);
    },

    drawImage(ctx, W, H) {
      const o = this.o, pad = 30, side = Math.min((W - pad * 3) / 2 - 140, H - 300, 420), y0 = 110;
      const ib = { x: pad, y: y0, s: side }, ob = { x: pad * 2 + side, y: y0, s: side };
      this.imgBoxes = [ib, ob];
      const sum = o.kv.reduce((a, v) => a + v, 0), zeroSum = Math.abs(sum) < 1e-6;
      const toCanvas = (data, abs) => {
        const c = this.cvs = this.cvs || {}, key = abs ? 'o' : data === this.img ? 'i' : 'o';
        let cv = c[key]; if (!cv) { cv = document.createElement('canvas'); cv.width = IMG; cv.height = IMG; c[key] = cv; }
        const cx = cv.getContext('2d'), img = cx.createImageData(IMG, IMG), d = img.data;
        for (let k = 0; k < IMG * IMG; k++) { const v = clamp(abs ? Math.abs(data[k]) * 1.5 : data[k], 0, 1) * 255; d[k * 4] = d[k * 4 + 1] = d[k * 4 + 2] = v; d[k * 4 + 3] = 255; }
        cx.putImageData(img, 0, 0); return cv;
      };
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(toCanvas(this.img, false), ib.x, ib.y, side, side);
      ctx.drawImage(toCanvas(this.imgOut, zeroSum), ob.x, ob.y, side, side);
      ctx.imageSmoothingEnabled = true;
      const t = (txt, x, y) => { ctx.fillStyle = '#c9cfda'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(txt, x, y); };
      t('Input image', ib.x, ib.y - 6); t(`Output: input * kernel${zeroSum ? '  (shown as |value|)' : ''}`, ob.x, ob.y - 6);
      // highlight the hovered pixel's 3×3 window on both images
      const [hx, hy] = o.hover, cs = side / IMG;
      for (const b of [ib, ob]) { ctx.strokeStyle = COL.y; ctx.lineWidth = 2; ctx.strokeRect(b.x + (hx - 1) * cs, b.y + (hy - 1) * cs, 3 * cs, 3 * cs); }
      // the arithmetic for that pixel
      const ax = ob.x + side + pad, aw = W - ax - pad, at = (x, y) => this.img[clamp(y, 0, IMG - 1) * IMG + clamp(x, 0, IMG - 1)];
      if (aw > 150) {
        t(`Pixel (${hx}, ${hy}): flipped kernel × neighbourhood`, ax, y0 - 6);
        const cell = Math.min(46, aw / 3.4);
        let total = 0;
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) {
          const kv = o.kv[(1 - j) * 3 + (1 - i)], pv = at(hx + i, hy + j), x = ax + (i + 1) * cell, y = y0 + (j + 1) * cell;
          total += kv * pv;
          const g = Math.round(pv * 255); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x, y, cell - 3, cell - 3);
          ctx.fillStyle = pv > 0.55 ? '#0b0e14' : '#e8ebf1'; ctx.font = '10.5px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(pv.toFixed(2), x + cell / 2 - 1.5, y + cell / 2 - 7);
          ctx.fillStyle = kv > 0 ? '#e09b14' : kv < 0 ? '#5b9bff' : '#7d8696'; ctx.fillText(`×${+kv.toFixed(3)}`, x + cell / 2 - 1.5, y + cell / 2 + 8);
        }
        ctx.fillStyle = COL.y; ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`sum = ${total.toFixed(3)}`, ax, y0 + 3 * cell + 10);
        ctx.fillStyle = '#9aa3b2'; ctx.font = '11.5px system-ui';
        ctx.fillText('grey = pixel value · orange/blue = weight', ax, y0 + 3 * cell + 32);
        ctx.fillText(`kernel sum = ${+sum.toFixed(3)} ${zeroSum ? '(edge-type)' : Math.abs(sum - 1) < 1e-6 ? '(keeps brightness)' : ''}`, ax, y0 + 3 * cell + 50);
      }
    },

    updateHud() {
      const o = this.o;
      let html;
      if (o.mode === 'slide') {
        const i = clamp(Math.round((o.t - T0) / DT), 0, NT - 1);
        html = `t = <b>${o.t.toFixed(2)}</b> · output (f * g)(t) = green area = <b>${this.y[i].toFixed(3)}</b>`;
      } else if (o.mode === 'theorem') {
        const s = this.theorem();
        html = `direct sum (256² = 65,536 multiply-adds) vs. FFT route (~6,000): largest relative difference <b>${s.err.toExponential(1)}</b>, i.e. identical up to rounding`;
      } else {
        html = `kernel: ${K2[o.k2] ? K2[o.k2].label : 'custom'} · 9 weights · applied at all ${(IMG * IMG).toLocaleString()} pixels`;
      }
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    pointer(type, x, y) {
      const o = this.o;
      if (o.mode === 'slide' && this.boxes) {
        const { top, bot } = this.boxes, inside = (b) => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
        if (type === 'down' && (inside(top) || inside(bot))) this.drag = true;
        if (type === 'up' || type === 'leave') { this.drag = false; return; }
        if (this.drag) o.t = clamp(T0 + ((x - top.x) / top.w) * (T1 - T0), T0, T1);
      } else if (o.mode === 'image' && this.imgBoxes) {
        for (const b of this.imgBoxes) {
          if (x >= b.x && x < b.x + b.s && y >= b.y && y < b.y + b.s) { o.hover = [clamp(Math.floor(((x - b.x) / b.s) * IMG), 0, IMG - 1), clamp(Math.floor(((y - b.y) / b.s) * IMG), 0, IMG - 1)]; }
        }
      }
    },
  };

  sim._internals = { KERNELS, SIGNALS, K2, conv2, fft, NT, DT, T0 };
  (window.Sims = window.Sims || {}).conv = sim;
})();
