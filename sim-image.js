/* Image denoising: noise models, convolution filters, median, 2D Fourier masking, bilateral, total variation.
 * Images are 256×256 grayscale Float32 in [0, 1].
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const N = 256, TAU = Math.PI * 2;
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  function rng(seed) {
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); };

  // ------------------------------------------------------------------ test images
  function canvasToGray(cv) {
    const d = cv.getContext('2d').getImageData(0, 0, N, N).data, out = new Float32Array(N * N);
    for (let k = 0; k < N * N; k++) out[k] = (0.299 * d[k * 4] + 0.587 * d[k * 4 + 1] + 0.114 * d[k * 4 + 2]) / 255;
    return out;
  }
  function scene(kind) {
    const cv = document.createElement('canvas'); cv.width = N; cv.height = N;
    const c = cv.getContext('2d');
    if (kind === 'text') {
      c.fillStyle = '#f2f0ea'; c.fillRect(0, 0, N, N);
      c.fillStyle = '#1d1d1d'; c.textBaseline = 'top';
      const lines = [['bold 26px Georgia, serif', 'Denoising'], ['15px Georgia, serif', 'Every image is a grid'], ['15px Georgia, serif', 'of numbers. Noise adds'], ['15px Georgia, serif', 'random numbers on top.'],
        ['12px Georgia, serif', 'Filters try to take them'], ['12px Georgia, serif', 'away again without erasing'], ['12px Georgia, serif', 'the real details: edges,'], ['12px Georgia, serif', 'letters, and textures.'],
        ['10px Georgia, serif', 'Small print is the hardest test.'], ['9px Georgia, serif', 'Median, Fourier, bilateral, TV.']];
      let y = 12;
      for (const [font, txt] of lines) { c.font = font; c.fillText(txt, 14, y); y += parseInt(font.match(/(\d+)px/)[1], 10) + 7; }
    } else {
      const g = c.createLinearGradient(0, 0, N, N); g.addColorStop(0, '#2c2c2c'); g.addColorStop(1, '#a8a8a8');
      c.fillStyle = g; c.fillRect(0, 0, N, N);
      const r = c.createRadialGradient(62, 56, 6, 74, 70, 52); r.addColorStop(0, '#ffffff'); r.addColorStop(1, '#3a3a3a');
      c.fillStyle = r; c.beginPath(); c.arc(74, 70, 50, 0, TAU); c.fill();                      // shaded sphere
      c.fillStyle = '#141414'; c.fillRect(146, 22, 92, 78);                                       // dark block
      c.fillStyle = '#e8e8e8'; c.beginPath(); c.arc(192, 61, 22, 0, TAU); c.fill();
      for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { c.fillStyle = (i + j) % 2 ? '#202020' : '#e0e0e0'; c.fillRect(18 + i * 9, 150 + j * 9, 9, 9); } // checker
      c.strokeStyle = '#f4f4f4'; c.lineWidth = 1;
      for (let k = 0; k < 9; k++) { c.beginPath(); c.moveTo(118 + k * 4, 140); c.lineTo(118 + k * 4, 240); c.stroke(); } // fine lines
      c.fillStyle = '#ffffff'; c.font = 'bold 52px system-ui, sans-serif'; c.textBaseline = 'alphabetic'; c.fillText('Hi!', 168, 212);
      c.fillStyle = '#101010'; c.font = 'bold 13px system-ui, sans-serif'; c.fillText('edges & detail', 158, 240);
    }
    return canvasToGray(cv);
  }

  // ------------------------------------------------------------------ FFT (radix-2) and 2D FFT
  function fft1(re, im, inv) {
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
  function fft2(re, im, inv) {
    const r = new Float64Array(N), i2 = new Float64Array(N);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) { r[x] = re[y * N + x]; i2[x] = im[y * N + x]; }
      fft1(r, i2, inv);
      for (let x = 0; x < N; x++) { re[y * N + x] = r[x]; im[y * N + x] = i2[x]; }
    }
    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) { r[y] = re[y * N + x]; i2[y] = im[y * N + x]; }
      fft1(r, i2, inv);
      for (let y = 0; y < N; y++) { re[y * N + x] = r[y]; im[y * N + x] = i2[y]; }
    }
  }
  const freqOf = (k) => (k < N / 2 ? k : k - N); // index -> signed frequency (cycles per image)

  // ------------------------------------------------------------------ filters
  const at = (src, x, y) => src[clamp(y, 0, N - 1) * N + clamp(x, 0, N - 1)];

  function convolveSeparable(src, kernel) {
    const r = (kernel.length - 1) / 2, tmp = new Float32Array(N * N), out = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let s = 0; for (let k = -r; k <= r; k++) s += kernel[k + r] * at(src, x + k, y); tmp[y * N + x] = s;
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let s = 0; for (let k = -r; k <= r; k++) s += kernel[k + r] * at(tmp, x, y + k); out[y * N + x] = s;
    }
    return out;
  }
  const boxKernel = (r) => new Float32Array(2 * r + 1).fill(1 / (2 * r + 1));
  function gaussKernel(sigma) {
    const r = Math.max(1, Math.ceil(3 * sigma)), k = new Float32Array(2 * r + 1);
    let s = 0;
    for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
    return k.map((v) => v / s);
  }

  function median(src, r) {
    const out = new Float32Array(N * N), win = new Float32Array((2 * r + 1) ** 2), mid = (win.length - 1) / 2;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      let n = 0;
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) win[n++] = at(src, x + dx, y + dy);
      win.sort();
      out[y * N + x] = win[mid];
    }
    return out;
  }

  function bilateral(src, sigmaS, sigmaR) {
    const r = Math.max(1, Math.ceil(1.5 * sigmaS)), ws = [], out = new Float32Array(N * N);
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) ws.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS))]);
    const lut = new Float32Array(1025);
    for (let k = 0; k <= 1024; k++) { const d = k / 1024; lut[k] = Math.exp(-(d * d) / (2 * sigmaR * sigmaR)); }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const c = src[y * N + x];
      let s = 0, wsum = 0;
      for (const [dx, dy, w0] of ws) {
        const v = at(src, x + dx, y + dy), w = w0 * lut[Math.min(1024, (Math.abs(v - c) * 1024) | 0)];
        s += w * v; wsum += w;
      }
      out[y * N + x] = s / wsum;
    }
    return out;
  }

  // Total-variation denoising (ROF model) by Chambolle's projection algorithm:
  //   min_u ½‖u − f‖² + λ·Σ|∇u|      — iterated a few steps per frame so you can watch it
  function tvState(f, lambda) { return { f, lambda, px: new Float32Array(N * N), py: new Float32Array(N * N), u: Float32Array.from(f), iter: 0 }; }
  function tvStep(s, iters) {
    const { f, lambda, px, py } = s, tau = 0.125, div = new Float32Array(N * N), w = new Float32Array(N * N);
    for (let it = 0; it < iters; it++) {
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const k = y * N + x;
        div[k] = (x < N - 1 ? px[k] : 0) - (x > 0 ? px[k - 1] : 0) + (y < N - 1 ? py[k] : 0) - (y > 0 ? py[k - N] : 0);
        w[k] = div[k] - f[k] / lambda;
      }
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const k = y * N + x, gx = x < N - 1 ? w[k + 1] - w[k] : 0, gy = y < N - 1 ? w[k + N] - w[k] : 0, m = 1 + tau * Math.hypot(gx, gy);
        px[k] = (px[k] + tau * gx) / m; py[k] = (py[k] + tau * gy) / m;
      }
      s.iter++;
    }
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const k = y * N + x;
      const d = (x < N - 1 ? px[k] : 0) - (x > 0 ? px[k - 1] : 0) + (y < N - 1 ? py[k] : 0) - (y > 0 ? py[k - N] : 0);
      s.u[k] = f[k] - lambda * d;
    }
  }

  const psnr = (a, b) => { let e = 0; for (let k = 0; k < a.length; k++) { const d = clamp(a[k], 0, 1) - b[k]; e += d * d; } return 10 * Math.log10(1 / Math.max(1e-12, e / a.length)); };

  const SMOOTH = [
    { value: 'none', label: 'None' }, { value: 'box', label: 'Box' }, { value: 'gauss', label: 'Gaussian' },
    { value: 'bilateral', label: 'Bilateral' }, { value: 'tv', label: 'Total variation' },
  ];

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'image',
    hint: 'Click bright dots in the spectrum (bottom-right) to notch them out · click again to remove',
    o: {
      scene: 'card', noiseG: 0.08, noiseSP: 0.03, stripeA: 0.18, stripeK: 26, stripeAng: 35, seed: 7,
      median: true, medianR: 1,
      freq: true, lowpass: 128, ideal: false, notches: [],
      smooth: 'tv', boxR: 1, gaussS: 1, bilS: 2, bilR: 0.12, tvL: 0.08,
      topLeft: 'clean', specOf: 'noisy',
    },
    W: 0, H: 0, dirty: true, hudTick: 0,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.clean) { this.clean = scene(this.o.scene); this.makeNoisy(); this.autoNotch(); }
      this.buildPanel();
    },
    unmount() {},
    resize(W, H) { this.W = W; this.H = H; },
    changed() { this.dirty = true; },

    makeNoisy() {
      const o = this.o, r = rng(o.seed), out = new Float32Array(N * N);
      const a = (o.stripeAng * Math.PI) / 180, fx = o.stripeK * Math.cos(a), fy = o.stripeK * Math.sin(a);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const k = y * N + x;
        let v = this.clean[k] + o.noiseG * gauss(r) + o.stripeA * Math.sin((TAU * (fx * x + fy * y)) / N);
        const u = r();
        if (u < o.noiseSP / 2) v = 0; else if (u < o.noiseSP) v = 1;
        out[k] = v; // not clipped: clipping would create extra stripe harmonics (display clips instead)
      }
      this.noisy = out;
      this.spec = { noisy: this.spectrum(out) };
      this.dirty = true;
    },

    spectrum(img) { // log magnitude, fft-shifted for display
      const re = Float64Array.from(img), im = new Float64Array(N * N);
      fft2(re, im, false);
      const out = new Float32Array(N * N);
      for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) {
        const k = v * N + u, sx = (u + N / 2) % N, sy = (v + N / 2) % N;
        out[sy * N + sx] = Math.log(1 + Math.hypot(re[k], im[k]));
      }
      return out;
    },

    mask(ku, kv) {
      const o = this.o;
      let m = 1;
      if (o.lowpass < 128) {
        const rho = Math.hypot(ku, kv);
        m *= o.ideal ? (rho <= o.lowpass ? 1 : 0) : Math.exp(-Math.LN2 * (rho / o.lowpass) ** 2);
      }
      for (const [nu, nv] of o.notches) {
        for (const s of [1, -1]) {
          const d2 = (ku - s * nu) ** 2 + (kv - s * nv) ** 2;
          m *= o.ideal ? (d2 <= 9 ? 0 : 1) : 1 - Math.exp(-d2 / (2 * 2.5 * 2.5));
        }
      }
      return m;
    },

    freqFilter(src) {
      const re = Float64Array.from(src), im = new Float64Array(N * N);
      fft2(re, im, false);
      for (let v = 0; v < N; v++) for (let u = 0; u < N; u++) { const k = v * N + u, m = this.mask(freqOf(u), freqOf(v)); re[k] *= m; im[k] *= m; }
      fft2(re, im, true);
      return Float32Array.from(re);
    },

    // find the bright isolated peaks that periodic noise makes in the spectrum
    autoNotch() {
      const s = this.spec.noisy, cand = [];
      let sum = 0, sum2 = 0, n = 0;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const ku = x - N / 2, kv = y - N / 2;
        if (Math.hypot(ku, kv) < 10) continue;
        const v = s[y * N + x]; sum += v; sum2 += v * v; n++;
      }
      const mean = sum / n, sd = Math.sqrt(sum2 / n - mean * mean);
      // Periodic noise covers the whole image, so its peak is a needle: far above the ring 3–5 bins around it.
      // Real structure (a checkerboard patch, lines of text) makes broader bumps. Measured: noise ≥ 3.1, image ≤ 2.4.
      for (let y = 6; y < N - 6; y++) for (let x = 6; x < N - 6; x++) {
        const ku = x - N / 2, kv = y - N / 2, v = s[y * N + x];
        if (Math.hypot(ku, kv) < 10 || v < mean + 4 * sd) continue;
        if (kv < 0 || (kv === 0 && ku < 0)) continue; // keep one of each mirror pair
        let isMax = true;
        for (let dy = -2; dy <= 2 && isMax; dy++) for (let dx = -2; dx <= 2; dx++) if ((dx || dy) && s[(y + dy) * N + x + dx] > v) { isMax = false; break; }
        if (!isMax) continue;
        let rs = 0, rn = 0;
        for (let dy = -5; dy <= 5; dy++) for (let dx = -5; dx <= 5; dx++) { const r = Math.hypot(dx, dy); if (r >= 3 && r <= 5) { rs += s[(y + dy) * N + x + dx]; rn++; } }
        if (v - rs / rn > 2.8) cand.push([ku, kv, v]);
      }
      cand.sort((a, b) => b[2] - a[2]);
      this.o.notches = cand.slice(0, 6).map(([u, v]) => [u, v]);
      this.dirty = true;
    },

    run() {
      const o = this.o;
      let cur = this.noisy;
      if (o.median) cur = median(cur, o.medianR);
      if (o.freq) cur = this.freqFilter(cur);
      this.stage2 = cur;
      this.tv = null;
      if (o.smooth === 'box') cur = convolveSeparable(cur, boxKernel(o.boxR));
      else if (o.smooth === 'gauss') cur = convolveSeparable(cur, gaussKernel(o.gaussS));
      else if (o.smooth === 'bilateral') cur = bilateral(cur, o.bilS, o.bilR);
      else if (o.smooth === 'tv') { this.tv = tvState(cur, o.tvL); cur = this.tv.u; }
      this.result = cur;
      this.spec.result = null;
      this.dirty = false;
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const ch = () => this.changed(), noiseCh = () => { this.makeNoisy(); };

      const s1 = UI.section('Image');
      const file = el('input'); file.type = 'file'; file.accept = 'image/*'; file.hidden = true;
      file.addEventListener('change', () => this.loadFile(file.files[0]));
      s1.append(UI.segmented({
        options: [{ value: 'card', label: 'Test card' }, { value: 'text', label: 'Printed text' }, { value: 'upload', label: '📁 Your photo…' }],
        value: o.scene, onChange: (v) => { if (v === 'upload') { file.click(); return; } o.scene = v; this.clean = scene(v); this.makeNoisy(); this.autoNotch(); },
      }), file);
      const n0 = el('p', 'note'); n0.innerHTML = 'An image is just a grid of numbers (here 256 × 256 brightness values). Your photo never leaves the browser.';
      s1.append(n0);

      const s2 = UI.section('Add noise');
      s2.append(UI.slider({ label: 'Gaussian noise σ (sensor grain)', min: 0, max: 0.3, step: 0.005, value: o.noiseG, format: (v) => v.toFixed(3), onInput: (v) => { o.noiseG = v; noiseCh(); } }));
      s2.append(UI.slider({ label: 'Salt & pepper (dead pixels)', min: 0, max: 0.25, step: 0.005, value: o.noiseSP, format: (v) => `${(v * 100).toFixed(1)}%`, onInput: (v) => { o.noiseSP = v; noiseCh(); } }));
      s2.append(UI.slider({ label: 'Periodic stripes (interference)', min: 0, max: 0.4, step: 0.01, value: o.stripeA, format: (v) => v.toFixed(2), onInput: (v) => { o.stripeA = v; noiseCh(); } }));
      s2.append(UI.slider({ label: 'Stripe frequency', min: 6, max: 90, step: 1, value: o.stripeK, format: (v) => `${v} cycles`, onInput: (v) => { o.stripeK = v; noiseCh(); } }));
      s2.append(UI.slider({ label: 'Stripe angle', min: 0, max: 180, step: 1, value: o.stripeAng, format: (v) => `${v}°`, onInput: (v) => { o.stripeAng = v; noiseCh(); } }));

      const s3 = UI.section('Denoising pipeline (runs top to bottom)');
      s3.append(UI.termCard({
        color: '#e5484d', tex: String.raw`y = \operatorname{median}\{x_{\text{neighbours}}\}`, title: '1 · Median filter',
        body: 'Each pixel becomes the <b>middle value</b> of its neighbourhood. A white or black outlier never wins that vote, so it is the classic cure for salt & pepper, and it keeps edges sharp.',
        controls: [UI.check({ label: 'On', checked: o.median, onChange: (v) => { o.median = v; ch(); } }),
          UI.slider({ label: 'Radius', min: 1, max: 3, step: 1, value: o.medianR, format: (v) => `${v} → ${(2 * v + 1)}×${(2 * v + 1)} window`, onInput: (v) => { o.medianR = v; ch(); } })],
      }));
      const fr = el('div', 'controls-row');
      fr.append(UI.button('🔍 Find periodic peaks', () => { o.freq = true; this.autoNotch(); this.buildPanel(); }), UI.button('Clear notches', () => { o.notches = []; ch(); }));
      s3.append(UI.termCard({
        color: '#3d8bfd', tex: String.raw`Y(u,v) = H(u,v)\,X(u,v)`, title: '2 · Fourier mask (2D FFT)',
        body: 'The spectrum (bottom-right) shows how much of every 2D stripe pattern the image contains: the centre is smooth shading, the edges are fine detail. ' +
          'Periodic noise is a <b>pair of bright dots</b>, so notch them out (click them, or auto-find). A low-pass keeps only the centre, which blurs. An ideal (sharp) cutoff makes ripples near edges: the Gibbs effect again.',
        controls: [UI.check({ label: 'On', checked: o.freq, onChange: (v) => { o.freq = v; ch(); } }),
          UI.slider({ label: 'Low-pass radius', min: 4, max: 128, step: 1, value: o.lowpass, format: (v) => (v >= 128 ? 'off' : `${v} cycles`), onInput: (v) => { o.lowpass = v; ch(); } }),
          UI.check({ label: 'Ideal (sharp) cutoff', checked: o.ideal, onChange: (v) => { o.ideal = v; ch(); } }), fr],
      }));
      const sm = UI.termCard({ color: '#2fa36b', tex: String.raw`\text{smooth}`, title: '3 · Smoothing', body: '', controls: [] });
      sm.append(UI.segmented({ options: SMOOTH, value: o.smooth, onChange: (v) => { o.smooth = v; ch(); this.buildPanel(); } }));
      const info = el('p', 'term-body'), texEl = el('div', 'eq');
      texEl.style.marginTop = '8px';
      const texts = {
        none: ['', 'No smoothing: see what stages 1–2 achieve on their own.'],
        box: [String.raw`y = \tfrac{1}{(2r+1)^2}\sum x_{\text{window}} = h_{\text{box}} * x`, 'Plain average of the neighbourhood: a <b>convolution</b> with a flat kernel. Removes grain but blurs every edge equally.'],
        gauss: [String.raw`h * x \;\Longleftrightarrow\; H \cdot X`, 'Weighted average with bell-curve weights. By the convolution theorem this is the same as multiplying the spectrum by a Gaussian, i.e. a smooth low-pass.'],
        bilateral: [String.raw`w = e^{-\frac{\|\Delta p\|^2}{2\sigma_s^2}}\, e^{-\frac{(\Delta I)^2}{2\sigma_r^2}}`, 'Average only neighbours that are <b>near and similar in brightness</b>. Across an edge the brightness jumps, so those pixels get ~0 weight and the edge survives. Phone "beauty" filters use this.'],
        tv: [String.raw`\min_u\ \tfrac12\|u - f\|^2 + \lambda \sum |\nabla u|`, 'Denoising as <b>optimization</b>: stay close to the noisy image f but keep the total variation small. The L1-type penalty (like lasso) prefers flat areas with crisp edges, a "cartoon" look. Watch it iterate.'],
      }[o.smooth];
      if (texts[0]) UI.tex(texEl, texts[0], true); else texEl.hidden = true;
      info.innerHTML = texts[1];
      sm.append(texEl, info);
      if (o.smooth === 'box') sm.append(UI.slider({ label: 'Radius r', min: 1, max: 5, step: 1, value: o.boxR, format: (v) => `${v}`, onInput: (v) => { o.boxR = v; ch(); } }));
      if (o.smooth === 'gauss') sm.append(UI.slider({ label: 'σ (pixels)', min: 0.3, max: 4, step: 0.1, value: o.gaussS, format: (v) => v.toFixed(1), onInput: (v) => { o.gaussS = v; ch(); } }));
      if (o.smooth === 'bilateral') {
        sm.append(UI.slider({ label: 'σs (distance)', min: 0.5, max: 4, step: 0.1, value: o.bilS, format: (v) => v.toFixed(1), onInput: (v) => { o.bilS = v; ch(); } }));
        sm.append(UI.slider({ label: 'σr (brightness)', min: 0.02, max: 0.5, step: 0.01, value: o.bilR, format: (v) => v.toFixed(2), onInput: (v) => { o.bilR = v; ch(); } }));
      }
      if (o.smooth === 'tv') sm.append(UI.slider({ label: 'λ (strength)', min: 0.01, max: 0.3, step: 0.005, value: o.tvL, format: (v) => v.toFixed(3), onInput: (v) => { o.tvL = v; ch(); } }));
      s3.append(sm);

      const s4 = UI.section('Compare');
      s4.append(UI.segmented({ options: [{ value: 'clean', label: 'Top-left: original' }, { value: 'error', label: 'Top-left: error map' }], value: o.topLeft, onChange: (v) => { o.topLeft = v; } }));
      s4.append(UI.segmented({ options: [{ value: 'noisy', label: 'Spectrum of noisy' }, { value: 'result', label: 'Spectrum of result' }], value: o.specOf, onChange: (v) => { o.specOf = v; } }));
      const n4 = el('p', 'note');
      n4.innerHTML = '<b>PSNR</b> (in the readout) measures closeness to the original in dB: higher is better, and +3 dB means half the error. ' +
        'Rules of thumb: salt & pepper → median; stripes → notch; grain → bilateral or TV. The error map shows <i>where</i> a filter hurts (usually at edges).';
      s4.append(n4);
      p.append(s1, s2, s3, s4);
    },

    loadFile(f) {
      if (!f) return;
      const img = new Image(), url = URL.createObjectURL(f);
      img.onload = () => {
        const cv = document.createElement('canvas'); cv.width = N; cv.height = N;
        const c = cv.getContext('2d'), s = Math.max(N / img.width, N / img.height);
        c.drawImage(img, (N - img.width * s) / 2, (N - img.height * s) / 2, img.width * s, img.height * s);
        URL.revokeObjectURL(url);
        this.o.scene = 'upload'; this.clean = canvasToGray(cv); this.makeNoisy(); this.autoNotch(); this.buildPanel();
      };
      img.src = url;
    },

    // ---------------------------------------------------------------- loop & drawing
    frame(ctx, W, H) {
      if (!this.noisy) return;
      if (this.dirty) this.run();
      if (this.tv && this.tv.iter < 200) { tvStep(this.tv, 8); this.spec.result = null; }
      this.draw(ctx, W, H);
      if (++this.hudTick % 8 === 0) this.updateHud();
    },

    updateHud() {
      const pn = psnr(this.noisy, this.clean), pr = psnr(this.result, this.clean);
      let html = `PSNR noisy <b>${pn.toFixed(1)} dB</b> → denoised <b>${pr.toFixed(1)} dB</b> <span class="${pr > pn ? 'dim' : 'warn'}">(${pr >= pn ? '+' : ''}${(pr - pn).toFixed(1)} dB)</span>`;
      if (this.tv) html += `<div class="dim">total variation: iteration ${this.tv.iter}${this.tv.iter >= 200 ? ' (converged)' : '…'}</div>`;
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    toCanvas(key, data, map, ver) { // repaint a tile only when its version changes
      this.cv = this.cv || {};
      let c = this.cv[key];
      if (!c) { c = document.createElement('canvas'); c.width = N; c.height = N; this.cv[key] = c; }
      if (ver !== undefined && c.ver === ver) return c;
      c.ver = ver;
      const cx = c.getContext('2d'), img = cx.createImageData(N, N), d = img.data;
      for (let k = 0; k < N * N; k++) {
        const [r, g, b] = map ? map(k) : (() => { const v = clamp(data[k], 0, 1) * 255; return [v, v, v]; })();
        d[k * 4] = r; d[k * 4 + 1] = g; d[k * 4 + 2] = b; d[k * 4 + 3] = 255;
      }
      cx.putImageData(img, 0, 0);
      return c;
    },

    draw(ctx, W, H) {
      const o = this.o, pad = 18, top = 100; // below the 2-line HUD
      const T = Math.floor(Math.min((W - pad * 3) / 2, (H - top - pad * 2 - 40) / 2));
      const x0 = (W - (T * 2 + pad)) / 2, tiles = [[x0, top], [x0 + T + pad, top], [x0, top + T + pad + 18], [x0 + T + pad, top + T + pad + 18]];
      this.specTile = { x: tiles[3][0], y: tiles[3][1], T };
      const label = (txt, [x, y]) => { ctx.fillStyle = '#c9cfda'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(txt, x, y - 5); };
      ctx.imageSmoothingEnabled = false;

      // version tags so unchanged tiles are not repainted
      this.ver = this.ver || new WeakMap();
      const tag = (arr) => { if (!this.ver.has(arr)) this.ver.set(arr, Math.random()); return this.ver.get(arr); };
      const resVer = `${tag(this.result)}:${this.tv ? this.tv.iter : 0}`;
      if (o.topLeft === 'clean') { ctx.drawImage(this.toCanvas('clean', this.clean, null, tag(this.clean)), tiles[0][0], tiles[0][1], T, T); label('Original (the truth)', tiles[0]); }
      else {
        const e = this.result, c = this.clean;
        ctx.drawImage(this.toCanvas('err', null, (k) => { const d = Math.min(1, Math.abs(clamp(e[k], 0, 1) - c[k]) * 4); return [d * 255, d * 120, 40 * d]; }, resVer + tag(this.clean)), tiles[0][0], tiles[0][1], T, T);
        label('Error |denoised − original| × 4', tiles[0]);
      }
      ctx.drawImage(this.toCanvas('noisy', this.noisy, null, tag(this.noisy)), tiles[1][0], tiles[1][1], T, T); label('Noisy', tiles[1]);
      ctx.drawImage(this.toCanvas('result', this.result, null, resVer), tiles[2][0], tiles[2][1], T, T); label('Denoised', tiles[2]);

      // spectrum with the mask overlaid
      if (o.specOf === 'result' && !this.spec.result) this.spec.result = this.spectrum(this.result);
      const sp = o.specOf === 'result' ? this.spec.result : this.spec.noisy;
      let mx = 0;
      for (let k = 0; k < sp.length; k++) if (k !== (N / 2) * N + N / 2) mx = Math.max(mx, sp[k]);
      ctx.drawImage(this.toCanvas('spec', null, (k) => {
        const x = k % N, y = (k / N) | 0, t = Math.min(1, sp[k] / mx) ** 1.6, m = o.freq ? this.mask(x - N / 2, y - N / 2) : 1;
        const v = t * 255;
        return m < 0.5 ? [v * 0.5 + 90 * (1 - m), v * 0.35, v * 0.35] : [v, v, v];
      }, `${tag(sp)}|${o.freq}|${o.lowpass}|${o.ideal}|${JSON.stringify(o.notches)}`), tiles[3][0], tiles[3][1], T, T);
      label(`Spectrum of ${o.specOf} (log |F|) · centre = smooth, edges = fine detail`, tiles[3]);
      // notch markers
      if (o.freq) {
        const sc = T / N;
        ctx.strokeStyle = '#ff8a8d'; ctx.lineWidth = 1.5;
        for (const [u, v] of o.notches) for (const s of [1, -1]) {
          ctx.beginPath(); ctx.arc(tiles[3][0] + (N / 2 + s * u + 0.5) * sc, tiles[3][1] + (N / 2 + s * v + 0.5) * sc, 7, 0, TAU); ctx.stroke();
        }
        if (o.lowpass < 128) {
          ctx.strokeStyle = 'rgba(91,155,255,0.9)'; ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.arc(tiles[3][0] + T / 2, tiles[3][1] + T / 2, o.lowpass * sc, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      ctx.imageSmoothingEnabled = true;
    },

    pointer(type, x, y) {
      if (type !== 'down' || !this.specTile) return;
      const { x: tx, y: ty, T } = this.specTile;
      if (x < tx || x > tx + T || y < ty || y > ty + T) return;
      const o = this.o, ku = Math.round(((x - tx) / T) * N - N / 2), kv = Math.round(((y - ty) / T) * N - N / 2);
      if (Math.hypot(ku, kv) < 3) return; // never notch the DC term (average brightness)
      const hit = o.notches.findIndex(([u, v]) => Math.hypot(u - ku, v - kv) < 5 || Math.hypot(u + ku, v + kv) < 5);
      if (hit >= 0) o.notches.splice(hit, 1); else o.notches.push([ku, kv]);
      o.freq = true;
      this.changed();
    },
  };

  sim._internals = { fft1, fft2, median, bilateral, tvState, tvStep, psnr, scene, N };
  (window.Sims = window.Sims || {}).image = sim;
})();
