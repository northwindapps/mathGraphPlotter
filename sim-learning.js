/* Machine-learning explorer: how weights move during training.
 *  1. Fit a line   – 2 weights (w, b): the data view and the loss surface side by side.
 *  2. Optimizer race – GD, momentum and Adam on tricky 2D loss surfaces.
 *  3. Tiny neural net – 2 → H → 1 classifier, its neurons, and a 2D slice of its loss landscape.
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const SEQ = UI.lut(UI.SEQ_STOPS);
  const TAU = Math.PI * 2;
  const COL = { a: '#3d8bfd', b: '#e09b14', fit: '#f5c451', gd: '#e5484d', mom: '#2fa36b', adam: '#f5c451', path: '#e8ebf1' };
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  function rng(seed) { // mulberry32: small, seedable
    return () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); };

  // Heat map of a scalar field on an nx×ny grid (row 0 = top), log-scaled, with contour bands.
  function heatCanvas(nx, ny, f) {
    const vals = new Float64Array(nx * ny);
    let mn = Infinity, mx = -Infinity;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = f(i, j); vals[j * nx + i] = v;
      if (Number.isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    }
    const eps = 1e-3 * (mx - mn + 1e-12), tr = (v) => Math.log(v - mn + eps), lo = tr(mn), hi = tr(mx);
    const cv = document.createElement('canvas'); cv.width = nx; cv.height = ny;
    const cx = cv.getContext('2d'), img = cx.createImageData(nx, ny), d = img.data;
    for (let k = 0; k < vals.length; k++) {
      const t = Number.isFinite(vals[k]) ? clamp((tr(vals[k]) - lo) / (hi - lo || 1), 0, 1) : 1;
      const idx = (t * 255) | 0, band = (t * 14) % 1 < 0.07 ? 0.7 : 1;
      d[k * 4] = SEQ[idx * 3] * band; d[k * 4 + 1] = SEQ[idx * 3 + 1] * band; d[k * 4 + 2] = SEQ[idx * 3 + 2] * band; d[k * 4 + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
    return { canvas: cv, min: mn, max: mx };
  }

  // A rectangle with a data-space mapping
  function box(x, y, w, h, x0, x1, y0, y1) {
    return {
      x, y, w, h, x0, x1, y0, y1,
      sx: (v) => x + ((v - x0) / (x1 - x0)) * w,
      sy: (v) => y + h - ((v - y0) / (y1 - y0)) * h,
      dx: (px) => x0 + ((px - x) / w) * (x1 - x0),
      dy: (py) => y0 + ((y + h - py) / h) * (y1 - y0),
      inside: (px, py) => px >= x && px <= x + w && py >= y && py <= y + h,
    };
  }

  function frameBox(ctx, b, title) {
    ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(b.x, b.y, b.w, b.h);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
    if (title) {
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(title, b.x, b.y - 6);
    }
  }
  function axisLabels(ctx, b, xl, yl) {
    ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(xl, b.x + b.w - 4, b.y + b.h + 4);
    ctx.save(); ctx.translate(b.x - 6, b.y + 4); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(yl, 0, 0); ctx.restore();
  }
  function polyline(ctx, pts, color, width) {
    if (pts.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.beginPath();
    pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }
  function dot(ctx, x, y, r, fill, stroke) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
  }
  function arrow(ctx, x0, y0, x1, y1, color) {
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
    ctx.moveTo(x1 - 8 * Math.cos(a - 0.45), y1 - 8 * Math.sin(a - 0.45)); ctx.lineTo(x1, y1); ctx.lineTo(x1 - 8 * Math.cos(a + 0.45), y1 - 8 * Math.sin(a + 0.45));
    ctx.stroke();
  }
  // simple loss-vs-step chart; series = [{ data, color }]
  function lossChart(ctx, b, series, title, logScale = true) {
    frameBox(ctx, b, title);
    let n = 2, lo = Infinity, hi = -Infinity;
    const tf = (v) => (logScale ? Math.log10(Math.max(v, 1e-9)) : v);
    for (const s of series) { n = Math.max(n, s.data.length); for (const v of s.data) if (Number.isFinite(v)) { lo = Math.min(lo, tf(v)); hi = Math.max(hi, tf(v)); } }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
    ctx.font = '10.5px system-ui'; ctx.fillStyle = '#7d8696'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let k = 0; k <= 3; k++) {
      const v = lo + ((hi - lo) * k) / 3, py = b.y + b.h - 14 - ((v - lo) / (hi - lo)) * (b.h - 24);
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.beginPath(); ctx.moveTo(b.x, py); ctx.lineTo(b.x + b.w, py); ctx.stroke();
      ctx.fillText(logScale ? (Math.pow(10, v) < 0.01 ? Math.pow(10, v).toExponential(0) : +Math.pow(10, v).toPrecision(2) + '') : v.toFixed(2), b.x + 4, py);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(`step ${n - 1}`, b.x + b.w - 4, b.y + b.h - 1);
    ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    for (const s of series) {
      const pts = [];
      const stride = Math.max(1, Math.floor(s.data.length / 600));
      for (let k = 0; k < s.data.length; k += stride) {
        const v = s.data[k];
        if (!Number.isFinite(v)) break;
        pts.push([b.x + 34 + (k / (n - 1)) * (b.w - 40), b.y + b.h - 14 - ((tf(v) - lo) / (hi - lo)) * (b.h - 24)]);
      }
      polyline(ctx, pts, s.color, 2);
    }
    ctx.restore();
  }

  // ================================================================== 1. Fit a line
  const LINE = {
    W: [-3, 3], B: [-2.5, 2.5],
    init(o) {
      const r = rng(7);
      o.pts = Array.from({ length: 14 }, () => { const x = -1 + 2 * r(); return [x, 0.8 * x + 0.35 + 0.25 * gauss(r)]; });
      this.restart(o, -2.2, 1.9);
      this.heat = null;
    },
    restart(o, w, b) { o.w = w; o.b = b; o.path = [[w, b]]; o.losses = [this.loss(o, w, b)]; o.diverged = false; },
    loss(o, w, b) { let s = 0; for (const [x, y] of o.pts) s += (w * x + b - y) ** 2; return s / Math.max(1, o.pts.length); },
    grad(o, w, b) {
      let gw = 0, gb = 0;
      for (const [x, y] of o.pts) { const e = w * x + b - y; gw += 2 * e * x; gb += 2 * e; }
      const n = Math.max(1, o.pts.length); return [gw / n, gb / n];
    },
    optimum(o) { // closed-form least squares
      const n = o.pts.length; if (n < 2) return null;
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (const [x, y] of o.pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
      const d = n * sxx - sx * sx; if (Math.abs(d) < 1e-9) return null;
      const w = (n * sxy - sx * sy) / d; return [w, (sy - w * sx) / n];
    },
    step(o) {
      if (o.diverged) return;
      const [gw, gb] = this.grad(o, o.w, o.b);
      o.w -= o.lr * gw; o.b -= o.lr * gb;
      const L = this.loss(o, o.w, o.b);
      if (!Number.isFinite(L) || L > 1e6 || Math.abs(o.w) > 1e4) { o.diverged = true; return; }
      o.path.push([o.w, o.b]); o.losses.push(L);
      if (o.path.length > 3000) { o.path.splice(1, 1); }
    },
    draw(ctx, W, H, sim) {
      const o = sim.o.line, wide = W > 760, pad = 40;
      const topH = H * 0.66;
      const side = Math.min((wide ? W / 2 : W) - pad * 1.6, topH - 70);
      const dBox = box(pad, 70, side, side, -1.3, 1.3, -1.8, 1.8);
      const lBox = wide ? box(W / 2 + pad * 0.4, 70, side, side, this.W[0], this.W[1], this.B[0], this.B[1])
        : box(pad, 70 + side + 40, side, side * 0.8, this.W[0], this.W[1], this.B[0], this.B[1]);
      sim.regions = { dBox, lBox };
      // ---- data view
      frameBox(ctx, dBox, 'Data space: the line y = w·x + b');
      ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(dBox.sx(0), dBox.y); ctx.lineTo(dBox.sx(0), dBox.y + dBox.h); ctx.moveTo(dBox.x, dBox.sy(0)); ctx.lineTo(dBox.x + dBox.w, dBox.sy(0)); ctx.stroke();
      const opt = this.optimum(o);
      ctx.save(); ctx.beginPath(); ctx.rect(dBox.x, dBox.y, dBox.w, dBox.h); ctx.clip();
      if (opt) {
        ctx.setLineDash([5, 5]); polyline(ctx, [[dBox.sx(-1.3), dBox.sy(opt[0] * -1.3 + opt[1])], [dBox.sx(1.3), dBox.sy(opt[0] * 1.3 + opt[1])]], 'rgba(255,255,255,0.35)', 1.5); ctx.setLineDash([]);
      }
      for (const [x, y] of o.pts) polyline(ctx, [[dBox.sx(x), dBox.sy(y)], [dBox.sx(x), dBox.sy(o.w * x + o.b)]], 'rgba(229,72,77,0.7)', 1.5); // residuals
      polyline(ctx, [[dBox.sx(-1.3), dBox.sy(o.w * -1.3 + o.b)], [dBox.sx(1.3), dBox.sy(o.w * 1.3 + o.b)]], COL.fit, 3);
      ctx.restore();
      for (const [x, y] of o.pts) dot(ctx, dBox.sx(x), dBox.sy(y), 5.5, '#e8ebf1', '#0b0e14');
      ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('red: errors · dashed: best possible line', dBox.x + 6, dBox.y + 6);
      // ---- loss surface over (w, b)
      if (!this.heat || this.heat.n !== o.pts.length || this.heat.sig !== this.sig(o)) {
        const nx = 160, ny = Math.round((160 * lBox.h) / lBox.w);
        this.heat = { ...heatCanvas(nx, ny, (i, j) => this.loss(o, this.W[0] + ((i + 0.5) / nx) * (this.W[1] - this.W[0]), this.B[1] - ((j + 0.5) / ny) * (this.B[1] - this.B[0]))), n: o.pts.length, sig: this.sig(o) };
      }
      ctx.imageSmoothingEnabled = true; ctx.drawImage(this.heat.canvas, lBox.x, lBox.y, lBox.w, lBox.h);
      frameBox(ctx, lBox, 'Weight space: loss L(w, b), darker = lower');
      axisLabels(ctx, lBox, 'w (slope) →', 'b (intercept) →');
      ctx.save(); ctx.beginPath(); ctx.rect(lBox.x, lBox.y, lBox.w, lBox.h); ctx.clip();
      if (opt) { ctx.fillStyle = '#ffffff'; ctx.font = '18px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', lBox.sx(opt[0]), lBox.sy(opt[1])); }
      polyline(ctx, o.path.map(([w, b]) => [lBox.sx(w), lBox.sy(b)]), 'rgba(255,255,255,0.85)', 1.6);
      for (const [w, b] of o.path.slice(-40)) dot(ctx, lBox.sx(w), lBox.sy(b), 2, 'rgba(255,255,255,0.8)');
      const [gw, gb] = this.grad(o, o.w, o.b), gl = Math.hypot(gw, gb);
      if (gl > 1e-6 && !o.diverged) {
        const k = Math.min(60, 25 + gl * 8) / gl;
        arrow(ctx, lBox.sx(o.w), lBox.sy(o.b), lBox.sx(o.w) - gw * k, lBox.sy(o.b) + gb * k, '#7cf0a8');
      }
      dot(ctx, lBox.sx(o.w), lBox.sy(o.b), 7, COL.fit, '#0b0e14');
      ctx.restore();
      ctx.fillStyle = '#7cf0a8'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('green arrow: −∇L, the downhill direction · ★ best fit · click to restart from there', lBox.x + 6, lBox.y + lBox.h + 18);
      // ---- loss curve
      const cy = wide ? 70 + side + 46 : lBox.y + lBox.h + 46;
      if (H - cy - 30 > 50) lossChart(ctx, { x: pad, y: cy, w: W - pad * 2, h: H - cy - 30 }, [{ data: o.losses, color: COL.fit }], 'Loss L = mean squared error, per step (log scale)');
    },
    sig(o) { return o.pts.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(';'); },
    pointer(type, x, y, sim) {
      const o = sim.o.line, { dBox, lBox } = sim.regions || {};
      if (!dBox) return;
      if (type === 'down') {
        if (lBox.inside(x, y)) { this.restart(o, lBox.dx(x), lBox.dy(y)); return; }
        if (dBox.inside(x, y)) {
          const k = o.pts.findIndex(([px, py]) => Math.hypot(dBox.sx(px) - x, dBox.sy(py) - y) < 12);
          if (k >= 0) sim.drag = k;
          else { o.pts.push([dBox.dx(x), dBox.dy(y)]); sim.drag = o.pts.length - 1; }
          o.losses = [this.loss(o, o.w, o.b)]; o.path = [[o.w, o.b]];
        }
      } else if (type === 'move' && sim.drag != null) {
        o.pts[sim.drag] = [dBox.dx(x), dBox.dy(y)];
      } else if (type === 'up' || type === 'leave') {
        if (sim.drag != null && !dBox.inside(x, y) && o.pts.length > 2) o.pts.splice(sim.drag, 1);
        if (sim.drag != null) { o.losses = [this.loss(o, o.w, o.b)]; o.path = [[o.w, o.b]]; o.diverged = false; }
        sim.drag = null;
      }
    },
  };

  // ================================================================== 2. Optimizer race
  const SURFACES = {
    valley: {
      label: 'Narrow valley', X: [-4, 4], Y: [-2, 2], start: [-3.6, 1.5], lr: 0.95, lrAdam: 0.1,
      f: (x, y) => 0.01 * x * x + y * y, // 100× steeper across than along
      about: 'A long, narrow valley: very steep across, almost flat along. Plain GD zig-zags across the walls while crawling along the floor. ' +
        'Momentum builds up speed along the floor; Adam rescales each direction separately.',
    },
    rosen: {
      label: 'Banana (Rosenbrock)', logHeight: true, X: [-2, 2], Y: [-1, 3], start: [-1.6, 2.6], lr: 0.0018, lrAdam: 0.05,
      f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
      about: 'A curved, flat-bottomed valley: a classic optimizer stress test. Finding the valley is easy; following its bend to the minimum at (1, 1) is hard.',
    },
    saddle: {
      label: 'Saddle point', X: [-2.5, 2.5], Y: [-2.5, 2.5], start: [-2.2, 0.001], lr: 0.02, lrAdam: 0.05,
      f: (x, y) => x * x - y * y + 0.25 * y ** 4,
      about: 'Downhill in x, but at the centre the slope in y is almost zero: a saddle. In high dimensions saddles are far more common than local minima, ' +
        'and they are where plain GD wastes the most time. Adam boosts the tiny gradient and escapes first.',
    },
    wells: {
      label: 'Many minima', X: [-3, 3], Y: [-3, 3], start: [0, 2.8], lr: 0.15, lrAdam: 0.05,
      f: (x, y) => 0.06 * (x * x + y * y) - 1.2 * Math.exp(-((x - 1.4) ** 2 + (y - 1.2) ** 2) / 0.5)
        - 0.9 * Math.exp(-((x + 1.3) ** 2 + (y + 0.6) ** 2) / 0.4) - 1.6 * Math.exp(-((x - 0.4) ** 2 + (y + 1.9) ** 2) / 0.7)
        + 0.12 * Math.sin(2.2 * x) * Math.cos(2.2 * y),
      about: 'Bumpy terrain with several valleys. Where you end up depends on where you start (click to try!) and on momentum, which can roll over small bumps.',
    },
  };
  const OPTS = [
    { key: 'gd', label: 'Gradient descent', color: COL.gd },
    { key: 'mom', label: 'Momentum', color: COL.mom },
    { key: 'adam', label: 'Adam', color: COL.adam },
  ];
  const RACE = {
    grad(S, x, y) { const h = 1e-5; return [(S.f(x + h, y) - S.f(x - h, y)) / (2 * h), (S.f(x, y + h) - S.f(x, y - h)) / (2 * h)]; },
    restart(o, x, y) {
      const S = SURFACES[o.surface];
      [x, y] = x == null ? S.start : [x, y];
      o.runs = Object.fromEntries(OPTS.map(({ key }) => [key, { p: [x, y], v: [0, 0], m: [0, 0], s: [0, 0], t: 0, path: [[x, y]], loss: [S.f(x, y)], dead: false }]));
      o.iter = 0;
    },
    step(o) {
      const S = SURFACES[o.surface];
      for (const { key } of OPTS) {
        const r = o.runs[key];
        if (!o.enabled[key] || r.dead) continue;
        const g = this.grad(S, r.p[0], r.p[1]);
        if (key === 'gd') { r.p[0] -= o.lr * g[0]; r.p[1] -= o.lr * g[1]; }
        else if (key === 'mom') {
          for (let k = 0; k < 2; k++) { r.v[k] = o.beta * r.v[k] - o.lr * g[k]; r.p[k] += r.v[k]; }
        } else {
          r.t++;
          for (let k = 0; k < 2; k++) {
            r.m[k] = 0.9 * r.m[k] + 0.1 * g[k];
            r.s[k] = 0.999 * r.s[k] + 0.001 * g[k] * g[k];
            const mh = r.m[k] / (1 - 0.9 ** r.t), sh = r.s[k] / (1 - 0.999 ** r.t);
            r.p[k] -= (o.lrAdam * mh) / (Math.sqrt(sh) + 1e-8);
          }
        }
        const L = S.f(r.p[0], r.p[1]);
        if (!Number.isFinite(L) || Math.abs(r.p[0]) > 50 || Math.abs(r.p[1]) > 50) { r.dead = true; continue; }
        r.path.push([r.p[0], r.p[1]]); r.loss.push(L);
        if (r.path.length > 4000) { r.path.splice(1, 1); }
      }
      o.iter++;
    },
    draw(ctx, W, H, sim) {
      const o = sim.o.race, S = SURFACES[o.surface], wide = W > 760;
      const mapW = wide ? W * 0.6 - 50 : W - 70, mapH = wide ? H - 136 : H * 0.55;
      const aspect = (S.X[1] - S.X[0]) / (S.Y[1] - S.Y[0]);
      let bw = mapW, bh = bw / aspect;
      if (bh > mapH) { bh = mapH; bw = bh * aspect; }
      const mBox = box(46, 100, bw, bh, S.X[0], S.X[1], S.Y[0], S.Y[1]); // below the (2-line) HUD
      sim.regions = { mBox };
      if (!this.heat || this.heat.key !== o.surface || this.heat.w !== Math.round(bw)) {
        const nx = 180, ny = Math.round(180 / aspect);
        const h = heatCanvas(nx, ny, (i, j) => S.f(S.X[0] + ((i + 0.5) / nx) * (S.X[1] - S.X[0]), S.Y[1] - ((j + 0.5) / ny) * (S.Y[1] - S.Y[0])));
        this.heat = { ...h, key: o.surface, w: Math.round(bw) };
        S.fmin = h.min;
      }
      if (o.view === '3d') {
        sim.regions.tBox = { x: 46, y: 100, w: mapW, h: mapH };
        this.terrain(ctx, sim.regions.tBox, o, S);
      } else {
        ctx.imageSmoothingEnabled = true; ctx.drawImage(this.heat.canvas, mBox.x, mBox.y, mBox.w, mBox.h);
        frameBox(ctx, mBox, `Loss surface: ${S.label} (darker = lower) · click to choose the start`);
        axisLabels(ctx, mBox, 'weight 1 →', 'weight 2 →');
        ctx.save(); ctx.beginPath(); ctx.rect(mBox.x, mBox.y, mBox.w, mBox.h); ctx.clip();
        for (const { key, color } of OPTS) {
          const r = o.runs[key];
          if (!o.enabled[key]) continue;
          polyline(ctx, r.path.map(([x, y]) => [mBox.sx(x), mBox.sy(y)]), color, 2);
          dot(ctx, mBox.sx(r.p[0]), mBox.sy(r.p[1]), 6.5, color, '#0b0e14');
        }
        const s0 = o.runs.gd.path[0];
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(mBox.sx(s0[0]), mBox.sy(s0[1]), 8, 0, TAU); ctx.stroke();
        ctx.restore();
      }
      // loss chart
      const cb = wide ? { x: W * 0.6 + 10, y: 100, w: W * 0.4 - 30, h: H - 136 } : { x: 46, y: mBox.y + mBox.h + 40, w: W - 70, h: H - mBox.y - mBox.h - 70 };
      if (cb.h > 60) {
        const fmin = S.fmin - 1e-6;
        lossChart(ctx, cb, OPTS.filter(({ key }) => o.enabled[key]).map(({ key, color }) => ({ data: o.runs[key].loss.map((v) => v - fmin), color })),
          'Loss above the lowest point (log scale)');
        ctx.font = '11px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        let ly = cb.y + 8;
        for (const { key, label, color } of OPTS) {
          if (!o.enabled[key]) continue;
          ctx.fillStyle = color; ctx.fillText(`${label}${o.runs[key].dead ? ' (diverged!)' : ''}`, cb.x + cb.w - 8, ly); ly += 16;
        }
      }
    },
    // 3D view: the loss surface as a shaded landscape with the optimizers rolling on it
    terrain(ctx, b, o, S) {
      const ar = (S.Y[1] - S.Y[0]) / (S.X[1] - S.X[0]), nx = 46, ny = Math.max(16, Math.round(46 * ar));
      if (!this.tgrid || this.tgrid.key !== o.surface) {
        const F = [];
        let mn = Infinity, mx = -Infinity;
        for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
          const v = S.f(S.X[0] + (i / nx) * (S.X[1] - S.X[0]), S.Y[0] + (j / ny) * (S.Y[1] - S.Y[0]));
          F.push(v); mn = Math.min(mn, v); mx = Math.max(mx, v);
        }
        this.tgrid = { key: o.surface, F, mn, mx };
      }
      const { F, mn, mx } = this.tgrid, logH = !!S.logHeight, eps = 1e-3 * (mx - mn);
      const lo = logH ? Math.log(eps) : 0, hi = logH ? Math.log(mx - mn + eps) : mx - mn;
      const Zn = (f) => clamp(((logH ? Math.log(Math.max(f - mn, 0) + eps) : f - mn) - lo) / (hi - lo || 1), 0, 1.15); // 0…1 height
      const cam = o.cam, cyw = Math.cos(cam.yaw), syw = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const Sc = Math.min(b.w, b.h * 1.25) * 0.4, ox = b.x + b.w / 2, oy = b.y + b.h * 0.56;
      const world = (x, y, f) => [((clamp(x, S.X[0], S.X[1]) - S.X[0]) / (S.X[1] - S.X[0])) * 2 - 1, (((clamp(y, S.Y[0], S.Y[1]) - S.Y[0]) / (S.Y[1] - S.Y[0])) * 2 - 1) * ar, Zn(f) * 0.9 - 0.45];
      const proj = ([X, Y, Z]) => { const x = X * cyw - Y * syw, d = X * syw + Y * cyw; return [ox + x * Sc, oy - (Z * cp + d * sp) * Sc, d * cp - Z * sp]; };
      // mesh, painter's algorithm, with simple light shading so the shape reads as 3D
      const Wp = [];
      for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) {
        const x = S.X[0] + (i / nx) * (S.X[1] - S.X[0]), y = S.Y[0] + (j / ny) * (S.Y[1] - S.Y[0]);
        Wp.push(world(x, y, F[j * (nx + 1) + i]));
      }
      const L = [-0.45, -0.55, 0.7], Ln = Math.hypot(...L), quads = [];
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const k = j * (nx + 1) + i, a = Wp[k], bq = Wp[k + 1], c = Wp[k + nx + 2], d = Wp[k + nx + 1];
        const ux = [bq[0] - a[0], bq[1] - a[1], bq[2] - a[2]], uy = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        const nrm = [ux[1] * uy[2] - ux[2] * uy[1], ux[2] * uy[0] - ux[0] * uy[2], ux[0] * uy[1] - ux[1] * uy[0]], nl = Math.hypot(...nrm) || 1;
        const shade = 0.55 + 0.45 * Math.max(0, (nrm[0] * L[0] + nrm[1] * L[1] + nrm[2] * L[2]) / (nl * Ln));
        const pts = [a, bq, c, d].map(proj);
        quads.push({ pts, depth: (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4, t: (a[2] + bq[2] + c[2] + d[2]) / 4 / 0.9 + 0.5, shade });
      }
      quads.sort((p, q) => q.depth - p.depth);
      ctx.lineWidth = 0.5;
      for (const q of quads) {
        const idx = (clamp(q.t, 0, 1) * 255) | 0;
        ctx.fillStyle = `rgb(${(SEQ[idx * 3] * q.shade) | 0},${(SEQ[idx * 3 + 1] * q.shade) | 0},${(SEQ[idx * 3 + 2] * q.shade) | 0})`;
        ctx.strokeStyle = 'rgba(0,0,0,0.25)';
        ctx.beginPath(); q.pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      // trails and balls, lifted slightly above the surface
      const lift = (x, y) => proj((() => { const w = world(x, y, S.f(x, y)); w[2] += 0.015; return w; })());
      for (const { key, color } of OPTS) {
        const r = o.runs[key];
        if (!o.enabled[key]) continue;
        const stride = Math.max(1, Math.floor(r.path.length / 800));
        const pts = []; for (let k = 0; k < r.path.length; k += stride) pts.push(lift(...r.path[k]));
        pts.push(lift(...r.p));
        polyline(ctx, pts.map(([x, y]) => [x, y]), color, 2.2);
      }
      const s0 = lift(...o.runs.gd.path[0]);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(s0[0], s0[1], 8, 0, TAU); ctx.stroke();
      for (const { key, color } of OPTS) {
        const r = o.runs[key];
        if (!o.enabled[key]) continue;
        const [x, y] = lift(...r.p), g = ctx.createRadialGradient(x - 3, y - 3, 1, x, y, 9);
        g.addColorStop(0, '#ffffff'); g.addColorStop(0.35, color); g.addColorStop(1, '#0b0e14');
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 8, 0, TAU); ctx.fill();
      }
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`3D terrain: ${S.label} · height = loss${logH ? ' (log-scaled, the walls are steep)' : ''} · drag to rotate`, b.x, b.y - 6);
    },
    pointer(type, x, y, sim) {
      const o = sim.o.race, { mBox } = sim.regions || {};
      if (o.view === '3d') { // drag rotates the terrain
        if (type === 'down') sim.rotRace = { x, y, yaw: o.cam.yaw, pitch: o.cam.pitch };
        else if (type === 'move' && sim.rotRace) { o.cam.yaw = sim.rotRace.yaw - (x - sim.rotRace.x) * 0.008; o.cam.pitch = clamp(sim.rotRace.pitch + (y - sim.rotRace.y) * 0.008, 0.1, 1.5); }
        else if (type === 'up' || type === 'leave') sim.rotRace = null;
        return;
      }
      if (type === 'down' && mBox && mBox.inside(x, y)) this.restart(o, mBox.dx(x), mBox.dy(y));
    },
  };

  // ================================================================== 3. Tiny neural network
  const DATASETS = {
    circle: { label: 'Circle', gen: (r) => { const a = TAU * r(), inner = r() < 0.5, rad = inner ? 0.5 * Math.sqrt(r()) : 0.8 + 0.4 * r(); return [rad * Math.cos(a), rad * Math.sin(a), inner ? 0 : 1]; } },
    xor: { label: 'XOR', gen: (r) => { const x = 2.4 * r() - 1.2, y = 2.4 * r() - 1.2; return [x + 0.06 * Math.sign(x), y + 0.06 * Math.sign(y), x * y > 0 ? 0 : 1]; } },
    moons: { label: 'Two moons', gen: (r) => { const c = r() < 0.5 ? 0 : 1, a = Math.PI * r(), n = () => 0.09 * gauss(r); return c ? [0.9 * Math.cos(a) - 0.45 + n(), 0.9 * Math.sin(a) - 0.25 + n(), 0] : [0.9 - 0.9 * Math.cos(a) - 0.45 + n(), -0.9 * Math.sin(a) + 0.35 + n(), 1]; } },
    spiral: { label: 'Spiral (hard)', gen: (r) => { const c = r() < 0.5 ? 0 : 1, t = r(), a = 3.2 * Math.PI * t + c * Math.PI, rad = 0.1 + 1.05 * t; return [rad * Math.cos(a) + 0.04 * gauss(r), rad * Math.sin(a) + 0.04 * gauss(r), c]; } },
  };

  const NET = {
    R: 1.5,
    nParams: (H) => 4 * H + 1,
    init(o, seed) {
      const r = rng(seed || 1 + Math.floor(Math.random() * 1e6)), H = o.H, th = new Float64Array(this.nParams(H));
      for (let k = 0; k < 2 * H; k++) th[k] = 1.2 * gauss(r);        // W1
      for (let k = 0; k < H; k++) th[2 * H + k] = 0.3 * gauss(r);    // b1
      for (let k = 0; k < H; k++) th[3 * H + k] = 0.6 * gauss(r);    // W2
      o.theta = th; o.opt = { v: new Float64Array(th.length), m: new Float64Array(th.length), s: new Float64Array(th.length), t: 0 };
      o.losses = [this.lossGrad(o, th)[0]]; o.accs = []; o.traj = [Float64Array.from(th)]; o.steps = 0; o.slice = null;
    },
    data(o, key, seed = 3) {
      const r = rng(seed); o.dataset = key;
      o.pts = Array.from({ length: key === 'spiral' ? 260 : 200 }, () => DATASETS[key].gen(r));
    },
    forward(th, H, x, y, hid) {
      let z = th[4 * H];
      for (let j = 0; j < H; j++) {
        const h = Math.tanh(th[2 * j] * x + th[2 * j + 1] * y + th[2 * H + j]);
        if (hid) hid[j] = h;
        z += th[3 * H + j] * h;
      }
      return 1 / (1 + Math.exp(-z));
    },
    lossGrad(o, th, grad) { // binary cross-entropy + backprop
      const H = o.H, hid = new Float64Array(H), N = o.pts.length;
      let L = 0, correct = 0;
      if (grad) grad.fill(0);
      for (const [x, y, c] of o.pts) {
        const p = this.forward(th, H, x, y, hid), pc = clamp(p, 1e-7, 1 - 1e-7);
        L -= c ? Math.log(pc) : Math.log(1 - pc);
        if ((p > 0.5) === (c === 1)) correct++;
        if (!grad) continue;
        const dz = (p - c) / N;
        grad[4 * H] += dz;
        for (let j = 0; j < H; j++) {
          grad[3 * H + j] += dz * hid[j];
          const dpre = dz * th[3 * H + j] * (1 - hid[j] * hid[j]);
          grad[2 * j] += dpre * x; grad[2 * j + 1] += dpre * y; grad[2 * H + j] += dpre;
        }
      }
      return [L / N, correct / N];
    },
    step(o) {
      const th = o.theta, g = new Float64Array(th.length), st = o.opt;
      const [L, acc] = this.lossGrad(o, th, g);
      if (o.optimizer === 'gd') for (let k = 0; k < th.length; k++) th[k] -= o.lr * g[k];
      else if (o.optimizer === 'mom') for (let k = 0; k < th.length; k++) { st.v[k] = 0.9 * st.v[k] - o.lr * g[k]; th[k] += st.v[k]; }
      else {
        st.t++;
        for (let k = 0; k < th.length; k++) {
          st.m[k] = 0.9 * st.m[k] + 0.1 * g[k]; st.s[k] = 0.999 * st.s[k] + 0.001 * g[k] * g[k];
          th[k] -= (o.lr * st.m[k] / (1 - 0.9 ** st.t)) / (Math.sqrt(st.s[k] / (1 - 0.999 ** st.t)) + 1e-8);
        }
      }
      o.steps++;
      o.losses.push(L); o.accs.push(acc);
      if (o.losses.length > 5000) { o.losses.splice(1, 1); o.accs.splice(0, 1); }
      if (o.steps % 4 === 0) { o.traj.push(Float64Array.from(th)); if (o.traj.length > 800) o.traj.splice(1, 1); }
    },
    snapshot(o) { // 2D slice of the loss landscape around the current weights
      const th = o.theta, n = th.length, r = rng(11 + o.steps);
      const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
      const dir = () => { const d = Float64Array.from({ length: n }, () => gauss(r)); const k = norm(th) / norm(d); return d.map((x) => x * k); };
      const d1 = dir(), d2 = dir(), G = 41, span = 1, vals = new Float64Array(G * G), tmp = new Float64Array(n);
      const h = heatCanvas(G, G, (i, j) => {
        const a = -span + (2 * span * i) / (G - 1), b = span - (2 * span * j) / (G - 1);
        for (let k = 0; k < n; k++) tmp[k] = th[k] + a * d1[k] + b * d2[k];
        return this.lossGrad(o, tmp)[0];
      });
      void vals;
      // project the training path onto the plane (least squares in the span of d1, d2)
      const dot2 = (u, v) => { let s = 0; for (let k = 0; k < n; k++) s += u[k] * v[k]; return s; };
      const g11 = dot2(d1, d1), g12 = dot2(d1, d2), g22 = dot2(d2, d2), det = g11 * g22 - g12 * g12;
      const path = o.traj.concat([th]).map((t) => {
        const rv = t.map((x, k) => x - th[k]), p1 = dot2(rv, d1), p2 = dot2(rv, d2);
        return [(g22 * p1 - g12 * p2) / det, (g11 * p2 - g12 * p1) / det];
      });
      o.slice = { canvas: h.canvas, path, span, dims: n };
    },
    neuronThumbs(o) {
      const H = o.H, th = o.theta, G = 22, R = this.R, thumbs = [];
      for (let j = 0; j <= H; j++) {
        const cv = document.createElement('canvas'); cv.width = G; cv.height = G;
        const cx = cv.getContext('2d'), img = cx.createImageData(G, G), d = img.data;
        for (let b = 0; b < G; b++) for (let a = 0; a < G; a++) {
          const x = -R + (2 * R * (a + 0.5)) / G, y = R - (2 * R * (b + 0.5)) / G;
          const v = j < H ? Math.tanh(th[2 * j] * x + th[2 * j + 1] * y + th[2 * H + j]) : 2 * this.forward(th, H, x, y) - 1;
          const [r, g, bl] = v > 0 ? [224, 155, 20] : [61, 139, 253], k = (b * G + a) * 4, t = Math.abs(v);
          d[k] = 11 + (r - 11) * t; d[k + 1] = 14 + (g - 14) * t; d[k + 2] = 20 + (bl - 20) * t; d[k + 3] = 255;
        }
        cx.putImageData(img, 0, 0); thumbs.push(cv);
      }
      return thumbs;
    },
    draw(ctx, W, H, sim) {
      const o = sim.o.net, wide = W > 820, R = this.R;
      const side = wide ? Math.min(W * 0.46, H - 136) : Math.min(W - 60, H * 0.46);
      const dBox = box(40, 100, side, side, -R, R, -R, R);
      sim.regions = { dBox };
      // decision regions (cached briefly)
      if (!this.region || sim.frameNo % 3 === 0) {
        const G = 64, cv = this.region || document.createElement('canvas'); cv.width = G; cv.height = G;
        const cx = cv.getContext('2d'), img = cx.createImageData(G, G), d = img.data;
        for (let b = 0; b < G; b++) for (let a = 0; a < G; a++) {
          const p = this.forward(o.theta, o.H, -R + (2 * R * (a + 0.5)) / G, R - (2 * R * (b + 0.5)) / G);
          const t = Math.abs(p - 0.5) * 2, [r, g, bl] = p > 0.5 ? [224, 155, 20] : [61, 139, 253], k = (b * G + a) * 4;
          const edge = t < 0.08 ? 1.6 : 1;
          d[k] = Math.min(255, (11 + (r - 11) * 0.45 * t) * edge); d[k + 1] = Math.min(255, (14 + (g - 14) * 0.45 * t) * edge); d[k + 2] = Math.min(255, (20 + (bl - 20) * 0.45 * t) * edge); d[k + 3] = 255;
        }
        cx.putImageData(img, 0, 0); this.region = cv;
      }
      ctx.imageSmoothingEnabled = true; ctx.drawImage(this.region, dBox.x, dBox.y, dBox.w, dBox.h);
      frameBox(ctx, dBox, 'What the network predicts (background) vs. the data (dots)');
      for (const [x, y, c] of o.pts) dot(ctx, dBox.sx(x), dBox.sy(y), 3.8, c ? COL.b : COL.a, '#0b0e14');

      // right column: network diagram, landscape slice, loss chart
      const rx = wide ? dBox.x + dBox.w + 40 : 40, ry = wide ? 100 : dBox.y + dBox.h + 40, rw = wide ? W - rx - 24 : W - 64, rh = wide ? H - 136 : H - ry - 30;
      const netH = Math.max(150, rh * 0.46);
      this.drawNet(ctx, { x: rx, y: ry, w: rw, h: netH }, o, sim);
      const by = ry + netH + 50, bh = rh - netH - 50;
      if (bh > 60) {
        const sw = Math.min(bh, rw * 0.5 - 10);
        const sBox = box(rx, by, sw, sw, -1, 1, -1, 1);
        if (o.slice) {
          ctx.drawImage(o.slice.canvas, sBox.x, sBox.y, sBox.w, sBox.h);
          frameBox(ctx, sBox, `A 2D slice of the ${o.slice.dims}-D loss landscape`);
          ctx.save(); ctx.beginPath(); ctx.rect(sBox.x, sBox.y, sBox.w, sBox.h); ctx.clip();
          polyline(ctx, o.slice.path.map(([a, b]) => [sBox.sx(a), sBox.sy(b)]), 'rgba(255,255,255,0.9)', 1.6);
          const s0 = o.slice.path[0];
          ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sBox.sx(s0[0]), sBox.sy(s0[1]), 5, 0, TAU); ctx.stroke();
          dot(ctx, sBox.sx(0), sBox.sy(0), 5, COL.fit, '#0b0e14');
          ctx.restore();
        } else {
          frameBox(ctx, sBox, '2D slice of the loss landscape');
          ctx.fillStyle = '#9aa3b2'; ctx.font = '12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText('Train for a while, then press', sBox.x + sBox.w / 2, sBox.y + sBox.h / 2 - 9);
          ctx.fillText('“📸 Slice the landscape”', sBox.x + sBox.w / 2, sBox.y + sBox.h / 2 + 9);
        }
        lossChart(ctx, { x: rx + sw + 20, y: by, w: rw - sw - 20, h: sw }, [{ data: o.losses, color: COL.fit }], 'Training loss (cross-entropy)');
      }
    },
    drawNet(ctx, b, o, sim) {
      const H = o.H, th = o.theta, T = 34;
      if (!this.thumbs || sim.frameNo % 6 === 0 || this.thumbs.length !== H + 1) this.thumbs = this.neuronThumbs(o);
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`The network: 2 inputs → ${H} hidden neurons (tanh) → 1 output · ${this.nParams(H)} weights`, b.x, b.y - 6);
      const xIn = b.x + 30, xHid = b.x + b.w * 0.5, xOut = b.x + b.w - 40;
      const yIn = [b.y + b.h * 0.35, b.y + b.h * 0.65];
      const gap = Math.min(T + 10, (b.h - 10) / H), yHid = Array.from({ length: H }, (_, j) => b.y + b.h / 2 + (j - (H - 1) / 2) * gap);
      const edge = (x0, y0, x1, y1, w) => {
        ctx.strokeStyle = w > 0 ? 'rgba(224,155,20,0.85)' : 'rgba(61,139,253,0.85)';
        ctx.lineWidth = clamp(Math.abs(w) * 1.6, 0.4, 7);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      };
      for (let j = 0; j < H; j++) {
        edge(xIn + 12, yIn[0], xHid - T / 2, yHid[j], th[2 * j]);
        edge(xIn + 12, yIn[1], xHid - T / 2, yHid[j], th[2 * j + 1]);
        edge(xHid + T / 2, yHid[j], xOut - T / 2, b.y + b.h / 2, th[3 * H + j]);
      }
      ['x₁', 'x₂'].forEach((lab, k) => {
        dot(ctx, xIn, yIn[k], 12, '#272e3b', '#8e98aa');
        ctx.fillStyle = '#e8ebf1'; ctx.font = '12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(lab, xIn, yIn[k]);
      });
      const sz = Math.min(T, gap - 4);
      ctx.imageSmoothingEnabled = false;
      for (let j = 0; j < H; j++) {
        ctx.drawImage(this.thumbs[j], xHid - sz / 2, yHid[j] - sz / 2, sz, sz);
        ctx.strokeStyle = '#8e98aa'; ctx.lineWidth = 1; ctx.strokeRect(xHid - sz / 2, yHid[j] - sz / 2, sz, sz);
      }
      ctx.drawImage(this.thumbs[H], xOut - T / 2 - 6, b.y + b.h / 2 - T / 2 - 6, T + 12, T + 12);
      ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 1.5; ctx.strokeRect(xOut - T / 2 - 6, b.y + b.h / 2 - T / 2 - 6, T + 12, T + 12);
      ctx.imageSmoothingEnabled = true;
      ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText('output', xOut, b.y + b.h / 2 + T / 2 + 10);
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('edge thickness = |weight| · orange +, blue −', b.x, b.y + b.h + 16);
    },
    pointer(type, x, y, sim) {
      const o = sim.o.net, { dBox } = sim.regions || {};
      if (type === 'down' && dBox && dBox.inside(x, y)) { o.pts.push([dBox.dx(x), dBox.dy(y), o.addClass]); }
    },
  };

  // ================================================================== 4. Lagrange multipliers
  // Optimize f(x, y) while staying on the path g(x, y) = c.  Optimum: ∇f = λ∇g,  and λ = d(best f)/dc.
  const LOSS_RL = (x, y) => (x - 2.4) ** 2 + 3 * (y - 0.1) ** 2 + 1.6 * (x - 2.4) * (y - 0.1); // shared by ridge & lasso
  const LAG_PROBS = {
    hill: {
      label: 'Hill & path', kind: 'max', xl: 'x →', yl: 'y →', R: [-2.5, 2.5, -2.5, 2.5], c: [1.3, 0.3, 2.1], cLabel: 'path size c', s0: 0.62, closed: true,
      f: (x, y) => 1.2 * Math.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / 0.6) + 0.7 * Math.exp(-((x + 0.9) ** 2 + (y + 0.7) ** 2) / 0.4),
      g: (x, y) => Math.hypot(x / 1.15, y / 0.8),
      curve: (s, c) => [c * 1.15 * Math.cos(TAU * s), c * 0.8 * Math.sin(TAU * s)],
      about: 'You must walk on an elliptical path (the white loop) across a landscape with two hills. Where on the path are you highest? ' +
        'Not at a hilltop: at the place where the path just <b>touches</b> a contour line.',
    },
    fence: {
      label: 'Fence: max area', kind: 'max', xl: 'width x →', yl: 'height y →', R: [0, 5, 0, 5], c: [6, 2, 9.6], cLabel: 'fence length P', s0: 0.15, closed: false,
      f: (x, y) => x * y, g: (x, y) => 2 * x + 2 * y,
      curve: (s, c) => { const x = 0.02 + s * (c / 2 - 0.04); return [x, c / 2 - x]; },
      about: 'A rectangle with sides x and y, and fence of length P: maximize the area xy subject to 2x + 2y = P. ' +
        'The answer is a square, x = y = P/4, and λ = P/8 is how much area one more metre of fence buys.',
    },
    ridge: {
      label: 'Ridge (L2)', kind: 'min', xl: 'weight w₁ →', yl: 'weight w₂ →', R: [-1.5, 3.5, -2.5, 2.5], c: [1.2, 0.2, 2.4], cLabel: 'weight budget r', s0: 0.2, closed: true,
      f: LOSS_RL, g: (x, y) => Math.hypot(x, y),
      curve: (s, c) => [c * Math.cos(TAU * s), c * Math.sin(TAU * s)],
      about: 'Two model weights (w₁, w₂). The contours are the training loss, whose best point is outside. Ridge regression limits the weights to a ' +
        '<b>circle</b> ‖w‖₂ = r. The solution sits where a loss contour touches the circle: both weights shrink, but neither becomes exactly zero.',
    },
    lasso: {
      label: 'Lasso (L1) → sparsity', kind: 'min', xl: 'weight w₁ →', yl: 'weight w₂ →', R: [-1.5, 3.5, -2.5, 2.5], c: [1.2, 0.2, 2.4], cLabel: 'weight budget r', s0: 0.2, closed: true,
      f: LOSS_RL, g: (x, y) => Math.abs(x) + Math.abs(y),
      curve: (s, c) => {
        const k = Math.floor(((s % 1) + 1) % 1 * 4), u = (((s % 1) + 1) % 1) * 4 - k;
        const P = [[c, 0], [0, c], [-c, 0], [0, -c], [c, 0]];
        return [P[k][0] + (P[k + 1][0] - P[k][0]) * u, P[k][1] + (P[k + 1][1] - P[k][1]) * u];
      },
      about: 'Same loss, but the budget is the <b>diamond</b> |w₁| + |w₂| = r (L1). Its sharp corners sit on the axes, and contours very often touch a corner first, ' +
        'so one weight becomes <b>exactly 0</b>. That is why lasso selects features: it switches unimportant ones off completely.',
    },
  };

  const LAG = {
    P: (o) => LAG_PROBS[o.prob],
    setProb(o, key) { const P = LAG_PROBS[key]; Object.assign(o, { prob: key, c: P.c[0], s: P.s0, solving: false, eta: 1 }); this.heat = null; },
    grad(fn, x, y) { const h = 1e-5; return [(fn(x + h, y) - fn(x - h, y)) / (2 * h), (fn(x, y + h) - fn(x, y - h)) / (2 * h)]; },
    wrap(o, s) { return this.P(o).closed ? ((s % 1) + 1) % 1 : clamp(s, 0, 1); },
    tangent(o, s) { // dγ/ds
      const P = this.P(o), h = 1e-4, a = P.curve(this.wrap(o, s - h), o.c), b = P.curve(this.wrap(o, s + h), o.c);
      return [(b[0] - a[0]) / (2 * h), (b[1] - a[1]) / (2 * h)];
    },
    samples(o, c = o.c, n = 800) {
      const P = this.P(o), out = [];
      for (let i = 0; i <= n; i++) { const s = P.closed ? i / n : i / n, [x, y] = P.curve(s, c); out.push({ s, x, y, f: P.f(x, y) }); }
      return out;
    },
    best(o, c = o.c) {
      const P = this.P(o), sm = this.samples(o, c, 2000);
      return sm.reduce((b, q) => ((P.kind === 'max' ? q.f > b.f : q.f < b.f) ? q : b));
    },
    info(o, s = o.s) {
      const P = this.P(o), [x, y] = P.curve(s, o.c), gf = this.grad(P.f, x, y), gg = this.grad(P.g, x, y);
      const gg2 = gg[0] ** 2 + gg[1] ** 2, lam = (gf[0] * gg[0] + gf[1] * gg[1]) / Math.max(gg2, 1e-12);
      const T = this.tangent(o, s), tl = Math.hypot(...T) || 1, tc = (gf[0] * T[0] + gf[1] * T[1]) / tl;
      return { x, y, f: P.f(x, y), gf, gg, lam, tc, T: [T[0] / tl, T[1] / tl], resid: Math.hypot(gf[0] - lam * gg[0], gf[1] - lam * gg[1]) };
    },
    step(o) { // slide along the path in the improving direction (projected gradient), halving the step on overshoot
      const P = this.P(o), dir = P.kind === 'max' ? 1 : -1;
      for (let k = 0; k < 4; k++) {
        const T = this.tangent(o, o.s), dfds = (this.grad(P.f, ...P.curve(o.s, o.c))[0] * T[0] + this.grad(P.f, ...P.curve(o.s, o.c))[1] * T[1]);
        const ds = clamp(dir * 0.02 * o.eta * dfds / Math.max(T[0] ** 2 + T[1] ** 2, 1e-6), -0.008, 0.008);
        if (o.lastSign && Math.sign(ds) !== o.lastSign) o.eta *= 0.5;
        o.lastSign = Math.sign(ds);
        o.s = this.wrap(o, o.s + ds);
        if (Math.abs(ds) < 1e-7) { o.solving = false; break; }
      }
    },
    shadowPrice(o) { // numerical d(best f)/dc, to compare with λ*
      const P = this.P(o), d = 0.004 * (P.c[2] - P.c[1]);
      const lo = { ...o, c: o.c - d }, hi = { ...o, c: o.c + d };
      return (this.best(hi, hi.c).f - this.best(lo, lo.c).f) / (2 * d);
    },
    camera(o, b) {
      const cam = o.cam, cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const S = b.w * 0.36, ox = b.x + b.w / 2, oy = b.y + b.h * 0.58;
      return (X, Y, Z) => { const x = X * cy - Y * sy, dd = X * sy + Y * cy; return [ox + x * S, oy - (Z * cp + dd * sp) * S, dd * cp - Z * sp]; };
    },
    draw(ctx, W, H, sim) {
      const o = sim.o.lag, P = this.P(o), wide = W > 780;
      const side = wide ? Math.min(W * 0.55, H - 190) : Math.min(W - 60, H * 0.5);
      const mBox = box(46, 130, side, side, P.R[0], P.R[1], P.R[2], P.R[3]); // below the HUD
      sim.regions = { mBox };
      const sm = this.samples(o), best = this.best(o), I = this.info(o);
      // local extrema along the path = all points where the path is tangent to a contour
      const tang = [];
      for (let i = 1; i < sm.length - 1; i++) if ((sm[i].f - sm[i - 1].f) * (sm[i + 1].f - sm[i].f) < 0) tang.push(sm[i]);
      if (o.view === '2d') {
        if (!this.heat || this.heat.key !== o.prob) {
          const n = 170;
          this.heat = { ...heatCanvas(n, n, (i, j) => P.f(P.R[0] + ((i + 0.5) / n) * (P.R[1] - P.R[0]), P.R[3] - ((j + 0.5) / n) * (P.R[3] - P.R[2]))), key: o.prob };
        }
        ctx.imageSmoothingEnabled = true; ctx.drawImage(this.heat.canvas, mBox.x, mBox.y, mBox.w, mBox.h);
        frameBox(ctx, mBox, `${P.kind === 'max' ? 'Maximize' : 'Minimize'} f (contours; ${P.kind === 'max' ? 'brighter = higher' : 'darker = lower'}) on the white path g = c`);
        axisLabels(ctx, mBox, P.xl, P.yl);
        ctx.save(); ctx.beginPath(); ctx.rect(mBox.x, mBox.y, mBox.w, mBox.h); ctx.clip();
        ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(mBox.sx(0), mBox.y); ctx.lineTo(mBox.sx(0), mBox.y + mBox.h); ctx.moveTo(mBox.x, mBox.sy(0)); ctx.lineTo(mBox.x + mBox.w, mBox.sy(0)); ctx.stroke();
        polyline(ctx, sm.map((q) => [mBox.sx(q.x), mBox.sy(q.y)]), '#ffffff', 2.6);
        for (const q of tang) { ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(mBox.sx(q.x), mBox.sy(q.y), 6, 0, TAU); ctx.stroke(); }
        ctx.fillStyle = COL.fit; ctx.font = '20px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', mBox.sx(best.x), mBox.sy(best.y));
        // gradient arrows at the current point
        const px = mBox.sx(I.x), py = mBox.sy(I.y), m = Math.max(Math.hypot(...I.gf), Math.abs(I.lam) * Math.hypot(...I.gg), 1e-9), Lp = 64;
        const arr = (v, color) => { const k = Lp / m; if (Math.hypot(...v) * k > 3) arrow(ctx, px, py, px + v[0] * k, py - v[1] * k, color); };
        arr([I.lam * I.gg[0], I.lam * I.gg[1]], '#7cf0a8');
        arr(I.gf, COL.fit);
        arr([I.T[0] * I.tc, I.T[1] * I.tc], '#ff6b6f');
        dot(ctx, px, py, 6.5, '#ffffff', '#0b0e14');
        ctx.restore();
        ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const ly = mBox.y + mBox.h + 20;
        ctx.fillStyle = COL.fit; ctx.fillText('→ ∇f', mBox.x, ly);
        ctx.fillStyle = '#7cf0a8'; ctx.fillText('→ λ∇g', mBox.x + 48, ly);
        ctx.fillStyle = '#ff6b6f'; ctx.fillText('→ part of ∇f along the path (0 at the optimum)', mBox.x + 108, ly);
        ctx.fillStyle = '#c9cfda'; ctx.fillText('★ best · ○ every point where the path is tangent to a contour · drag the white dot', mBox.x, ly + 16);
      } else {
        // 3D: the landscape with the path lifted onto it
        const n = 40, Pj = this.camera(o, mBox), fx = (x) => ((x - P.R[0]) / (P.R[1] - P.R[0])) * 2 - 1, fy = (y) => ((y - P.R[2]) / (P.R[3] - P.R[2])) * 2 - 1;
        let fmin = Infinity, fmax = -Infinity;
        const grid = [];
        for (let j = 0; j <= n; j++) { const row = []; for (let i = 0; i <= n; i++) { const x = P.R[0] + (i / n) * (P.R[1] - P.R[0]), y = P.R[2] + (j / n) * (P.R[3] - P.R[2]), f = P.f(x, y); fmin = Math.min(fmin, f); fmax = Math.max(fmax, f); row.push([x, y, f]); } grid.push(row); }
        const fz = (f) => ((f - fmin) / (fmax - fmin || 1)) * 0.9 - 0.5;
        const quads = [];
        for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
          const q = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]].map(([x, y, f]) => Pj(fx(x), fy(y), fz(f)));
          quads.push({ q, d: (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4, t: (grid[j][i][2] - fmin) / (fmax - fmin || 1) });
        }
        quads.sort((a, b) => b.d - a.d);
        ctx.lineWidth = 0.5;
        for (const { q, t } of quads) {
          const idx = (clamp(t, 0, 1) * 255) | 0;
          ctx.fillStyle = `rgb(${SEQ[idx * 3]},${SEQ[idx * 3 + 1]},${SEQ[idx * 3 + 2]})`; ctx.strokeStyle = 'rgba(0,0,0,0.3)';
          ctx.beginPath(); q.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.fill(); ctx.stroke();
        }
        const floor = sm.map((q) => Pj(fx(q.x), fy(q.y), -0.55)), lift = sm.map((q) => Pj(fx(q.x), fy(q.y), fz(q.f) + 0.01));
        ctx.setLineDash([4, 4]); polyline(ctx, floor, 'rgba(255,255,255,0.45)', 1.5); ctx.setLineDash([]);
        polyline(ctx, lift, '#ffffff', 3);
        const cur = Pj(fx(I.x), fy(I.y), fz(I.f)), base = Pj(fx(I.x), fy(I.y), -0.55);
        polyline(ctx, [base, cur], 'rgba(255,255,255,0.5)', 1);
        dot(ctx, cur[0], cur[1], 6.5, '#ffffff', '#0b0e14');
        const bp = Pj(fx(best.x), fy(best.y), fz(best.f));
        ctx.fillStyle = COL.fit; ctx.font = '22px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', bp[0], bp[1] - 4);
        frameBox(ctx, { x: mBox.x, y: mBox.y, w: 0, h: 0 }, `3D: the path g = c (dashed, on the floor) lifted onto the surface f · drag to rotate`);
      }
      // f along the path
      const pb = wide ? { x: mBox.x + mBox.w + 40, y: 130, w: W - mBox.x - mBox.w - 64, h: Math.min(260, H * 0.34) } : { x: 46, y: mBox.y + mBox.h + 64, w: W - 70, h: H - mBox.y - mBox.h - 90 };
      if (pb.h > 60 && pb.w > 120) {
        frameBox(ctx, pb, 'f along the path (walking the white curve)');
        let lo = Infinity, hi = -Infinity;
        for (const q of sm) { lo = Math.min(lo, q.f); hi = Math.max(hi, q.f); }
        const pad = (hi - lo) * 0.1 || 1, X = (s) => pb.x + 10 + s * (pb.w - 20), Y = (f) => pb.y + pb.h - 8 - ((f - lo + pad) / (hi - lo + 2 * pad)) * (pb.h - 16);
        polyline(ctx, sm.map((q) => [X(q.s), Y(q.f)]), '#ffffff', 2);
        for (const q of tang) { ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(X(q.s), pb.y); ctx.lineTo(X(q.s), pb.y + pb.h); ctx.stroke(); ctx.setLineDash([]); }
        ctx.fillStyle = COL.fit; ctx.font = '18px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', X(best.s), Y(best.f));
        dot(ctx, X(o.s), Y(I.f), 5.5, '#ffffff', '#0b0e14');
        ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText('dashed: flat spots of this curve = tangencies on the map', pb.x + 6, pb.y + pb.h + 6);
      }
    },
    pointer(type, x, y, sim) {
      const o = sim.o.lag, { mBox } = sim.regions || {};
      if (!mBox) return;
      if (o.view === '3d') {
        if (type === 'down') sim.rot = { x, y, yaw: o.cam.yaw, pitch: o.cam.pitch };
        else if (type === 'move' && sim.rot) { o.cam.yaw = sim.rot.yaw - (x - sim.rot.x) * 0.008; o.cam.pitch = clamp(sim.rot.pitch + (y - sim.rot.y) * 0.008, 0.05, 1.5); }
        else sim.rot = null;
        return;
      }
      const nearest = () => {
        let bs = null, bd = Infinity;
        for (const q of this.samples(o, o.c, 600)) { const d = Math.hypot(mBox.sx(q.x) - x, mBox.sy(q.y) - y); if (d < bd) { bd = d; bs = q.s; } }
        return [bs, bd];
      };
      if (type === 'down') { const [s, d] = nearest(); if (d < 22) { sim.dragLag = true; o.s = s; o.solving = false; } }
      else if (type === 'move' && sim.dragLag) o.s = nearest()[0];
      else if (type === 'up' || type === 'leave') sim.dragLag = false;
    },
  };

  // ================================================================== module
  const MODES = [
    { value: 'line', label: 'Fit a line', hint: 'Drag data points · click empty space to add · drag a point out to delete · click the loss map to restart there' },
    { value: 'race', label: 'Optimizer race', hint: 'Click anywhere on the loss surface to start all optimizers from there' },
    { value: 'net', label: 'Tiny neural network', hint: 'Click the data area to add points of the selected class' },
    { value: 'lagrange', label: 'Lagrange multipliers', hint: 'Drag the white dot along the path · or let it slide uphill/downhill on its own' },
  ];

  const sim = {
    id: 'learning',
    get hint() { return this.o.mode === 'race' && this.o.race.view === '3d' ? 'Drag to rotate the terrain · switch to the contour map to pick a new start' : MODES.find((m) => m.value === this.o.mode).hint; },
    o: {
      mode: 'line', running: true,
      line: { lr: 0.3, speed: 8 },
      race: { surface: 'valley', lr: 0.95, lrAdam: 0.1, beta: 0.9, speed: 30, enabled: { gd: true, mom: true, adam: true }, view: '2d', cam: { yaw: -0.65, pitch: 0.75 } },
      net: { H: 6, dataset: 'circle', optimizer: 'adam', lr: 0.03, speed: 4, addClass: 1 },
      lag: { prob: 'hill', view: '2d', cam: { yaw: -0.6, pitch: 0.75 } },
    },
    acc: 0, frameNo: 0, regions: null, drag: null,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.ready) {
        LINE.init(this.o.line);
        RACE.restart(this.o.race);
        NET.data(this.o.net, this.o.net.dataset); NET.init(this.o.net, 5);
        LAG.setProb(this.o.lag, this.o.lag.prob);
        this.ready = true;
      }
      this.buildPanel();
    },
    unmount() { this.drag = null; },
    resize(W, H) { this.W = W; this.H = H; },

    setMode(m) { this.o.mode = m; this.acc = 0; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({ options: MODES, value: o.mode, onChange: (v) => this.setMode(v) }));
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Train', () => { o.running = !o.running; play.textContent = o.running ? '⏸ Pause' : '▶ Train'; });
      row.append(play, UI.button('Step ›', () => this.stepOnce()));
      s0.append(row);
      p.append(s0);
      if (o.mode === 'line') this.panelLine(p);
      else if (o.mode === 'race') this.panelRace(p);
      else if (o.mode === 'net') this.panelNet(p);
      else this.panelLag(p);
    },
    stepOnce() {
      const o = this.o;
      if (o.mode === 'line') LINE.step(o.line); else if (o.mode === 'race') RACE.step(o.race); else if (o.mode === 'net') NET.step(o.net); else LAG.step(o.lag);
    },

    panelLag(p) {
      const o = this.o.lag, P = LAG_PROBS[o.prob];
      const s1 = UI.section('Constrained optimization');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered} \text{${P.kind === 'max' ? 'maximize' : 'minimize'} } f(x,y) \ \text{ subject to } \ g(x,y) = c \\[4pt]
        \textcolor{#f5c451}{\nabla f} = \lambda\, \textcolor{#7cf0a8}{\nabla g}, \qquad \lambda = \frac{d f^{*}}{dc} \end{gathered}`, true);
      s1.append(eq);
      s1.append(UI.segmented({ options: Object.entries(LAG_PROBS).map(([value, q]) => ({ value, label: q.label })), value: o.prob, onChange: (v) => { LAG.setProb(o, v); this.buildPanel(); } }));
      const ab = el('p', 'note'); ab.innerHTML = P.about; s1.append(ab);
      s1.append(UI.slider({ label: P.cLabel, min: P.c[1], max: P.c[2], step: (P.c[2] - P.c[1]) / 200, value: o.c, format: (v) => v.toFixed(2), onInput: (v) => { o.c = v; } }));
      const r = el('div', 'controls-row');
      r.append(UI.button(P.kind === 'max' ? '▶ Slide uphill along the path' : '▶ Slide downhill along the path', () => { o.solving = true; o.eta = 1; o.lastSign = 0; }));
      s1.append(r, UI.segmented({ options: [{ value: '2d', label: 'Contour map' }, { value: '3d', label: '3D landscape' }], value: o.view, onChange: (v) => { o.view = v; } }));

      const s2 = UI.section('Why it works');
      s2.append(UI.termCard({ color: COL.fit, tex: String.raw`\nabla f \parallel \nabla g`, title: 'The gradients line up',
        body: 'If ∇f had any component <b>along</b> the path (red arrow), you could slide a little that way and do better. At the best point that component is zero, so ∇f points straight ' +
          'across the path, and so does ∇g (it is always perpendicular to its own level curve g = c). Two vectors perpendicular to the same curve are parallel: <b>∇f = λ∇g</b>. ' +
          'Geometrically, the path just touches a contour line there.' }));
      s2.append(UI.termCard({ color: '#7cf0a8', tex: String.raw`\mathcal{L}(x,y,\lambda) = f - \lambda\,(g - c)`, title: 'λ is a price',
        body: 'Setting all derivatives of the Lagrangian 𝓛 to zero gives the tangency condition plus the constraint itself. And ∂𝓛/∂c = λ, so λ says how much the best value f* ' +
          'improves per unit of loosening the constraint: a <b>shadow price</b>. Move the c slider; the readout compares λ with the measured change in f*.' }));
      s2.append(UI.termCard({ color: COL.b, tex: String.raw`\|w\|_2 \le r \;\;\text{vs.}\;\; \|w\|_1 \le r`, title: 'In machine learning',
        body: '<b>Regularization is a constraint</b>: "fit the data, but keep the weights small." λ becomes the regularization strength. ' +
          'Compare <b>Ridge</b> and <b>Lasso</b>: the L1 diamond has corners on the axes, so the optimum often lands exactly on one, with a weight of exactly 0. ' +
          'At a corner ∇g has no single direction (any normal in the corner\'s "fan" counts), so ∇f need not line up with one particular ∇g there. ' +
          'Lagrange multipliers also give SVMs (max margin), PCA (maximize variance with ‖w‖ = 1 gives Σw = λw, so λ is the variance) and softmax (maximum entropy).' }));
      p.append(s1, s2);
    },

    panelLine(p) {
      const o = this.o.line;
      const s1 = UI.section('Gradient descent');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered} L(w,b) = \frac{1}{N}\sum_i \big(\underbrace{w x_i + b}_{\text{prediction}} - y_i\big)^2 \\[4pt]
        \begin{pmatrix} w \\ b \end{pmatrix} \leftarrow \begin{pmatrix} w \\ b \end{pmatrix} - \textcolor{#f5c451}{\eta}\, \textcolor{#7cf0a8}{\nabla L} \end{gathered}`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Training = walking downhill on the loss surface. Every choice of the two weights (w, b) is a point on the <b>right</b> and a line on the <b>left</b>. ' +
        'The gradient ∇L points uphill; each step moves a little the other way. Watch both views move together.';
      s1.append(eq, n1);
      s1.append(UI.slider({ label: 'Learning rate η', min: 0.01, max: 1.2, step: 0.01, value: o.lr, format: (v) => v.toFixed(2), onInput: (v) => { o.lr = v; } }));
      s1.append(UI.slider({ label: 'Steps per second', min: 1, max: 60, step: 1, value: o.speed, format: (v) => `${v}`, onInput: (v) => { o.speed = v; } }));
      const r = el('div', 'controls-row');
      r.append(UI.button('↺ Restart weights', () => LINE.restart(o, -2.2, 1.9)), UI.button('New data', () => { o.pts = []; const rr = rng(Math.random() * 1e9 | 0); const m = 2 * rr() - 1, c = rr() - 0.5; for (let k = 0; k < 14; k++) { const x = -1 + 2 * rr(); o.pts.push([x, m * x + c + 0.25 * gauss(rr)]); } LINE.restart(o, -2.2, 1.9); }));
      s1.append(r);
      const s2 = UI.section('Try this');
      s2.append(UI.termCard({ color: COL.fit, tex: String.raw`\eta`, title: 'Learning rate',
        body: 'Small η: safe but slow. Around η ≈ 1 the steps overshoot the valley and bounce from wall to wall; a bit higher and they <b>diverge</b>, flying off to infinity. ' +
          'The limit is set by the steepest direction of the bowl (here, by how spread out the x values are).' }));
      s2.append(UI.termCard({ color: '#7cf0a8', tex: String.raw`\nabla L = \Big(\tfrac{\partial L}{\partial w}, \tfrac{\partial L}{\partial b}\Big)`, title: 'Why it is a differential equation',
        body: 'With tiny steps, gradient descent becomes <b>gradient flow</b>: dθ/dt = −∇L(θ), an ODE that slides downhill. Adding momentum gives m·θ″ + c·θ′ = −∇L, ' +
          'the same damped mass-spring equation as the <b>Circuits</b> tab, so learning rate and momentum are really damping knobs.' }));
      s2.append(UI.termCard({ color: '#e5484d', tex: String.raw`y_i - \hat y_i`, title: 'Drag the data',
        body: 'Move a point and the whole loss surface reshapes: the ★ (the exact least-squares answer) moves with it. Gradient descent never sees the surface, only the slope where it stands.' }));
      p.append(s1, s2);
    },

    panelRace(p) {
      const o = this.o.race, S = SURFACES[o.surface];
      const s1 = UI.section('Loss surface');
      s1.append(UI.segmented({ options: Object.entries(SURFACES).map(([value, s]) => ({ value, label: s.label })), value: o.surface,
        onChange: (v) => { o.surface = v; o.lr = SURFACES[v].lr; o.lrAdam = SURFACES[v].lrAdam; RACE.restart(o); this.buildPanel(); } }));
      const n = el('p', 'note'); n.innerHTML = S.about; s1.append(n);
      s1.append(UI.slider({ label: 'Steps per second', min: 1, max: 120, step: 1, value: o.speed, format: (v) => `${v}`, onInput: (v) => { o.speed = v; } }));
      s1.append(UI.button('↺ Restart race', () => RACE.restart(o, o.runs.gd.path[0][0], o.runs.gd.path[0][1])));
      s1.append(UI.segmented({ options: [{ value: '2d', label: 'Contour map' }, { value: '3d', label: '3D terrain' }], value: o.view, onChange: (v) => { o.view = v; document.getElementById('stageHint').textContent = this.hint; } }));
      s1.append(el('p', 'note', 'In 3D, drag to rotate. Switch back to the contour map to click a new starting point.'));
      const s2 = UI.section('The optimizers');
      const toggle = (key) => UI.check({ label: 'Race this one', checked: o.enabled[key], onChange: (v) => { o.enabled[key] = v; } });
      s2.append(UI.termCard({ color: COL.gd, tex: String.raw`\theta \leftarrow \theta - \eta\,\nabla L`, title: 'Gradient descent',
        body: 'Step straight downhill, by an amount proportional to the slope. Simple, but it zig-zags in narrow valleys and crawls on flat ground.',
        controls: [toggle('gd'), UI.slider({ label: 'Learning rate η (GD & momentum)', min: S.lr / 20, max: S.lr * 2.2, step: S.lr / 200, value: o.lr, format: (v) => v.toPrecision(2), onInput: (v) => { o.lr = v; } })] }));
      s2.append(UI.termCard({ color: COL.mom, tex: String.raw`v \leftarrow \beta v - \eta\nabla L,\;\; \theta \leftarrow \theta + v`, title: 'Momentum (heavy ball)',
        body: 'A ball with mass: it keeps some of its previous velocity. The zig-zags cancel out while the consistent downhill push adds up. ' +
          'Physically it is m·θ″ + c·θ′ = −∇L with damping set by β: β near 1 is underdamped (overshoots and rolls back), small β is overdamped.',
        controls: [toggle('mom'), UI.slider({ label: 'Momentum β', min: 0, max: 0.99, step: 0.01, value: o.beta, format: (v) => v.toFixed(2), onInput: (v) => { o.beta = v; } })] }));
      s2.append(UI.termCard({ color: COL.adam, tex: String.raw`\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat s}+\epsilon}`, title: 'Adam',
        body: 'Momentum (m) plus a per-weight scale (s = running average of g²). Each weight gets its own step size: tiny gradients are boosted, huge ones tamed. ' +
          'That is why it escapes saddles fast. It is the default optimizer for most deep learning.',
        controls: [toggle('adam'), UI.slider({ label: 'Learning rate η (Adam)', min: 0.005, max: 0.5, step: 0.005, value: o.lrAdam, format: (v) => v.toFixed(3), onInput: (v) => { o.lrAdam = v; } })] }));
      p.append(s1, s2);
    },

    panelNet(p) {
      const o = this.o.net;
      const s1 = UI.section('Data & network');
      s1.append(UI.segmented({ options: Object.entries(DATASETS).map(([value, d]) => ({ value, label: d.label })), value: o.dataset,
        onChange: (v) => { NET.data(o, v); NET.init(o); this.buildPanel(); } }));
      s1.append(UI.slider({ label: 'Hidden neurons', min: 1, max: 10, step: 1, value: o.H, format: (v) => `${v} → ${NET.nParams(v)} weights`, onInput: (v) => { o.H = v; NET.init(o); } }));
      const r = el('div', 'controls-row');
      r.append(UI.button('🎲 New random weights', () => NET.init(o)), UI.button('📸 Slice the landscape', () => NET.snapshot(o)));
      s1.append(r);
      s1.append(el('p', 'note', 'Add points by clicking the data area. Class of new points:'));
      s1.append(UI.segmented({ options: [{ value: 0, label: '● blue' }, { value: 1, label: '● orange' }], value: o.addClass, onChange: (v) => { o.addClass = v; } }));

      const s2 = UI.section('Training');
      s2.append(UI.segmented({ options: [{ value: 'gd', label: 'Gradient descent' }, { value: 'mom', label: 'Momentum' }, { value: 'adam', label: 'Adam' }], value: o.optimizer,
        onChange: (v) => { o.optimizer = v; o.lr = v === 'adam' ? 0.03 : 0.5; this.buildPanel(); } }));
      s2.append(UI.slider({ label: 'Learning rate η', min: -3, max: 0.5, step: 0.01, value: Math.log10(o.lr), format: (v) => Math.pow(10, v).toPrecision(2), onInput: (v) => { o.lr = Math.pow(10, v); } }));
      s2.append(UI.slider({ label: 'Steps per frame', min: 1, max: 30, step: 1, value: o.speed, format: (v) => `${v}`, onInput: (v) => { o.speed = v; } }));

      const s3 = UI.section('How to read it');
      s3.append(UI.termCard({ color: COL.b, tex: String.raw`h_j = \tanh(w_{j1}x_1 + w_{j2}x_2 + b_j)`, title: 'Each hidden neuron',
        body: 'One neuron can only draw a single straight (soft) dividing line; its small square shows that line over the input space. ' +
          'The output adds the neurons up with its own weights, bending those straight lines into a curved boundary. Try 1 neuron on the circle: impossible. 3 or more: easy.' }));
      s3.append(UI.termCard({ color: COL.fit, tex: String.raw`L = -\tfrac1N\textstyle\sum [y\log p + (1-y)\log(1-p)]`, title: 'Loss & gradient',
        body: 'Cross-entropy punishes confident mistakes. Backpropagation computes ∂L/∂(every weight) with the chain rule, and then the same downhill step as in the other modes, just in ' +
          '<b>many</b> dimensions at once.' }));
      s3.append(UI.termCard({ color: '#e8ebf1', tex: String.raw`L(\theta^* + \alpha d_1 + \beta d_2)`, title: 'Seeing a many-dimensional landscape',
        body: 'We can\'t draw 25+ dimensions, so we pick two random directions d₁, d₂ through the current weights and plot the loss on that flat slice. The path the weights took is ' +
          'projected onto it (circle = start, yellow = now). This is how researchers make loss-landscape pictures of huge networks.' }));
      p.append(s1, s2, s3);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o;
      this.frameNo++;
      if (o.running && this.drag == null) {
        if (o.mode === 'line') { this.acc += o.line.speed / 60; while (this.acc >= 1) { LINE.step(o.line); this.acc--; } }
        else if (o.mode === 'race') { this.acc += o.race.speed / 60; let n = 0; while (this.acc >= 1 && n++ < 10) { RACE.step(o.race); this.acc--; } }
        else if (o.mode === 'net') for (let k = 0; k < o.net.speed; k++) NET.step(o.net);
      }
      if (o.mode === 'lagrange' && o.lag.solving) LAG.step(o.lag);
      if (o.mode === 'line') LINE.draw(ctx, W, H, this);
      else if (o.mode === 'race') RACE.draw(ctx, W, H, this);
      else if (o.mode === 'net') NET.draw(ctx, W, H, this);
      else LAG.draw(ctx, W, H, this);
      if (this.frameNo % 6 === 0) this.updateHud();
    },

    updateHud() {
      const o = this.o, parts = [];
      if (o.mode === 'line') {
        const l = o.line, L = LINE.loss(l, l.w, l.b), [gw, gb] = LINE.grad(l, l.w, l.b);
        parts.push(`step ${l.path.length - 1} · w = <b>${l.w.toFixed(3)}</b>, b = <b>${l.b.toFixed(3)}</b> · loss ${L.toPrecision(3)} · |∇L| ${Math.hypot(gw, gb).toPrecision(2)}`);
        if (l.diverged) parts.push('<div class="warn">Diverged: the learning rate is too high, so each step overshoots further. Lower η and restart.</div>');
      } else if (o.mode === 'race') {
        parts.push(`step ${o.race.iter} · ` + OPTS.filter(({ key }) => o.race.enabled[key]).map(({ key, label, color }) => {
          const r = o.race.runs[key]; return `<span style="color:${color}">${label}</span> ${r.dead ? 'diverged' : r.loss[r.loss.length - 1].toFixed(3)}`;
        }).join(' · '));
      } else if (o.mode === 'lagrange') {
        const l = o.lag, I = LAG.info(l), b = LAG.best(l), Ib = LAG.info(l, b.s);
        parts.push(`here: f = <b>${I.f.toFixed(3)}</b> · slope along path ${I.tc.toFixed(3)} · λ ≈ ${I.lam.toFixed(3)}`);
        parts.push(`<div>★ f* = ${b.f.toFixed(3)} at (${b.x.toFixed(2)}, ${b.y.toFixed(2)}) · <b>λ* = ${Ib.lam.toFixed(3)}</b> ≈ measured df*/dc ${LAG.shadowPrice(l).toFixed(3)}</div>`);
        if (!l.solving && Math.abs(I.tc) < 2e-3 && Math.abs(I.f - b.f) > 1e-3) {
          parts.push('<div class="warn">A tangency, but only a <b>local</b> optimum. Every tangency satisfies ∇f = λ∇g; compare them to find the ★.</div>');
        } else if (l.prob === 'lasso' && Math.min(Math.abs(b.x), Math.abs(b.y)) < 0.02) {
          parts.push('<div class="dim">★ is at a <b>corner</b>: one weight is exactly 0 (sparsity).</div>');
        } else if (l.prob === 'ridge') {
          parts.push('<div class="dim">★ on a smooth circle: both weights shrink, neither hits 0.</div>');
        }
      } else {
        const n = o.net, acc = n.accs.length ? n.accs[n.accs.length - 1] : 0;
        parts.push(`step ${n.steps} · loss <b>${n.losses[n.losses.length - 1].toFixed(3)}</b> · accuracy <b>${(acc * 100).toFixed(0)}%</b> · ${NET.nParams(n.H)} weights moving at once`);
      }
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    pointer(type, x, y) {
      const m = this.o.mode;
      if (m === 'line') LINE.pointer(type, x, y, this);
      else if (m === 'race') RACE.pointer(type, x, y, this);
      else if (m === 'net') NET.pointer(type, x, y, this);
      else LAG.pointer(type, x, y, this);
    },
  };

  sim._internals = { LINE, RACE, NET, SURFACES, LAG, LAG_PROBS }; // exposed for automated checks
  (window.Sims = window.Sims || {}).learning = sim;
})();
