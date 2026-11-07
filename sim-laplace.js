/* Laplace explorer: drag poles and zeros on the s-plane and see the system respond.
 *   H(s) = K·Π(s − z)/Π(s − p),   h(t) = Σ r_k e^{p_k t}   (residues r_k)
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const SEQ = UI.lut(UI.SEQ_STOPS);
  const COL = { imp: '#f5c451', step: '#3d8bfd', ghost: '#c9cfda', freq: '#2fa36b', jw: '#5b9bff', pole: '#ffffff', unstable: '#e5484d' };
  const RE = [-5, 2.5], IM = 5, TMAX = 10, WMAX = 5;
  const TAU = Math.PI * 2;
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  // complex helpers on [re, im]
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
  const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];

  const PRESETS = {
    first: { label: '1st order (RC filter)', items: [['p', -1, 0]],
      about: 'One real pole at s = −1. The response is a pure decay e<sup>−t</sup>, with no oscillation. A low-pass filter: it passes slow signals and weakens fast ones.' },
    spring: { label: 'Spring, light damping', items: [['p', -0.4, 2.5]],
      about: 'A pole pair close to the jω axis: a bouncy spring. The height ω sets how fast it oscillates; the small σ means it rings for a long time. Note the resonance peak in the frequency response.' },
    heavy: { label: 'Heavy damping', items: [['p', -0.7, 0], ['p', -3, 0]],
      about: 'Two real poles: like a door closer. No oscillation at all, just a slow creep, dominated by the pole nearest the axis.' },
    osc: { label: 'No damping', items: [['p', 0, 2]],
      about: 'Poles exactly on the jω axis: σ = 0, so nothing decays. It oscillates forever, and the frequency response is infinite at ω = 2.' },
    unstable: { label: 'Unstable', items: [['p', 0.25, 2]],
      about: 'Poles in the right half: σ > 0, so the oscillation grows exponentially. Every unstable system (a falling pendulum, audio feedback squeal) has a pole on this side.' },
    notch: { label: 'Resonance + notch', items: [['p', -0.3, 1.5], ['p', -1.5, 0], ['z', 0, 3.5]],
      about: 'Zeros on the jω axis make |H| exactly zero there: that frequency is blocked (a notch filter, like removing 50/60 Hz mains hum). The pole pair adds a resonance peak at ω ≈ 1.5.' },
  };

  // ------------------------------------------------------------------ the system
  function expand(items, kind) {
    const out = [];
    for (const it of items) {
      if (it.kind !== kind) continue;
      if (it.im > 1e-9) { out.push([it.re, it.im], [it.re, -it.im]); } else out.push([it.re, 0]);
    }
    return out;
  }

  function residues(poles, zeros, K) {
    const p = poles.map((q) => q.slice());
    for (let i = 0; i < p.length; i++) {   // nudge repeated poles apart (residue formula needs distinct poles)
      for (let j = 0; j < i; j++) {
        if (Math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1]) < 2e-3) p[i][0] += 4e-3 * (i + 1);
      }
    }
    return p.map((pk, k) => {
      let num = [K, 0], den = [1, 0];
      for (const z of zeros) num = cmul(num, csub(pk, z));
      p.forEach((pi, i) => { if (i !== k) den = cmul(den, csub(pk, pi)); });
      return { p: pk, r: cdiv(num, den) };
    });
  }

  function evalH(poles, zeros, K, re, im) {
    let nr = K, ni = 0, dr = 1, di = 0;
    for (const z of zeros) { const a = re - z[0], b = im - z[1]; const t = nr * a - ni * b; ni = nr * b + ni * a; nr = t; }
    for (const p of poles) { const a = re - p[0], b = im - p[1]; const t = dr * a - di * b; di = dr * b + di * a; dr = t; }
    const d = dr * dr + di * di;
    return Math.sqrt((nr * nr + ni * ni) / d);
  }

  function analyse(items) {
    const poles = expand(items, 'p'), zeros = expand(items, 'z');
    // gain: unit DC gain when possible, otherwise unit peak gain
    let K = 1;
    const nearZero = poles.some((p) => Math.hypot(...p) < 1e-3);
    let h0 = null;
    if (!nearZero) {
      let num = [1, 0], den = [1, 0];
      for (const z of zeros) num = cmul(num, [-z[0], -z[1]]);
      for (const p of poles) den = cmul(den, [-p[0], -p[1]]);
      h0 = cdiv(num, den)[0];
    }
    if (h0 !== null && Math.abs(h0) > 1e-6) K = 1 / h0;
    else {
      let m = 0;
      for (let i = 0; i <= 300; i++) { const v = evalH(poles, zeros, 1, 0, (i / 300) * WMAX); if (Number.isFinite(v)) m = Math.max(m, v); }
      K = m > 0 ? 1 / m : 1;
    }
    const imp = residues(poles, zeros, K);
    const stepPoles = [...poles, nearZero ? [-1e-3, 0] : [0, 0]];
    const step = residues(stepPoles, zeros, K);
    const sumAt = (res, t) => { let v = 0; for (const { p, r } of res) { const e = Math.exp(p[0] * t); v += e * (r[0] * Math.cos(p[1] * t) - r[1] * Math.sin(p[1] * t)); } return v; };
    const n = 500, hs = new Float64Array(n + 1), ss = new Float64Array(n + 1);
    for (let i = 0; i <= n; i++) { const t = (i / n) * TMAX; hs[i] = sumAt(imp, t); ss[i] = sumAt(step, t); }
    const fr = new Float64Array(401);
    for (let i = 0; i <= 400; i++) fr[i] = evalH(poles, zeros, K, 0, (i / 400) * WMAX);
    return { poles, zeros, K, hs, ss, fr, stable: poles.every((p) => p[0] < -1e-9) };
  }

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'laplace',
    hint: 'Drag the × poles and ○ zeros · hover the plane to see e^(st) · drag a point off the plane to delete it',
    o: { preset: 'spring', view: '2d', yaw: -0.6, pitch: 0.75 },
    items: [], sys: null, W: 0, H: 0, hover: null, drag: null, dirty: true,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.items.length) this.loadPreset(this.o.preset); else this.buildPanel();
    },
    unmount() { this.drag = null; this.hover = null; },
    resize(W, H) { this.W = W; this.H = H; this.dirty = true; },

    loadPreset(key) {
      this.o.preset = key;
      this.items = PRESETS[key].items.map(([k, re, im]) => ({ kind: k === 'p' ? 'pole' : 'zero', re, im }));
      this.update();
      this.buildPanel();
    },
    update() {
      this.sys = analyse(this.items.map((i) => ({ kind: i.kind === 'pole' ? 'p' : 'z', re: i.re, im: i.im })));
      this.dirty = true;
      this.heat = null;
      this.updateFormula();
    },
    countOf(kind) { return this.items.filter((i) => i.kind === kind).reduce((n, i) => n + (i.im > 1e-9 ? 2 : 1), 0); },
    add(kind, pair) {
      const nz = this.countOf('zero') + (kind === 'zero' ? (pair ? 2 : 1) : 0);
      const np = this.countOf('pole') + (kind === 'pole' ? (pair ? 2 : 1) : 0);
      if (nz >= np) { this.flash('Need more poles than zeros (otherwise h(t) contains an impulse). Add a pole first.'); return; }
      if (np > 8) { this.flash('That is plenty of poles!'); return; }
      this.items.push({ kind, re: kind === 'pole' ? -1.5 : -0.5, im: pair ? (kind === 'pole' ? 1.5 : 3) : 0 });
      this.update();
      this.buildPanel();
    },
    flash(msg) { this.msg = { text: msg, until: performance.now() + 3500 }; },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();

      const s1 = UI.section('Laplace transform');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`X(s) = \int_0^\infty x(t)\,e^{-st}\,dt, \qquad s = \textcolor{#e09b14}{\sigma} + i\textcolor{${COL.jw}}{\omega}`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Fourier asks “how much of each pure oscillation e<sup>iωt</sup>?” Laplace asks the same about every ' +
        '<b>growing or decaying</b> oscillation <b>e<sup>st</sup> = e<sup>σt</sup>(cos ωt + i sin ωt)</b>. Each point of the s-plane is one such signal: ' +
        '<b style="color:#e09b14">σ</b> (left ↔ right) sets decay or growth, and <b style="color:#5b9bff">ω</b> (up ↕ down) sets the oscillation speed. ' +
        '<b>Hover the plane</b> to see the signal for that point.';
      s1.append(eq, n1);

      const s2 = UI.section('A system H(s)');
      s2.append(UI.segmented({
        options: Object.entries(PRESETS).map(([value, pr]) => ({ value, label: pr.label })),
        value: o.preset, onChange: (v) => this.loadPreset(v),
      }));
      const about = el('p', 'note');
      about.innerHTML = PRESETS[o.preset] ? PRESETS[o.preset].about : '';
      this.formulaEl = el('div', 'eq');
      this.formulaEl.style.marginTop = '10px';
      const row = el('div', 'controls-row');
      row.append(
        UI.button('+ Pole pair', () => this.add('pole', true)), UI.button('+ Real pole', () => this.add('pole', false)),
        UI.button('+ Zero pair', () => this.add('zero', true)), UI.button('+ Real zero', () => this.add('zero', false)),
      );
      s2.append(about, this.formulaEl, row);
      this.updateFormula();

      const s3 = UI.section('How to read it');
      s3.append(UI.termCard({
        color: '#ffffff', tex: String.raw`\times\ \text{poles}`, title: 'Poles = natural modes',
        body: 'Where H(s) blows up. Each pole p adds a term e<sup>pt</sup> to the impulse response: further <b>left</b> means it dies out faster, ' +
          'further from the <b>real axis</b> means it oscillates faster. Complex poles always come in mirror-image pairs.',
      }));
      s3.append(UI.termCard({
        color: COL.unstable, tex: String.raw`\operatorname{Re}(p) < 0`, title: 'Stability',
        body: 'Stable means every pole sits in the left half-plane. One pole to the right and the response grows without limit. ' +
          'Tuning the gains in the <b>Control</b> tab is really moving the closed-loop poles around this plane.',
      }));
      s3.append(UI.termCard({
        color: COL.freq, tex: String.raw`\circ\ \text{zeros}`, title: 'Zeros = blocked signals',
        body: 'Where H(s) = 0. A zero on the jω axis completely blocks that frequency, and one near it weakens it.',
      }));
      s3.append(UI.termCard({
        color: COL.jw, tex: String.raw`H(i\omega)`, title: 'Fourier is a slice of Laplace',
        body: 'Walk up the <b>jω axis</b> (σ = 0) and read |H|: that is the frequency response, i.e. the Fourier transform of h(t). ' +
          'Switch to the <b>3D landscape</b>: poles are tent poles pushing the surface up, zeros pin it down, and the blue cut along the axis is the frequency-response curve.',
      }));

      const s4 = UI.section('View');
      s4.append(UI.segmented({
        options: [{ value: '2d', label: 's-plane (heat map)' }, { value: '3d', label: '3D landscape |H(s)|' }],
        value: o.view, onChange: (v) => { o.view = v; this.dirty = true; },
      }));
      s4.append(el('p', 'note', 'In 3D, drag to rotate. Switch back to 2D to move poles and zeros.'));

      p.append(s1, s2, s3, s4);
    },

    updateFormula() {
      if (!this.formulaEl || !this.sys) return;
      const f = (v) => (Math.abs(v) < 0.005 ? '0' : (+v.toFixed(2)).toString());
      const factor = (it) => {
        if (it.im > 1e-9) { // (s − p)(s − p̄) = s² − 2σ s + |p|²
          const b = -2 * it.re, c = it.re * it.re + it.im * it.im;
          return `(s^2 ${b >= 0 ? '+' : '-'} ${f(Math.abs(b))}s + ${f(c)})`;
        }
        return Math.abs(it.re) < 0.005 ? 's' : `(s ${it.re <= 0 ? '+' : '-'} ${f(Math.abs(it.re))})`;
      };
      const num = this.items.filter((i) => i.kind === 'zero').map(factor).join('') || '1';
      const den = this.items.filter((i) => i.kind === 'pole').map(factor).join('') || '1';
      UI.tex(this.formulaEl, String.raw`H(s) = ${f(this.sys.K)}\cdot\frac{${num}}{${den}}`, true);
    },

    // ---------------------------------------------------------------- layout
    layout(W, H) {
      const wide = W > 760, pad = 14;
      const plane = wide ? { x: pad, y: pad + 44, w: W * 0.52 - pad * 1.5, h: H - pad * 2 - 44 } : { x: pad, y: pad + 44, w: W - pad * 2, h: H * 0.5 - pad - 44 };
      const right = wide ? { x: W * 0.52 + pad / 2, y: pad, w: W * 0.48 - pad * 1.5, h: H - pad * 2 - 22 } : { x: pad, y: H * 0.5 + pad / 2, w: W - pad * 2, h: H * 0.5 - pad * 1.5 - 22 };
      const th = (right.h - pad) * 0.55;
      const time = { x: right.x, y: right.y + (wide ? 44 : 0), w: right.w, h: th - (wide ? 44 : 0) };
      const freq = { x: right.x, y: right.y + th + pad, w: right.w, h: right.h - th - pad };
      const sc = Math.min(plane.w / (RE[1] - RE[0]), plane.h / (2 * IM));
      const ox = plane.x + (plane.w - sc * (RE[1] - RE[0])) / 2 - RE[0] * sc, oy = plane.y + plane.h / 2;
      return { plane, time, freq, sc, ox, oy };
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      if (!this.sys) return;
      const L = (this.L = this.layout(W, H));
      if (this.o.view === '2d') this.drawPlane(ctx, L); else this.draw3D(ctx, L);
      this.drawTime(ctx, L.time);
      this.drawFreq(ctx, L.freq);
      this.updateHud();
    },

    updateHud() {
      const parts = [], s = this.sys;
      parts.push(s.stable ? '<b>Stable</b> <span class="dim">(all poles on the left)</span>' : '<span class="warn"><b>Not stable</b>: a pole is on or right of the jω axis</span>');
      const focus = this.drag ? this.drag.item : this.hoverItem;
      if (focus && focus.kind === 'pole') {
        const sg = focus.re, w = focus.im, wn = Math.hypot(sg, w);
        const decay = sg < 0 ? `decays like e<sup>${sg.toFixed(2)}t</sup>` : sg > 0 ? `<span class="warn">grows like e<sup>${sg.toFixed(2)}t</sup></span>` : 'never decays';
        parts.push(`<div>× pole ${sg.toFixed(2)} ${w > 1e-9 ? `± ${w.toFixed(2)}i` : ''}: ${decay}` +
          (w > 1e-9 ? `, oscillates every ${(TAU / w).toFixed(2)} s, damping ratio ζ = ${(-sg / wn).toFixed(2)}` : '') + '</div>');
      } else if (focus) {
        parts.push(`<div>○ zero at ${focus.re.toFixed(2)} ${focus.im > 1e-9 ? `± ${focus.im.toFixed(2)}i` : ''}</div>`);
      } else if (this.hover && this.o.view === '2d') {
        parts.push(`<div class="dim">s = ${this.hover.re.toFixed(2)} ${this.hover.im >= 0 ? '+' : '−'} ${Math.abs(this.hover.im).toFixed(2)}i → e<sup>st</sup> drawn faintly in the time plot</div>`);
      }
      if (this.msg && performance.now() < this.msg.until) parts.push(`<div class="warn">${this.msg.text}</div>`);
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    toS(L, re, im) { return [L.ox + re * L.sc, L.oy - im * L.sc]; },
    fromS(L, x, y) { return [(x - L.ox) / L.sc, (L.oy - y) / L.sc]; },

    drawPlane(ctx, L) {
      const { plane, sc } = L, s = this.sys;
      const x0 = L.ox + RE[0] * sc, x1 = L.ox + RE[1] * sc, y0 = L.oy - IM * sc, y1 = L.oy + IM * sc;
      // heat map of log|H(s)|, cached until something changes
      const cell = 3, cw = Math.ceil((x1 - x0) / cell), ch = Math.ceil((y1 - y0) / cell);
      if (!this.heat || this.heat.w !== cw || this.heat.h !== ch) {
        const off = document.createElement('canvas'); off.width = cw; off.height = ch;
        const octx = off.getContext('2d'), img = octx.createImageData(cw, ch), d = img.data;
        for (let j = 0; j < ch; j++) for (let i = 0; i < cw; i++) {
          const re = RE[0] + ((i + 0.5) / cw) * (RE[1] - RE[0]), im = IM - ((j + 0.5) / ch) * 2 * IM;
          const v = Math.log10(evalH(s.poles, s.zeros, s.K, re, im));
          const t = clamp((v + 1.6) / 3.2, 0, 1), idx = (t * 255) | 0, k = (j * cw + i) * 4;
          let r = SEQ[idx * 3], g = SEQ[idx * 3 + 1], b = SEQ[idx * 3 + 2];
          if (re > 0) { r = r * 0.8 + 40; g *= 0.8; b *= 0.8; } // tint the unstable half
          d[k] = r; d[k + 1] = g; d[k + 2] = b; d[k + 3] = 255;
        }
        octx.putImageData(img, 0, 0);
        this.heat = { w: cw, h: ch, canvas: off };
      }
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.heat.canvas, x0, y0, x1 - x0, y1 - y0);
      // grid & axes
      ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1;
      ctx.font = '10.5px system-ui, sans-serif'; ctx.fillStyle = '#9aa3b2';
      for (let r = Math.ceil(RE[0]); r <= RE[1]; r++) {
        const [x] = this.toS(L, r, 0);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        if (r) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(r), x, L.oy + 4); }
      }
      for (let i = -IM; i <= IM; i++) {
        const [, y] = this.toS(L, 0, i);
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        if (i) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(`${i}i`, L.ox - 5, y); }
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.beginPath(); ctx.moveTo(x0, L.oy); ctx.lineTo(x1, L.oy); ctx.stroke();
      ctx.strokeStyle = COL.jw; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(L.ox, y0); ctx.lineTo(L.ox, y1); ctx.stroke();
      ctx.font = '600 11.5px system-ui, sans-serif'; ctx.fillStyle = COL.jw; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('jω axis = Fourier', L.ox + 6, y0 + 4);
      ctx.fillStyle = '#c9cfda'; ctx.textBaseline = 'bottom';
      ctx.fillText('← decaying (stable)', x0 + 6, y1 - 6);
      ctx.fillStyle = '#ff8a8d'; ctx.textAlign = 'right';
      ctx.fillText('growing →', x1 - 6, y1 - 6);
      ctx.fillStyle = '#c9cfda'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('s-plane · brightness = |H(s)|', plane.x, y0 - 6);

      // hover marker
      if (this.hover && !this.drag) {
        const [hx, hy] = this.toS(L, this.hover.re, this.hover.im);
        ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(hx, hy, 5, 0, TAU); ctx.stroke();
      }
      // poles and zeros (with mirror images)
      for (const it of this.items) {
        for (const sign of it.im > 1e-9 ? [1, -1] : [1]) {
          const [x, y] = this.toS(L, it.re, it.im * sign);
          const active = this.drag && this.drag.item === it, hot = this.hoverItem === it;
          ctx.lineWidth = active || hot ? 3.5 : 2.5;
          ctx.strokeStyle = it.kind === 'pole' && it.re >= 0 ? '#ff6b6f' : COL.pole;
          ctx.globalAlpha = sign < 0 ? 0.75 : 1;
          if (it.kind === 'pole') {
            ctx.beginPath(); ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); ctx.stroke();
          } else {
            ctx.strokeStyle = '#7cf0a8';
            ctx.beginPath(); ctx.arc(x, y, 7, 0, TAU); ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }
      }
    },

    draw3D(ctx, L) {
      const { plane } = L, s = this.sys, o = this.o, N = 46;
      const cx = plane.x + plane.w / 2, cy = plane.y + plane.h * 0.58, S = Math.min(plane.w, plane.h) * 0.42;
      const cyaw = Math.cos(o.yaw), syaw = Math.sin(o.yaw), cp = Math.cos(o.pitch), sp = Math.sin(o.pitch);
      const zOf = (re, im) => clamp(Math.log10(evalH(s.poles, s.zeros, s.K, re, im)), -1.4, 1.6);
      const proj = (re, im, z) => {
        const x = ((re - (RE[0] + RE[1]) / 2) / ((RE[1] - RE[0]) / 2)) * 0.85, y = im / IM, zz = z * 0.32;
        const X = x * cyaw - y * syaw, Y = x * syaw + y * cyaw;
        return [cx + X * S, cy - (zz * cp + Y * sp) * S, Y * cp - zz * sp];
      };
      const grid = [];
      for (let j = 0; j <= N; j++) {
        const row = [];
        for (let i = 0; i <= N; i++) {
          const re = RE[0] + (i / N) * (RE[1] - RE[0]), im = -IM + (j / N) * 2 * IM, z = zOf(re, im);
          row.push({ p: proj(re, im, z), z, re });
        }
        grid.push(row);
      }
      const quads = [];
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const a = grid[j][i], b = grid[j][i + 1], c = grid[j + 1][i + 1], d = grid[j + 1][i];
        quads.push({ pts: [a.p, b.p, c.p, d.p], depth: (a.p[2] + b.p[2] + c.p[2] + d.p[2]) / 4, z: (a.z + b.z + c.z + d.z) / 4, re: a.re });
      }
      quads.sort((q1, q2) => q2.depth - q1.depth);
      ctx.lineWidth = 0.6;
      for (const q of quads) {
        const t = clamp((q.z + 1.4) / 3, 0, 1), idx = (t * 255) | 0;
        let r = SEQ[idx * 3], g = SEQ[idx * 3 + 1], b = SEQ[idx * 3 + 2];
        if (q.re > 0) { r = Math.min(255, r * 0.85 + 40); g *= 0.85; b *= 0.85; }
        ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
        ctx.strokeStyle = 'rgba(0,0,0,0.35)';
        ctx.beginPath(); q.pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      // the jω slice = frequency response
      ctx.strokeStyle = COL.jw; ctx.lineWidth = 3; ctx.beginPath();
      for (let k = 0; k <= 200; k++) {
        const im = -IM + (k / 200) * 2 * IM, [x, y] = proj(0, im, zOf(0, im));
        if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillStyle = '#c9cfda'; ctx.fillText('3D landscape: height = log |H(s)| · drag to rotate', plane.x, plane.y + plane.h - 22);
      ctx.fillStyle = COL.jw; ctx.fillText('blue: slice along the jω axis = frequency response', plane.x, plane.y + plane.h - 4);
      // axis labels
      const lab = (re, im, text) => { const [x, y] = proj(re, im, -1.4); ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.fillText(text, x, y); };
      lab(RE[0], 0, 'σ −'); lab(RE[1], 0, 'σ +'); lab(0, IM, 'jω'); lab(0, -IM, '−jω');
    },

    plotBox(ctx, b, title, legend) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(title, b.x + 8, b.y + 7);
      ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right';
      let lx = b.x + b.w - 8;
      for (const [label, color, dashed] of [...legend].reverse()) {
        ctx.fillStyle = '#aab2c0'; ctx.fillText(label, lx, b.y + 7);
        lx -= ctx.measureText(label).width + 6;
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [4, 3] : []);
        ctx.beginPath(); ctx.moveTo(lx - 14, b.y + 13); ctx.lineTo(lx, b.y + 13); ctx.stroke(); ctx.setLineDash([]);
        lx -= 22;
      }
    },

    line(ctx, b, data, xmax, lo, hi, color, width, dash) {
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y + 22, b.w, b.h - 22); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
      const n = data.length - 1, ix = b.x + 30, iw = b.w - 40, iy = b.y + 28, ih = b.h - 50;
      for (let i = 0; i <= n; i++) {
        const v = clamp(data[i], lo - (hi - lo), hi + (hi - lo));
        const px = ix + (i / n) * iw, py = iy + ih - ((v - lo) / (hi - lo)) * ih;
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke(); ctx.restore();
    },

    axesLabels(ctx, b, lo, hi, xmax, xlabel) {
      const ix = b.x + 30, iw = b.w - 40, iy = b.y + 28, ih = b.h - 50;
      ctx.font = '10.5px system-ui, sans-serif'; ctx.fillStyle = '#7d8696';
      const zeroY = iy + ih - ((0 - lo) / (hi - lo)) * ih;
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
      if (zeroY >= iy && zeroY <= iy + ih) { ctx.beginPath(); ctx.moveTo(ix, zeroY); ctx.lineTo(ix + iw, zeroY); ctx.stroke(); }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [lo, (lo + hi) / 2, hi]) ctx.fillText(+v.toFixed(2) + '', ix - 4, iy + ih - ((v - lo) / (hi - lo)) * ih);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let k = 0; k <= 5; k++) ctx.fillText(+(k / 5 * xmax).toFixed(1) + '', ix + (k / 5) * iw, iy + ih + 4);
      ctx.textAlign = 'right'; ctx.fillText(xlabel, ix + iw, iy + ih + 15);
    },

    drawTime(ctx, b) {
      const s = this.sys;
      let lo = 0, hi = 1;
      for (let i = 0; i < s.hs.length; i++) { lo = Math.min(lo, s.hs[i], s.ss[i]); hi = Math.max(hi, s.hs[i], s.ss[i]); }
      lo = Math.max(lo, -6); hi = Math.min(hi, 6);
      const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
      this.plotBox(ctx, b, 'Time response', [['impulse h(t)', COL.imp], ['step', COL.step], ['e^(st) at cursor', COL.ghost, true]]);
      this.axesLabels(ctx, b, lo, hi, TMAX, 'time t (s)');
      if (this.hover && !this.drag && this.o.view === '2d') {
        const { re, im } = this.hover, n = 400, g = new Float64Array(n + 1);
        const amp = Math.min(hi, -lo, 1) * 0.9 || 0.9;
        for (let i = 0; i <= n; i++) { const t = (i / n) * TMAX; g[i] = amp * Math.exp(re * t) * Math.cos(im * t); }
        this.line(ctx, b, g, TMAX, lo, hi, COL.ghost, 1.4, [4, 3]);
      }
      this.line(ctx, b, s.ss, TMAX, lo, hi, COL.step, 2);
      this.line(ctx, b, s.hs, TMAX, lo, hi, COL.imp, 2.2);
    },

    drawFreq(ctx, b) {
      const s = this.sys;
      let hi = 1;
      for (const v of s.fr) if (Number.isFinite(v)) hi = Math.max(hi, v);
      hi = Math.min(hi * 1.1, 12);
      this.plotBox(ctx, b, 'Frequency response |H(iω)| (= |H| along the jω axis)', [['gain', COL.freq]]);
      this.axesLabels(ctx, b, 0, hi, WMAX, 'frequency ω (rad/s)');
      this.line(ctx, b, s.fr, WMAX, 0, hi, COL.freq, 2.2);
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      const L = this.L;
      if (!L) return;
      if (this.o.view === '3d') {
        if (type === 'down') this.rot = { x, y, yaw: this.o.yaw, pitch: this.o.pitch };
        else if (type === 'move' && this.rot) {
          this.o.yaw = this.rot.yaw + (x - this.rot.x) * 0.01;
          this.o.pitch = clamp(this.rot.pitch + (y - this.rot.y) * 0.01, 0.15, 1.45);
        } else this.rot = null;
        return;
      }
      const [re, im] = this.fromS(L, x, y);
      const inPlane = re >= RE[0] && re <= RE[1] && Math.abs(im) <= IM;
      const pick = () => {
        let best = null, bd = 14;
        for (const it of this.items) for (const sign of it.im > 1e-9 ? [1, -1] : [1]) {
          const [px, py] = this.toS(L, it.re, it.im * sign), d = Math.hypot(px - x, py - y);
          if (d < bd) { bd = d; best = it; }
        }
        return best;
      };
      if (type === 'down') {
        const it = pick();
        if (it) this.drag = { item: it, flip: im < 0 };
      } else if (type === 'move') {
        this.hover = inPlane ? { re, im } : null;
        this.hoverItem = this.drag ? null : pick();
        if (this.drag) {
          const it = this.drag.item;
          it.re = clamp(re, RE[0] - 1, RE[1] + 1);
          let ni = Math.abs(im);
          if (ni < 0.12) ni = 0; // snap to the real axis
          const wasPair = it.im > 1e-9;
          it.im = ni;
          // keep "more poles than zeros" when a real zero becomes a pair
          if (it.kind === 'zero' && !wasPair && ni > 0 && this.countOf('zero') >= this.countOf('pole')) it.im = 0;
          if (it.kind === 'pole' && wasPair && ni === 0 && this.countOf('zero') >= this.countOf('pole')) it.im = 0.12;
          this.update();
        }
      } else if (type === 'up' || type === 'leave') {
        if (this.drag && !inPlane) {
          const it = this.drag.item;
          const after = this.items.filter((i) => i !== it);
          const np = after.filter((i) => i.kind === 'pole').reduce((n, i) => n + (i.im > 1e-9 ? 2 : 1), 0);
          const nz = after.filter((i) => i.kind === 'zero').reduce((n, i) => n + (i.im > 1e-9 ? 2 : 1), 0);
          if (np > nz && np > 0) { this.items = after; this.update(); this.buildPanel(); }
          else { it.re = -1; it.im = it.im; this.update(); this.flash('Can\'t remove that one: H(s) needs more poles than zeros.'); }
        }
        this.drag = null;
        if (type === 'leave') { this.hover = null; this.hoverItem = null; }
      }
      if (this.o.preset && this.drag) this.o.preset = null;
    },
  };

  (window.Sims = window.Sims || {}).laplace = sim;
})();
