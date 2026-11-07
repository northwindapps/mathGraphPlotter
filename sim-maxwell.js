/* 2D Maxwell's equations, FDTD (Yee, 1966), TMz polarisation:
 * E points out of/into the screen (Ez), B circulates in the plane (Hx, Hy).
 * Normalised units: c = 1, cell size = 1, Courant number Sc = c·dt/dx = 0.5.
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { far: '#e5484d', amp: '#3d8bfd', gauss: '#2fa36b', mat: '#e09b14' };
  const SEQ = UI.lut(UI.SEQ_STOPS), DIV = UI.lut(UI.DIV_STOPS);
  const TARGET_CELLS = 60000;
  const SC = 0.5;
  const VAC = 0, METAL = 1, GLASS = 2;

  class FDTD {
    constructor(nx, ny) {
      this.nx = nx; this.ny = ny;
      const n = nx * ny, F = () => new Float32Array(n);
      this.ez = F(); this.hx = F(); this.hy = F(); this.avg = F();
      this.ca = F(); this.cb = F(); this.da = F(); this.db = F(); this.sponge = F();
      this.mat = new Uint8Array(n);
      this.sources = [];
      this.t = 0;
      this.lambda = 24;
      this.n = 1.5;
      // Absorbing border ("sponge"): graded loss so waves leave instead of bouncing back.
      const L = (this.L = Math.max(12, Math.min(30, Math.round(Math.min(nx, ny) * 0.08))));
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const d = Math.min(i, j, nx - 1 - i, ny - 1 - j);
        this.sponge[i + nx * j] = d < L ? 0.45 * ((L - d) / L) ** 3 : 0;
      }
      this.updateCoeffs();
    }

    updateCoeffs() {
      const eps = this.n * this.n, metal = [];
      for (let c = 0; c < this.ez.length; c++) {
        const s = this.sponge[c], e = this.mat[c] === GLASS ? eps : 1;
        this.ca[c] = (1 - s) / (1 + s);
        this.cb[c] = SC / e / (1 + s);
        this.da[c] = (1 - s) / (1 + s);
        this.db[c] = SC / (1 + s);
        if (this.mat[c] === METAL) metal.push(c);
      }
      this.metalIdx = Int32Array.from(metal);
    }

    clearFields() {
      for (const a of [this.ez, this.hx, this.hy, this.avg]) a.fill(0);
      this.t = 0;
      for (const s of this.sources) s.t0 = 0;
    }

    addPoint(i, j) {
      const { nx, ny } = this;
      i = Math.max(1, Math.min(nx - 2, Math.round(i)));
      j = Math.max(1, Math.min(ny - 2, Math.round(j)));
      this.sources.push({ kind: 'point', i, j, cells: Int32Array.of(i + nx * j), amp: 1, t0: this.t });
    }

    addLine(i) {
      const { nx, ny, L } = this, cells = [];
      i = Math.max(1, Math.min(nx - 2, Math.round(i)));
      for (let j = L; j < ny - L; j++) cells.push(i + nx * j);
      this.sources.push({ kind: 'line', i, j0: L, j1: ny - L, cells: Int32Array.from(cells), amp: 0.25, t0: this.t });
    }

    paint(ci, cj, r, mat) {
      const { nx, ny } = this;
      for (let j = Math.max(0, Math.floor(cj - r)); j <= Math.min(ny - 1, Math.ceil(cj + r)); j++) {
        for (let i = Math.max(0, Math.floor(ci - r)); i <= Math.min(nx - 1, Math.ceil(ci + r)); i++) {
          if ((i - ci) ** 2 + (j - cj) ** 2 <= r * r) this.mat[i + nx * j] = mat;
        }
      }
      if (mat === VAC) {
        this.sources = this.sources.filter((s) =>
          s.kind === 'point' ? (s.i - ci) ** 2 + (s.j - cj) ** 2 > r * r : Math.abs(s.i - ci) > r);
      }
    }

    step() {
      const { nx, ny, ez, hx, hy, ca, cb, da, db } = this;
      // Faraday:  ∂B/∂t = −∇×E
      for (let j = 0; j < ny - 1; j++) {
        for (let c = j * nx, e = c + nx; c < e; c++) hx[c] = da[c] * hx[c] - db[c] * (ez[c + nx] - ez[c]);
      }
      for (let j = 0; j < ny; j++) {
        for (let c = j * nx, e = c + nx - 1; c < e; c++) hy[c] = da[c] * hy[c] + db[c] * (ez[c + 1] - ez[c]);
      }
      // Ampère–Maxwell:  ∂D/∂t = ∇×H − J
      for (let j = 1; j < ny; j++) {
        for (let c = j * nx + 1, e = j * nx + nx; c < e; c++) {
          ez[c] = ca[c] * ez[c] + cb[c] * ((hy[c] - hy[c - 1]) - (hx[c] - hx[c - nx]));
        }
      }
      // J: oscillating currents at the sources (smoothly switched on)
      const omega = (2 * Math.PI) / this.lambda, ramp = 2 * this.lambda;
      for (const s of this.sources) {
        const age = this.t - s.t0;
        const env = age < ramp ? 0.5 - 0.5 * Math.cos((Math.PI * age) / ramp) : 1;
        const val = s.amp * env * Math.sin(omega * age) * SC;
        for (let k = 0; k < s.cells.length; k++) ez[s.cells[k]] += val;
      }
      // Perfect conductor: E = 0 inside metal
      for (let k = 0; k < this.metalIdx.length; k++) ez[this.metalIdx[k]] = 0;
      // Running average of E² (what a light detector / your eye sees)
      const a = this.avg;
      for (let c = 0; c < a.length; c++) a[c] += 0.01 * (ez[c] * ez[c] - a[c]);
      this.t += SC;
    }
  }

  // Presets in grid coordinates. lam = wavelength as a fraction of grid height.
  const PRESETS = {
    point: {
      label: 'Point source', lam: 1 / 9,
      about: 'One oscillating charge radiates circular waves that travel outward at the speed of light.',
      build(s) { s.addPoint(s.nx * 0.5, s.ny * 0.5); },
    },
    two: {
      label: 'Two sources', lam: 1 / 10,
      about: 'Two sources in step. Where crests meet crests the waves add up; where crests meet troughs they cancel. ' +
        'Switch to <b>Intensity</b> to see the bright and dark fringes clearly.',
      build(s) { const d = 1.5 * s.lambda; s.addPoint(s.nx * 0.45, s.ny / 2 - d); s.addPoint(s.nx * 0.45, s.ny / 2 + d); },
    },
    slit: {
      label: 'Double slit', lam: 1 / 14,
      about: "Young's experiment: a plane wave hits a metal wall with two slits. Each slit acts as a new source, " +
        'and the two interfere. Try <b>Intensity</b> view.',
      build(s) {
        const x0 = Math.round(s.nx * 0.3), half = Math.max(2, Math.round(s.lambda * 0.35)), d = Math.round(s.lambda * 1.6);
        for (let j = 0; j < s.ny; j++) {
          const y = j - s.ny / 2;
          if (Math.abs(y - d) <= half || Math.abs(y + d) <= half) continue;
          for (let i = x0; i < x0 + 3; i++) s.mat[i + s.nx * j] = METAL;
        }
        s.addLine(s.nx * 0.1);
      },
    },
    refraction: {
      label: 'Refraction', lam: 1 / 12,
      about: 'Light enters glass at an angle. It slows to c/n, so its wavelength shrinks and the wavefront bends. ' +
        'Drag the <b>n</b> slider to change how strongly it bends.',
      build(s) {
        for (let j = 0; j < s.ny; j++) for (let i = 0; i < s.nx; i++) {
          if (i > s.nx * 0.5 + (j - s.ny / 2) * 0.6) s.mat[i + s.nx * j] = GLASS;
        }
        s.addLine(s.nx * 0.12);
      },
    },
    lens: {
      label: 'Lens', lam: 1 / 18,
      about: 'A convex glass lens is thicker in the middle, so the centre of the wave is delayed the most. ' +
        'The flat wavefront curves and converges to a focus.',
      build(s) {
        const R = 0.45 * s.ny, h = 0.065 * s.ny, xc = s.nx * 0.33, yc = s.ny / 2;
        for (let j = 0; j < s.ny; j++) for (let i = 0; i < s.nx; i++) {
          const d1 = Math.hypot(i - (xc + R - h), j - yc), d2 = Math.hypot(i - (xc - R + h), j - yc);
          if (d1 < R && d2 < R) s.mat[i + s.nx * j] = GLASS;
        }
        s.addLine(s.nx * 0.1);
      },
    },
    dish: {
      label: 'Parabolic dish', lam: 1 / 16,
      about: 'A source at the focus of a metal parabola: every reflected ray leaves parallel, giving a beam. ' +
        'Satellite dishes and car headlights work this way.',
      build(s) {
        const F = 0.14 * s.ny, xv = s.nx * 0.12, yc = s.ny / 2, span = 0.42 * s.ny;
        for (let j = 0; j < s.ny; j++) {
          const y = j - yc;
          if (Math.abs(y) > span) continue;
          const xp = xv + (y * y) / (4 * F), slope = Math.abs(y) / (2 * F), thick = 1.6 * Math.sqrt(1 + slope * slope);
          for (let i = Math.floor(xp - thick); i <= Math.ceil(xp + thick); i++) {
            if (i >= 0 && i < s.nx && Math.abs(i - xp) <= thick) s.mat[i + s.nx * j] = METAL;
          }
        }
        s.addPoint(xv + F, yc);
      },
    },
  };

  const TOOLS = [
    { value: 'point', label: '• Source', title: 'Click to place an oscillating point source' },
    { value: 'line', label: '| Plane wave', title: 'Click to place a vertical line of sources (a flat wavefront)' },
    { value: 'metal', label: 'Metal', title: 'Paint perfectly conducting metal (reflects)' },
    { value: 'glass', label: 'Glass', title: 'Paint glass (slows light to c/n)' },
    { value: 'erase', label: 'Erase', title: 'Remove material and sources' },
  ];
  const SPEEDS = [{ value: 0.34, label: 'Slow-mo' }, { value: 3, label: 'Normal' }, { value: 8, label: 'Fast' }];

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'em',
    hint: 'Click / drag on the canvas with the selected tool',
    o: { preset: 'slit', tool: 'point', view: 'field', arrows: false, n: 1.5, brush: 4, running: true, speed: 3, lamS: null },
    s: null, W: 0, H: 0, acc: 0, hudTick: 0, drag: null,
    scale: { e: 0.05, i: 0.002, h: 0.05 },

    mount(panel, hud) { this.panel = panel; this.hud = hud; this.buildPanel(); this.updateHud(); },
    unmount() { this.drag = null; },

    resize(W, H) {
      this.W = W; this.H = H;
      const cell = Math.sqrt((W * H) / TARGET_CELLS);
      const nx = Math.max(60, Math.round(W / cell)), ny = Math.max(60, Math.round(H / cell));
      if (!this.s || this.s.nx !== nx || this.s.ny !== ny) this.loadPreset(this.o.preset, nx, ny);
    },

    loadPreset(key, nx = this.s?.nx, ny = this.s?.ny) {
      if (!nx) return;
      const pr = PRESETS[key];
      this.o.preset = key;
      const s = (this.s = new FDTD(nx, ny));
      s.n = this.o.n;
      s.lambda = Math.round(pr.lam * ny);
      this.o.lamS = s.lambda;
      pr.build(s);
      s.updateCoeffs();
      this.scale = { e: 0.05, i: 0.002, h: 0.05 };
      this.off = document.createElement('canvas');
      this.off.width = nx; this.off.height = ny;
      this.octx = this.off.getContext('2d');
      this.img = this.octx.createImageData(nx, ny);
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();

      const s1 = UI.section('Experiment');
      s1.append(UI.segmented({
        options: Object.entries(PRESETS).map(([value, pr]) => ({ value, label: pr.label })),
        value: o.preset, onChange: (v) => { this.loadPreset(v); this.buildPanel(); },
      }));
      const about = el('p', 'note');
      about.innerHTML = PRESETS[o.preset].about;
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => {
        o.running = !o.running;
        play.textContent = o.running ? '⏸ Pause' : '▶ Play';
      });
      row.append(play,
        UI.button('Step ›', () => { this.s.step(); }),
        UI.button('↺ Restart', () => this.s.clearFields()),
        UI.button('Clear all', () => { this.s.mat.fill(0); this.s.sources = []; this.s.updateCoeffs(); this.s.clearFields(); }));
      s1.append(about, row,
        UI.segmented({ options: SPEEDS, value: o.speed, onChange: (v) => { o.speed = v; } }));

      const s2 = UI.section("Maxwell's equations");
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered}
        \textcolor{${COL.gauss}}{\nabla\cdot\mathbf{D} = \rho \qquad \nabla\cdot\mathbf{B} = 0} \\
        \textcolor{${COL.far}}{\nabla\times\mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}} \\
        \textcolor{${COL.amp}}{\nabla\times\mathbf{H} = \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}} \\
        \textcolor{${COL.mat}}{\mathbf{D} = \varepsilon\mathbf{E},\quad \mathbf{B} = \mu\mathbf{H}}
      \end{gathered}`, true);
      s2.append(eq, el('p', 'note',
        'This is a 2D slice. E points straight into or out of the screen (red / blue), and B loops around in the plane (turn on the arrows).'));

      s2.append(UI.termCard({
        color: COL.far, tex: String.raw`\nabla\times\mathbf{E} = -\tfrac{\partial \mathbf{B}}{\partial t}`, title: "Faraday's law",
        body: 'Wherever E varies from place to place (it "curls"), B changes in time. Every step, the simulation uses this law to update B from E.',
      }));
      s2.append(UI.termCard({
        color: COL.amp, tex: String.raw`\nabla\times\mathbf{H} = \mathbf{J} + \tfrac{\partial \mathbf{D}}{\partial t}`, title: 'Ampère–Maxwell law',
        body: 'Wherever B curls, E changes in time. Maxwell added the <b>∂D/∂t</b> term. Without it a changing E could not create B, ' +
          'the hand-off below would break, and there would be no light. <b>J</b> is the oscillating current in each source.',
        controls: [UI.slider({
          label: 'Source wavelength λ', min: 8, max: 60, step: 1, value: o.lamS || 24,
          format: (v) => `${v} cells`, onInput: (v) => { o.lamS = v; this.s.lambda = v; },
        })],
      }));
      s2.append(UI.termCard({
        color: COL.gauss, tex: String.raw`\nabla\cdot\mathbf{D} = \rho,\;\nabla\cdot\mathbf{B} = 0`, title: "Gauss's laws",
        body: 'Charges are where field lines begin and end, and magnetic field lines never end. ' +
          'There is no free charge here, and in this 2D slice both laws hold automatically.',
      }));
      s2.append(UI.termCard({
        color: COL.mat, tex: String.raw`\varepsilon = \varepsilon_0 n^2,\;\; v = c/n`, title: 'Materials',
        body: 'Glass has a larger ε, so light slows to c/n and its wavelength shrinks, which is why rays bend (refraction). ' +
          'Metal forces E = 0, so it reflects.',
        controls: [UI.slider({
          label: 'Glass refractive index n', min: 1, max: 3, step: 0.05, value: o.n,
          format: (v) => v.toFixed(2), onInput: (v) => { o.n = v; this.s.n = v; this.s.updateCoeffs(); },
        })],
      }));
      const how = el('p', 'note');
      how.innerHTML = '<b>How it is solved:</b> FDTD (Yee, 1966). Space is a grid. Each step, Faraday updates B from how E curls, ' +
        'then Ampère–Maxwell updates E from how B curls. That endless hand-off between E and B <i>is</i> the wave. ' +
        'Use <b>Slow-mo</b> or <b>Step</b> to watch it.';
      s2.append(how);

      const s3 = UI.section('Draw');
      s3.append(UI.segmented({ options: TOOLS, value: o.tool, onChange: (v) => { o.tool = v; } }));
      s3.append(UI.slider({ label: 'Brush size', min: 1, max: 15, step: 1, value: o.brush, format: (v) => `${v} cells`, onInput: (v) => { o.brush = v; } }));

      const s4 = UI.section('Show');
      s4.append(UI.segmented({
        options: [{ value: 'field', label: 'Electric field E' }, { value: 'intensity', label: 'Intensity (brightness)' }],
        value: o.view, onChange: (v) => { o.view = v; this.updateHud(); },
      }));
      s4.append(UI.check({ label: 'Magnetic field arrows (B)', checked: o.arrows, onChange: (v) => { o.arrows = v; } }));

      p.append(s1, s2, s3, s4);
    },

    updateHud() {
      if (!this.hud || !this.s) return;
      const o = this.o, parts = [];
      parts.push(`λ = ${this.s.lambda} cells in vacuum · glass n = ${o.n.toFixed(2)} → λ in glass = ${(this.s.lambda / o.n).toFixed(1)}`);
      parts.push(o.view === 'field'
        ? UI.legend(UI.DIV_STOPS, 'E out of screen', 'E into screen')
        : UI.legend(UI.SEQ_STOPS, 'dark', 'bright'));
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const s = this.s;
      if (!s) return;
      if (this.o.running) {
        this.acc += this.o.speed;
        while (this.acc >= 1) { s.step(); this.acc -= 1; }
      }
      this.render(ctx, W, H);
      if (++this.hudTick % 10 === 0) this.updateHud();
    },

    render(ctx, W, H) {
      const s = this.s, o = this.o, { nx, ny, ez, avg, mat, L } = s, data = this.img.data;
      const smooth = (key, val) => (this.scale[key] = Math.max(1e-6, this.scale[key] * 0.95 + val * 0.05));

      // Scale by RMS over the interior (a peak would be dominated by the source itself)
      let sum = 0, cnt = 0;
      for (let j = L; j < ny - L; j += 2) for (let i = L; i < nx - L; i += 2) {
        const c = i + nx * j;
        sum += o.view === 'field' ? ez[c] * ez[c] : avg[c];
        cnt++;
      }
      const scale = o.view === 'field' ? smooth('e', 2.5 * Math.sqrt(sum / cnt)) : smooth('i', 4 * (sum / cnt));

      for (let c = 0, k = 0; c < ez.length; c++, k += 4) {
        let r, g, b;
        if (mat[c] === METAL) { r = 150; g = 156; b = 168; }
        else {
          if (o.view === 'field') {
            const idx = Math.max(0, Math.min(255, (0.5 + (0.5 * ez[c]) / scale) * 255)) | 0;
            r = DIV[idx * 3]; g = DIV[idx * 3 + 1]; b = DIV[idx * 3 + 2];
          } else {
            const idx = Math.min(255, Math.sqrt(avg[c] / scale) * 255) | 0; // sqrt ≈ how the eye sees brightness
            r = SEQ[idx * 3]; g = SEQ[idx * 3 + 1]; b = SEQ[idx * 3 + 2];
          }
          if (mat[c] === GLASS) { r = r * 0.85 + 22; g = g * 0.85 + 40; b = b * 0.85 + 52; }
        }
        data[k] = r; data[k + 1] = g; data[k + 2] = b; data[k + 3] = 255;
      }
      this.octx.putImageData(this.img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.off, 0, 0, W, H);

      const sx = W / nx, sy = H / ny;
      // sources
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      for (const src of s.sources) {
        if (src.kind === 'point') {
          ctx.beginPath(); ctx.arc((src.i + 0.5) * sx, (src.j + 0.5) * sy, 3, 0, Math.PI * 2); ctx.fill();
          ctx.beginPath(); ctx.arc((src.i + 0.5) * sx, (src.j + 0.5) * sy, 7, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo((src.i + 0.5) * sx, src.j0 * sy); ctx.lineTo((src.i + 0.5) * sx, src.j1 * sy); ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // B arrows
      if (o.arrows) {
        const step = 26;
        let m = 0;
        for (let c = 0; c < ez.length; c += 7) { const h = s.hx[c] * s.hx[c] + s.hy[c] * s.hy[c]; m += h; }
        const hs = smooth('h', 2.5 * Math.sqrt(m / (ez.length / 7)));
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let y = step / 2; y < H; y += step) for (let x = step / 2; x < W; x += step) {
          const i = Math.min(nx - 1, (x / sx) | 0), j = Math.min(ny - 1, (y / sy) | 0), c = i + nx * j;
          const bx = s.hx[c], by = s.hy[c], mag = Math.hypot(bx, by);
          if (mag < hs * 0.08 || mat[c] === METAL) continue;
          const len = Math.min(1, mag / hs) * step * 0.85, dx = bx / mag, dy = by / mag;
          const x1 = x + (dx * len) / 2, y1 = y + (dy * len) / 2;
          ctx.moveTo(x - (dx * len) / 2, y - (dy * len) / 2); ctx.lineTo(x1, y1);
          const hh = Math.min(5, len * 0.35);
          ctx.moveTo(x1, y1); ctx.lineTo(x1 - hh * (dx + dy * 0.6), y1 - hh * (dy - dx * 0.6));
          ctx.moveTo(x1, y1); ctx.lineTo(x1 - hh * (dx - dy * 0.6), y1 - hh * (dy + dx * 0.6));
        }
        ctx.stroke();
      }
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      const s = this.s;
      if (!s) return;
      const ci = (x / this.W) * s.nx - 0.5, cj = (y / this.H) * s.ny - 0.5, o = this.o;
      const matOf = { metal: METAL, glass: GLASS, erase: VAC };
      if (type === 'down') {
        if (o.tool === 'point') { s.addPoint(ci, cj); return; }
        if (o.tool === 'line') { s.addLine(ci); return; }
        s.paint(ci, cj, o.brush, matOf[o.tool]);
        s.updateCoeffs();
        this.drag = { ci, cj };
      } else if (type === 'move' && this.drag) {
        const d = this.drag, n = Math.max(1, Math.ceil(Math.hypot(ci - d.ci, cj - d.cj)));
        for (let k = 1; k <= n; k++) s.paint(d.ci + ((ci - d.ci) * k) / n, d.cj + ((cj - d.cj) * k) / n, o.brush, matOf[o.tool]);
        s.updateCoeffs();
        this.drag = { ci, cj };
      } else if (type === 'up' || type === 'leave') {
        this.drag = null;
      }
    },
  };

  (window.Sims = window.Sims || {}).em = sim;
})();
