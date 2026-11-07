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
  const TAU = Math.PI * 2;
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

  // ------------------------------------------------------------------ 3D plane wave
  // Travels along +x. E lies in the y–z plane, B = x̂ × E / c (drawn scaled up: in SI units B is tiny).
  const PLANE = {
    hint: 'Drag to rotate · use the view buttons to look along the beam',
    X0: -1.7, X1: 1.7, PX: 0.15, // beam extent and polarizer position
    // E-field (y, z components) at position x and time t, before any polarizer
    E(q, x) {
      const ph = TAU * q.waves * ((x - this.X0) / (this.X1 - this.X0)) - q.t, A = 0.55;
      if (q.pol === 'circular') return [A * Math.cos(ph), A * Math.sin(ph)];
      const a = q.pol === 'tilted' ? (q.tilt * Math.PI) / 180 : 0, s = A * Math.sin(ph);
      return [s * Math.cos(a), s * Math.sin(a)];
    },
    field(q, x) { // after the polarizer only the component along its axis survives
      let [ey, ez] = this.E(q, x);
      if (q.polarizer && x > this.PX) {
        const a = (q.polAngle * Math.PI) / 180, ay = Math.cos(a), az = Math.sin(a), d = ey * ay + ez * az;
        ey = d * ay; ez = d * az;
      }
      return { E: [ey, ez], B: [-ez, ey] }; // B = x̂ × E: (0, −Ez, Ey)
    },
    transmitted(q) { // time-averaged intensity fraction after the polarizer (Malus's law)
      if (!q.polarizer) return 1;
      if (q.pol === 'circular') return 0.5;
      const a = ((q.pol === 'tilted' ? q.tilt : 0) - q.polAngle) * (Math.PI / 180);
      return Math.cos(a) ** 2;
    },
    camera(W, H, q) {
      const c = q.cam, cy = Math.cos(c.yaw), sy = Math.sin(c.yaw), cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const S = Math.min(W * 0.27, H * 0.42), ox = W / 2, oy = H * 0.55;
      // world: X along the beam, Y up, Z depth → screen
      return (X, Y, Z) => { const x = X * cy - Z * sy, d = X * sy + Z * cy; return [ox + x * S, oy - (Y * cp + d * sp) * S, d * cp - Y * sp]; };
    },
    frame(ctx, W, H, q) {
      if (q.running) q.t += (q.speed * TAU) / 120;
      if (q.cam.target) {
        const c = q.cam; c.yaw += (c.target.yaw - c.yaw) * 0.1; c.pitch += (c.target.pitch - c.pitch) * 0.1;
        if (Math.abs(c.target.yaw - c.yaw) + Math.abs(c.target.pitch - c.pitch) < 1e-3) { c.yaw = c.target.yaw; c.pitch = c.target.pitch; c.target = null; }
      }
      const P = this.camera(W, H, q), line = (pts, color, width, dash) => {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
        pts.forEach((p3, k) => { const [x, y] = P(...p3); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); ctx.setLineDash([]);
      };
      const label = (p3, txt, color, align = 'left') => { const [x, y] = P(...p3); ctx.fillStyle = color; ctx.font = '600 12px system-ui'; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(txt, x, y); };
      // faint reference planes: E's vertical plane and B's horizontal plane
      for (const [k, col] of [[0, 'rgba(229,72,77,0.07)'], [1, 'rgba(61,139,253,0.07)']]) {
        const c = k === 0 ? [[this.X0, -0.7, 0], [this.X1, -0.7, 0], [this.X1, 0.7, 0], [this.X0, 0.7, 0]] : [[this.X0, 0, -0.7], [this.X1, 0, -0.7], [this.X1, 0, 0.7], [this.X0, 0, 0.7]];
        ctx.fillStyle = col; ctx.beginPath(); c.forEach((p3, i) => { const [x, y] = P(...p3); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.closePath(); ctx.fill();
      }
      // propagation axis
      line([[this.X0 - 0.1, 0, 0], [this.X1 + 0.25, 0, 0]], 'rgba(255,255,255,0.55)', 1.5);
      const tip = P(this.X1 + 0.25, 0, 0), base = P(this.X1 + 0.1, 0, 0), ang = Math.atan2(tip[1] - base[1], tip[0] - base[0]);
      ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.moveTo(tip[0], tip[1]); ctx.lineTo(tip[0] - 11 * Math.cos(ang - 0.4), tip[1] - 11 * Math.sin(ang - 0.4)); ctx.lineTo(tip[0] - 11 * Math.cos(ang + 0.4), tip[1] - 11 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      label([this.X1 + 0.32, 0, 0], 'travel direction (speed c)', '#e8ebf1');
      // polarizer frame with its transmission axis
      if (q.polarizer) {
        const a = (q.polAngle * Math.PI) / 180, R = 0.75, sq = [[-R, -R], [R, -R], [R, R], [-R, R]].map(([y, z]) => [this.PX, y, z]);
        ctx.fillStyle = 'rgba(200,210,225,0.08)'; ctx.beginPath(); sq.forEach((p3, i) => { const [x, y] = P(...p3); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.closePath(); ctx.fill();
        line([...sq, sq[0]], 'rgba(200,210,225,0.5)', 1.2);
        for (let k = -4; k <= 4; k++) { // slits along the transmission axis
          const off = k * 0.15, py = -Math.sin(a) * off, pz = Math.cos(a) * off;
          line([[this.PX, py - Math.cos(a) * 0.7, pz - Math.sin(a) * 0.7], [this.PX, py + Math.cos(a) * 0.7, pz + Math.sin(a) * 0.7]], 'rgba(200,210,225,0.22)', 1);
        }
        line([[this.PX, -Math.cos(a) * 0.85, -Math.sin(a) * 0.85], [this.PX, Math.cos(a) * 0.85, Math.sin(a) * 0.85]], '#e8ebf1', 2.2, [6, 4]);
        label([this.PX, 0.95, 0], 'polarizer', '#c9cfda', 'center');
      }
      // field vectors along the beam
      const n = 64, Ecurve = [], Bcurve = [];
      for (let i = 0; i <= n; i++) {
        const x = this.X0 + (i / n) * (this.X1 - this.X0), f = this.field(q, x);
        Ecurve.push([x, f.E[0], f.E[1]]); Bcurve.push([x, f.B[0], f.B[1]]);
        if (i % 2 === 0) {
          if (q.showB) line([[x, 0, 0], [x, f.B[0], f.B[1]]], 'rgba(61,139,253,0.55)', 1.4);
          line([[x, 0, 0], [x, f.E[0], f.E[1]]], 'rgba(229,72,77,0.65)', 1.6);
        }
      }
      if (q.showB) line(Bcurve, '#5b9bff', 2.6);
      line(Ecurve, '#ff6b6f', 2.8);
      // Poynting vector S = E × B / μ0 : always along +x, size ∝ |E|²
      if (q.showS) {
        for (let i = 2; i < n; i += 6) {
          const x = this.X0 + (i / n) * (this.X1 - this.X0), f = this.field(q, x), s = (f.E[0] ** 2 + f.E[1] ** 2) / 0.3025; // normalised by A²
          if (s < 0.02) continue;
          const a = P(x, -0.9, 0), b = P(x + 0.28 * s, -0.9, 0);
          ctx.strokeStyle = '#f5c451'; ctx.fillStyle = '#f5c451'; ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
          const g = Math.atan2(b[1] - a[1], b[0] - a[0]);
          ctx.beginPath(); ctx.moveTo(...b); ctx.lineTo(b[0] - 8 * Math.cos(g - 0.45), b[1] - 8 * Math.sin(g - 0.45)); ctx.lineTo(b[0] - 8 * Math.cos(g + 0.45), b[1] - 8 * Math.sin(g + 0.45)); ctx.closePath(); ctx.fill();
        }
        label([this.X0, -1.05, 0], 'S = E × B / μ₀ (energy flow)', '#f5c451');
      }
      // legend
      ctx.font = '600 12.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = '#ff6b6f'; ctx.fillText('E  electric field', 20, H - 64);
      if (q.showB) { ctx.fillStyle = '#5b9bff'; ctx.fillText('B  magnetic field (drawn ×c so it is visible)', 20, H - 44); }
    },
    panel(p, sim) {
      const q = sim.o.plane;
      const s1 = UI.section('A light wave in 3D');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered} \textcolor{#ff6b6f}{\mathbf{E}} = E_0 \sin(kx - \omega t)\,\hat{\mathbf y}, \quad \textcolor{#5b9bff}{\mathbf{B}} = \tfrac{E_0}{c} \sin(kx - \omega t)\,\hat{\mathbf z} \\[4pt]
        \mathbf{S} = \tfrac{1}{\mu_0}\,\mathbf{E}\times\mathbf{B}, \qquad c = \tfrac{\omega}{k} = \tfrac{1}{\sqrt{\mu_0\varepsilon_0}} \end{gathered}`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'The textbook picture of light, live. The <b style="color:#ff6b6f">electric</b> and <b style="color:#5b9bff">magnetic</b> fields are <b>perpendicular to each other and to the direction of travel</b>, ' +
        'and they peak together (in phase). The 2D simulation is a slice of exactly this wave, with E pointing out of the screen.';
      s1.append(eq, n1);
      const views = el('div', 'controls-row');
      const fly = (yaw, pitch) => { q.cam.target = { yaw, pitch }; };
      views.append(UI.button('3D', () => fly(-0.55, 0.32)), UI.button('Look along the beam', () => fly(-Math.PI / 2 + 0.0001, 0.0001)), UI.button('Side (see E)', () => fly(0, 0)), UI.button('Top (see B)', () => fly(0, Math.PI / 2 - 0.001)));
      const row = el('div', 'controls-row');
      const play = UI.button(q.running ? '⏸ Pause' : '▶ Play', () => { q.running = !q.running; play.textContent = q.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play);
      s1.append(views, row);
      s1.append(UI.slider({ label: 'Wavelengths in view', min: 0.5, max: 5, step: 0.1, value: q.waves, format: (v) => v.toFixed(1), onInput: (v) => { q.waves = v; } }));
      s1.append(UI.slider({ label: 'Speed', min: 0.1, max: 3, step: 0.1, value: q.speed, format: (v) => `${v.toFixed(1)}×`, onInput: (v) => { q.speed = v; } }));
      s1.append(UI.check({ label: 'Show B', checked: q.showB, onChange: (v) => { q.showB = v; } }));
      s1.append(UI.check({ label: 'Show energy flow (Poynting vector)', checked: q.showS, onChange: (v) => { q.showS = v; } }));

      const s2 = UI.section('Polarization');
      s2.append(UI.segmented({ options: [{ value: 'vertical', label: 'Vertical' }, { value: 'tilted', label: 'Tilted' }, { value: 'circular', label: 'Circular' }], value: q.pol, onChange: (v) => { q.pol = v; sim.buildPanel(); } }));
      if (q.pol === 'tilted') s2.append(UI.slider({ label: 'Tilt of E', min: 0, max: 180, step: 1, value: q.tilt, format: (v) => `${v}°`, onInput: (v) => { q.tilt = v; } }));
      const pn = el('p', 'note');
      pn.innerHTML = q.pol === 'circular'
        ? '<b>Circular:</b> E keeps its length and rotates as the wave moves, tracing a corkscrew, the same helix as e<sup>iωt</sup> in the Fourier tab. It is two perpendicular linear waves a quarter-cycle apart. Look along the beam to see the circle.'
        : '<b>Linear:</b> E always swings back and forth along one line. Look along the beam and the tip of E draws a straight line; its angle is the polarization.';
      s2.append(pn);
      s2.append(UI.termCard({
        color: '#c9cfda', tex: String.raw`I = I_0 \cos^2\theta`, title: "Polarizer & Malus's law",
        body: 'A polarizing filter only lets through the part of E along its axis. Amplitude scales by cos θ, so <b>intensity</b> (∝ E²) scales by <b>cos²θ</b>. At 90° nothing passes. ' +
          'Circular light always loses half. Polarized sunglasses block glare (which is mostly horizontally polarized); LCD screens and 3D cinema glasses work the same way.',
        controls: [
          UI.check({ label: 'Put a polarizer in the beam', checked: q.polarizer, onChange: (v) => { q.polarizer = v; } }),
          UI.slider({ label: 'Polarizer axis', min: 0, max: 180, step: 1, value: q.polAngle, format: (v) => `${v}°`, onInput: (v) => { q.polAngle = v; } }),
        ],
      }));
      const s3 = UI.section('Why it travels');
      s3.append(UI.termCard({
        color: '#ff6b6f', tex: String.raw`\nabla\times\mathbf{E} = -\tfrac{\partial\mathbf{B}}{\partial t}, \;\; \nabla\times\mathbf{B} = \mu_0\varepsilon_0\tfrac{\partial\mathbf{E}}{\partial t}`, title: 'E and B keep each other going',
        body: 'Where E varies along the beam, B must change in time (Faraday); where B varies, E must change in time (Ampère–Maxwell). Each one regenerates the other a little further on, so the pattern ' +
          'moves without any wire or medium. Plug the sine wave in and the equations demand speed ω/k = 1/√(μ₀ε₀) ≈ 3 × 10⁸ m/s. Maxwell computed this and realised <b>light is an electromagnetic wave</b>.',
      }));
      p.append(s1, s2, s3);
    },
    hud(q) {
      const tr = this.transmitted(q);
      let html = `E ⟂ B ⟂ direction of travel · E and B in phase · |B| = |E|/c`;
      if (q.polarizer) {
        let d = ((((q.pol === 'tilted' ? q.tilt : 0) - q.polAngle) % 180) + 180) % 180;
        if (d > 90) d = 180 - d; // angle between the light's E and the filter axis, 0…90°
        html += `<div>after the polarizer: intensity <b>${(tr * 100).toFixed(1)}%</b> ${q.pol === 'circular' ? '(circular light: always 50%)' : `= cos²(${d.toFixed(0)}°)`}</div>`;
      }
      return html;
    },
    pointer(type, x, y, q) {
      if (type === 'down') { this.rot = { x, y, yaw: q.cam.yaw, pitch: q.cam.pitch }; q.cam.target = null; }
      else if (type === 'move' && this.rot) { q.cam.yaw = this.rot.yaw - (x - this.rot.x) * 0.008; q.cam.pitch = Math.max(-1.5, Math.min(1.5, this.rot.pitch + (y - this.rot.y) * 0.008)); }
      else if (type === 'up' || type === 'leave') this.rot = null;
    },
  };

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'em',
    get hint() { return this.o.view3 === 'plane' ? PLANE.hint : 'Click / drag on the canvas with the selected tool'; },
    o: {
      preset: 'slit', tool: 'point', view: 'field', arrows: false, n: 1.5, brush: 4, running: true, speed: 3, lamS: null, view3: 'sim',
      plane: { pol: 'vertical', tilt: 45, waves: 2, speed: 1, showB: true, showS: false, polarizer: false, polAngle: 30, t: 0, running: true, cam: { yaw: -0.55, pitch: 0.32 } },
    },
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

      const s0 = UI.section('View');
      s0.append(UI.segmented({
        options: [{ value: 'sim', label: '2D simulation (FDTD)' }, { value: 'plane', label: '3D plane wave' }],
        value: o.view3, onChange: (v) => {
          o.view3 = v; this.buildPanel();
          document.getElementById('stageHint').textContent = this.hint;
        },
      }));
      p.append(s0);
      if (o.view3 === 'plane') { PLANE.panel(p, this); return; }

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
      if (!this.hud) return;
      if (this.o.view3 === 'plane') { const html = PLANE.hud(this.o.plane); if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; } return; }
      if (!this.s) return;
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
      if (this.o.view3 === 'plane') {
        PLANE.frame(ctx, W, H, this.o.plane);
        if (++this.hudTick % 6 === 0) this.updateHud();
        return;
      }
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
      if (this.o.view3 === 'plane') { PLANE.pointer(type, x, y, this.o.plane); return; }
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

  sim._internals = { PLANE };
  (window.Sims = window.Sims || {}).em = sim;
})();
