/* 2D incompressible Navier–Stokes, "Stable Fluids" (Jos Stam, 1999).
 *   ∂u/∂t + (u·∇)u = −∇p/ρ + ν∇²u + f,   ∇·u = 0
 * Grid units: 1 cell = 1 length unit, velocities in cells/second.
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { adv: '#e5484d', pres: '#3d8bfd', visc: '#2fa36b', force: '#e09b14' };
  const SEQ = UI.lut(UI.SEQ_STOPS), DIV = UI.lut(UI.DIV_STOPS);
  const TARGET_CELLS = 19000;
  const VORT_SCALE = 12;
  const D_CYL = 0.15; // cylinder diameter, in units of the tank height
  const nuOf = (s) => (s <= 0 ? 0 : 1e-5 * Math.pow(10, s * 3.5)); // slider 0..1 -> ν (height²/s)

  const SCENARIOS = {
    stir: {
      label: 'Stir a tank', mode: 'box', nuS: 0.15, vortS: 0.3, U: 0.5,
      about: 'A closed tank of still fluid. Drag to stir it and drop dye. Watch swirls form, merge and slowly die out.',
    },
    tunnel: {
      label: 'Wind tunnel', mode: 'tunnel', nuS: 0.3, vortS: 0, U: 0.5,
      about: 'Fluid flows left → right past a cylinder. With low viscosity (high Reynolds number) the wake sheds vortices ' +
        'alternately: a <b>von Kármán vortex street</b>. Raise ν and the flow turns smooth and symmetric.',
    },
    shear: {
      label: 'Shear layer', mode: 'shear', nuS: 0, vortS: 0, U: 0.5,
      about: 'Two layers slide past each other. A tiny ripple grows into rolling waves: the ' +
        '<b>Kelvin–Helmholtz instability</b>, the same physics as wind over water or billowing clouds.',
    },
  };
  const VIEWS = [
    { value: 'dye', label: 'Dye' },
    { value: 'speed', label: 'Speed |u|' },
    { value: 'vorticity', label: 'Swirl (vorticity)' },
    { value: 'pressure', label: 'Pressure p' },
  ];

  // ------------------------------------------------------------------ solver
  class Fluid {
    constructor(nx, ny, mode) {
      this.nx = nx; this.ny = ny; this.w = nx + 2; this.mode = mode;
      const n = (nx + 2) * (ny + 2);
      const F = () => new Float32Array(n);
      this.u = F(); this.v = F(); this.u0 = F(); this.v0 = F();
      this.p = F(); this.div = F(); this.tmp = F(); this.tmp2 = F(); this.curl = F();
      this.dye = [F(), F(), F()];
      this.solid = new Uint8Array(n);
      this.inflow = 0;
      this.rebuildMask();
    }

    rebuildMask() {
      const { nx, ny, w, solid } = this;
      const fluid = [], nb = [], solids = [];
      for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
        const c = i + w * j;
        if (solid[c]) { solids.push(c); continue; }
        fluid.push(c);
        nb.push(4 - solid[c - 1] - solid[c + 1] - solid[c - w] - solid[c + w]);
      }
      this.fluid = Int32Array.from(fluid);
      this.nb = Float32Array.from(nb);
      this.solidIdx = Int32Array.from(solids);
    }

    // b: 0 scalar, 1 = x-velocity, 2 = y-velocity, 3 = pressure
    setBnd(b, x) {
      const { nx, ny, w, mode } = this;
      for (let j = 1; j <= ny; j++) {
        const L = w * j, R = nx + 1 + w * j;
        if (mode === 'shear') { x[L] = x[nx + w * j]; x[R] = x[1 + w * j]; } // periodic left/right
        else if (mode === 'tunnel') {
          x[L] = b === 1 ? this.inflow : b === 2 ? 0 : x[L + 1];                  // inflow
          x[R] = b === 3 ? -x[R - 1] : x[R - 1];                                   // open outflow (p = 0)
        } else {
          x[L] = b === 1 ? -x[L + 1] : x[L + 1];                                   // solid walls
          x[R] = b === 1 ? -x[R - 1] : x[R - 1];
        }
      }
      for (let i = 0; i <= nx + 1; i++) { // top & bottom walls (free-slip)
        const T = i, B = i + w * (ny + 1);
        x[T] = b === 2 ? -x[T + w] : x[T + w];
        x[B] = b === 2 ? -x[B - w] : x[B - w];
      }
      if (b === 1 || b === 2) for (let k = 0; k < this.solidIdx.length; k++) x[this.solidIdx[k]] = 0; // no-slip obstacle
    }

    diffuse(b, x, x0, a, iters) {
      const { fluid, w } = this, inv = 1 / (1 + 4 * a);
      for (let it = 0; it < iters; it++) {
        for (let k = 0; k < fluid.length; k++) {
          const c = fluid[k];
          x[c] = (x0[c] + a * (x[c - 1] + x[c + 1] + x[c - w] + x[c + w])) * inv;
        }
        this.setBnd(b, x);
      }
    }

    // Semi-Lagrangian: trace each cell backwards along the velocity and sample there.
    advect(b, d, d0, u, v, dt) {
      const { nx, ny, w, solid } = this, periodic = this.mode === 'shear';
      for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
        const c = i + w * j;
        if (solid[c]) { d[c] = 0; continue; }
        let x = i - dt * u[c], y = j - dt * v[c];
        if (periodic) x = (((x - 0.5) % nx) + nx) % nx + 0.5;
        else x = x < 0.5 ? 0.5 : x > nx + 0.5 ? nx + 0.5 : x;
        y = y < 0.5 ? 0.5 : y > ny + 0.5 ? ny + 0.5 : y;
        const i0 = x | 0, j0 = y | 0, s1 = x - i0, t1 = y - j0, s0 = 1 - s1, t0 = 1 - t1;
        const a = i0 + w * j0;
        d[c] = s0 * (t0 * d0[a] + t1 * d0[a + w]) + s1 * (t0 * d0[a + 1] + t1 * d0[a + 1 + w]);
      }
      this.setBnd(b, d);
    }

    // MacCormack advection for dye: forward + backward trace to cancel most of the blur,
    // clamped to the values it was interpolated from so it can't overshoot.
    advectSharp(d, u, v, dt) {
      const { nx, ny, w, solid, tmp, tmp2 } = this, periodic = this.mode === 'shear';
      tmp.set(d);                              // φ
      this.advect(0, tmp2, tmp, u, v, dt);     // φ̂  (forward)
      this.advect(0, d, tmp2, u, v, -dt);      // φ̃  (back again)
      for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
        const c = i + w * j;
        if (solid[c]) { d[c] = 0; continue; }
        let x = i - dt * u[c], y = j - dt * v[c];
        if (periodic) x = (((x - 0.5) % nx) + nx) % nx + 0.5;
        else x = x < 0.5 ? 0.5 : x > nx + 0.5 ? nx + 0.5 : x;
        y = y < 0.5 ? 0.5 : y > ny + 0.5 ? ny + 0.5 : y;
        const a = (x | 0) + w * (y | 0);
        const p0 = tmp[a], p1 = tmp[a + 1], p2 = tmp[a + w], p3 = tmp[a + w + 1];
        const mn = Math.min(p0, p1, p2, p3), mx = Math.max(p0, p1, p2, p3);
        const val = tmp2[c] + 0.5 * (tmp[c] - d[c]);
        d[c] = val < mn ? mn : val > mx ? mx : val;
      }
      this.setBnd(0, d);
    }

    // Solve ∇²p = ∇·u, then u ← u − ∇p, so the result has no divergence.
    project(iters) {
      const { u, v, p, div, w, fluid, nb, solid } = this;
      for (let k = 0; k < fluid.length; k++) {
        const c = fluid[k];
        div[c] = -0.5 * (u[c + 1] - u[c - 1] + v[c + w] - v[c - w]);
      }
      this.setBnd(0, div);
      this.setBnd(3, p);
      for (let it = 0; it < iters; it++) {
        for (let k = 0; k < fluid.length; k++) {
          const c = fluid[k], n = nb[k];
          if (!n) continue;
          let s = div[c];
          if (!solid[c - 1]) s += p[c - 1];
          if (!solid[c + 1]) s += p[c + 1];
          if (!solid[c - w]) s += p[c - w];
          if (!solid[c + w]) s += p[c + w];
          p[c] = s / n;
        }
        this.setBnd(3, p);
      }
      for (let k = 0; k < fluid.length; k++) {
        const c = fluid[k], pc = p[c];
        const pl = solid[c - 1] ? pc : p[c - 1], pr = solid[c + 1] ? pc : p[c + 1];
        const pu = solid[c - w] ? pc : p[c - w], pd = solid[c + w] ? pc : p[c + w];
        u[c] -= 0.5 * (pr - pl);
        v[c] -= 0.5 * (pd - pu);
      }
      this.setBnd(1, u);
      this.setBnd(2, v);
    }

    // Vorticity confinement: re-injects small swirls lost to numerical blur (not part of N–S).
    confine(eps, dt) {
      const { u, v, w, curl, fluid } = this;
      for (let k = 0; k < fluid.length; k++) {
        const c = fluid[k];
        curl[c] = 0.5 * ((v[c + 1] - v[c - 1]) - (u[c + w] - u[c - w]));
      }
      for (let k = 0; k < fluid.length; k++) {
        const c = fluid[k];
        const dx = Math.abs(curl[c + 1]) - Math.abs(curl[c - 1]);
        const dy = Math.abs(curl[c + w]) - Math.abs(curl[c - w]);
        const len = Math.hypot(dx, dy) + 1e-5;
        u[c] += dt * eps * (dy / len) * curl[c];
        v[c] -= dt * eps * (dx / len) * curl[c];
      }
      this.setBnd(1, u);
      this.setBnd(2, v);
    }

    step(dt, o) {
      const { u, v, u0, v0 } = this;
      if (o.vort > 0) this.confine(o.vort, dt);
      if (o.nu > 0) {                                     // ν∇²u   (viscosity)
        u0.set(u); v0.set(v);
        this.diffuse(1, u, u0, o.nu * dt, 15);
        this.diffuse(2, v, v0, o.nu * dt, 15);
      }
      if (o.pressure) this.project(24);                   // −∇p, ∇·u = 0
      if (o.advection) {                                  // (u·∇)u
        u0.set(u); v0.set(v);
        this.advect(1, u, u0, u0, v0, dt);
        this.advect(2, v, v0, u0, v0, dt);
      }
      if (o.pressure) this.project(24);
      for (const d of this.dye) {                         // dye is just carried along
        this.advectSharp(d, u, v, dt);
        if (o.fade < 1) for (let k = 0; k < d.length; k++) d[k] *= o.fade;
      }
    }

    // Gaussian "push": nudge velocity toward (vx, vy) and add dye around (cx, cy).
    splat(cx, cy, vx, vy, color, r, strength = 1, dyeAmount = strength) {
      const { nx, ny, w, u, v, solid, dye } = this, r2 = r * r, dyeK = dyeAmount / strength;
      const i0 = Math.max(1, Math.floor(cx - 2 * r)), i1 = Math.min(nx, Math.ceil(cx + 2 * r));
      const j0 = Math.max(1, Math.floor(cy - 2 * r)), j1 = Math.min(ny, Math.ceil(cy + 2 * r));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const c = i + w * j;
        if (solid[c]) continue;
        const g = Math.exp(-((i - cx) ** 2 + (j - cy) ** 2) / r2) * strength;
        if (g < 0.01) continue;
        if (vx !== null) { u[c] += (vx - u[c]) * g; v[c] += (vy - v[c]) * g; }
        if (color) for (let ch = 0; ch < 3; ch++) dye[ch][c] = Math.min(1.4, dye[ch][c] + color[ch] * g * dyeK);
      }
    }
  }

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'fluid',
    hint: 'Drag to push the fluid · click to drop dye',
    o: { scenario: 'stir', advection: true, pressure: true, nuS: 0.15, vortS: 0.3, force: 1, U: 0.5, view: 'dye', arrows: false, running: true },
    f: null, W: 0, H: 0, t: 0, hudTick: 0, drag: null,
    scale: { speed: 1, vort: 1, pres: 1 },
    img: null, off: null, octx: null,

    mount(panel, hud) { this.panel = panel; this.hud = hud; this.buildPanel(); this.updateHud(); },
    unmount() { this.drag = null; },

    resize(W, H) {
      this.W = W; this.H = H;
      const ny = Math.max(40, Math.round(Math.sqrt((TARGET_CELLS * H) / W)));
      const nx = Math.max(40, Math.round((ny * W) / H));
      if (!this.f || this.f.nx !== nx || this.f.ny !== ny) this.reset(nx, ny);
    },

    reset(nx = this.f?.nx, ny = this.f?.ny) {
      if (!nx) return;
      const sc = SCENARIOS[this.o.scenario];
      const f = (this.f = new Fluid(nx, ny, sc.mode));
      this.t = 0;
      this.scale = { speed: 1, vort: 1, pres: 1 };
      this.off = document.createElement('canvas');
      this.off.width = nx; this.off.height = ny;
      this.octx = this.off.getContext('2d');
      this.img = this.octx.createImageData(nx, ny);
      const { w } = f;

      if (sc.mode === 'tunnel') {
        const cx = nx * 0.22, cy = ny * 0.5 + 0.7, r = (D_CYL * ny) / 2; // tiny offset lets symmetry break
        this.cyl = { cx, cy, r };
        for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
          if ((i - cx) ** 2 + (j - cy) ** 2 < r * r) f.solid[i + w * j] = 1;
        }
        f.rebuildMask();
        f.inflow = this.o.U * ny;
        f.u.fill(f.inflow);
        f.setBnd(1, f.u);
      } else if (sc.mode === 'shear') {
        const U = this.o.U * ny, delta = 0.04 * ny;
        for (let j = 1; j <= ny; j++) for (let i = 1; i <= nx; i++) {
          const c = i + w * j, y = j - ny / 2;
          f.u[c] = -U * Math.tanh(y / delta);
          f.v[c] = 0.12 * U * Math.sin((2 * Math.PI * 2 * i) / nx) * Math.exp(-((y / (0.1 * ny)) ** 2));
          const mix = 0.5 - 0.5 * Math.tanh(y / 2); // 1 = top layer (orange), 0 = bottom (blue)
          f.dye[0][c] = 0.95 * mix + 0.15 * (1 - mix);
          f.dye[1][c] = 0.55 * mix + 0.45 * (1 - mix);
          f.dye[2][c] = 0.15 * mix + 0.95 * (1 - mix);
        }
      } else {
        const r = 0.06 * ny, sp = 0.9 * ny;
        f.splat(nx * 0.3, ny * 0.45, sp, 0, [1, 0.45, 0.1], r, 1);
        f.splat(nx * 0.7, ny * 0.55, -sp, 0, [0.15, 0.5, 1], r, 1);
      }
    },

    setScenario(v) {
      const sc = SCENARIOS[v];
      Object.assign(this.o, { scenario: v, nuS: sc.nuS, vortS: sc.vortS, U: sc.U });
      this.f = null;
      this.resize(this.W, this.H);
      this.buildPanel();
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();

      const s1 = UI.section('Scenario');
      s1.append(UI.segmented({
        options: Object.entries(SCENARIOS).map(([value, s]) => ({ value, label: s.label })),
        value: o.scenario, onChange: (v) => this.setScenario(v),
      }));
      const about = el('p', 'note');
      about.innerHTML = SCENARIOS[o.scenario].about;
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => {
        o.running = !o.running;
        play.textContent = o.running ? '⏸ Pause' : '▶ Play';
      });
      row.append(play, UI.button('↺ Reset', () => this.reset()));
      s1.append(about, row);

      const s2 = UI.section('The equations');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\begin{gathered}
        \frac{\partial \mathbf{u}}{\partial t} + \textcolor{${COL.adv}}{(\mathbf{u}\cdot\nabla)\mathbf{u}}
        = \textcolor{${COL.pres}}{-\tfrac{1}{\rho}\nabla p} + \textcolor{${COL.visc}}{\nu\nabla^2\mathbf{u}} + \textcolor{${COL.force}}{\mathbf{f}} \\
        \textcolor{${COL.pres}}{\nabla\cdot\mathbf{u} = 0}
      \end{gathered}`, true);
      s2.append(eq, el('p', 'note',
        'u is the velocity at each point, p the pressure. The left side is how a fluid parcel accelerates; ' +
        'each coloured term is a cause. Switch them off one at a time to see what each does.'));

      s2.append(UI.termCard({
        color: COL.adv, tex: String.raw`(\mathbf{u}\cdot\nabla)\mathbf{u}`, title: 'Advection',
        body: 'Moving fluid carries its own momentum along. This is the only <b>nonlinear</b> term, and it is what ' +
          'stretches and folds swirls into turbulence. Without it, motion just fades in place.',
        controls: [UI.check({ label: 'Include this term', checked: o.advection, onChange: (v) => { o.advection = v; } })],
      }));
      s2.append(UI.termCard({
        color: COL.pres, tex: String.raw`-\tfrac{1}{\rho}\nabla p,\;\; \nabla\cdot\mathbf{u}=0`, title: 'Pressure / incompressibility',
        body: "Liquid can't be squeezed. Pressure builds up wherever flow would pile up and pushes it aside, so what " +
          'flows in must flow out. Switch it off and the dye bunches up and thins out like a gas.',
        controls: [UI.check({ label: 'Include this term', checked: o.pressure, onChange: (v) => { o.pressure = v; } })],
      }));
      s2.append(UI.termCard({
        color: COL.visc, tex: String.raw`\nu\nabla^2\mathbf{u}`, title: 'Viscosity',
        body: 'Internal friction that evens out differences in velocity. Honey has a large ν, water and air a tiny one.',
        controls: [UI.slider({
          label: 'ν (viscosity)', min: 0, max: 1, step: 0.01, value: o.nuS,
          format: (v) => (nuOf(v) === 0 ? '0' : nuOf(v).toExponential(1)), onInput: (v) => { o.nuS = v; },
        })],
      }));
      const forceCtl = [UI.slider({
        label: 'Push strength', min: 0, max: 3, step: 0.05, value: o.force,
        format: (v) => v.toFixed(2) + '×', onInput: (v) => { o.force = v; },
      })];
      if (o.scenario === 'tunnel') {
        forceCtl.push(UI.slider({
          label: 'Inflow speed U', min: 0.05, max: 1, step: 0.01, value: o.U,
          format: (v) => v.toFixed(2), onInput: (v) => { o.U = v; },
        }));
      }
      s2.append(UI.termCard({
        color: COL.force, tex: String.raw`\mathbf{f}`, title: 'External force',
        body: o.scenario === 'tunnel' ? 'The inflow on the left keeps the fluid moving. You can also drag to push it.' : "That's you: drag across the fluid to push it.",
        controls: forceCtl,
      }));

      const s3 = UI.section('Show');
      s3.append(UI.segmented({ options: VIEWS, value: o.view, onChange: (v) => { o.view = v; this.updateHud(); } }));
      s3.append(UI.check({ label: 'Velocity arrows', checked: o.arrows, onChange: (v) => { o.arrows = v; } }));

      const s4 = UI.section('Under the hood');
      s4.append(UI.slider({
        label: 'Vorticity boost', min: 0, max: 1, step: 0.01, value: o.vortS,
        format: (v) => v.toFixed(2), onInput: (v) => { o.vortS = v; },
      }));
      const n1 = el('p', 'note');
      n1.innerHTML = '<b>Not</b> part of Navier–Stokes: a common graphics trick that puts back small swirls the coarse grid smears out.';
      const n2 = el('p', 'note');
      n2.innerHTML = `Solved with <b>Stable Fluids</b> (Jos Stam, 1999) on a ${this.f ? `${this.f.nx}×${this.f.ny}` : ''} grid. ` +
        'Each frame: add forces → viscosity → advection → solve for the pressure that removes any divergence.';
      s4.append(n1, n2);

      p.append(s1, s2, s3, s4);
    },

    updateHud() {
      if (!this.hud) return;
      const o = this.o, parts = [];
      if (o.scenario === 'tunnel') {
        const nu = nuOf(o.nuS);
        const Re = nu > 0 ? (o.U * D_CYL) / nu : Infinity;
        parts.push(`Reynolds number <b>Re = U·D/ν ≈ ${Number.isFinite(Re) ? Math.round(Re).toLocaleString() : '∞'}</b>`);
        parts.push(`<div class="dim">${Re < 60 ? 'Low Re: viscosity wins and the flow stays smooth.' : 'High Re: inertia wins, so expect an unsteady, swirling wake.'}</div>`);
      }
      if (!o.pressure) parts.push('<div class="warn">Pressure off: ∇·u ≠ 0, so the fluid is no longer incompressible.</div>');
      if (!o.advection) parts.push('<div class="warn">Advection off: the flow no longer carries its own momentum.</div>');
      if (o.view === 'speed') parts.push(UI.legend(UI.SEQ_STOPS, 'still', 'fast'));
      if (o.view === 'vorticity') parts.push(UI.legend(UI.DIV_STOPS, '↻ clockwise', 'counter-clockwise ↺'));
      if (o.view === 'pressure') parts.push(UI.legend(UI.DIV_STOPS, 'low', 'high'));
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const f = this.f;
      if (!f) return;
      const o = this.o;
      if (o.running) {
        const dt = 1 / 60;
        f.inflow = o.U * f.ny;
        if (f.mode === 'tunnel') this.injectTunnelDye();
        f.step(dt, {
          nu: nuOf(o.nuS) * f.ny * f.ny, // convert height²/s to cells²/s
          vort: o.vortS * VORT_SCALE,
          advection: o.advection, pressure: o.pressure,
          fade: f.mode === 'box' ? 0.9995 : 1,
        });
        this.t += dt;
      }
      this.render(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    injectTunnelDye() {
      // thin smoke streaks, like a real wind tunnel
      const f = this.f, { ny, w, dye } = f, spacing = ny / 12;
      for (let j = 1; j <= ny; j++) {
        const pos = (j - 1) / spacing, k = Math.floor(pos);
        if ((pos - k) * spacing > 2.2) continue;
        const col = UI.hsl(190 + k * 14, 0.85, 0.62);
        for (let i = 1; i <= 2; i++) for (let ch = 0; ch < 3; ch++) dye[ch][i + w * j] = col[ch];
      }
    },

    maxSpeed() {
      const { u, v, fluid } = this.f;
      let m = 0;
      for (let k = 0; k < fluid.length; k += 3) { const c = fluid[k]; const s = u[c] * u[c] + v[c] * v[c]; if (s > m) m = s; }
      return Math.sqrt(m);
    },

    render(ctx, W, H) {
      const f = this.f, o = this.o, { nx, ny, w, u, v, p, solid, dye } = f, data = this.img.data;
      const smooth = (key, val) => (this.scale[key] = Math.max(1e-6, this.scale[key] * 0.92 + val * 0.08));
      let sSpeed = 1;
      if (o.view === 'speed' || o.arrows) sSpeed = smooth('speed', this.maxSpeed());

      let sv = 1;
      if (o.view === 'vorticity' || o.view === 'pressure') {
        let m = 0;
        for (let k = 0; k < f.fluid.length; k += 2) {
          const c = f.fluid[k];
          const val = o.view === 'pressure' ? p[c] : 0.5 * ((v[c + 1] - v[c - 1]) - (u[c + w] - u[c - w]));
          if (Math.abs(val) > m) m = Math.abs(val);
        }
        sv = smooth(o.view === 'pressure' ? 'pres' : 'vort', m) * 0.6;
      }

      let k = 0;
      for (let j = 1; j <= ny; j++) {
        for (let i = 1; i <= nx; i++, k += 4) {
          const c = i + w * j;
          if (solid[c]) { data[k] = 96; data[k + 1] = 102; data[k + 2] = 116; data[k + 3] = 255; continue; }
          let r, g, b;
          if (o.view === 'dye') {
            // sqrt lifts faint, well-mixed dye so thin wisps stay visible
            r = 11 + Math.sqrt(dye[0][c]) * 235; g = 14 + Math.sqrt(dye[1][c]) * 235; b = 20 + Math.sqrt(dye[2][c]) * 235;
          } else if (o.view === 'speed') {
            const idx = Math.min(255, (Math.hypot(u[c], v[c]) / sSpeed) * 255) | 0;
            r = SEQ[idx * 3]; g = SEQ[idx * 3 + 1]; b = SEQ[idx * 3 + 2];
          } else {
            // screen y points down, so flip the sign to keep "counter-clockwise = positive"
            const val = o.view === 'pressure'
              ? (o.pressure ? p[c] : 0)
              : -0.5 * ((v[c + 1] - v[c - 1]) - (u[c + w] - u[c - w]));
            const idx = Math.max(0, Math.min(255, (0.5 + (0.5 * val) / sv) * 255)) | 0;
            r = DIV[idx * 3]; g = DIV[idx * 3 + 1]; b = DIV[idx * 3 + 2];
          }
          data[k] = r; data[k + 1] = g; data[k + 2] = b; data[k + 3] = 255;
        }
      }
      this.octx.putImageData(this.img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.off, 0, 0, W, H);

      const sx = W / nx, sy = H / ny;
      if (f.mode === 'tunnel' && this.cyl) { // crisp outline of the obstacle
        ctx.beginPath();
        ctx.arc((this.cyl.cx - 0.5) * sx, (this.cyl.cy - 0.5) * sy, this.cyl.r * sx, 0, Math.PI * 2);
        ctx.fillStyle = '#5f6674';
        ctx.fill();
      }

      if (o.arrows) {
        const step = 28;
        ctx.strokeStyle = 'rgba(255,255,255,0.75)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let y = step / 2; y < H; y += step) for (let x = step / 2; x < W; x += step) {
          const i = Math.min(nx, Math.max(1, Math.round(x / sx + 0.5))), j = Math.min(ny, Math.max(1, Math.round(y / sy + 0.5)));
          const c = i + w * j;
          if (solid[c]) continue;
          const mag = Math.hypot(u[c], v[c]);
          if (mag < sSpeed * 0.02) continue;
          const len = Math.min(1, mag / sSpeed) * step * 0.85, dx = u[c] / mag, dy = v[c] / mag;
          const x0 = x - (dx * len) / 2, y0 = y - (dy * len) / 2, x1 = x + (dx * len) / 2, y1 = y + (dy * len) / 2;
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
          const hs = Math.min(5, len * 0.35);
          ctx.moveTo(x1, y1); ctx.lineTo(x1 - hs * (dx + dy * 0.6), y1 - hs * (dy - dx * 0.6));
          ctx.moveTo(x1, y1); ctx.lineTo(x1 - hs * (dx - dy * 0.6), y1 - hs * (dy + dx * 0.6));
        }
        ctx.stroke();
      }
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      const f = this.f;
      if (!f) return;
      const cx = (x / this.W) * f.nx + 0.5, cy = (y / this.H) * f.ny + 0.5;
      const r = Math.max(3, 0.035 * f.ny), now = performance.now();
      if (type === 'down') {
        this.hue = ((this.hue || 0) + 47) % 360;
        f.splat(cx, cy, null, null, UI.hsl(this.hue, 0.85, 0.55), r * 1.2, 0.9);
        this.drag = { cx, cy, t: now };
      } else if (type === 'move' && this.drag) {
        const d = this.drag, dt = Math.max(1, now - d.t) / 1000;
        let vx = (cx - d.cx) / dt, vy = (cy - d.cy) / dt;
        const m = Math.hypot(vx, vy), cap = 4 * f.ny;
        if (m > cap) { vx *= cap / m; vy *= cap / m; }
        this.hue = (this.hue + 1.5) % 360;
        const steps = Math.max(1, Math.ceil(Math.hypot(cx - d.cx, cy - d.cy) / (r * 0.5))); // fill gaps on fast drags
        for (let s = 1; s <= steps; s++) {
          const px = d.cx + ((cx - d.cx) * s) / steps, py = d.cy + ((cy - d.cy) * s) / steps;
          f.splat(px, py, vx * this.o.force, vy * this.o.force, UI.hsl(this.hue, 0.85, 0.55), r, 0.35, 0.9 / steps);
        }
        this.drag = { cx, cy, t: now };
      } else if (type === 'up' || type === 'leave') {
        this.drag = null;
      }
    },
  };

  (window.Sims = window.Sims || {}).fluid = sim;
})();
