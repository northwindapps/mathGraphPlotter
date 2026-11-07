/* Feedback control playground: a PID controller driving three different "plants".
 *   u(t) = Kp·e + Ki·∫e dt + Kd·de/dt,   e = r − y
 * Physics runs at 960 Hz (16 sub-steps per frame); the controller runs every sub-step.
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { p: '#e5484d', i: '#3d8bfd', d: '#2fa36b', plant: '#e09b14', y: '#f5c451', r: '#aab2c0', u: '#e8ebf1' };
  const DT = 1 / 960, SUB = 16, WINDOW = 10;
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const deg = (r) => (r * 180) / Math.PI, rad = (d) => (d * Math.PI) / 180;
  const gauss = () => { let a = 0; while (!a) a = Math.random(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * Math.random()); };

  // ------------------------------------------------------------------ plants
  const PLANTS = {
    servo: {
      label: 'Servo motor',
      about: 'A DC motor turns an arm to a target angle, like a robot joint or the servo in an RC car. ' +
        'An encoder measures the angle; the controller sets the motor voltage.',
      yLabel: 'angle θ', uLabel: 'voltage V', gainUnits: ['V/rad', 'V/(rad·s)', 'V·s/rad'],
      gainMax: [200, 200, 6], presetDefault: 2,
      presets: [
        { label: 'P only', g: [20, 0, 0] }, { label: 'PD', g: [20, 0, 1] },
        { label: 'PID', g: [20, 40, 1] }, { label: 'Too aggressive', g: [150, 0, 0] },
      ],
      limit: { label: 'Voltage limit', min: 2, max: 48, step: 1, value: 12, unit: 'V' },
      dist: { label: 'Load torque', min: -1.5, max: 1.5, step: 0.05, value: 0, unit: 'N·m' },
      option: { key: 'gravity', label: 'Arm has weight (gravity pulls it down)', value: true },
      steps: [rad(60), rad(150), rad(-30), rad(90)], stepEvery: 3, noise: rad(2),
      tex: String.raw`J\ddot\theta = \tfrac{K_t}{R}(V - K_e\dot\theta) - b\dot\theta - mg\ell\cos\theta + \tau_{\rm load}`,
      body: 'Voltage drives current, current makes torque, torque accelerates the arm. The spinning motor also generates a ' +
        'back-voltage (K<sub>e</sub>θ̇) that acts like a brake. With the weight on, gravity keeps pulling the arm down: ' +
        'try <b>P only</b> and notice it stops a little short of the target.',
      init: () => ({ th: -Math.PI / 2, w: 0 }),
      y: (s) => s.th,
      angular: true, show: (y) => deg(y), fmt: (v) => `${v.toFixed(1)}°`, uFmt: (u) => `${u.toFixed(1)} V`,
      limits: (o) => [-o.limit, o.limit],
      step(s, u, dt, o) {
        const J = 0.01, b = 0.02, Kt = 0.5, Ke = 0.5, R = 2, mgl = 1.0;
        const torque = (Kt * (u - Ke * s.w)) / R - b * s.w - (o.gravity ? mgl * Math.cos(s.th) : 0) + o.dist;
        s.w += (torque / J) * dt;
        s.th += s.w * dt;
      },
      poke(s) { s.w += 14; },
    },

    drone: {
      label: 'Drone altitude',
      about: 'A quadcopter holds a target height. An altimeter measures height; the controller sets total rotor thrust. ' +
        'Gravity never stops pulling, which is exactly why the I term matters here.',
      yLabel: 'altitude z', uLabel: 'thrust T', gainUnits: ['N/m', 'N/(m·s)', 'N·s/m'],
      gainMax: [40, 20, 20], presetDefault: 2,
      presets: [
        { label: 'P only', g: [8, 0, 0] }, { label: 'PD', g: [8, 0, 5] },
        { label: 'PID', g: [8, 4, 5] }, { label: 'Windup trap', g: [8, 10, 3] },
      ],
      limit: { label: 'Max thrust', min: 11, max: 40, step: 1, value: 20, unit: 'N' },
      dist: { label: 'Vertical wind', min: -6, max: 6, step: 0.1, value: 0, unit: 'N' },
      option: { key: 'ff', label: 'Feed-forward: add hover thrust mg', value: false },
      steps: [2, 6, 3, 8], stepEvery: 5, noise: 0.12,
      tex: String.raw`m\ddot z = T - mg - c\,\dot z + F_{\rm wind}`,
      body: 'Thrust minus weight accelerates the drone, and air drag slows it. With <b>P only</b> the drone hovers <i>below</i> the target, ' +
        'because some error is needed to produce the thrust that cancels gravity. I removes that offset by learning it; ' +
        '<b>feed-forward</b> removes it by knowing the physics in advance. Try <b>Windup trap</b> with anti-windup off.',
      init: () => ({ z: 0, v: 0 }),
      y: (s) => s.z,
      show: (y) => y, fmt: (v) => `${v.toFixed(2)} m`, uFmt: (u) => `${u.toFixed(1)} N`,
      limits: (o) => { const ff = o.ff ? 9.81 : 0; return [-ff, o.limit - ff]; },
      thrust: (u, o) => u + (o.ff ? 9.81 : 0),
      step(s, u, dt, o) {
        const T = this.thrust(u, o);
        s.v += (T - 9.81 - 0.4 * s.v + o.dist) * dt; // m = 1 kg
        s.z += s.v * dt;
        if (s.z < 0) { s.z = 0; if (s.v < 0) s.v = 0; }
      },
      poke(s) { s.v -= 4; },
    },

    pendulum: {
      label: 'Balancing robot',
      about: 'An inverted pendulum on a cart, like a Segway or a rocket balancing on its thrust. Upright is <b>unstable</b>: ' +
        'left alone it falls. The PID loop reads the lean angle and pushes the cart back under the pole.',
      yLabel: 'lean φ', uLabel: 'cart force F', gainUnits: ['N/rad', 'N/(rad·s)', 'N·s/rad'], flip: true,
      gainMax: [150, 100, 20], presetDefault: 2,
      presets: [
        { label: 'Too weak', g: [10, 0, 2] }, { label: 'P only', g: [40, 0, 0] },
        { label: 'PD', g: [40, 0, 5] }, { label: 'PID', g: [40, 5, 5] },
      ],
      limit: { label: 'Max force', min: 5, max: 80, step: 1, value: 40, unit: 'N' },
      dist: null,
      option: { key: 'outer', label: 'Outer loop: lean toward the cart target', value: true },
      steps: [-1.2, 1.2], stepEvery: 6, noise: rad(1),
      tex: String.raw`\ell\ddot\varphi = g\sin\varphi - \ddot x\cos\varphi`,
      body: 'Gravity tips the pole further the more it leans; the only cure is to move the cart underneath it. ' +
        'Here the error is φ − φ<sub>target</sub>: if it leans right, push right. To travel right, the outer loop asks for a slight lean ' +
        'to the right, so the cart first has to <b>back up</b> to create that lean. Watch for it! With <b>P only</b> nothing damps ' +
        'the wobble, so it grows until the pole falls; D supplies the damping. <b>Too weak</b> can\'t out-push gravity at all.',
      init: () => ({ x: 0, xv: 0, p: 0.08, pw: 0 }),
      y: (s) => s.p,
      angular: true, show: (y) => deg(y), fmt: (v) => `${v.toFixed(1)}°`, uFmt: (u) => `${u.toFixed(1)} N`,
      limits: (o) => [-o.limit, o.limit],
      reference(s, sim) {
        if (!sim.o.outer) return 0;
        return clamp(0.1 * (sim.r - s.x) - 0.15 * s.xv, -0.25, 0.25);
      },
      step(s, u, dt) {
        const M = 1, m = 0.3, l = 0.8, g = 9.81;
        const sp = Math.sin(s.p), cp = Math.cos(s.p);
        const xa = (u - 0.5 * s.xv + m * sp * (l * s.pw * s.pw - g * cp)) / (M + m * sp * sp);
        const pa = (g * sp - xa * cp) / l;
        s.xv += xa * dt; s.x += s.xv * dt;
        s.pw += pa * dt; s.p += s.pw * dt;
        if (Math.abs(s.x) > 2.4) { s.x = Math.sign(s.x) * 2.4; s.xv = 0; }
        if (Math.abs(s.p) > Math.PI / 2) { s.p = (Math.sign(s.p) * Math.PI) / 2; s.pw = 0; } // lying on the cart
      },
      poke(s) { s.pw += (Math.random() < 0.5 ? -1 : 1) * 1.3; },
    },
  };

  // ------------------------------------------------------------------ PID
  class PID {
    constructor() { this.reset(); }
    reset() { this.i = 0; this.dF = 0; this.prev = null; this.P = this.I = this.D = this.u = 0; this.sat = false; }
    update(e, dSig, dt, o, lo, hi) {
      if (this.prev === null) this.prev = dSig;
      const raw = (dSig - this.prev) / dt;
      this.prev = dSig;
      this.dF += ((raw - this.dF) * dt) / (0.015 + dt); // derivative is low-pass filtered (as in real controllers)
      this.i += e * dt;
      this.P = o.kp * e;
      this.I = o.ki * this.i;
      this.D = o.kd * this.dF;
      const u = this.P + this.I + this.D, us = clamp(u, lo, hi);
      // Anti-windup (conditional integration): stop integrating while saturated in the same direction.
      if (o.antiWindup && us !== u && Math.sign(e) === Math.sign(u - us)) {
        this.i -= e * dt;
        this.I = o.ki * this.i;
      }
      this.sat = us !== u;
      this.u = us;
      return us;
    }
  }

  // ------------------------------------------------------------------ module
  const sim = {
    id: 'control',
    hint: 'Drag in the scene to move the target · Poke to add a disturbance',
    o: { plant: 'servo', kp: 20, ki: 40, kd: 1, antiWindup: true, dMeas: false, noise: 0, auto: true, running: true, limit: 12, dist: 0 },
    pid: new PID(), s: null, t: 0, r: 0, hist: [], W: 0, H: 0, hudTick: 0, range: null,

    get P() { return PLANTS[this.o.plant]; },

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.s) this.setPlant(this.o.plant); else this.buildPanel();
    },
    unmount() { this.dragging = false; },
    resize(W, H) { this.W = W; this.H = H; },

    setPlant(key) {
      const P = PLANTS[key], o = this.o;
      o.plant = key;
      [o.kp, o.ki, o.kd] = P.presets[P.presetDefault].g;
      o.limit = P.limit.value;
      o.dist = P.dist ? P.dist.value : 0;
      o[P.option.key] = P.option.value;
      this.reset();
      this.buildPanel();
    },

    reset() {
      const P = this.P;
      this.s = P.init();
      this.pid.reset();
      this.t = 0;
      this.hist = [];
      this.range = null;
      this.stepIdx = 0;
      this.nextStep = P.stepEvery;
      this.setR(P.steps[0]);
    },

    ref() { return this.P.reference ? this.P.reference(this.s, this) : this.r; },

    setR(v) {
      this.r = v;
      const y = this.P.y(this.s);
      this.metric = { t0: this.t, y0: y, r: v, t10: null, t90: null, peak: 0, lastOut: this.t };
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, P = this.P, p = this.panel;
      if (!p) return;
      p.replaceChildren();

      const s1 = UI.section('System');
      s1.append(UI.segmented({
        options: Object.entries(PLANTS).map(([value, pl]) => ({ value, label: pl.label })),
        value: o.plant, onChange: (v) => this.setPlant(v),
      }));
      const about = el('p', 'note');
      about.innerHTML = P.about;
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => {
        o.running = !o.running;
        play.textContent = o.running ? '⏸ Pause' : '▶ Play';
      });
      row.append(play, UI.button('↺ Reset', () => this.reset()), UI.button('⚡ Poke', () => P.poke(this.s)));
      this.autoCheck = UI.check({ label: 'Auto-step the target', checked: o.auto, onChange: (v) => { o.auto = v; this.nextStep = this.t + P.stepEvery; } });
      s1.append(about, row, this.autoCheck);

      // live block diagram
      const s2 = UI.section('The feedback loop');
      const wrap = el('div');
      wrap.innerHTML = `
        <svg class="loop-diagram" viewBox="-14 0 354 128" role="img" aria-label="Feedback loop: target, sum, PID, plant, sensor">
          <defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" class="bd-head"/></marker></defs>
          <text x="16" y="30" class="bd-lbl">r</text><text x="16" y="58" class="bd-val" data-v="r"></text>
          <path d="M28,40 H56" class="bd-line" marker-end="url(#ah)"/>
          <circle cx="68" cy="40" r="11" class="bd-box"/><text x="68" y="41" class="bd-txt">Σ</text>
          <text x="52" y="62" class="bd-lbl">−</text>
          <path d="M79,40 H118" class="bd-line" marker-end="url(#ah)"/>
          <text x="99" y="32" class="bd-lbl">e</text><text x="99" y="58" class="bd-val" data-v="e"></text>
          <rect x="120" y="24" width="56" height="32" rx="6" class="bd-box"/><text x="148" y="41" class="bd-txt">PID</text>
          <path d="M176,40 H212" class="bd-line" marker-end="url(#ah)"/>
          <text x="194" y="32" class="bd-lbl">u</text><text x="194" y="58" class="bd-val" data-v="u"></text>
          <rect x="214" y="24" width="60" height="32" rx="6" class="bd-box"/><text x="244" y="41" class="bd-txt">Plant</text>
          <path d="M274,40 H330" class="bd-line" marker-end="url(#ah)"/>
          <text x="306" y="32" class="bd-lbl">y</text><text x="306" y="58" class="bd-val" data-v="y"></text>
          <path d="M300,40 V104 H232" class="bd-line" marker-end="url(#ah)"/>
          <rect x="164" y="90" width="66" height="28" rx="6" class="bd-box"/><text x="197" y="105" class="bd-txt">Sensor</text>
          <path d="M164,104 H68 V53" class="bd-line" marker-end="url(#ah)"/>
          <text x="116" y="98" class="bd-lbl" data-v="noise"></text>
        </svg>`;
      this.bd = Object.fromEntries([...wrap.querySelectorAll('[data-v]')].map((n) => [n.dataset.v, n]));
      s2.append(wrap, el('p', 'note',
        'The sensor measures the output y, the controller compares it with the target r, and acts on the error e. ' +
        'The plant changes, the sensor sees it, and around it goes, hundreds of times a second.'));

      // controller
      const s3 = UI.section('The controller');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`u(t) = \textcolor{${COL.p}}{K_p\,e(t)} + \textcolor{${COL.i}}{K_i\!\int_0^t\! e\,d\tau} + \textcolor{${COL.d}}{K_d\,\frac{de}{dt}}`, true);
      s3.append(eq);
      s3.append(UI.segmented({
        options: P.presets.map((pr, i) => ({ value: i, label: pr.label })),
        value: -1, onChange: (i) => {
          [o.kp, o.ki, o.kd] = P.presets[i].g;
          this.sliders.kp.set(o.kp); this.sliders.ki.set(o.ki); this.sliders.kd.set(o.kd);
        },
      }));
      const gainSlider = (key, idx, name) => UI.slider({
        label: `${name}  (${P.gainUnits[idx]})`, min: 0, max: P.gainMax[idx], step: P.gainMax[idx] / 400, value: o[key],
        format: (v) => (v < 10 ? v.toFixed(2) : v.toFixed(1)), onInput: (v) => { o[key] = v; },
      });
      this.sliders = { kp: gainSlider('kp', 0, 'Kp'), ki: gainSlider('ki', 1, 'Ki'), kd: gainSlider('kd', 2, 'Kd') };
      s3.append(UI.termCard({
        color: COL.p, tex: String.raw`K_p\,e`, title: 'P: the present',
        body: 'Push in proportion to the error now. More Kp gives a faster response but more overshoot. On its own it often settles ' +
          '<i>near</i> the target, not on it, whenever something (like gravity) keeps pushing back.',
        controls: [this.sliders.kp],
      }));
      s3.append(UI.termCard({
        color: COL.i, tex: String.raw`K_i\!\int e\,d\tau`, title: 'I: the past',
        body: 'Adds up the error over time and keeps pushing until it is truly zero, which removes steady offsets. ' +
          'Too much causes slow oscillation, and when the actuator is maxed out the sum can grow huge: <b>integral windup</b>.',
        controls: [this.sliders.ki, UI.check({ label: 'Anti-windup', checked: o.antiWindup, onChange: (v) => { o.antiWindup = v; } })],
      }));
      s3.append(UI.termCard({
        color: COL.d, tex: String.raw`K_d\,\tfrac{de}{dt}`, title: 'D: the future',
        body: 'Reacts to how fast the error is changing, like a brake that eases off before overshooting. ' +
          'It also amplifies sensor noise (turn the noise up and watch the green line).',
        controls: [this.sliders.kd, UI.check({
          label: 'D on measurement (no kick when the target jumps)', checked: o.dMeas, onChange: (v) => { o.dMeas = v; },
        })],
      }));

      // plant
      const s4 = UI.section('The plant (the physics)');
      const plantCtl = [
        UI.slider({ label: `${P.limit.label}`, min: P.limit.min, max: P.limit.max, step: P.limit.step, value: o.limit,
          format: (v) => `${v} ${P.limit.unit}`, onInput: (v) => { o.limit = v; } }),
      ];
      if (P.dist) {
        plantCtl.push(UI.slider({ label: P.dist.label, min: P.dist.min, max: P.dist.max, step: P.dist.step, value: o.dist,
          format: (v) => `${v.toFixed(2)} ${P.dist.unit}`, onInput: (v) => { o.dist = v; } }));
      }
      plantCtl.push(UI.check({ label: P.option.label, checked: o[P.option.key], onChange: (v) => { o[P.option.key] = v; } }));
      s4.append(UI.termCard({ color: COL.plant, tex: P.tex, title: P.label, body: P.body, controls: plantCtl }));

      const s5 = UI.section('Real-world trouble');
      s5.append(UI.slider({ label: 'Sensor noise', min: 0, max: 1, step: 0.01, value: o.noise,
        format: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { o.noise = v; } }));
      const where = el('p', 'note');
      where.innerHTML = '<b>Where else?</b> The same loop runs cruise control, thermostats, 3D-printer hot-ends, hard-disk heads, ' +
        'quadcopters, CPU fan and clock control, TCP congestion control and cloud autoscalers ' +
        '(the Kubernetes HPA is essentially a P controller).';
      s5.append(where);

      p.append(s1, s2, s3, s4, s5);
    },

    // ---------------------------------------------------------------- simulation
    frame(ctx, W, H) {
      if (!this.s) return;
      const o = this.o, P = this.P, pid = this.pid;
      if (o.running) {
        for (let k = 0; k < SUB; k++) {
          if (o.auto && this.t >= this.nextStep) {
            this.stepIdx++;
            this.setR(P.steps[this.stepIdx % P.steps.length]);
            this.nextStep = this.t + P.stepEvery;
          }
          const r = this.ref();
          const y = P.y(this.s) + (o.noise > 0 ? o.noise * P.noise * gauss() : 0);
          const e = P.flip ? y - r : r - y;
          const dSig = o.dMeas ? (P.flip ? y : -y) : e;
          const [lo, hi] = P.limits(o);
          const u = pid.update(e, dSig, DT, o, lo, hi);
          P.step(this.s, u, DT, o);
          this.t += DT;
        }
        const y = P.y(this.s);
        this.hist.push({ t: this.t, r: this.ref(), y, u: pid.u, P: pid.P, I: pid.I, D: pid.D, sat: pid.sat });
        while (this.hist.length && this.hist[0].t < this.t - WINDOW - 0.5) this.hist.shift();
        this.trackMetric(y);
      }
      this.draw(ctx, W, H);
      if (++this.hudTick % 6 === 0) { this.updateHud(); this.updateDiagram(); }
    },

    trackMetric(y) {
      const m = this.metric, span = m.r - m.y0;
      if (!m || Math.abs(span) < 1e-6 || this.P.flip) return;
      const prog = (y - m.y0) / span;
      if (m.t10 === null && prog >= 0.1) m.t10 = this.t;
      if (m.t90 === null && prog >= 0.9) m.t90 = this.t;
      m.peak = Math.max(m.peak, prog - 1);
      if (Math.abs(prog - 1) > 0.02) m.lastOut = this.t;
    },

    updateHud() {
      const P = this.P, o = this.o, m = this.metric, s = this.s, parts = [];
      const last = this.hist[this.hist.length - 1];
      if (P.flip) {
        parts.push(`Lean <b>${P.fmt(deg(s.p))}</b> · cart x = ${s.x.toFixed(2)} m → target ${this.r.toFixed(1)} m`);
        if (Math.abs(s.p) > 1.2) parts.push('<div class="warn">It fell over! The gains can\'t catch it. Press Reset.</div>');
      } else {
        const y = P.y(s), settled = this.t - m.lastOut > 0.3;
        const f = (v) => (v == null ? '—' : `${v.toFixed(2)} s`);
        parts.push(`<b>Step response</b> <span class="dim">(target ${P.fmt(P.show(m.r))})</span>`);
        parts.push(`<div>Rise time ${f(m.t10 !== null && m.t90 !== null ? m.t90 - m.t10 : null)} · ` +
          `Overshoot ${m.t90 !== null ? Math.max(0, m.peak * 100).toFixed(0) + '%' : '—'} · ` +
          `Settling ${settled ? f(m.lastOut - m.t0) : '—'}</div>`);
        parts.push(`<div>Error now <b>${P.fmt(P.show(m.r) - P.show(y))}</b>${settled && Math.abs(P.show(m.r) - P.show(y)) > (P.angular ? 0.5 : 0.03) ? ' <span class="warn">steady-state error</span>' : ''}</div>`);
      }
      if (last && last.sat) parts.push(`<div class="warn">Actuator saturated: the ${P.uLabel.split(' ')[0]} is maxed out.</div>`);
      if (!o.antiWindup && Math.abs(this.pid.I) > Math.abs(P.limits(o)[1]) * 1.5) parts.push('<div class="warn">Integral wound up!</div>');
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    updateDiagram() {
      if (!this.bd || !this.hist.length) return;
      const P = this.P, h = this.hist[this.hist.length - 1];
      const e = P.flip ? h.y - h.r : h.r - h.y;
      this.bd.r.textContent = P.fmt(P.show(h.r));
      this.bd.e.textContent = P.fmt(P.show(e));
      this.bd.u.textContent = P.uFmt(h.u);
      this.bd.u.classList.toggle('sat', h.sat);
      this.bd.y.textContent = P.fmt(P.show(h.y));
      this.bd.noise.textContent = this.o.noise > 0 ? '+ noise' : '';
    },

    // ---------------------------------------------------------------- drawing
    draw(ctx, W, H) {
      const narrow = W < 560;
      const sceneH = Math.round(H * (narrow ? 0.46 : 0.52));
      const scene = { x: 0, y: 0, w: W, h: sceneH };
      this.scene = scene;
      this[`draw_${this.o.plant}`](ctx, scene);

      const pad = 12, ph = (H - sceneH - pad * 3 - 22) / 2; // 22px left for the hint line
      const plotW = W - pad * 2;
      this.plotOutput(ctx, pad, sceneH + pad, plotW, ph);
      this.plotControl(ctx, pad, sceneH + pad * 2 + ph, plotW, ph);
    },

    axes(ctx, x, y, w, h, lo, hi, fmt, title, legend) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)';
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      const span = hi - lo, raw = span / 4, p10 = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 5, 10].map((m) => m * p10).find((s) => s >= raw);
      ctx.font = '10.5px system-ui, sans-serif';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        const py = y + h - ((v - lo) / span) * h;
        ctx.strokeStyle = Math.abs(v) < step * 1e-6 ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.06)';
        ctx.beginPath(); ctx.moveTo(x, Math.round(py) + 0.5); ctx.lineTo(x + w, Math.round(py) + 0.5); ctx.stroke();
        ctx.fillStyle = '#7d8696';
        if (py > y + 8 && py < y + h - 6) ctx.fillText(fmt(v), x + 5, py);
      }
      // seconds ticks
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      for (let s = 0; s <= WINDOW; s += 2) {
        const px = x + w - (s / WINDOW) * w;
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, y); ctx.lineTo(Math.round(px) + 0.5, y + h); ctx.stroke();
        if (s > 0 && s < WINDOW) { ctx.fillStyle = '#5f6878'; ctx.fillText(`−${s}s`, px, y + h - 2); }
      }
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillStyle = '#c9cfda';
      ctx.font = '600 11.5px system-ui, sans-serif';
      ctx.fillText(title, x + 44, y + 6);
      // legend (right-aligned)
      ctx.font = '11px system-ui, sans-serif';
      let lx = x + w - 8;
      ctx.textAlign = 'right';
      for (const [label, color, dashed] of [...legend].reverse()) {
        ctx.fillStyle = '#aab2c0';
        ctx.fillText(label, lx, y + 6);
        lx -= ctx.measureText(label).width + 6;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash(dashed ? [4, 3] : []);
        ctx.beginPath(); ctx.moveTo(lx - 14, y + 12); ctx.lineTo(lx, y + 12); ctx.stroke();
        ctx.setLineDash([]);
        lx -= 24;
      }
    },

    series(ctx, x, y, w, h, lo, hi, get, color, width, dash) {
      const hs = this.hist;
      if (hs.length < 2) return;
      ctx.save();
      ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < hs.length; i++) {
        const px = x + w - ((this.t - hs[i].t) / WINDOW) * w;
        const py = y + h - ((get(hs[i]) - lo) / (hi - lo)) * h;
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.restore();
    },

    plotOutput(ctx, x, y, w, h) {
      const P = this.P;
      let lo = Infinity, hi = -Infinity;
      for (const s of this.hist) {
        const a = P.show(s.y), b = P.show(s.r);
        lo = Math.min(lo, a, b); hi = Math.max(hi, a, b);
      }
      if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
      const minSpan = P.angular ? 20 : 2;
      if (hi - lo < minSpan) { const c = (hi + lo) / 2; lo = c - minSpan / 2; hi = c + minSpan / 2; }
      const pad = (hi - lo) * 0.12;
      const target = [lo - pad, hi + pad];
      this.range = this.range ? this.range.map((v, i) => v + (target[i] - v) * 0.08) : target;
      const [a, b] = this.range;
      this.axes(ctx, x, y, w, h, a, b, (v) => (P.angular ? `${Math.round(v)}°` : `${+v.toFixed(1)}`),
        `Output ${P.yLabel}`, [['target r', COL.r, true], ['measured y', COL.y, false]]);
      this.series(ctx, x, y, w, h, a, b, (s) => P.show(s.r), COL.r, 1.5, [5, 4]);
      this.series(ctx, x, y, w, h, a, b, (s) => P.show(s.y), COL.y, 2.2);
    },

    plotControl(ctx, x, y, w, h) {
      const P = this.P, [l0, h0] = P.limits(this.o);
      const pad = (h0 - l0) * 0.15, lo = Math.min(l0, 0) - pad, hi = h0 + pad;
      this.axes(ctx, x, y, w, h, lo, hi, (v) => `${+v.toFixed(1)}`, `Control ${P.uLabel}`,
        [['P', COL.p], ['I', COL.i], ['D', COL.d], ['u (sent)', COL.u]]);
      // actuator limits
      ctx.strokeStyle = 'rgba(224,155,20,0.55)';
      ctx.setLineDash([2, 4]);
      ctx.lineWidth = 1;
      for (const v of [l0, h0]) {
        const py = y + h - ((v - lo) / (hi - lo)) * h;
        ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x + w, py); ctx.stroke();
      }
      ctx.setLineDash([]);
      this.series(ctx, x, y, w, h, lo, hi, (s) => s.P, COL.p, 1.2);
      this.series(ctx, x, y, w, h, lo, hi, (s) => s.I, COL.i, 1.2);
      this.series(ctx, x, y, w, h, lo, hi, (s) => s.D, COL.d, 1.2);
      this.series(ctx, x, y, w, h, lo, hi, (s) => s.u, COL.u, 2);
    },

    draw_servo(ctx, sc) {
      const { s, o } = this, cx = sc.x + sc.w / 2, cy = sc.y + sc.h * 0.54;
      const R = Math.min(sc.w * 0.3, sc.h * 0.36);
      this.servoGeom = { cx, cy };
      // dial
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R * 1.12, 0, Math.PI * 2); ctx.stroke();
      ctx.font = '10.5px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let a = -150; a <= 180; a += 30) {
        const t = rad(a), c = Math.cos(t), sn = Math.sin(t);
        ctx.beginPath(); ctx.moveTo(cx + c * R * 1.08, cy - sn * R * 1.08); ctx.lineTo(cx + c * R * 1.16, cy - sn * R * 1.16); ctx.stroke();
        ctx.fillStyle = '#6c7586';
        ctx.fillText(`${a}°`, cx + c * R * 1.27, cy - sn * R * 1.27);
      }
      // target arm
      const r = this.r;
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = COL.r; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(r) * R, cy - Math.sin(r) * R); ctx.stroke();
      ctx.setLineDash([]);
      // error arc
      const th = s.th;
      ctx.strokeStyle = 'rgba(229,72,77,0.8)'; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.42, -Math.max(th, r), -Math.min(th, r)); ctx.stroke();
      ctx.fillStyle = '#e5484d'; ctx.font = 'italic 13px Georgia, serif';
      const mid = (th + r) / 2;
      if (Math.abs(th - r) > 0.15) ctx.fillText('e', cx + Math.cos(mid) * R * 0.55, cy - Math.sin(mid) * R * 0.55);
      // arm
      const ax = cx + Math.cos(th) * R, ay = cy - Math.sin(th) * R;
      ctx.strokeStyle = '#d6dbe4'; ctx.lineWidth = 10; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.lineCap = 'butt';
      if (o.gravity) {
        ctx.fillStyle = '#8e98aa';
        ctx.beginPath(); ctx.arc(ax, ay, 13, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ax, ay + 16); ctx.lineTo(ax, ay + 34); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ax - 4, ay + 29); ctx.lineTo(ax, ay + 35); ctx.lineTo(ax + 4, ay + 29); ctx.stroke();
      }
      // motor hub (spinning marks show speed)
      ctx.fillStyle = '#272e3b'; ctx.strokeStyle = '#4a5466'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.16, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#d6dbe4';
      ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
      // voltage gauge
      const [, vmax] = P_limits(this);
      const gx = sc.x + sc.w - 34, gh = sc.h * 0.6, gy = sc.y + sc.h * 0.2, frac = this.pid.u / vmax;
      this.gauge(ctx, gx, gy, gh, frac, P_fmtU(this));
    },

    // Actuator gauge. Bipolar (−max…+max, zero in the middle) or unipolar (0…max).
    gauge(ctx, gx, gy, gh, frac, label, unipolar = false) {
      ctx.fillStyle = 'rgba(255,255,255,0.06)';
      ctx.fillRect(gx, gy, 10, gh);
      const mid = unipolar ? gy + gh : gy + gh / 2;
      const len = unipolar ? clamp(frac, 0, 1) * gh : (clamp(frac, -1, 1) * gh) / 2;
      ctx.fillStyle = Math.abs(frac) >= 0.999 ? '#e09b14' : '#e8ebf1';
      ctx.fillRect(gx, Math.min(mid, mid - len), 10, Math.abs(len));
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.beginPath(); ctx.moveTo(gx - 3, mid + 0.5); ctx.lineTo(gx + 13, mid + 0.5); ctx.stroke();
      ctx.fillStyle = '#aab2c0'; ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('actuator', gx + 10, gy - 10);
      ctx.fillText(label, gx - 6, unipolar ? mid - len : mid);
    },

    draw_drone(ctx, sc) {
      const { s, o } = this, P = this.P;
      const top = sc.y + 22, gy = sc.y + sc.h - 20, zMax = 10;
      const toY = (z) => gy - (z / zMax) * (gy - top);
      this.droneGeom = { toZ: (py) => clamp(((gy - py) / (gy - top)) * zMax, 0, zMax) };
      // ground and scale
      ctx.fillStyle = '#161b24'; ctx.fillRect(sc.x, gy, sc.w, sc.y + sc.h - gy);
      ctx.strokeStyle = '#3a4352'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sc.x, gy); ctx.lineTo(sc.x + sc.w, gy); ctx.stroke();
      ctx.font = '10.5px system-ui, sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const sx = sc.x + Math.min(Math.max(sc.w * 0.2, 300), sc.w * 0.35); // keep the scale clear of the HUD
      for (let z = 0; z <= zMax; z += 2) {
        const py = toY(z);
        ctx.strokeStyle = 'rgba(255,255,255,0.07)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx, py); ctx.lineTo(sc.x + sc.w * 0.85, py); ctx.stroke();
        ctx.fillStyle = '#6c7586'; ctx.fillText(`${z} m`, sx - 6, py);
      }
      // target line
      const ty = toY(this.r);
      ctx.setLineDash([6, 5]); ctx.strokeStyle = COL.r; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(sx, ty); ctx.lineTo(sc.x + sc.w * 0.85, ty); ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = 'left'; ctx.fillStyle = COL.r; ctx.fillText(`target ${this.r.toFixed(1)} m`, sc.x + sc.w * 0.85 + 6, ty);
      // shadow
      const cx = sc.x + sc.w * 0.52, dy = toY(s.z) - 8;
      ctx.fillStyle = `rgba(0,0,0,${0.5 - s.z * 0.035})`;
      ctx.beginPath(); ctx.ellipse(cx, gy + 4, 46 - s.z * 2, 5, 0, 0, Math.PI * 2); ctx.fill();
      // thrust arrows
      const T = P.thrust(this.pid.u, o), tl = (T / 20) * 34;
      ctx.strokeStyle = 'rgba(224,155,20,0.85)'; ctx.lineWidth = 3;
      for (const ox of [-38, 38]) {
        ctx.beginPath(); ctx.moveTo(cx + ox, dy + 8); ctx.lineTo(cx + ox, dy + 8 + tl); ctx.stroke();
      }
      // body
      ctx.strokeStyle = '#8e98aa'; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(cx - 38, dy); ctx.lineTo(cx + 38, dy); ctx.stroke();
      ctx.fillStyle = '#d6dbe4';
      ctx.beginPath(); ctx.roundRect(cx - 16, dy - 7, 32, 14, 4); ctx.fill();
      // rotors (blur width wobbles with thrust)
      const spin = 0.6 + 0.4 * Math.sin(this.t * 60);
      ctx.fillStyle = `rgba(200,210,225,${0.25 + 0.3 * Math.min(1, T / 20)})`;
      for (const ox of [-38, 38]) { ctx.beginPath(); ctx.ellipse(cx + ox, dy - 6, 22 * (0.8 + 0.2 * spin), 3.5, 0, 0, Math.PI * 2); ctx.fill(); }
      // wind
      if (Math.abs(o.dist) > 0.05) {
        ctx.strokeStyle = 'rgba(120,180,255,0.6)'; ctx.lineWidth = 2;
        const dir = Math.sign(o.dist), len = 10 + Math.abs(o.dist) * 5;
        for (const ox of [-90, 90]) {
          const x0 = cx + ox, y0 = dy + (dir > 0 ? 20 : -20);
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 - dir * len);
          ctx.moveTo(x0 - 4, y0 - dir * (len - 6)); ctx.lineTo(x0, y0 - dir * len); ctx.lineTo(x0 + 4, y0 - dir * (len - 6)); ctx.stroke();
        }
      }
      this.gauge(ctx, sc.x + sc.w - 34, sc.y + sc.h * 0.2, sc.h * 0.6, T / o.limit, `T = ${T.toFixed(1)} N`, true);
    },

    draw_pendulum(ctx, sc) {
      const { s } = this, ty = sc.y + sc.h * 0.8, cx0 = sc.x + sc.w / 2, scale = (sc.w * 0.86) / 5.2;
      const toX = (x) => cx0 + x * scale;
      this.pendGeom = { toM: (px) => clamp((px - cx0) / scale, -2.2, 2.2) };
      // track
      ctx.strokeStyle = '#3a4352'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(toX(-2.6), ty + 14); ctx.lineTo(toX(2.6), ty + 14); ctx.stroke();
      ctx.fillStyle = '#4a5466';
      for (const e of [-2.6, 2.6]) ctx.fillRect(toX(e) - (e < 0 ? 0 : 6), ty - 10, 6, 26);
      ctx.font = '10.5px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let m = -2; m <= 2; m++) { ctx.fillStyle = '#6c7586'; ctx.fillText(`${m} m`, toX(m), ty + 22); }
      // cart target marker
      const tx = toX(this.r);
      ctx.fillStyle = COL.r;
      ctx.beginPath(); ctx.moveTo(tx, ty + 16); ctx.lineTo(tx - 7, ty + 26); ctx.lineTo(tx + 7, ty + 26); ctx.fill();
      // cart
      const px = toX(s.x);
      ctx.fillStyle = '#d6dbe4';
      ctx.beginPath(); ctx.roundRect(px - 36, ty - 12, 72, 22, 5); ctx.fill();
      ctx.fillStyle = '#4a5466';
      for (const o of [-22, 22]) { ctx.beginPath(); ctx.arc(px + o, ty + 11, 6, 0, Math.PI * 2); ctx.fill(); }
      // force arrow
      const F = this.pid.u, fl = (F / this.o.limit) * 60;
      if (Math.abs(fl) > 2) {
        ctx.strokeStyle = '#e09b14'; ctx.lineWidth = 3;
        const sx = px + Math.sign(fl) * 38;
        ctx.beginPath(); ctx.moveTo(sx, ty - 1); ctx.lineTo(sx + fl, ty - 1);
        ctx.moveTo(sx + fl - Math.sign(fl) * 7, ty - 7); ctx.lineTo(sx + fl, ty - 1); ctx.lineTo(sx + fl - Math.sign(fl) * 7, ty + 5);
        ctx.stroke();
      }
      // pole
      const L = Math.min(0.8 * scale * 1.6, sc.h * 0.62), pivY = ty - 12;
      const ref = this.ref();
      ctx.setLineDash([5, 5]); ctx.strokeStyle = 'rgba(170,178,192,0.6)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, pivY); ctx.lineTo(px + Math.sin(ref) * L, pivY - Math.cos(ref) * L); ctx.stroke();
      ctx.setLineDash([]);
      const bx = px + Math.sin(s.p) * L, by = pivY - Math.cos(s.p) * L;
      ctx.strokeStyle = '#f5c451'; ctx.lineWidth = 6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(px, pivY); ctx.lineTo(bx, by); ctx.stroke(); ctx.lineCap = 'butt';
      ctx.fillStyle = '#f5c451';
      ctx.beginPath(); ctx.arc(bx, by, 11, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#272e3b';
      ctx.beginPath(); ctx.arc(px, pivY, 4, 0, Math.PI * 2); ctx.fill();
      const [, fmax] = P_limits(this);
      this.gauge(ctx, sc.x + sc.w - 34, sc.y + sc.h * 0.2, sc.h * 0.55, F / fmax, P_fmtU(this));
    },

    // ---------------------------------------------------------------- input
    pointer(type, x, y) {
      if (!this.scene) return;
      if (type === 'down') {
        if (y > this.scene.h) return; // clicks on the plots do nothing
        this.dragging = true;
        this.o.auto = false;
        const box = this.autoCheck && this.autoCheck.querySelector('input');
        if (box) box.checked = false;
      }
      if (type === 'up' || type === 'leave') { this.dragging = false; return; }
      if (!this.dragging) return;
      let r;
      if (this.o.plant === 'servo') r = Math.atan2(-(y - this.servoGeom.cy), x - this.servoGeom.cx);
      else if (this.o.plant === 'drone') r = this.droneGeom.toZ(y);
      else r = this.pendGeom.toM(x);
      if (type === 'down' || Math.abs(r - this.r) > 1e-3) this.setR(r);
    },
  };

  const P_limits = (sim) => sim.P.limits(sim.o);
  const P_fmtU = (sim) => sim.P.uFmt(sim.pid.u);

  (window.Sims = window.Sims || {}).control = sim;
})();
