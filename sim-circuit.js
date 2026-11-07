/* RC, RL, LC, series RLC and parallel RLC circuits as differential equations.
 *   Series   (KVL): L·q'' + R·q' + q/C = V(t),        i = q'
 *   Parallel (KCL): C·v' + v/R + i_L = I(t),  L·i_L' = v
 * Integrated with RK4 (20 sub-steps per frame). Component values are chosen so
 * things happen in seconds; real circuits usually run in micro- to milliseconds.
 */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { R: '#e5484d', L: '#e09b14', C: '#3d8bfd', V: '#e8ebf1', i: '#f5c451', E: '#e8ebf1' };
  const WINDOW = 10, SUB = 20, TAU = Math.PI * 2, BG = '#0b0e14';
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const fmt = (v, d = 2) => (Math.abs(v) < 0.005 && d <= 2 ? '0' : v.toFixed(d));

  const CIRCUITS = {
    rc: {
      label: 'RC', has: { R: true, L: false, C: true }, R: 10, L: 1, C: 0.1, amp: 10,
      about: 'A resistor charging a capacitor: a camera flash, the debounce filter on a button, or the timing of a 555 chip. ' +
        'Flip the switch: the capacitor fills fast at first, then ever more slowly.',
      ode: String.raw`\textcolor{${COL.R}}{R\,\frac{dq}{dt}} + \textcolor{${COL.C}}{\frac{q}{C}} = V(t)`,
      sol: String.raw`v_C(t) = V\left(1 - e^{-t/RC}\right),\qquad \tau = RC`,
      solNote: 'First order (only dq/dt appears), so the solution is a single exponential. After one time constant τ it is 63% of the way there; after 5τ it is essentially done.',
    },
    rl: {
      label: 'RL', has: { R: true, L: true, C: false }, R: 2, L: 2, C: 0.1, amp: 10,
      about: 'A resistor and an inductor: a relay coil, a motor winding, a solenoid. The inductor resists any change in current, ' +
        'so the current builds up gradually and keeps flowing for a moment after the switch opens.',
      ode: String.raw`\textcolor{${COL.L}}{L\,\frac{di}{dt}} + \textcolor{${COL.R}}{R\,i} = V(t)`,
      sol: String.raw`i(t) = \frac{V}{R}\left(1 - e^{-tR/L}\right),\qquad \tau = \frac{L}{R}`,
      solNote: 'Also first order, so it is the same exponential shape as RC, but now it is the current that lags behind, not the voltage.',
    },
    lc: {
      label: 'LC', has: { R: false, L: true, C: true }, R: 0, L: 1, C: 0.1, amp: 10,
      about: 'No resistor, so nothing loses energy. It sloshes back and forth between the capacitor (electric field) and the inductor ' +
        '(magnetic field) forever: an electrical pendulum. This is how radio tuners and oscillators pick a frequency.',
      ode: String.raw`\textcolor{${COL.L}}{L\,\frac{d^2q}{dt^2}} + \textcolor{${COL.C}}{\frac{q}{C}} = V(t)`,
      sol: String.raw`q(t) = A\cos(\omega_0 t + \phi),\qquad \omega_0 = \frac{1}{\sqrt{LC}}`,
      solNote: 'Second order with no damping term: the textbook simple harmonic oscillator, exactly like a frictionless mass on a spring. Watch the energy plot.',
    },
    rlc: {
      label: 'Series RLC', has: { R: true, L: true, C: true }, R: 1, L: 1, C: 0.1, amp: 10,
      about: 'All three in one loop, so the <b>same current</b> flows through each. The resistor decides the character of the response: ' +
        'ringing (underdamped), the fastest settle with no overshoot (critically damped), or a slow creep (overdamped).',
      ode: String.raw`\textcolor{${COL.L}}{L\,\frac{d^2q}{dt^2}} + \textcolor{${COL.R}}{R\,\frac{dq}{dt}} + \textcolor{${COL.C}}{\frac{q}{C}} = V(t)`,
      sol: String.raw`q(t) = A e^{s_1 t} + B e^{s_2 t},\qquad Ls^2 + Rs + \tfrac{1}{C} = 0`,
      solNote: 'Guess q = e^{st} and the derivatives turn into powers of s, so the ODE becomes a quadratic. Its two roots s₁, s₂ decide everything ' +
        '(the same poles as in the Laplace tab).',
    },
    prlc: {
      label: 'Parallel RLC', parallel: true, has: { R: true, L: true, C: true }, R: 10, L: 1, C: 0.1, amp: 1,
      about: 'R, L and C side by side across the same two wires, driven by a <b>current source</b>. Every branch sees the <b>same voltage</b> v, ' +
        'and the branch currents must add up to the source current (Kirchhoff\'s current law). This "tank circuit" is what tunes a radio to one station.',
      ode: String.raw`\textcolor{${COL.C}}{C\,\frac{dv}{dt}} + \textcolor{${COL.R}}{\frac{v}{R}} + \textcolor{${COL.L}}{\frac{1}{L}\!\int\! v\,dt} = I(t)`,
      sol: String.raw`C\,\ddot v + \tfrac{1}{R}\,\dot v + \tfrac{1}{L}\,v = \dot I,\qquad Cs^2 + \tfrac{1}{R}s + \tfrac{1}{L} = 0`,
      solNote: 'Differentiate once and you get the same second-order equation as the series circuit, with the roles swapped: ' +
        'C plays the mass, 1/L the spring and 1/R the damper. That is why a bigger R now means LESS damping.',
    },
  };

  const SOURCES = [
    { value: 'dc', label: 'DC + switch' },
    { value: 'square', label: 'Square wave' },
    { value: 'sine', label: 'AC sine' },
  ];

  const sim = {
    id: 'circuit',
    hint: 'Flip the switch, change R, L, C, or try the AC source and sweep its frequency',
    o: { circuit: 'rlc', R: 1, L: 1, C: 0.1, V: 10, source: 'dc', freq: 0.5, autoFlip: true, analogy: true, plot2: 'energy', speed: 1, running: true },
    st: { q: 0, i: 0, v: 0, iL: 0 }, t: 0, sw: false, nextFlip: 0.5, hist: [], dotOff: 0, segOff: new Array(10).fill(0),
    scales: { i: 1 }, hudTick: 0, heat: 0,

    get C0() { return CIRCUITS[this.o.circuit]; },
    get par() { return !!this.C0.parallel; },

    mount(panel, hud) { this.panel = panel; this.hud = hud; if (!this.loaded) this.setCircuit(this.o.circuit); else this.buildPanel(); },
    unmount() {},
    resize(W, H) { this.W = W; this.H = H; },

    setCircuit(key) {
      const c = CIRCUITS[key], o = this.o;
      Object.assign(o, { circuit: key, R: c.R, L: c.L, C: c.C, V: c.amp });
      this.loaded = true;
      this.scales = { i: c.parallel ? 0.2 : 1 };
      this.reset();
      this.buildPanel();
    },
    reset() {
      this.st = { q: 0, i: 0, v: 0, iL: 0 };
      this.t = 0; this.sw = false; this.nextFlip = 0.5; this.hist = []; this.heat = 0;
    },

    // source value: a voltage for series circuits, a current for the parallel one
    src(t) {
      const o = this.o;
      if (o.source === 'dc') return this.sw ? o.V : 0;
      if (o.source === 'square') return Math.sin(TAU * o.freq * t) >= 0 ? o.V : 0;
      return o.V * Math.sin(TAU * o.freq * t);
    },

    // ---------------------------------------------------------------- integration (RK4)
    rk4(state, f, dt) {
      const t = this.t, n = state.length, add = (a, k, h) => a.map((x, j) => x + h * k[j]);
      const k1 = f(state, t), k2 = f(add(state, k1, dt / 2), t + dt / 2), k3 = f(add(state, k2, dt / 2), t + dt / 2), k4 = f(add(state, k3, dt), t + dt);
      const out = new Array(n);
      for (let j = 0; j < n; j++) out[j] = state[j] + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
      return out;
    },

    step(dt) {
      const o = this.o, h = this.C0.has, s = this.st;
      if (this.par) {                                   // state (v, i_L)
        [s.v, s.iL] = this.rk4([s.v, s.iL], ([v, iL], t) => [(this.src(t) - v / o.R - iL) / o.C, v / o.L], dt);
      } else if (!h.L) {                                // RC: first order in q, i is algebraic
        const di = (q, t) => (this.src(t) - q / o.C) / o.R;
        [s.q] = this.rk4([s.q], ([q], t) => [di(q, t)], dt);
        s.i = di(s.q, this.t + dt);
      } else {                                          // RL, LC, series RLC: state (q, i)
        [s.q, s.i] = this.rk4([s.q, s.i], ([q, i], t) => [i, (this.src(t) - (h.R ? o.R * i : 0) - (h.C ? q / o.C : 0)) / o.L], dt);
      }
      this.t += dt;
    },

    // instantaneous values (series: voltages; parallel: branch currents)
    probe() {
      const o = this.o, h = this.C0.has, s = this.st, S = this.src(this.t);
      if (this.par) {
        const iR = s.v / o.R, iC = S - iR - s.iL;
        return { S, v: s.v, iR, iL: s.iL, iC, P: s.v * iR, iInd: s.iL, EC: 0.5 * o.C * s.v * s.v, EL: 0.5 * o.L * s.iL * s.iL };
      }
      const vC = h.C ? s.q / o.C : 0, vR = h.R ? o.R * s.i : 0, vL = h.L ? S - vR - vC : 0;
      return { S, vC, vR, vL, i: s.i, q: s.q, P: vR * s.i, iInd: s.i, EC: h.C ? (s.q * s.q) / (2 * o.C) : 0, EL: h.L ? 0.5 * o.L * s.i * s.i : 0 };
    },

    roots() {
      const o = this.o, h = this.C0.has;
      if (!this.par && !h.L) return { kind: 'first', s: [[-1 / (o.R * o.C), 0]], tau: o.R * o.C };
      if (!this.par && !h.C) return { kind: 'first', s: [[-o.R / o.L, 0]], tau: o.L / o.R };
      // series: L s² + R s + 1/C      parallel: C s² + (1/R) s + 1/L
      const a = this.par ? o.C : o.L, b = this.par ? 1 / o.R : (h.R ? o.R : 0), c = this.par ? 1 / o.L : 1 / o.C;
      const disc = b * b - 4 * a * c, w0 = 1 / Math.sqrt(o.L * o.C);
      const zeta = this.par ? (1 / (2 * o.R)) * Math.sqrt(o.L / o.C) : ((h.R ? o.R : 0) / 2) * Math.sqrt(o.C / o.L);
      const s = disc >= 0
        ? [[(-b + Math.sqrt(disc)) / (2 * a), 0], [(-b - Math.sqrt(disc)) / (2 * a), 0]]
        : [[-b / (2 * a), Math.sqrt(-disc) / (2 * a)], [-b / (2 * a), -Math.sqrt(-disc) / (2 * a)]];
      let kind = 'under';
      if (zeta === 0) kind = 'undamped'; else if (Math.abs(zeta - 1) < 0.03) kind = 'critical'; else if (zeta > 1) kind = 'over';
      return { kind, s, w0, zeta };
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, c = this.C0, p = this.panel, par = this.par;
      if (!p) return;
      p.replaceChildren();

      const s1 = UI.section('Circuit');
      s1.append(UI.segmented({ options: Object.entries(CIRCUITS).map(([value, cc]) => ({ value, label: cc.label })), value: o.circuit, onChange: (v) => this.setCircuit(v) }));
      const about = el('p', 'note'); about.innerHTML = c.about;
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => { o.running = !o.running; play.textContent = o.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play, UI.button('↺ Reset', () => this.reset()));
      s1.append(about, row, UI.slider({ label: 'Simulation speed', min: 0.1, max: 2, step: 0.05, value: o.speed, format: (v) => `${v.toFixed(2)}×`, onInput: (v) => { o.speed = v; } }));

      const s2 = UI.section('The differential equation');
      const eq = el('div', 'eq'); UI.tex(eq, c.ode, true);
      const law = el('p', 'note');
      law.innerHTML = par
        ? '<b>Kirchhoff\'s current law:</b> the current flowing into the top wire must flow out again, so the three branch currents add up to the source current. ' +
          'Each branch\'s current depends on a different operation on the shared voltage v: the capacitor on its <i>derivative</i> dv/dt, the resistor on v itself, ' +
          'the inductor on its <i>integral</i> ∫v dt. Mixing those is what makes it a differential equation.'
        : '<b>Kirchhoff\'s voltage law:</b> going once around the loop, the voltage drops add up to the source voltage. ' +
          'Each component\'s drop depends on a different <i>derivative</i> of the charge q, and that is what turns a circuit into a differential equation: ' +
          'the capacitor on q itself, the resistor on its rate i = dq/dt, the inductor on the rate of the rate di/dt = d²q/dt².';
      s2.append(eq, law);

      const rSlider = UI.slider({ label: 'R', min: par ? 0.5 : (c.has.L && c.has.C ? 0 : 0.5), max: par ? 50 : 20, step: 0.1, value: o.R, format: (v) => `${v.toFixed(1)} Ω`, onInput: (v) => { o.R = v; this.refreshChar(); } });
      const lSlider = UI.slider({ label: 'L', min: 0.1, max: 3, step: 0.05, value: o.L, format: (v) => `${v.toFixed(2)} H`, onInput: (v) => { o.L = v; this.refreshChar(); } });
      const cSlider = UI.slider({ label: 'C', min: 0.01, max: 0.5, step: 0.005, value: o.C, format: (v) => `${(v * 1000).toFixed(0)} mF`, onInput: (v) => { o.C = v; this.refreshChar(); } });
      if (par) {
        s2.append(UI.termCard({ color: COL.R, tex: String.raw`i_R = \frac{v}{R}`, title: 'Resistor branch: a leak',
          body: 'Lets energy leak away as heat. In parallel a <b>large</b> R leaks <b>less</b>, so it damps <b>less</b>: the opposite of the series circuit. ' +
            'An ideal tank (R → ∞) would ring forever.', controls: [rSlider] }));
        s2.append(UI.termCard({ color: COL.L, tex: String.raw`i_L = \frac{1}{L}\!\int\! v\,dt`, title: 'Inductor branch: a spring',
          body: 'Its current is the running total of the voltage applied to it. In this circuit it acts like the <b>spring</b>, always pulling v back toward 0. At DC it is a short circuit.', controls: [lSlider] }));
        s2.append(UI.termCard({ color: COL.C, tex: String.raw`i_C = C\,\frac{dv}{dt}`, title: 'Capacitor branch: a mass',
          body: 'Current flows only while the voltage is <b>changing</b>, so it resists sudden changes of v the way a mass resists sudden changes of speed.', controls: [cSlider] }));
      } else {
        if (c.has.R) s2.append(UI.termCard({ color: COL.R, tex: String.raw`v_R = R\,i`, title: 'Resistor: friction',
          body: 'Turns electrical energy into heat (watch it glow). It is the only part that <b>loses</b> energy, so it is what makes things settle down.', controls: [rSlider] }));
        if (c.has.L) s2.append(UI.termCard({ color: COL.L, tex: String.raw`v_L = L\,\frac{di}{dt}`, title: 'Inductor: inertia',
          body: 'A coil that fights any <b>change</b> in current, storing energy in its magnetic field (½Li²). Current through an inductor can\'t jump, just like a heavy mass can\'t change speed instantly.', controls: [lSlider] }));
        if (c.has.C) s2.append(UI.termCard({ color: COL.C, tex: String.raw`v_C = \frac{q}{C}`, title: 'Capacitor: spring',
          body: 'Two plates that store charge and energy in their electric field (½Cv²). The more charge it holds, the harder it pushes back, just like a spring. Its voltage can\'t jump.', controls: [cSlider] }));
      }

      const s3 = UI.section('Solution');
      this.charEl = el('div', 'eq');
      this.charNote = el('p', 'note');
      const sn = el('p', 'note'); sn.textContent = c.solNote;
      s3.append(this.charEl, sn, this.charNote);
      this.refreshChar();

      const s4 = UI.section(par ? 'Current source I(t)' : 'Source V(t)');
      s4.append(UI.segmented({ options: SOURCES, value: o.source, onChange: (v) => { o.source = v; if (v === 'sine') o.plot2 = 'resonance'; this.buildPanel(); } }));
      s4.append(UI.slider({
        label: par ? 'Amplitude I' : 'Amplitude V', min: 0, max: par ? 2 : 10, step: par ? 0.01 : 0.1, value: o.V,
        format: (v) => (par ? `${v.toFixed(2)} A` : `${v.toFixed(1)} V`), onInput: (v) => { o.V = v; },
      }));
      if (o.source === 'dc') {
        const r2 = el('div', 'controls-row');
        r2.append(UI.button('⏻ Flip switch', () => { this.sw = !this.sw; o.autoFlip = false; this.flipBox.querySelector('input').checked = false; }));
        s4.append(r2);
        this.flipBox = UI.check({ label: 'Flip automatically every 6 s', checked: o.autoFlip, onChange: (v) => { o.autoFlip = v; this.nextFlip = this.t + 6; } });
        s4.append(this.flipBox);
      } else {
        s4.append(UI.slider({ label: 'Frequency f', min: 0.05, max: 2, step: 0.01, value: o.freq, format: (v) => `${v.toFixed(2)} Hz`, onInput: (v) => { o.freq = v; } }));
        if (o.source === 'sine' && c.has.L && c.has.C) {
          const n = el('p', 'note'), f0 = (1 / (TAU * Math.sqrt(o.L * o.C))).toFixed(2);
          n.innerHTML = par
            ? `<b>Resonance:</b> near f₀ = 1/(2π√LC) ≈ <b>${f0} Hz</b> the inductor and capacitor currents cancel each other, the tank\'s impedance peaks at R, ` +
              'and the <b>voltage</b> swells. That is how a radio\'s tank circuit picks out one station. Larger R gives a sharper peak.'
            : `<b>Resonance:</b> sweep f toward f₀ = 1/(2π√LC) ≈ <b>${f0} Hz</b> and the current swells. ` +
              'There the inductor\'s and capacitor\'s voltages cancel each other and only R limits the current. Less R gives a taller, sharper peak.';
          s4.append(n);
        }
      }
      if (par) s4.append(el('p', 'note', 'Why a current source? With an ideal voltage source across them, each branch would just follow that voltage on its own and nothing interesting would happen.'));

      const s5 = UI.section('Show');
      s5.append(UI.segmented({
        options: [{ value: 'current', label: par ? 'Voltage v(t)' : 'Current i(t)' }, { value: 'energy', label: 'Energy' }, { value: 'resonance', label: 'Resonance curve' }],
        value: o.plot2, onChange: (v) => { o.plot2 = v; },
      }));
      s5.append(UI.check({ label: 'Mechanical twin: mass–spring–damper', checked: o.analogy, onChange: (v) => { o.analogy = v; } }));
      s5.append(UI.termCard({
        color: '#aab2c0', tex: String.raw`m\ddot x + c\,\dot x + k\,x = F`, title: 'Same equation, different physics',
        body: par
          ? 'For the parallel circuit the matching is: capacitance C ↔ mass m · 1/R ↔ damper c · 1/L ↔ spring k · voltage v ↔ velocity · current source I ↔ force F.'
          : 'Inductance L ↔ mass m · resistance R ↔ damper c · 1/C ↔ spring stiffness k · charge q ↔ position x · current i ↔ velocity v · voltage V ↔ force F. ' +
            'Engineers once built <i>analog computers</i> from RLC circuits to simulate car suspensions this way.',
      }));
      s5.append(UI.termCard({
        color: '#7cf0a8', tex: String.raw`\text{series} \leftrightarrow \text{parallel}`, title: 'Duality',
        body: 'Swap voltage ↔ current, L ↔ C, R ↔ 1/R and KVL ↔ KCL, and the series circuit turns into the parallel one. Every result for one has a mirror-image result for the other. ' +
          'Compare the two resonance curves: series peaks in <b>current</b>, parallel peaks in <b>voltage</b>.',
      }));

      p.append(s1, s2, s3, s4, s5);
    },

    refreshChar() {
      if (!this.charEl) return;
      const c = this.C0, o = this.o, r = this.roots(), par = this.par;
      let tex = c.sol;
      const f = (v) => (+v.toFixed(2)).toString();
      if (r.kind === 'first') tex += String.raw`\\[4pt] \tau = ${f(r.tau)}\,\text{s},\quad s = -1/\tau = ${f(r.s[0][0])}`;
      else {
        const [s1, s2] = r.s;
        const rootTex = s1[1] ? `${f(s1[0])} \\pm ${f(Math.abs(s1[1]))}i` : `${f(s1[0])},\\ ${f(s2[0])}`;
        tex += String.raw`\\[4pt] s = ${rootTex},\quad \omega_0 = ${f(r.w0)},\ \zeta = ${f(r.zeta)}`;
      }
      UI.tex(this.charEl, String.raw`\begin{gathered}${tex}\end{gathered}`, true);
      const msg = {
        first: 'One real negative root, so a single smooth exponential with no oscillation.',
        undamped: 'Purely imaginary roots: ζ = 0, an oscillation that never dies (the poles sit on the jω axis).',
        under: `Complex roots → <b>underdamped</b> (ζ < 1): it rings at ${f(Math.abs(r.s?.[0]?.[1] || 0))} rad/s while the envelope decays like e<sup>${f(r.s?.[0]?.[0] || 0)}t</sup>.`,
        critical: '<b>Critically damped</b> (ζ ≈ 1): the two roots merge, giving the fastest settle without overshoot. Car suspensions and meter needles aim for this.',
        over: '<b>Overdamped</b> (ζ > 1): two real roots, a slow creep with no oscillation. The slower root (closer to 0) dominates.',
      }[r.kind];
      let extra = '';
      if (r.kind !== 'first' && c.has.R) {
        extra = par
          ? ` Critical damping happens at R = ½√(L/C) = <b>${f(0.5 * Math.sqrt(o.L / o.C))} Ω</b>; <i>raise</i> R above that and it rings.`
          : ` Critical damping happens at R = 2√(L/C) = <b>${f(2 * Math.sqrt(o.L / o.C))} Ω</b>.`;
      }
      this.charNote.innerHTML = msg + extra;
    },

    // ---------------------------------------------------------------- loop
    parCurrents(p) { // currents on the 10 wire segments of the parallel schematic (see drawParallel)
      return [p.S, p.S, p.S - p.iR, p.iC, p.iR, p.iL, p.iC, p.iC, p.iC + p.iL, p.S];
    },

    frame(ctx, W, H) {
      const o = this.o;
      if (o.running) {
        const dt = (o.speed / 60) / SUB;
        for (let k = 0; k < SUB; k++) {
          if (o.source === 'dc' && o.autoFlip && this.t >= this.nextFlip) { this.sw = !this.sw; this.nextFlip = this.t + 6; }
          this.step(dt);
        }
        const p = this.probe(), fdt = o.speed / 60;
        this.heat += p.P * fdt;
        this.hist.push({ t: this.t, ...p });
        while (this.hist.length && this.hist[0].t < this.t - WINDOW - 0.5) this.hist.shift();
        if (this.par) this.parCurrents(p).forEach((cur, k) => { this.segOff[k] += clamp(cur * 110, -420, 420) * fdt; });
        else this.dotOff += clamp(p.i * 45, -420, 420) * fdt;
        this.scales.i = Math.max(Math.abs(p.iInd), this.scales.i * 0.995, 0.05);
      }
      this.draw(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    updateHud() {
      const p = this.probe(), o = this.o, r = this.roots(), parts = [];
      const sw = o.source === 'dc' ? ` · switch <b>${this.sw ? 'ON' : 'OFF'}</b>` : '';
      parts.push(this.par
        ? `v = <b>${fmt(p.v)} V</b> across every branch · I<sub>source</sub> = ${fmt(p.S)} A${sw}`
        : `i = <b>${fmt(p.i)} A</b> through every part · V<sub>source</sub> = ${fmt(p.S, 1)} V${sw}`);
      if (r.kind === 'first') parts.push(`<div>time constant τ = ${fmt(r.tau)} s</div>`);
      else parts.push(`<div>ω₀ = ${fmt(r.w0)} rad/s (period ${fmt(TAU / r.w0)} s) · ζ = ${fmt(r.zeta)} · <b>${{ undamped: 'undamped', under: 'underdamped', critical: 'critically damped', over: 'overdamped' }[r.kind]}</b></div>`);
      if (this.C0.has.R) parts.push(`<div class="dim">heat dissipated in R so far: ${fmt(this.heat, 1)} J</div>`);
      const html = parts.join('');
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    // ---------------------------------------------------------------- drawing
    draw(ctx, W, H) {
      const o = this.o, topH = Math.round(H * 0.5);
      const showTwin = o.analogy && W >= 640;
      const sch = { x: 0, y: 70, w: showTwin ? W * 0.56 : W, h: topH - 70 };
      if (this.par) this.drawParallel(ctx, sch); else this.drawSeries(ctx, sch);
      this.drawRoots(ctx, sch);
      if (showTwin) this.drawTwin(ctx, { x: W * 0.56, y: 70, w: W * 0.44, h: topH - 70 });
      const pad = 12, ph = (H - topH - pad * 3 - 22) / 2;
      const b1 = { x: pad, y: topH + pad, w: W - pad * 2, h: ph }, b2 = { x: pad, y: topH + pad * 2 + ph, w: W - pad * 2, h: ph };
      if (this.par) this.plotBranches(ctx, b1); else this.plotVoltages(ctx, b1);
      if (o.plot2 === 'current') this.plotMain(ctx, b2);
      else if (o.plot2 === 'energy') this.plotEnergy(ctx, b2);
      else this.plotResonance(ctx, b2);
    },

    // loop geometry shared by both schematics: leaves room on the left for the root plot
    loopBox(b) {
      const lw = Math.min(b.w - 230, 440), lh = Math.min(b.h * 0.66, 220);
      const x0 = b.x + b.w - lw - 70, y0 = b.y + (b.h - lh) / 2 + 4; // right margin keeps the inductor's field off the twin
      return { x0, x1: x0 + lw, y0, y1: y0 + lh, lw, lh, mx: x0 + lw / 2, my: y0 + lh / 2 };
    },

    label(ctx, txt, x, y, color, align = 'center') {
      ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillStyle = color; ctx.fillText(txt, x, y);
    },

    resistor(ctx, x, y, vertical, power) {
      const w = 84, pw = clamp(power, 0, 1);
      ctx.fillStyle = BG;
      if (vertical) ctx.fillRect(x - 14, y - w / 2 - 4, 28, w + 8); else ctx.fillRect(x - w / 2 - 4, y - 14, w + 8, 28);
      ctx.save();
      ctx.shadowColor = `rgba(255,120,60,${pw})`; ctx.shadowBlur = 22 * pw;
      ctx.strokeStyle = `rgb(${200 + 55 * pw},${110 + 40 * (1 - pw)},${90 * (1 - pw) + 60})`; ctx.lineWidth = 3;
      ctx.beginPath();
      for (let k = 0; k <= 8; k++) {
        const along = -w / 2 + (k * w) / 8, side = k === 0 || k === 8 ? 0 : k % 2 ? -11 : 11;
        const px = vertical ? x + side : x + along, py = vertical ? y + along : y + side;
        if (k) ctx.lineTo(px, py); else ctx.moveTo(vertical ? x : x - w / 2 - 4, vertical ? y - w / 2 - 4 : y);
        if (k === 0) ctx.lineTo(px, py);
      }
      ctx.lineTo(vertical ? x : x + w / 2 + 4, vertical ? y + w / 2 + 4 : y);
      ctx.stroke();
      ctx.restore();
    },

    inductor(ctx, x, y, field) {
      const hgt = 88, iy0 = y - hgt / 2;
      for (let k = 1; k <= 3; k++) {
        ctx.strokeStyle = `rgba(224,155,20,${(0.5 * field) / k})`; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.ellipse(x, y, 20 + k * 13, hgt / 2 + k * 9, 0, 0, TAU); ctx.stroke();
      }
      ctx.fillStyle = BG; ctx.fillRect(x - 6, iy0 - 4, 26, hgt + 8);
      ctx.strokeStyle = COL.L; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(x, iy0 - 4);
      for (let k = 0; k < 4; k++) ctx.arc(x, iy0 + hgt / 8 + (k * hgt) / 4, hgt / 8, -Math.PI / 2, Math.PI / 2, false);
      ctx.lineTo(x, iy0 + hgt + 4); ctx.stroke();
    },

    // capacitor: plates across the wire; `charge` in −2…2, positive = plate "a" (right / top) is +
    capacitor(ctx, x, y, vertical, charge) {
      const gap = 14, ph = 48, n = Math.round(Math.min(1, Math.abs(charge)) * 5);
      ctx.fillStyle = BG;
      if (vertical) ctx.fillRect(x - ph / 2 - 2, y - gap / 2 - 3, ph + 4, gap + 6); else ctx.fillRect(x - gap / 2 - 3, y - ph / 2 - 2, gap + 6, ph + 4);
      ctx.fillStyle = `rgba(61,139,253,${0.12 + 0.25 * Math.min(1, Math.abs(charge))})`;
      if (vertical) ctx.fillRect(x - ph / 2, y - gap / 2, ph, gap); else ctx.fillRect(x - gap / 2, y - ph / 2, gap, ph);
      ctx.strokeStyle = COL.C; ctx.lineWidth = 4; ctx.beginPath();
      if (vertical) { ctx.moveTo(x - ph / 2, y - gap / 2); ctx.lineTo(x + ph / 2, y - gap / 2); ctx.moveTo(x - ph / 2, y + gap / 2); ctx.lineTo(x + ph / 2, y + gap / 2); }
      else { ctx.moveTo(x - gap / 2, y - ph / 2); ctx.lineTo(x - gap / 2, y + ph / 2); ctx.moveTo(x + gap / 2, y - ph / 2); ctx.lineTo(x + gap / 2, y + ph / 2); }
      ctx.stroke();
      ctx.font = 'bold 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let k = 0; k < n; k++) {
        const along = -ph / 2 + ((k + 0.5) * ph) / 5;
        const [ax, ay, bx, by] = vertical ? [x + along, y - gap / 2 - 9, x + along, y + gap / 2 + 9] : [x + gap / 2 + 9, y + along, x - gap / 2 - 9, y + along];
        ctx.fillStyle = '#ff8a8d'; ctx.fillText(charge > 0 ? '+' : '−', ax, ay);
        ctx.fillStyle = '#8ab8ff'; ctx.fillText(charge > 0 ? '−' : '+', bx, by);
      }
    },

    dots(ctx, segs, offs) {
      ctx.fillStyle = COL.i;
      segs.forEach(([ax, ay, bx, by], k) => {
        const len = Math.hypot(bx - ax, by - ay), off = ((offs[k] % 26) + 26) % 26;
        for (let s = off; s < len; s += 26) { ctx.beginPath(); ctx.arc(ax + ((bx - ax) * s) / len, ay + ((by - ay) * s) / len, 3.2, 0, TAU); ctx.fill(); }
      });
    },

    drawSeries(ctx, b) {
      const o = this.o, h = this.C0.has, p = this.probe(), L = this.loopBox(b);
      const { x0, x1, y0, y1, mx, my } = L;
      // one loop: four segments, all carrying the same current (clockwise = positive)
      const segs = [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]];
      let acc = this.dotOff;
      const offs = segs.map(([ax, ay, bx, by]) => { const o2 = acc; acc -= Math.hypot(bx - ax, by - ay); return o2; });
      ctx.strokeStyle = '#5c6575'; ctx.lineWidth = 3; ctx.lineJoin = 'round';
      ctx.strokeRect(x0, y0, L.lw, L.lh);
      this.dots(ctx, segs, offs);

      if (h.R) {
        this.resistor(ctx, mx, y0, false, p.P / Math.max(1e-6, (o.V * o.V) / Math.max(o.R, 0.5)));
        this.label(ctx, `R = ${o.R.toFixed(1)} Ω`, mx, y0 - 28, COL.R);
        this.label(ctx, `v_R = ${fmt(p.vR)} V`, mx, y0 + 26, '#c9cfda');
      }
      if (h.L) {
        this.inductor(ctx, x1, my, clamp(Math.abs(p.i) / this.scales.i, 0, 1));
        this.label(ctx, `L = ${o.L.toFixed(2)} H`, x1 - 44, my - 12, COL.L, 'right');
        this.label(ctx, `v_L = ${fmt(p.vL)} V`, x1 - 44, my + 10, '#c9cfda', 'right');
      }
      if (h.C) {
        // current enters the right plate when i > 0, so that plate is + for q > 0
        this.capacitor(ctx, mx, y1, false, p.q / Math.max(1e-6, o.C * Math.max(o.V, 0.1)));
        this.label(ctx, `C = ${(o.C * 1000).toFixed(0)} mF`, mx, y1 + 40, COL.C);
        this.label(ctx, `v_C = ${fmt(p.vC)} V`, mx, y1 - 38, '#c9cfda');
      }
      // voltage source (left) + switch
      const sh = 56;
      ctx.fillStyle = BG; ctx.fillRect(x0 - 26, my - sh / 2, 52, sh);
      if (o.source === 'dc') {
        ctx.strokeStyle = this.sw ? '#e8ebf1' : '#5c6575'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(x0, my - sh / 2); ctx.lineTo(x0, my - 8); ctx.moveTo(x0 - 20, my - 8); ctx.lineTo(x0 + 20, my - 8); ctx.stroke();
        ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(x0 - 11, my + 4); ctx.lineTo(x0 + 11, my + 4); ctx.stroke();
        ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x0, my + 4); ctx.lineTo(x0, my + sh / 2); ctx.stroke();
        this.drawSwitch(ctx, x0 + 30, y0);
      } else this.acSymbol(ctx, x0, my, false);
      this.label(ctx, `V = ${fmt(p.S, 1)} V`, x0 - 32, my, '#e8ebf1', 'right');
    },

    drawSwitch(ctx, sx, y) {
      ctx.fillStyle = BG; ctx.fillRect(sx - 4, y - 20, 40, 26);
      ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(sx + 30, this.sw ? y : y - 16); ctx.stroke();
      ctx.fillStyle = '#e8ebf1'; ctx.beginPath(); ctx.arc(sx, y, 3.5, 0, TAU); ctx.arc(sx + 32, y, 3.5, 0, TAU); ctx.fill();
      this.label(ctx, this.sw ? 'ON' : 'OFF', sx + 16, y - 26, this.sw ? '#7cf0a8' : '#9aa3b2');
    },

    acSymbol(ctx, x, y, arrow) {
      ctx.fillStyle = BG; ctx.fillRect(x - 24, y - 26, 48, 52);
      ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(x, y, 20, 0, TAU); ctx.stroke();
      ctx.beginPath();
      if (arrow) { // current source: an arrow inside the circle
        ctx.moveTo(x, y + 12); ctx.lineTo(x, y - 12); ctx.moveTo(x - 6, y - 5); ctx.lineTo(x, y - 12); ctx.lineTo(x + 6, y - 5);
      } else if (this.o.source === 'sine') {
        for (let k = 0; k <= 20; k++) { const xx = x - 11 + (k * 22) / 20, yy = y - 7 * Math.sin((k / 20) * TAU); if (k) ctx.lineTo(xx, yy); else ctx.moveTo(xx, yy); }
      } else { ctx.moveTo(x - 11, y + 6); ctx.lineTo(x - 11, y - 6); ctx.lineTo(x, y - 6); ctx.lineTo(x, y + 6); ctx.lineTo(x + 11, y + 6); ctx.lineTo(x + 11, y - 6); }
      ctx.stroke();
    },

    drawParallel(ctx, b) {
      const o = this.o, p = this.probe(), L = this.loopBox(b);
      const { x0, x1, y0, y1, my, lw } = L;
      const xs = x0, xR = x0 + lw * 0.36, xL = x0 + lw * 0.68, xC = x1;
      // 10 segments; current direction: up through the source, along the top, down each branch, back along the bottom
      const segs = [
        [xs, y1, xs, y0], [xs, y0, xR, y0], [xR, y0, xL, y0], [xL, y0, xC, y0],
        [xR, y0, xR, y1], [xL, y0, xL, y1], [xC, y0, xC, y1],
        [xC, y1, xL, y1], [xL, y1, xR, y1], [xR, y1, xs, y1],
      ];
      ctx.strokeStyle = '#5c6575'; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); segs.forEach(([ax, ay, bx, by]) => { ctx.moveTo(ax, ay); ctx.lineTo(bx, by); }); ctx.stroke();
      ctx.lineCap = 'butt';
      this.dots(ctx, segs, this.segOff);
      ctx.fillStyle = '#8e98aa';
      for (const x of [xR, xL]) for (const y of [y0, y1]) { ctx.beginPath(); ctx.arc(x, y, 4.5, 0, TAU); ctx.fill(); } // junctions

      const iRef = Math.max(o.V, 0.05);
      this.resistor(ctx, xR, my, true, p.P / Math.max(1e-6, o.V * o.V * o.R));
      this.inductor(ctx, xL, my, clamp(Math.abs(p.iL) / this.scales.i, 0, 1));
      this.capacitor(ctx, xC, my, true, p.v / Math.max(0.05, iRef * o.R));
      this.acSymbol(ctx, xs, my, true);
      if (o.source === 'dc') this.label(ctx, this.sw ? 'source ON' : 'source OFF', xs, y1 + 18, this.sw ? '#7cf0a8' : '#9aa3b2');
      else this.label(ctx, o.source === 'sine' ? '∿ AC' : '⊓ square', xs, y1 + 18, '#9aa3b2');

      this.label(ctx, `I = ${fmt(p.S)} A`, xs - 30, my, '#e8ebf1', 'right');
      this.label(ctx, `R = ${o.R.toFixed(1)} Ω`, xR, y0 - 16, COL.R);
      this.label(ctx, `L = ${o.L.toFixed(2)} H`, xL, y0 - 16, COL.L);
      this.label(ctx, `C = ${(o.C * 1000).toFixed(0)} mF`, xC, y0 - 16, COL.C);
      this.label(ctx, `i_R = ${fmt(p.iR)}`, xR, y1 + 18, '#c9cfda');
      this.label(ctx, `i_L = ${fmt(p.iL)}`, xL, y1 + 18, '#c9cfda');
      this.label(ctx, `i_C = ${fmt(p.iC)}`, xC, y1 + 18, '#c9cfda');
      this.label(ctx, `v = ${fmt(p.v)} V`, (xs + xR) / 2 + 6, y0 + 22, '#e8ebf1');
      this.label(ctx, 'across every branch', (xs + xR) / 2 + 6, y0 + 38, '#9aa3b2');
    },

    drawRoots(ctx, b) {
      const r = this.roots(), bw = 132, bh = 104, bx = b.x + 12, by = b.y + b.h - bh - 4;
      ctx.fillStyle = 'rgba(255,255,255,0.04)'; ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
      let m = 1;
      for (const [re, im] of r.s) m = Math.max(m, Math.abs(re), Math.abs(im));
      m *= 1.25;
      const ox = bx + bw * 0.72, oy = by + bh / 2 + 6, sc = Math.min(bw * 0.68, bh * 0.4) / m;
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.beginPath(); ctx.moveTo(bx + 4, oy); ctx.lineTo(bx + bw - 4, oy); ctx.stroke();
      ctx.strokeStyle = 'rgba(91,155,255,0.8)';
      ctx.beginPath(); ctx.moveTo(ox, by + 18); ctx.lineTo(ox, by + bh - 4); ctx.stroke();
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
      for (const [re, im] of r.s) {
        const px = clamp(ox + re * sc, bx + 6, bx + bw - 6), py = oy - im * sc;
        ctx.beginPath(); ctx.moveTo(px - 5, py - 5); ctx.lineTo(px + 5, py + 5); ctx.moveTo(px + 5, py - 5); ctx.lineTo(px - 5, py + 5); ctx.stroke();
      }
      ctx.font = '10.5px system-ui'; ctx.fillStyle = '#9aa3b2'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('roots in the s-plane', bx + 6, by + 4);
    },

    drawTwin(ctx, b) {
      const o = this.o, h = this.C0.has, p = this.probe(), par = this.par;
      // what plays which role
      const map = par
        ? { mass: 'C', spring: '1/L', damper: '1/R', force: 'I', hasMass: true, hasSpring: true, hasDamper: true,
            disp: p.iL / Math.max(0.05, o.V), foot: 'velocity ↔ voltage v · position ↔ flux L·i_L' }
        : { mass: 'L', spring: '1/C', damper: 'R', force: 'V', hasMass: h.L, hasSpring: h.C, hasDamper: h.R,
            disp: h.C ? p.q / Math.max(1e-6, o.C * Math.max(o.V, 0.1)) : 0, foot: 'position x ↔ charge q · velocity ↔ current i' };
      const wallX = b.x + 18, cy = b.y + b.h * 0.5, mx0 = b.x + b.w * 0.55, mW = 54, mH = 44;
      const dx = clamp(map.disp * b.w * 0.16, -b.w * 0.3, b.w * 0.36), mxL = mx0 + dx;
      ctx.font = '600 11.5px system-ui'; ctx.fillStyle = '#c9cfda'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Mechanical twin', b.x + 8, b.y);
      // floor (scrolls for RL, where the "position" keeps growing)
      const floorY = cy + mH / 2 + 4;
      ctx.strokeStyle = '#3a4352'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(wallX, floorY); ctx.lineTo(b.x + b.w - 10, floorY); ctx.stroke();
      const off = map.hasSpring ? 0 : ((-(p.q || 0) * 40) % 18 + 18) % 18;
      ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
      for (let x = wallX + off; x < b.x + b.w - 10; x += 18) { ctx.beginPath(); ctx.moveTo(x, floorY); ctx.lineTo(x - 8, floorY + 8); ctx.stroke(); }
      ctx.fillStyle = '#3a4352'; ctx.fillRect(wallX - 10, cy - 52, 10, 104);
      if (map.hasSpring) {
        const sy = cy - 10, n = 12;
        ctx.strokeStyle = par ? COL.L : COL.C; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(wallX, sy);
        for (let k = 1; k < n; k++) ctx.lineTo(wallX + ((mxL - wallX) * k) / n, sy + (k % 2 ? -9 : 9));
        ctx.lineTo(mxL, sy); ctx.stroke();
        ctx.fillStyle = par ? COL.L : COL.C; ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(`spring k ↔ ${map.spring}`, (wallX + mxL) / 2, sy - 12);
      }
      if (map.hasDamper) {
        const dy = cy + 12, mid = wallX + (mxL - wallX) * 0.45;
        ctx.strokeStyle = COL.R; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(wallX, dy); ctx.lineTo(mid, dy);
        ctx.moveTo(mid - 14, dy - 8); ctx.lineTo(mid + 18, dy - 8); ctx.moveTo(mid - 14, dy + 8); ctx.lineTo(mid + 18, dy + 8); ctx.moveTo(mid - 14, dy - 8); ctx.lineTo(mid - 14, dy + 8);
        const px = clamp(mid + (mxL - mid) * 0.35, mid - 8, mid + 14);
        ctx.moveTo(px, dy - 6); ctx.lineTo(px, dy + 6); ctx.moveTo(px, dy); ctx.lineTo(mxL, dy); ctx.stroke();
        ctx.fillStyle = COL.R; ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
        ctx.fillText(`damper c ↔ ${map.damper}`, (wallX + mxL) / 2, dy + 12);
      }
      if (map.hasMass) {
        ctx.fillStyle = '#d6dbe4'; ctx.beginPath(); ctx.roundRect(mxL, cy - mH / 2, mW, mH, 5); ctx.fill();
        ctx.fillStyle = '#272e3b'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('m', mxL + mW / 2, cy);
        ctx.fillStyle = par ? COL.C : COL.L; ctx.font = '11px system-ui'; ctx.textBaseline = 'bottom';
        ctx.fillText(`mass m ↔ ${map.mass}`, mxL + mW / 2, cy - mH / 2 - 4);
      } else {
        ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5;
        ctx.strokeRect(mxL, cy - mH / 2, 12, mH); ctx.setLineDash([]);
        ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText('massless (no L)', mxL, cy - mH / 2 - 4);
      }
      const fl = (p.S / Math.max(1e-6, o.V)) * 46, ax = mxL + (map.hasMass ? mW : 12) + 6;
      if (Math.abs(fl) > 1) {
        ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(ax, cy); ctx.lineTo(ax + fl, cy);
        ctx.moveTo(ax + fl - Math.sign(fl) * 7, cy - 5); ctx.lineTo(ax + fl, cy); ctx.lineTo(ax + fl - Math.sign(fl) * 7, cy + 5); ctx.stroke();
        ctx.fillStyle = '#e8ebf1'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`force F ↔ ${map.force}`, ax + 4, cy - 8);
      }
      ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(map.foot, b.x + 8, b.y + b.h - 16);
    },

    // ---------------------------------------------------------------- plots
    frameBox(ctx, b, title, legend) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1; ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.w - 1, b.h - 1);
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 11.5px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(title, b.x + 44, b.y + 6);
      ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'right';
      let lx = b.x + b.w - 8;
      for (const [lab, color, dashed] of [...legend].reverse()) {
        ctx.fillStyle = '#aab2c0'; ctx.fillText(lab, lx, b.y + 6);
        lx -= ctx.measureText(lab).width + 6;
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dashed ? [4, 3] : []);
        ctx.beginPath(); ctx.moveTo(lx - 14, b.y + 12); ctx.lineTo(lx, b.y + 12); ctx.stroke(); ctx.setLineDash([]);
        lx -= 22;
      }
    },

    yAxis(ctx, b, lo, hi, unit) {
      const span = hi - lo, raw = span / 4, p10 = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 5, 10].map((m) => m * p10).find((s) => s >= raw);
      ctx.font = '10.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        const py = b.y + b.h - ((v - lo) / span) * b.h;
        ctx.strokeStyle = Math.abs(v) < step * 1e-6 ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.06)';
        ctx.beginPath(); ctx.moveTo(b.x, Math.round(py) + 0.5); ctx.lineTo(b.x + b.w, Math.round(py) + 0.5); ctx.stroke();
        ctx.fillStyle = '#7d8696';
        if (py > b.y + 8 && py < b.y + b.h - 6) ctx.fillText(`${+v.toFixed(3)}${unit}`, b.x + 4, py);
      }
    },

    timeSeries(ctx, b, lo, hi, key, color, width, dash) {
      const hs = this.hist;
      if (hs.length < 2) return;
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath();
      hs.forEach((s, k) => {
        const px = b.x + b.w - ((this.t - s.t) / WINDOW) * b.w, py = b.y + b.h - ((s[key] - lo) / (hi - lo)) * b.h;
        if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      });
      ctx.stroke(); ctx.restore();
    },

    autoRange(keys, name) {
      let lo = 0, hi = 0;
      for (const s of this.hist) for (const k of keys) { lo = Math.min(lo, s[k]); hi = Math.max(hi, s[k]); }
      if (hi - lo < 1e-3) hi = lo + (this.par ? 0.1 : 1);
      const pad = (hi - lo) * 0.12, t = [lo - pad, hi + pad], prev = this.scales[name];
      this.scales[name] = Array.isArray(prev) ? prev.map((v, k) => v + (t[k] - v) * 0.1) : t;
      return this.scales[name];
    },

    plotVoltages(ctx, b) {
      const h = this.C0.has, keys = ['S'], legend = [['source V', COL.V, true]];
      if (h.R) { keys.push('vR'); legend.push(['v_R', COL.R]); }
      if (h.L) { keys.push('vL'); legend.push(['v_L', COL.L]); }
      if (h.C) { keys.push('vC'); legend.push(['v_C', COL.C]); }
      const [lo, hi] = this.autoRange(keys, 'r1');
      this.frameBox(ctx, b, 'Voltages (they always add up to the source: KVL)', legend);
      this.yAxis(ctx, b, lo, hi, ' V');
      this.timeSeries(ctx, b, lo, hi, 'S', COL.V, 1.5, [5, 4]);
      if (h.R) this.timeSeries(ctx, b, lo, hi, 'vR', COL.R, 1.8);
      if (h.L) this.timeSeries(ctx, b, lo, hi, 'vL', COL.L, 1.8);
      if (h.C) this.timeSeries(ctx, b, lo, hi, 'vC', COL.C, 2.4);
    },

    plotBranches(ctx, b) {
      const [lo, hi] = this.autoRange(['S', 'iR', 'iL', 'iC'], 'r1');
      this.frameBox(ctx, b, 'Branch currents (they always add up to the source: KCL)', [['source I', COL.V, true], ['i_R', COL.R], ['i_L', COL.L], ['i_C', COL.C]]);
      this.yAxis(ctx, b, lo, hi, ' A');
      this.timeSeries(ctx, b, lo, hi, 'S', COL.V, 1.5, [5, 4]);
      this.timeSeries(ctx, b, lo, hi, 'iR', COL.R, 1.8);
      this.timeSeries(ctx, b, lo, hi, 'iC', COL.C, 1.8);
      this.timeSeries(ctx, b, lo, hi, 'iL', COL.L, 2.4);
    },

    plotMain(ctx, b) {
      const key = this.par ? 'v' : 'i', [lo, hi] = this.autoRange([key], 'r2');
      this.frameBox(ctx, b, this.par ? 'Voltage v(t), shared by every branch' : 'Current i(t) = dq/dt', [[key, COL.i]]);
      this.yAxis(ctx, b, lo, hi, this.par ? ' V' : ' A');
      this.timeSeries(ctx, b, lo, hi, key, COL.i, 2.4);
    },

    plotEnergy(ctx, b) {
      const h = this.C0.has;
      for (const s of this.hist) s.ET = s.EC + s.EL;
      const [lo, hi] = this.autoRange(['ET'], 'r3'), lo0 = Math.min(0, lo);
      const legend = [];
      if (h.C) legend.push(['capacitor ½Cv²', COL.C]);
      if (h.L) legend.push(['inductor ½Li²', COL.L]);
      legend.push(['total', COL.E, true]);
      this.frameBox(ctx, b, 'Stored energy (J)', legend);
      this.yAxis(ctx, b, lo0, hi, ' J');
      if (h.C) this.timeSeries(ctx, b, lo0, hi, 'EC', COL.C, 2.2);
      if (h.L) this.timeSeries(ctx, b, lo0, hi, 'EL', COL.L, 2.2);
      this.timeSeries(ctx, b, lo0, hi, 'ET', COL.E, 1.5, [5, 4]);
    },

    // steady-state response amplitude vs. frequency: series |I| = V/|Z|, parallel |V| = I/|Y|
    respAmp(f) {
      const o = this.o, h = this.C0.has, w = TAU * f;
      if (this.par) return o.V / Math.max(1e-4, Math.hypot(1 / o.R, w * o.C - 1 / (w * o.L)));
      const X = (h.L ? w * o.L : 0) - (h.C ? 1 / (w * o.C) : 0);
      return o.V / Math.max(1e-3, Math.hypot(h.R ? o.R : 0, X));
    },

    plotResonance(ctx, b) {
      const o = this.o, h = this.C0.has, fMax = 2, n = 300, par = this.par;
      const ys = Array.from({ length: n + 1 }, (_, k) => this.respAmp(0.01 + (k / n) * (fMax - 0.01)));
      const hi = Math.min(Math.max(...ys) * 1.12, 60);
      this.frameBox(ctx, b, par ? 'Resonance: steady-state VOLTAGE amplitude vs. source frequency' : 'Resonance: steady-state CURRENT amplitude vs. source frequency',
        [[par ? '|V(f)| = I/|Y|' : '|I(f)| = V/|Z|', COL.i]]);
      this.yAxis(ctx, b, 0, hi, par ? ' V' : ' A');
      const fx = (f) => b.x + 40 + (f / fMax) * (b.w - 50), fy = (v) => b.y + b.h - (Math.min(v, hi) / hi) * b.h;
      ctx.font = '10.5px system-ui'; ctx.fillStyle = '#7d8696'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      for (let f = 0; f <= fMax + 1e-9; f += 0.25) ctx.fillText(`${f} Hz`, fx(f), b.y + b.h - 2);
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      ctx.strokeStyle = COL.i; ctx.lineWidth = 2.2; ctx.beginPath();
      ys.forEach((v, k) => { const px = fx(0.01 + (k / n) * (fMax - 0.01)); if (k) ctx.lineTo(px, fy(v)); else ctx.moveTo(px, fy(v)); });
      ctx.stroke(); ctx.restore();
      if (h.L && h.C) {
        const f0 = 1 / (TAU * Math.sqrt(o.L * o.C));
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(fx(f0), b.y + 20); ctx.lineTo(fx(f0), b.y + b.h); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#c9cfda'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`f₀ = ${f0.toFixed(2)} Hz`, fx(f0) + 4, b.y + 22);
      }
      if (o.source === 'sine') {
        const v = this.respAmp(o.freq);
        ctx.fillStyle = '#e5484d'; ctx.beginPath(); ctx.arc(fx(o.freq), fy(v), 6, 0, TAU); ctx.fill();
        ctx.fillStyle = '#ff8a8d'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        ctx.fillText('your source ●', fx(o.freq) - 10, fy(v) + 6);
      } else {
        ctx.fillStyle = '#9aa3b2'; ctx.textAlign = 'right'; ctx.textBaseline = 'top';
        ctx.fillText('switch the source to AC sine to place yourself on this curve', b.x + b.w - 8, b.y + 24);
      }
    },

    pointer() {},
  };

  (window.Sims = window.Sims || {}).circuit = sim;
})();
