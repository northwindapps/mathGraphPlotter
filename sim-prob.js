/* Probability: Bayes' rule, the central limit theorem, and Markov chains. */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const TAU = Math.PI * 2;
  const COL = { sick: '#e5484d', healthy: '#5c6575', pos: '#f5c451', a: '#3d8bfd', b: '#2fa36b', c: '#e09b14', d: '#d6409f', walker: '#f5c451' };
  const STATE_COLORS = ['#3d8bfd', '#e09b14', '#2fa36b', '#d6409f', '#a78bfa'];
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

  // ------------------------------------------------------------------ CLT distributions
  const DISTS = {
    uniform: { label: 'Uniform', X: [0, 1], mu: 0.5, sd: Math.sqrt(1 / 12), sample: () => Math.random(), pdf: (x) => (x >= 0 && x <= 1 ? 1 : 0) },
    expo: { label: 'Exponential (skewed)', X: [0, 5], mu: 1, sd: 1, sample: () => -Math.log(1 - Math.random()), pdf: (x) => (x >= 0 ? Math.exp(-x) : 0) },
    dice: { label: 'One die', X: [0.5, 6.5], mu: 3.5, sd: Math.sqrt(35 / 12), discrete: [1, 2, 3, 4, 5, 6], sample: () => 1 + Math.floor(Math.random() * 6), pdf: () => 1 / 6 },
    bimodal: {
      label: 'Two humps', X: [0, 1], mu: 0.5, sd: Math.sqrt(0.0064 + 0.0625),
      sample: () => { const z = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(TAU * Math.random()); return (Math.random() < 0.5 ? 0.25 : 0.75) + 0.08 * z; },
      pdf: (x) => 0.5 * (Math.exp(-((x - 0.25) ** 2) / (2 * 0.0064)) + Math.exp(-((x - 0.75) ** 2) / (2 * 0.0064))) / Math.sqrt(TAU * 0.0064),
    },
    coin: { label: 'Unfair coin (p = 0.2)', X: [-0.5, 1.5], mu: 0.2, sd: Math.sqrt(0.16), discrete: [0, 1], sample: () => (Math.random() < 0.2 ? 1 : 0), pdf: (x) => (x === 1 ? 0.2 : 0.8) },
  };

  // ------------------------------------------------------------------ Markov presets
  function surferMatrix(d) {
    // links: A→B, A→C, B→C, C→A, D→C ; damping d, teleport (1−d)/n
    const L = [[0, 0.5, 0.5, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], n = 4;
    return L.map((row) => row.map((v) => d * v + (1 - d) / n));
  }
  const CHAINS = {
    weather: {
      label: 'Weather', states: ['☀ Sunny', '☁ Cloudy', '☂ Rainy'],
      P: () => [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]],
      about: 'Tomorrow\'s weather depends only on today\'s (the <b>Markov property</b>: no memory of the past). Each arrow is a probability, and each state\'s outgoing arrows add up to 1.',
    },
    surfer: {
      label: 'Web surfer (PageRank)', states: ['A', 'B', 'C', 'D'], damping: true,
      P: (o) => surferMatrix(o.damping),
      about: 'Four web pages. The surfer clicks a random link on the current page, but with probability 1 − d gets bored and jumps to a random page. ' +
        'The long-run share of time spent on each page is its <b>PageRank</b>. Page C, which everyone links to, wins; D, which nobody links to, only gets the random jumps.',
    },
    islands: {
      label: 'Two islands (slow mixing)', states: ['1', '2', '3', '4'],
      P: () => [[0.6, 0.38, 0.02, 0], [0.4, 0.6, 0, 0], [0, 0.02, 0.58, 0.4], [0, 0, 0.4, 0.6]],
      about: 'Two pairs of states, joined by a rare 2% bridge. The walker stays on one island for a long time, so the visit counts take ages to settle. ' +
        'How fast a chain "forgets" its start is set by its second-largest eigenvalue: here it is close to 1.',
    },
  };

  function stationary(P) { // left eigenvector for eigenvalue 1, by power iteration (the PageRank algorithm)
    const n = P.length; let p = new Array(n).fill(1 / n);
    for (let it = 0; it < 5000; it++) {
      const q = new Array(n).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q[j] += p[i] * P[i][j];
      let diff = 0; for (let j = 0; j < n; j++) diff += Math.abs(q[j] - p[j]);
      p = q; if (diff < 1e-14) break;
    }
    return p;
  }
  function mixingRate(P, pi) { // |λ₂| estimated from how fast the worst-case distance to π shrinks
    const n = P.length; let worst = 0;
    for (let s = 0; s < n; s++) {
      let p = new Array(n).fill(0); p[s] = 1;
      const dist = [];
      for (let k = 0; k < 400; k++) {
        dist.push(p.reduce((acc, v, j) => acc + Math.abs(v - pi[j]), 0));
        const q = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q[j] += p[i] * P[i][j]; p = q;
      }
      // geometric decay rate over a window where the distance is still well above round-off
      let k1 = 20; while (k1 < 380 && dist[k1] > 1e-9) k1++;
      const k0 = Math.max(1, Math.floor(k1 / 2));
      if (dist[k1] > 0 && dist[k0] > 0 && k1 > k0) worst = Math.max(worst, Math.pow(dist[k1] / dist[k0], 1 / (k1 - k0)));
    }
    return worst;
  }

  const MODES = [
    { value: 'bayes', label: "Bayes' rule", hint: 'Change how rare the disease is and how good the test is' },
    { value: 'clt', label: 'Central limit theorem', hint: 'Pick any distribution, then average n draws again and again' },
    { value: 'markov', label: 'Markov chains', hint: 'Watch the walker hop; compare its visit counts with the stationary distribution π' },
  ];

  const sim = {
    id: 'prob',
    get hint() { return MODES.find((m) => m.value === this.o.mode).hint; },
    o: {
      mode: 'bayes', running: true,
      bayes: { prev: 0.01, sens: 0.95, fpr: 0.05, prior: null, tests: 0 },
      clt: { dist: 'expo', n: 5, speed: 20 },
      markov: { chain: 'weather', damping: 0.85, speed: 4 },
    },
    W: 0, H: 0, hudTick: 0,

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.ready) { this.resetCLT(); this.resetMarkov(); this.ready = true; }
      this.buildPanel();
    },
    unmount() {},
    resize(W, H) { this.W = W; this.H = H; },

    // ---------------------------------------------------------------- Bayes
    bayes() {
      const b = this.o.bayes, p = b.prior ?? b.prev;
      const tp = p * b.sens, fp = (1 - p) * b.fpr, post = tp / (tp + fp);
      return { p, tp, fp, post, plus: tp + fp };
    },

    // ---------------------------------------------------------------- CLT
    resetCLT() { this.means = []; this.lastBatch = []; },
    stepCLT(k) {
      const c = this.o.clt, D = DISTS[c.dist];
      for (let r = 0; r < k; r++) {
        const batch = Array.from({ length: c.n }, () => D.sample());
        this.means.push(batch.reduce((a, v) => a + v, 0) / c.n);
        this.lastBatch = batch;
      }
      if (this.means.length > 200000) this.means.splice(0, this.means.length - 200000);
    },

    // ---------------------------------------------------------------- Markov
    resetMarkov() {
      const m = this.o.markov, C = CHAINS[m.chain];
      this.P = C.P(m);
      this.pi = stationary(this.P);
      this.lam2 = mixingRate(this.P, this.pi);
      this.walker = { state: 0, from: 0, t: 1, steps: 0, visits: new Array(C.states.length).fill(0) };
      this.walker.visits[0] = 1;
      this.mStep = 0;
    },
    hop() {
      const w = this.walker, row = this.P[w.state];
      let r = Math.random(), j = 0;
      while (j < row.length - 1 && r >= row[j]) { r -= row[j]; j++; }
      w.from = w.state; w.state = j; w.t = 0; w.steps++; w.visits[j]++;
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({ options: MODES, value: o.mode, onChange: (v) => { o.mode = v; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; } }));
      p.append(s0);
      if (o.mode === 'bayes') this.panelBayes(p);
      else if (o.mode === 'clt') this.panelCLT(p);
      else this.panelMarkov(p);
    },

    panelBayes(p) {
      const b = this.o.bayes;
      const s1 = UI.section("Bayes' rule");
      this.bayesTex = el('div', 'eq');
      s1.append(this.bayesTex);
      const n1 = el('p', 'note');
      n1.innerHTML = 'A disease affects 1% of people. A test catches 95% of sick people and wrongly flags 5% of healthy ones. You test positive: how likely are you to be sick? ' +
        'Most people guess 95%. Count the yellow-outlined squares: the true answer is far lower, because the healthy majority produces more false alarms than the rare disease produces true ones.';
      s1.append(n1);
      const upd = () => { b.prior = null; b.tests = 0; this.refreshBayes(); };
      s1.append(UI.slider({ label: 'How common the disease is (prior)', min: -3, max: Math.log10(0.5), step: 0.01, value: Math.log10(b.prev), format: (v) => pct(Math.pow(10, v), Math.pow(10, v) < 0.01 ? 2 : 1), onInput: (v) => { b.prev = Math.pow(10, v); upd(); } }));
      s1.append(UI.slider({ label: 'Sensitivity: P(+ | sick)', min: 0.5, max: 0.999, step: 0.001, value: b.sens, format: (v) => pct(v), onInput: (v) => { b.sens = v; upd(); } }));
      s1.append(UI.slider({ label: 'False-positive rate: P(+ | healthy)', min: 0.001, max: 0.3, step: 0.001, value: b.fpr, format: (v) => pct(v), onInput: (v) => { b.fpr = v; upd(); } }));
      const row = el('div', 'controls-row');
      row.append(UI.button('🔁 Positive again (second, independent test)', () => { b.prior = this.bayes().post; b.tests++; this.refreshBayes(); }),
        UI.button('Reset', () => { b.prior = null; b.tests = 0; this.refreshBayes(); }));
      s1.append(row);
      const s2 = UI.section('Why it matters');
      s2.append(UI.termCard({ color: COL.pos, tex: String.raw`\text{posterior} \propto \text{likelihood} \times \text{prior}`, title: 'Updating beliefs',
        body: 'Bayes\' rule turns "how likely is the evidence if X is true" into "how likely is X given the evidence". A second positive test uses the first answer as the new prior, and confidence jumps. ' +
          'That is how evidence accumulates.' }));
      s2.append(UI.termCard({ color: COL.a, tex: String.raw`P(\text{spam}\mid\text{words})`, title: 'In computer science',
        body: 'Spam filters (naive Bayes), medical AI, A/B testing, robot localization (where am I, given my sensors?) and every probabilistic ML model are Bayes\' rule at scale. ' +
          'The base-rate trap shown here is why rare-event detectors (fraud, intrusion, disease screening) drown in false alarms.' }));
      p.append(s1, s2);
      this.refreshBayes();
    },
    refreshBayes() {
      if (!this.bayesTex) return;
      const b = this.o.bayes, r = this.bayes(), f = (v) => (+v.toPrecision(3)).toString();
      UI.tex(this.bayesTex, String.raw`P(\text{sick}\mid +) = \frac{\overbrace{P(+\mid\text{sick})}^{${f(b.sens)}}\;\overbrace{P(\text{sick})}^{${f(r.p)}}}{\underbrace{P(+)}_{${f(r.plus)}}} = \mathbf{${f(r.post)}}`, true);
    },

    panelCLT(p) {
      const c = this.o.clt;
      const s1 = UI.section('Central limit theorem');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\bar X_n = \frac{1}{n}\sum_{i=1}^{n} X_i \;\xrightarrow{\;n\ \text{large}\;}\; \mathcal{N}\!\Big(\mu,\ \frac{\sigma^2}{n}\Big)`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'Draw n random numbers from the distribution at the top and average them. Do it thousands of times and histogram the averages (bottom). ' +
        'Whatever the starting shape (flat, lopsided, two humps, even a coin), the averages pile up into a <b>bell curve</b>, and it gets narrower like σ/√n. ' +
        'That is why the normal distribution is everywhere: measurement errors, test scores, heights are all sums of many small effects.';
      s1.append(eq, n1);
      s1.append(UI.segmented({ options: Object.entries(DISTS).map(([value, d]) => ({ value, label: d.label })), value: c.dist, onChange: (v) => { c.dist = v; this.resetCLT(); } }));
      s1.append(UI.slider({ label: 'n: how many draws are averaged', min: 1, max: 50, step: 1, value: c.n, format: (v) => `${v}`, onInput: (v) => { c.n = v; this.resetCLT(); } }));
      s1.append(UI.slider({ label: 'Averages per frame', min: 1, max: 200, step: 1, value: c.speed, format: (v) => `${v}`, onInput: (v) => { c.speed = v; } }));
      const row = el('div', 'controls-row');
      const play = UI.button(this.o.running ? '⏸ Pause' : '▶ Play', () => { this.o.running = !this.o.running; play.textContent = this.o.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play, UI.button('+10,000 at once', () => this.stepCLT(10000)), UI.button('↺ Clear', () => this.resetCLT()));
      s1.append(row);
      const s2 = UI.section('Try this');
      s2.append(UI.termCard({ color: COL.c, tex: String.raw`n = 1`, title: 'n = 1: no averaging',
        body: 'The histogram just copies the original shape. Then raise n to 2, 5, 30 and watch the bell appear. Skewed shapes (exponential, the unfair coin) need a larger n than symmetric ones.' }));
      s2.append(UI.termCard({ color: COL.a, tex: String.raw`\sigma/\sqrt{n}`, title: 'Why polls need ~1,000 people',
        body: 'Uncertainty shrinks only like 1/√n: 4× the data for half the error. That is the square-root law behind poll sizes, A/B-test durations and Monte-Carlo simulations.' }));
      p.append(s1, s2);
    },

    panelMarkov(p) {
      const m = this.o.markov, C = CHAINS[m.chain];
      const s1 = UI.section('Markov chain');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`p_{k+1} = p_k P, \qquad \pi = \pi P \;\;(\text{eigenvector with } \lambda = 1)`, true);
      s1.append(eq);
      s1.append(UI.segmented({ options: Object.entries(CHAINS).map(([value, q]) => ({ value, label: q.label })), value: m.chain, onChange: (v) => { m.chain = v; this.resetMarkov(); this.buildPanel(); } }));
      const ab = el('p', 'note'); ab.innerHTML = C.about; s1.append(ab);
      if (C.damping) s1.append(UI.slider({ label: 'Damping d (follow a link vs. jump randomly)', min: 0, max: 1, step: 0.01, value: m.damping, format: (v) => v.toFixed(2), onInput: (v) => { m.damping = v; this.resetMarkov(); } }));
      s1.append(UI.slider({ label: 'Hops per second', min: 1, max: 60, step: 1, value: m.speed, format: (v) => `${v}`, onInput: (v) => { m.speed = v; } }));
      const row = el('div', 'controls-row');
      const play = UI.button(this.o.running ? '⏸ Pause' : '▶ Play', () => { this.o.running = !this.o.running; play.textContent = this.o.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play, UI.button('+1,000 hops', () => { for (let k = 0; k < 1000; k++) this.hop(); this.walker.t = 1; }), UI.button('↺ Reset', () => this.resetMarkov()));
      s1.append(row);
      const s2 = UI.section('What to notice');
      s2.append(UI.termCard({ color: COL.walker, tex: String.raw`\frac{\text{visits}_j}{\text{steps}} \to \pi_j`, title: 'Long-run behaviour',
        body: 'The walker\'s path is random, but the fraction of time spent in each state settles to fixed numbers π (bars on the right), no matter where it started. ' +
          'π is the left eigenvector of P with eigenvalue 1: the same power iteration as in the Linear algebra tab.' }));
      s2.append(UI.termCard({ color: COL.d, tex: String.raw`|\lambda_2|`, title: 'Mixing speed',
        body: 'The distance to π shrinks roughly like |λ₂|<sup>k</sup>. Close to 1 (try <b>Two islands</b>) means the chain forgets its start slowly: the curves on the right crawl. ' +
          'Markov chains also power text prediction (n-gram models), queueing models of servers, and MCMC sampling in statistics.' }));
      p.append(s1, s2);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o;
      if (o.mode === 'clt' && o.running) this.stepCLT(o.clt.speed);
      if (o.mode === 'markov' && o.running) {
        const w = this.walker;
        w.t = Math.min(1, w.t + o.markov.speed / 60 * 1.6);
        this.mStep += o.markov.speed / 60;
        while (this.mStep >= 1) { this.hop(); this.mStep--; }
      }
      if (o.mode === 'bayes') this.drawBayes(ctx, W, H);
      else if (o.mode === 'clt') this.drawCLT(ctx, W, H);
      else this.drawMarkov(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    title(ctx, txt, x, y) { ctx.fillStyle = '#c9cfda'; ctx.font = '600 12px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(txt, x, y); },

    drawBayes(ctx, W, H) {
      // 2,000 people: with the default 1% / 95% / 5% every count is a whole number (20 sick, 19 caught, 99 false alarms)
      const b = this.o.bayes, r = this.bayes(), N = 2000, cols = 50, rows = 40;
      const gw = Math.min(W * 0.62, (H - 160) * (cols / rows)), cell = gw / cols, x0 = 30, y0 = 100;
      const sick = Math.round(r.p * N), tp = Math.round(sick * b.sens), fp = Math.round((N - sick) * b.fpr);
      this.title(ctx, `${N.toLocaleString()} people · ${sick} sick (red) · outlined = tested positive`, x0, y0 - 8);
      for (let k = 0; k < N; k++) {
        const isSick = k < sick, pos = isSick ? k < tp : k - sick < fp;
        const x = x0 + (k % cols) * cell, y = y0 + Math.floor(k / cols) * cell;
        ctx.fillStyle = isSick ? (pos ? COL.sick : 'rgba(229,72,77,0.45)') : (pos ? '#8a93a5' : 'rgba(92,101,117,0.35)');
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        if (pos) { ctx.strokeStyle = COL.pos; ctx.lineWidth = 1.6; ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); }
      }
      // the people who tested positive, side by side
      const bx = x0 + gw + 40, bw = Math.max(80, W - bx - 40), by = y0, bh = rows * cell;
      if (bw > 60) {
        this.title(ctx, 'Everyone who tested positive', bx, by - 8);
        const tpF = r.tp / r.plus;
        ctx.fillStyle = COL.sick; ctx.fillRect(bx, by + bh * (1 - tpF), bw, bh * tpF);
        ctx.fillStyle = '#8a93a5'; ctx.fillRect(bx, by, bw, bh * (1 - tpF));
        ctx.strokeStyle = COL.pos; ctx.lineWidth = 2; ctx.strokeRect(bx, by, bw, bh);
        ctx.font = '600 14px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = '#0b0e14';
        if (tpF > 0.08) ctx.fillText(`sick ${pct(tpF)}`, bx + bw / 2, by + bh * (1 - tpF / 2));
        if (1 - tpF > 0.08) ctx.fillText(`healthy ${pct(1 - tpF)}`, bx + bw / 2, by + bh * (1 - tpF) / 2);
        ctx.fillStyle = '#9aa3b2'; ctx.font = '11.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`≈ ${tp} true + ${fp} false positives`, bx, by + bh + 8);
        if (b.tests) ctx.fillText(`after ${b.tests + 1} positive tests (prior = last posterior)`, bx, by + bh + 26);
      }
      // natural-frequency tree under the grid
      const ty = y0 + rows * cell + 30;
      if (H - ty > 70) {
        ctx.font = '12.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const lines = [
          [`${N.toLocaleString()} people`, '#e8ebf1'],
          [`├─ ${sick} sick      → ${tp} test positive (sensitivity ${pct(b.sens, 0)}),  ${sick - tp} missed`, COL.sick],
          [`└─ ${N - sick} healthy → ${fp} test positive (false alarms ${pct(b.fpr, 1)}),  ${N - sick - fp} correctly negative`, '#aab2c0'],
          [`P(sick | +) = ${tp} / (${tp} + ${fp}) ≈ ${pct(tp / Math.max(1, tp + fp))}   (exact: ${pct(r.post)})`, COL.pos],
        ];
        lines.forEach(([t, c], k) => { ctx.fillStyle = c; ctx.fillText(t, x0, ty + k * 19); });
      }
    },

    drawCLT(ctx, W, H) {
      const c = this.o.clt, D = DISTS[c.dist], x0 = 50, w = W - 100;
      const sx = (v) => x0 + ((v - D.X[0]) / (D.X[1] - D.X[0])) * w;
      // top: the original distribution with the latest batch
      const ty = 110, th = Math.min(150, H * 0.2);
      this.title(ctx, `The distribution you draw from: ${D.label} (μ = ${D.mu.toFixed(2)}, σ = ${D.sd.toFixed(2)})`, x0, ty - 8);
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(x0, ty, w, th);
      let pmax = 0;
      if (D.discrete) pmax = Math.max(...D.discrete.map((v) => D.pdf(v)));
      else for (let k = 0; k <= 200; k++) pmax = Math.max(pmax, D.pdf(D.X[0] + (k / 200) * (D.X[1] - D.X[0])));
      const py = (v) => ty + th - (v / pmax) * (th - 14);
      if (D.discrete) {
        for (const v of D.discrete) { ctx.fillStyle = 'rgba(224,155,20,0.75)'; const bw = Math.max(6, w / (D.X[1] - D.X[0]) * 0.5); ctx.fillRect(sx(v) - bw / 2, py(D.pdf(v)), bw, ty + th - py(D.pdf(v))); }
      } else {
        ctx.fillStyle = 'rgba(224,155,20,0.35)'; ctx.strokeStyle = COL.c; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx(D.X[0]), ty + th);
        for (let k = 0; k <= 300; k++) { const v = D.X[0] + (k / 300) * (D.X[1] - D.X[0]); ctx.lineTo(sx(v), py(D.pdf(v))); }
        ctx.lineTo(sx(D.X[1]), ty + th); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      for (const v of this.lastBatch.slice(0, 50)) { ctx.fillStyle = '#e8ebf1'; ctx.beginPath(); ctx.arc(sx(clamp(v, D.X[0], D.X[1])), ty + th - 6 + (Math.random() - 0.5) * 4, 3, 0, TAU); ctx.fill(); }
      if (this.lastBatch.length) {
        const m = this.lastBatch.reduce((a, v) => a + v, 0) / this.lastBatch.length;
        ctx.strokeStyle = COL.pos; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(sx(m), ty); ctx.lineTo(sx(m), ty + th + 30); ctx.stroke();
        ctx.fillStyle = COL.pos; ctx.font = '11.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`average of these ${c.n} → drops into the histogram`, sx(m) + 6, ty + th + 6);
      }
      // bottom: histogram of the averages with the predicted normal curve
      const hy = ty + th + 70, hh = H - hy - 50, bins = 80, counts = new Array(bins).fill(0);
      this.title(ctx, `Histogram of ${this.means.length.toLocaleString()} averages (each of n = ${c.n} draws), with the bell curve N(μ, σ²/n)`, x0, hy - 8);
      ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(x0, hy, w, hh);
      const bwid = (D.X[1] - D.X[0]) / bins;
      for (const m of this.means) { const k = Math.floor((m - D.X[0]) / bwid); if (k >= 0 && k < bins) counts[k]++; }
      const sdm = D.sd / Math.sqrt(c.n), peak = 1 / (sdm * Math.sqrt(TAU));
      const dens = counts.map((k) => k / Math.max(1, this.means.length) / bwid);
      const ymax = Math.max(peak, ...dens) * 1.1, hyv = (v) => hy + hh - (v / ymax) * (hh - 10);
      ctx.fillStyle = 'rgba(61,139,253,0.75)';
      dens.forEach((d, k) => { if (d > 0) ctx.fillRect(sx(D.X[0] + k * bwid) + 0.5, hyv(d), Math.max(1, (w / bins) - 1), hy + hh - hyv(d)); });
      ctx.strokeStyle = '#e8ebf1'; ctx.lineWidth = 2.2; ctx.beginPath();
      for (let k = 0; k <= 400; k++) { const v = D.X[0] + (k / 400) * (D.X[1] - D.X[0]), y = hyv(peak * Math.exp(-((v - D.mu) ** 2) / (2 * sdm * sdm))); if (k) ctx.lineTo(sx(v), y); else ctx.moveTo(sx(v), y); }
      ctx.stroke();
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx(D.mu), hy); ctx.lineTo(sx(D.mu), hy + hh); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#7d8696'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const span = D.X[1] - D.X[0], stepv = span <= 2 ? 0.25 : 1;
      for (let v = Math.ceil(D.X[0] / stepv) * stepv; v <= D.X[1] + 1e-9; v += stepv) ctx.fillText(+v.toFixed(2) + '', sx(v), hy + hh + 4);
    },

    drawMarkov(ctx, W, H) {
      const m = this.o.markov, C = CHAINS[m.chain], n = C.states.length, P = this.P, w = this.walker;
      const wide = W > 820, gx = wide ? W * 0.3 : W / 2, gy = wide ? H * 0.52 : H * 0.33, R = Math.min(wide ? W * 0.2 : W * 0.3, H * 0.27);
      const pos = C.states.map((_, k) => { const a = -Math.PI / 2 + (k / n) * TAU + (n === 4 ? Math.PI / 4 : 0); return [gx + R * Math.cos(a), gy + R * Math.sin(a)]; });
      const NR = 30;
      // edges
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const pij = P[i][j];
        if (pij < 0.005) continue;
        ctx.strokeStyle = `rgba(200,210,225,${0.15 + 0.6 * pij})`; ctx.lineWidth = 1 + 5 * pij;
        if (i === j) { // self-loop
          const [x, y] = pos[i], a = Math.atan2(y - gy, x - gx), lx = x + Math.cos(a) * (NR + 14), ly = y + Math.sin(a) * (NR + 14);
          ctx.beginPath(); ctx.arc(lx, ly, 14, 0, TAU); ctx.stroke();
          ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(pij.toFixed(2), x + Math.cos(a) * (NR + 40), y + Math.sin(a) * (NR + 40));
          continue;
        }
        const [x1, y1] = pos[i], [x2, y2] = pos[j], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), nx = -dy / L, ny = dx / L, off = 10;
        const sx = x1 + (dx / L) * NR + nx * off, sy = y1 + (dy / L) * NR + ny * off, ex = x2 - (dx / L) * (NR + 4) + nx * off, ey = y2 - (dy / L) * (NR + 4) + ny * off;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); ctx.stroke();
        const a = Math.atan2(ey - sy, ex - sx);
        ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 10 * Math.cos(a - 0.4), ey - 10 * Math.sin(a - 0.4)); ctx.lineTo(ex - 10 * Math.cos(a + 0.4), ey - 10 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#9aa3b2'; ctx.font = '11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(pij.toFixed(2), (sx + ex) / 2 + nx * 12, (sy + ey) / 2 + ny * 12);
      }
      // nodes
      C.states.forEach((s, k) => {
        const [x, y] = pos[k];
        ctx.fillStyle = '#1b212c'; ctx.strokeStyle = STATE_COLORS[k]; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, NR, 0, TAU); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#e8ebf1'; ctx.font = '600 13px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y);
      });
      // the walker, gliding along its last hop
      const [fx, fy] = pos[w.from], [tx, ty] = pos[w.state], e = w.t < 1 ? 0.5 - 0.5 * Math.cos(Math.PI * w.t) : 1;
      const wx = fx + (tx - fx) * e, wy = fy + (ty - fy) * e - (w.from === w.state ? Math.sin(Math.PI * e) * 30 : 0);
      ctx.save(); ctx.shadowColor = COL.walker; ctx.shadowBlur = 16;
      ctx.fillStyle = COL.walker; ctx.beginPath(); ctx.arc(wx, wy, 9, 0, TAU); ctx.fill(); ctx.restore();

      // right: visit frequencies vs. π, and p_k converging
      const bx = wide ? W * 0.58 : 40, bw = wide ? W * 0.38 : W - 80, by = wide ? 110 : gy + R + 80, bh = wide ? (H - 200) * 0.45 : (H - by - 80) * 0.5;
      if (bh > 40) {
        this.title(ctx, `Fraction of time in each state after ${w.steps.toLocaleString()} hops vs. π`, bx, by - 8);
        const total = w.visits.reduce((a, v) => a + v, 0), slot = bw / n;
        C.states.forEach((s, k) => {
          const f = w.visits[k] / Math.max(1, total), x = bx + k * slot + slot * 0.18, bwid = slot * 0.64;
          ctx.fillStyle = STATE_COLORS[k]; ctx.globalAlpha = 0.8; ctx.fillRect(x, by + bh - f * bh, bwid, f * bh); ctx.globalAlpha = 1;
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x - 4, by + bh - this.pi[k] * bh); ctx.lineTo(x + bwid + 4, by + bh - this.pi[k] * bh); ctx.stroke();
          ctx.fillStyle = '#c9cfda'; ctx.font = '11.5px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText(`${s}`, x + bwid / 2, by + bh + 4);
          ctx.fillStyle = '#9aa3b2'; ctx.fillText(`π = ${this.pi[k].toFixed(3)}`, x + bwid / 2, by + bh + 20);
        });
        // p_k from the start state, converging to π
        const cy = by + bh + 80, ch = bh;
        if (H - cy > 60) {
          this.title(ctx, `p_k = p_0·P^k starting from "${C.states[0]}": every curve flattens onto π`, bx, cy - 8);
          ctx.fillStyle = 'rgba(255,255,255,0.03)'; ctx.fillRect(bx, cy, bw, ch);
          const K = 40; let p = new Array(n).fill(0); p[0] = 1; const hist = [p];
          for (let k = 0; k < K; k++) { const q = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) q[j] += p[i] * P[i][j]; p = q; hist.push(p); }
          for (let j = 0; j < n; j++) {
            ctx.strokeStyle = STATE_COLORS[j]; ctx.lineWidth = 2; ctx.beginPath();
            hist.forEach((pp, k) => { const x = bx + (k / K) * bw, y = cy + ch - pp[j] * ch; if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
            ctx.stroke();
            ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(bx, cy + ch - this.pi[j] * ch); ctx.lineTo(bx + bw, cy + ch - this.pi[j] * ch); ctx.stroke(); ctx.setLineDash([]);
          }
          ctx.fillStyle = '#7d8696'; ctx.font = '10.5px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(`k = ${K} steps`, bx + bw, cy + ch + 4);
        }
      }
    },

    updateHud() {
      const o = this.o;
      let html;
      if (o.mode === 'bayes') {
        const r = this.bayes();
        html = `P(sick | positive) = <b>${pct(r.post)}</b> · prior ${pct(r.p, r.p < 0.01 ? 2 : 1)} · ${pct(r.fp / r.plus, 0)} of positives are false alarms`;
      } else if (o.mode === 'clt') {
        const D = DISTS[o.clt.dist], ms = this.means, n = ms.length;
        if (n > 1) {
          const mean = ms.reduce((a, v) => a + v, 0) / n, sd = Math.sqrt(ms.reduce((a, v) => a + (v - mean) ** 2, 0) / (n - 1));
          const sk = ms.reduce((a, v) => a + ((v - mean) / sd) ** 3, 0) / n;
          html = `mean of averages <b>${mean.toFixed(3)}</b> (μ = ${D.mu.toFixed(3)}) · spread <b>${sd.toFixed(3)}</b> (σ/√n = ${(D.sd / Math.sqrt(o.clt.n)).toFixed(3)}) · skewness ${sk.toFixed(2)} (bell = 0)`;
        } else html = 'collecting averages…';
      } else {
        const w = this.walker, total = w.visits.reduce((a, v) => a + v, 0);
        const tv = 0.5 * w.visits.reduce((a, v, k) => a + Math.abs(v / total - this.pi[k]), 0);
        html = `${w.steps.toLocaleString()} hops · distance from π ${tv.toFixed(3)} · |λ₂| ≈ <b>${this.lam2.toFixed(3)}</b> (distance to π shrinks ×${this.lam2.toFixed(2)} per step)`;
      }
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    pointer() {},
  };

  sim._internals = { stationary, mixingRate, surferMatrix, DISTS, CHAINS };
  (window.Sims = window.Sims || {}).prob = sim;
})();
