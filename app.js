/* Math Graph Plotter
 * Type expressions; every free letter becomes a slider, `t` is animated time.
 * Rendering: HTML canvas. Parsing / symbolic derivatives: math.js. Pretty formulas: KaTeX.
 */
(() => {
  'use strict';

  // ---------------------------------------------------------------- constants
  const PALETTE = ['#2f7ae5', '#e5484d', '#2fa36b', '#e09b14', '#8e4ec6', '#0f9fae', '#d6409f', '#7a8a2e'];
  const DEFAULT_OPTS = { tangent: true, deriv: false, ghost: false, thetaMax: 2 };
  const GREEK = new Set(('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi rho sigma ' +
    'upsilon chi psi omega Gamma Delta Theta Lambda Xi Sigma Upsilon Phi Psi Omega').split(' '));

  // params: name -> [value, min, max, step?]
  // view:   [xmin, xmax] (equal aspect) or [xmin, xmax, ymin, ymax]
  const PRESETS = [
    { name: 'Traveling wave  A·sin(kx − ωt)', exprs: ['y = A*sin(k*x - w*t)'],
      params: { A: [1.5, 0, 3], k: [1, 0.1, 4], w: [2, 0, 6] }, view: [-10, 10] },
    { name: 'Standing wave = two waves added', exprs: ['y = sin(k*x - w*t)', 'y = sin(k*x + w*t)', 'y = sin(k*x - w*t) + sin(k*x + w*t)'],
      params: { k: [1, 0.2, 3], w: [1.5, 0, 5] }, view: [-10, 10] },
    { name: 'Beats: two close frequencies', exprs: ['y = sin(a*x) + sin(b*x - t)'],
      params: { a: [5, 1, 10], b: [5.6, 1, 10] }, view: [-20, 20, -3, 3] },
    { name: 'Quadratic  ax² + bx + c  (hover for slope)', exprs: ['f(x) = a*x^2 + b*x + c'],
      params: { a: [1, -3, 3], b: [0, -5, 5], c: [-2, -5, 5] }, view: [-6, 6] },
    { name: 'Sigmoid (logistic) & its derivative', exprs: ['y = 1/(1 + e^(-k*(x - x0)))'],
      params: { k: [1, 0.1, 10], x0: [0, -5, 5] }, view: [-8, 8, -0.4, 1.4], opts: { deriv: true } },
    { name: 'Normal distribution  N(μ, σ)', exprs: ['y = 1/(sigma*sqrt(2 pi)) * e^(-(x - mu)^2 / (2 sigma^2))'],
      params: { mu: [0, -4, 4], sigma: [1, 0.2, 3] }, view: [-6, 6, -0.15, 1.1] },
    { name: 'Damped oscillation', exprs: ['y = e^(-d*x) * cos(w*x - t)', 'y = e^(-d*x)', 'y = -e^(-d*x)'],
      params: { d: [0.3, 0, 2], w: [3, 0.5, 8] }, view: [-1, 14, -1.6, 1.6] },
    { name: 'Big-O growth rates', exprs: ['logn(n) = log2(n)', 'lin(n) = n', 'nlogn(n) = n*log2(n)', 'quad(n) = n^2', 'expo(n) = 2^n'],
      view: [-1, 16, -4, 70] },
    { name: "Amdahl's law (parallel speed-up)", exprs: ['S(n) = 1/((1 - p) + p/n)'],
      params: { p: [0.9, 0, 1] }, view: [-2, 64, -1, 22] },
    { name: 'Fourier series → square wave', exprs: ['y = sign(sin(x))', 'y = 4/pi*(sin(x) + sin(3x)/3 + sin(5x)/5 + sin(7x)/7 + sin(9x)/9)'],
      view: [-7, 7] },
    { name: 'Polar rose  r = cos(kθ)', exprs: ['r = cos(k*theta)'],
      params: { k: [4, 1, 9] }, view: [-1.8, 1.8], opts: { ghost: true } },
    { name: 'Archimedean spiral  r = aθ', exprs: ['r = a*theta'],
      params: { a: [0.3, 0.05, 1] }, view: [-8, 8], opts: { thetaMax: 6 } },
  ];

  // ---------------------------------------------------------------- state
  const state = {
    exprs: [],               // see parseExpr for fields
    params: new Map(),       // name -> { value, min, max, step, playing, dir, el }
    activeId: null,
    time: 0, timePlaying: true, timeSpeed: 1,
    view: { cx: 0, cy: 0, sx: 50, sy: 50 }, // world center + pixels per unit
    opts: { ...DEFAULT_OPTS },
    hover: null,
    ghosts: [],
  };
  const paramMemory = new Map(); // remembered slider configs (presets / removed params)
  const size = { w: 800, h: 600 };
  let nextId = 1, dirty = true, frameNo = 0, saveTimer = null, theme = {};

  // ---------------------------------------------------------------- DOM
  const $ = (s) => document.querySelector(s);
  const canvas = $('#plot'), ctx = canvas.getContext('2d'), stage = $('.stage');
  const exprListEl = $('#exprList'), paramListEl = $('#paramList'), readoutEl = $('#readout');
  const tValueEl = $('#tValue'), timePlayBtn = $('#timePlay');

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ---------------------------------------------------------------- coordinates
  const toSX = (x) => size.w / 2 + (x - state.view.cx) * state.view.sx;
  const toSY = (y) => size.h / 2 - (y - state.view.cy) * state.view.sy;
  const toWX = (px) => state.view.cx + (px - size.w / 2) / state.view.sx;
  const toWY = (py) => state.view.cy - (py - size.h / 2) / state.view.sy;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  function setViewRange(x0, x1, y0, y1) {
    const v = state.view;
    v.sx = size.w / (x1 - x0);
    v.cx = (x0 + x1) / 2;
    if (y0 == null) { v.sy = v.sx; v.cy = 0; }
    else { v.sy = size.h / (y1 - y0); v.cy = (y0 + y1) / 2; }
    dirty = true;
  }
  const currentRange = () => [toWX(0), toWX(size.w), toWY(size.h), toWY(0)];

  function zoomAt(px, py, fx, fy) {
    const v = state.view, wx = toWX(px), wy = toWY(py);
    v.sx = clamp(v.sx * fx, 1e-6, 1e9);
    v.sy = clamp(v.sy * fy, 1e-6, 1e9);
    v.cx = wx - (px - size.w / 2) / v.sx;
    v.cy = wy + (py - size.h / 2) / v.sy;
    dirty = true;
  }

  // ---------------------------------------------------------------- LaTeX helpers
  function nameTex(name) {
    const m = /^([A-Za-z]+)(\d+)$/.exec(name);
    if (m) return `${nameTex(m[1])}_{${m[2]}}`;
    if (GREEK.has(name)) return '\\' + name;
    if (name === 'pi' || name === 'tau' || name === 'phi') return '\\' + name;
    return name.length === 1 ? name : `\\mathit{${name}}`;
  }
  const texOpts = {
    parenthesis: 'auto',
    implicit: 'hide',
    handler: (node) => (node.isSymbolNode && !(node.name in math)) ? ' ' + nameTex(node.name) : undefined,
  };
  function renderTex(target, tex) {
    target.hidden = !tex;
    if (!tex) { target.textContent = ''; return; }
    if (window.katex) katex.render(tex, target, { throwOnError: false });
    else target.textContent = tex;
  }

  // ---------------------------------------------------------------- expressions
  function toNumber(v) {
    if (typeof v === 'number') return v;
    if (v && typeof v.re === 'number') return Math.abs(v.im) < 1e-9 ? v.re : NaN; // complex -> undefined
    if (typeof v === 'boolean') return v ? 1 : 0;
    return NaN;
  }
  const fnOf = (compiled, indep) => (x, scope) => {
    scope[indep] = x;
    try { return toNumber(compiled.evaluate(scope)); } catch { return NaN; }
  };
  const numericDerivative = (f) => (x, scope) => {
    const h = 1e-5 * Math.max(1, Math.abs(x));
    return (f(x + h, scope) - f(x - h, scope)) / (2 * h);
  };

  const LHS_RE = /^\s*([A-Za-z_]\w*)\s*(?:\(\s*([A-Za-z_]\w*)\s*\))?\s*=(?!=)([\s\S]*)$/;

  function makeExpr(text, visible = true) {
    const used = new Set(state.exprs.map((e) => e.color));
    const color = PALETTE.find((c) => !used.has(c)) || PALETTE[state.exprs.length % PALETTE.length];
    const e = { id: nextId++, text, color, visible };
    parseExpr(e);
    return e;
  }

  function parseExpr(e) {
    Object.assign(e, {
      kind: 'y', indep: 'x', node: null, f: null, df: null, symbols: [], usesT: false,
      parseError: null, evalError: null, tex: '', dTex: '', label: 'y', dLabel: 'dy/dx',
    });
    let src = e.text.replace(/θ/g, 'theta').replace(/π/g, 'pi').replace(/[·×]/g, '*').replace(/−/g, '-');
    if (!src.trim()) return;

    let lhs = 'y', arg = null;
    const m = LHS_RE.exec(src);
    if (m) {
      [, lhs, arg] = m;
      src = m[3];
      if (lhs === 'x' && !arg) { e.parseError = 'Curves of the form x = … are not supported yet.'; return; }
    }
    if ((lhs === 'r' && !arg) || arg === 'theta') e.kind = 'polar';
    e.indep = arg || (e.kind === 'polar' ? 'theta' : 'x');
    if (!src.trim()) return;

    const lhsTex = arg ? `${nameTex(lhs)}(${nameTex(arg)})` : nameTex(lhs);
    const dLhsTex = arg ? `${nameTex(lhs)}'(${nameTex(arg)})` : `\\frac{d${nameTex(lhs)}}{d${nameTex(e.indep)}}`;
    e.label = arg ? `${lhs}(${arg})` : lhs;
    e.dLabel = arg ? `${lhs}′(${arg})` : `d${lhs}/d${e.indep}`;

    try {
      const node = math.parse(src);
      const syms = new Set();
      node.traverse((n, path, parent) => {
        if (n.isSymbolNode && !(parent && parent.isFunctionNode && path === 'fn')) syms.add(n.name);
      });
      e.node = node;
      e.usesT = syms.has('t') && e.indep !== 't';
      e.symbols = [...syms].filter((s) => s !== e.indep && s !== 't' && !(s in math));
      e.f = fnOf(node.compile(), e.indep);
      e.tex = `${lhsTex} = ${node.toTex(texOpts)}`;
      try {
        const d = math.derivative(node, e.indep);
        e.df = fnOf(d.compile(), e.indep);
        e.dTex = `${dLhsTex} = ${d.toTex(texOpts)}`;
      } catch {
        e.df = numericDerivative(e.f); // e.g. functions math.js can't differentiate symbolically
      }
    } catch (err) {
      e.parseError = err.message;
      e.node = null;
    }
  }

  const exprError = (e) => e.parseError || e.evalError;
  const drawable = (e) => e.visible && e.node && !exprError(e);
  const activeExpr = () => state.exprs.find((e) => e.id === state.activeId) || null;

  // ---------------------------------------------------------------- parameters
  function syncParams() {
    const used = [];
    for (const e of state.exprs) for (const s of e.symbols) if (!used.includes(s)) used.push(s);
    for (const name of [...state.params.keys()]) {
      if (!used.includes(name)) { paramMemory.set(name, state.params.get(name)); state.params.delete(name); }
    }
    for (const name of used) {
      if (state.params.has(name)) continue;
      const mem = paramMemory.get(name);
      state.params.set(name, {
        value: 1, min: -5, max: 5, step: 'any', ...(mem || {}),
        playing: false, dir: 1, el: null,
      });
    }
  }

  function paramValue(p) {
    if (typeof p.step === 'number') return p.min + Math.round((p.value - p.min) / p.step) * p.step;
    return p.value;
  }
  function buildScope() {
    const s = { t: state.time };
    for (const [k, p] of state.params) s[k] = paramValue(p);
    return s;
  }

  function validateExprs() {
    const scope = buildScope();
    for (const e of state.exprs) {
      e.evalError = null;
      if (!e.node) continue;
      scope[e.indep] = 0.7;
      try { e.node.evaluate(scope); } catch (err) { e.evalError = err.message; }
    }
  }

  const fmtP = (v) => String(Math.round(v * 1000) / 1000);

  let renderedParamKeys = null;
  function renderParams(force = false) {
    const keys = [...state.params.keys()].join(',');
    if (!force && keys === renderedParamKeys) return;
    renderedParamKeys = keys;
    paramListEl.replaceChildren(...[...state.params].map(([n, p]) => buildParamRow(n, p)));
    $('#noParams').hidden = state.params.size > 0;
  }

  function buildParamRow(name, p) {
    const row = el('div', 'param-row');
    const head = el('div', 'param-head');
    const nameEl = el('span', 'pname');
    renderTex(nameEl, nameTex(name));
    const val = el('input', 'pval');
    Object.assign(val, { type: 'number', step: 'any', value: fmtP(p.value) });
    const play = el('button', 'icon-btn play', '▶');
    play.title = `Animate ${name} back and forth`;
    head.append(nameEl, el('span', 'peq', '='), val, play);

    const sl = el('div', 'param-slider');
    const minI = el('input', 'pbound'), maxI = el('input', 'pbound'), range = el('input', 'prange');
    Object.assign(minI, { type: 'number', step: 'any', value: p.min, title: 'Slider minimum' });
    Object.assign(maxI, { type: 'number', step: 'any', value: p.max, title: 'Slider maximum' });
    Object.assign(range, { type: 'range', min: p.min, max: p.max, step: p.step, value: p.value });
    range.setAttribute('aria-label', name);
    sl.append(minI, range, maxI);
    row.append(head, sl);

    const changed = () => { dirty = true; state.ghosts = []; scheduleSave(); };
    range.addEventListener('input', () => { p.value = parseFloat(range.value); val.value = fmtP(p.value); changed(); });
    val.addEventListener('input', () => {
      const v = parseFloat(val.value);
      if (!Number.isFinite(v)) return;
      p.value = v;
      if (v < p.min) { p.min = v; minI.value = v; range.min = v; }
      if (v > p.max) { p.max = v; maxI.value = v; range.max = v; }
      range.value = v;
      changed();
    });
    minI.addEventListener('change', () => {
      const v = parseFloat(minI.value);
      if (!Number.isFinite(v) || v >= p.max) { minI.value = p.min; return; }
      p.min = v; range.min = v;
      if (p.value < v) { p.value = v; val.value = fmtP(v); }
      range.value = p.value; changed();
    });
    maxI.addEventListener('change', () => {
      const v = parseFloat(maxI.value);
      if (!Number.isFinite(v) || v <= p.min) { maxI.value = p.max; return; }
      p.max = v; range.max = v;
      if (p.value > v) { p.value = v; val.value = fmtP(v); }
      range.value = p.value; changed();
    });
    play.addEventListener('click', () => {
      p.playing = !p.playing;
      play.textContent = p.playing ? '⏸' : '▶';
      play.classList.toggle('on', p.playing);
      if (!p.playing) scheduleSave();
    });
    p.el = { val, range };
    return row;
  }

  // ---------------------------------------------------------------- expression list UI
  function renderExprList() {
    exprListEl.replaceChildren(...state.exprs.map(buildExprRow));
  }

  function buildExprRow(e) {
    const row = el('div', 'expr-row');
    row.style.setProperty('--c', e.color);
    const sw = el('button', 'swatch');
    sw.title = 'Show / hide';
    sw.addEventListener('click', () => { e.visible = !e.visible; afterExprChange(); });

    const main = el('div', 'expr-main');
    const input = el('input', 'expr-input');
    Object.assign(input, { value: e.text, placeholder: 'y = a*sin(b*x)', spellcheck: false, autocomplete: 'off' });
    input.setAttribute('autocapitalize', 'off');
    input.addEventListener('input', () => { e.text = input.value; parseExpr(e); afterExprChange(); });
    input.addEventListener('focus', () => setActive(e.id));
    input.addEventListener('keydown', (ev) => {
      if (ev.key !== 'Enter') return;
      ev.preventDefault();
      const idx = state.exprs.indexOf(e);
      if (idx === state.exprs.length - 1) addExpr('');
      else state.exprs[idx + 1].dom.input.focus();
    });
    const tex = el('div', 'expr-tex'), deriv = el('div', 'expr-deriv'), err = el('div', 'expr-err');
    main.append(input, tex, deriv, err);

    const del = el('button', 'icon-btn del', '×');
    del.title = 'Remove';
    del.addEventListener('click', () => removeExpr(e));

    row.append(sw, main, del);
    e.dom = { row, input, tex, deriv, err };
    updateRow(e);
    return row;
  }

  function updateRow(e) {
    const d = e.dom;
    if (!d) return;
    const error = exprError(e);
    d.row.classList.toggle('active', e.id === state.activeId);
    d.row.classList.toggle('hidden-expr', !e.visible);
    d.err.textContent = error || '';
    d.err.hidden = !error;
    if (d.lastTex !== e.tex || d.lastErr !== error) renderTex(d.tex, error ? '' : e.tex);
    const dTex = e.id === state.activeId && !error ? e.dTex : '';
    if (d.lastDTex !== dTex) renderTex(d.deriv, dTex);
    Object.assign(d, { lastTex: e.tex, lastErr: error, lastDTex: dTex });
  }

  function afterExprChange(forceParams = false) {
    syncParams();
    validateExprs();
    renderParams(forceParams);
    state.exprs.forEach(updateRow);
    $('#timeHint').hidden = state.exprs.some((e) => e.usesT);
    state.ghosts = [];
    dirty = true;
    scheduleSave();
  }

  function setActive(id) {
    if (state.activeId === id) return;
    state.activeId = id;
    state.exprs.forEach(updateRow);
    dirty = true;
  }

  function addExpr(text) {
    const e = makeExpr(text);
    state.exprs.push(e);
    exprListEl.append(buildExprRow(e));
    afterExprChange();
    e.dom.input.focus();
  }

  function removeExpr(e) {
    state.exprs = state.exprs.filter((x) => x !== e);
    e.dom.row.remove();
    if (!state.exprs.length) { addExpr(''); return; }
    if (state.activeId === e.id) setActive(state.exprs[0].id);
    afterExprChange();
  }

  // ---------------------------------------------------------------- sampling
  function sampleExplicit(f, scope) {
    const out = [], step = 1.5, H = size.h, sy = state.view.sy;
    let prevX = NaN, prevY = NaN;
    for (let px = 0; px <= size.w + step; px += step) {
      const x = toWX(px), y = f(x, scope);
      // Big jump on screen: a steep curve or an asymptote/discontinuity?
      if (Number.isFinite(y) && Number.isFinite(prevY) && Math.abs(y - prevY) * sy > H / 2 &&
          isJump(f, scope, prevX, prevY, x, y)) out.push(NaN, NaN);
      out.push(x, y);
      prevX = x; prevY = y;
    }
    return out;
  }

  // Bisect toward the steepest half. For a continuous curve the gap shrinks as the
  // interval shrinks; across an asymptote (tan, 1/x) it stays large.
  function isJump(f, scope, xa, ya, xb, yb) {
    const gap0 = Math.abs(yb - ya);
    for (let i = 0; i < 12; i++) {
      const xm = (xa + xb) / 2, ym = f(xm, scope);
      if (!Number.isFinite(ym)) return true;
      if (Math.abs(ym - ya) > Math.abs(yb - ym)) { xb = xm; yb = ym; } else { xa = xm; ya = ym; }
    }
    return Math.abs(yb - ya) > gap0 * 0.25;
  }

  function samplePolar(f, scope) {
    const out = [], tmax = state.opts.thetaMax * Math.PI;
    const n = Math.min(12000, Math.round(700 * state.opts.thetaMax));
    const jump = Math.max(size.w, size.h);
    let pxp = NaN, pyp = NaN;
    for (let i = 0; i <= n; i++) {
      const th = (tmax * i) / n, r = f(th, scope);
      const x = r * Math.cos(th), y = r * Math.sin(th);
      if (Number.isFinite(pxp) && Math.hypot((x - pxp) * state.view.sx, (y - pyp) * state.view.sy) > jump) out.push(NaN, NaN);
      out.push(x, y);
      pxp = x; pyp = y;
    }
    return out;
  }

  // ---------------------------------------------------------------- drawing
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    theme = { bg: g('--canvas-bg'), minor: g('--grid-minor'), major: g('--grid-major'), axis: g('--axis'), label: g('--label'), guide: g('--guide') };
    dirty = true;
  }

  function strokeWorld(pts) {
    const lim = 4 * Math.max(size.w, size.h);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < pts.length; i += 2) {
      const x = pts[i], y = pts[i + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
      const sx = clamp(toSX(x), -lim, size.w + lim), sy = clamp(toSY(y), -lim, size.h + lim);
      if (pen) ctx.lineTo(sx, sy); else { ctx.moveTo(sx, sy); pen = true; }
    }
    ctx.stroke();
  }

  function niceStep(pxPerUnit) {
    const raw = 90 / pxPerUnit;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / p;
    const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
    return { major: m * p, minor: (m * p) / (m === 2 ? 4 : 5) };
  }
  function fmtTick(v, step) {
    if (Math.abs(v) < step * 1e-6) return '0';
    const a = Math.abs(v);
    if (a >= 1e6 || a < 1e-4) return v.toExponential(1).replace('e+', 'e');
    return v.toFixed(Math.max(0, -Math.floor(Math.log10(step))));
  }
  function fmtVal(v) {
    if (!Number.isFinite(v)) return 'undefined';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e6 || a < 1e-4)) return v.toExponential(3);
    return String(parseFloat(v.toFixed(4)));
  }

  function drawGrid() {
    const W = size.w, H = size.h;
    const gx = niceStep(state.view.sx), gy = niceStep(state.view.sy);
    const [x0, x1, y0, y1] = currentRange();
    const lines = (step, color) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (let i = Math.ceil(x0 / step.x); i * step.x <= x1; i++) { const s = Math.round(toSX(i * step.x)) + 0.5; ctx.moveTo(s, 0); ctx.lineTo(s, H); }
      for (let i = Math.ceil(y0 / step.y); i * step.y <= y1; i++) { const s = Math.round(toSY(i * step.y)) + 0.5; ctx.moveTo(0, s); ctx.lineTo(W, s); }
      ctx.stroke();
    };
    ctx.lineWidth = 1;
    lines({ x: gx.minor, y: gy.minor }, theme.minor);
    lines({ x: gx.major, y: gy.major }, theme.major);

    // axes
    const ax = Math.round(toSX(0)) + 0.5, ay = Math.round(toSY(0)) + 0.5;
    ctx.strokeStyle = theme.axis;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    if (ax >= 0 && ax <= W) { ctx.moveTo(ax, 0); ctx.lineTo(ax, H); }
    if (ay >= 0 && ay <= H) { ctx.moveTo(0, ay); ctx.lineTo(W, ay); }
    ctx.stroke();

    // labels (stick to the edge when an axis is off-screen)
    ctx.font = '11px system-ui, -apple-system, sans-serif';
    ctx.fillStyle = theme.label;
    ctx.strokeStyle = theme.bg;
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    const label = (txt, x, y) => { ctx.strokeText(txt, x, y); ctx.fillText(txt, x, y); };

    const ly = clamp(ay + 4, 4, H - 16);
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let i = Math.ceil(x0 / gx.major); i * gx.major <= x1; i++) {
      const sx = toSX(i * gx.major);
      if (i === 0 || sx < 16 || sx > W - 16) continue; // skip labels that would be clipped
      label(fmtTick(i * gx.major, gx.major), sx, ly);
    }
    const right = ax > W - 50;
    ctx.textAlign = right ? 'right' : 'left'; ctx.textBaseline = 'middle';
    const lx = right ? Math.min(ax, W) - 5 : Math.max(ax, 0) + 5;
    for (let i = Math.ceil(y0 / gy.major); i * gy.major <= y1; i++) {
      const sy = toSY(i * gy.major);
      if (i === 0 || sy < 8 || sy > H - 8) continue;
      label(fmtTick(i * gy.major, gy.major), lx, sy);
    }
    if (ax >= 0 && ax <= W && ay >= 0 && ay <= H) {
      ctx.textAlign = 'right'; ctx.textBaseline = 'top';
      label('0', ax - 4, ay + 4);
    }
  }

  function dot(x, y, color) {
    ctx.beginPath();
    ctx.arc(toSX(x), toSY(y), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = theme.bg;
    ctx.stroke();
  }

  function draw(animating) {
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, size.w, size.h);
    drawGrid();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    const scope = buildScope();
    const active = activeExpr();

    // ghost trails: earlier frames, fading out
    if (state.opts.ghost && state.ghosts.length) {
      ctx.lineWidth = 1.25;
      state.ghosts.forEach((g, i) => {
        ctx.globalAlpha = 0.04 + (0.3 * (i + 1)) / state.ghosts.length;
        ctx.strokeStyle = g.color;
        strokeWorld(g.pts);
      });
      ctx.globalAlpha = 1;
    }

    // derivative of the selected curve (dashed)
    if (state.opts.deriv && active && drawable(active) && active.kind !== 'polar') {
      ctx.setLineDash([7, 6]);
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = active.color;
      strokeWorld(sampleExplicit(active.df, scope));
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    const fresh = [];
    for (const e of state.exprs) {
      if (!drawable(e)) continue;
      const pts = e.kind === 'polar' ? samplePolar(e.f, scope) : sampleExplicit(e.f, scope);
      ctx.lineWidth = e === active ? 3 : 2.25;
      ctx.strokeStyle = e.color;
      strokeWorld(pts);
      fresh.push({ pts, color: e.color });
    }

    if (state.opts.ghost && animating && ++frameNo % 5 === 0) {
      state.ghosts.push(...fresh);
      const max = 30 * Math.max(1, fresh.length);
      if (state.ghosts.length > max) state.ghosts.splice(0, state.ghosts.length - max);
    }

    drawHover(scope, active);
  }

  function drawHover(scope, active) {
    if (!state.hover) { readoutEl.hidden = true; return; }
    const px = state.hover.px, x = toWX(px);
    const indep = active && active.kind !== 'polar' ? active.indep : 'x';
    const rows = [`<div class="ro-x">${indep} = ${fmtVal(x)}</div>`];

    ctx.strokeStyle = theme.guide;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, 0); ctx.lineTo(Math.round(px) + 0.5, size.h); ctx.stroke();
    ctx.setLineDash([]);

    const dots = [];
    for (const e of state.exprs) {
      if (!drawable(e) || e.kind === 'polar') continue;
      const y = e.f(x, scope);
      rows.push(`<div><span class="dot" style="background:${e.color}"></span>${e.label} = ${fmtVal(y)}</div>`);
      if (!Number.isFinite(y)) continue;
      dots.push([x, y, e.color]);
      if (e === active) {
        const m = e.df(x, scope);
        rows.push(`<div class="slope">slope ${e.dLabel} = ${fmtVal(m)}</div>`);
        if (state.opts.tangent && Number.isFinite(m)) {
          const xa = toWX(0), xb = toWX(size.w);
          ctx.strokeStyle = e.color;
          ctx.globalAlpha = 0.55;
          ctx.lineWidth = 1.5;
          strokeWorld([xa, y + m * (xa - x), xb, y + m * (xb - x)]);
          ctx.globalAlpha = 1;
        }
      }
    }
    dots.forEach(([dx, dy, c]) => dot(dx, dy, c));
    readoutEl.innerHTML = rows.join('');
    readoutEl.hidden = rows.length < 2;
  }

  // ---------------------------------------------------------------- animation loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    let animating = false;

    if (state.timePlaying && state.exprs.some((e) => drawable(e) && e.usesT)) {
      state.time += dt * state.timeSpeed;
      tValueEl.textContent = state.time.toFixed(2);
      animating = true;
    }
    for (const p of state.params.values()) {
      if (!p.playing) continue;
      p.value += (p.dir * (p.max - p.min) * dt * state.timeSpeed) / 4; // full sweep ≈ 4 s
      if (p.value >= p.max) { p.value = p.max; p.dir = -1; }
      if (p.value <= p.min) { p.value = p.min; p.dir = 1; }
      if (p.el) {
        p.el.range.value = p.value;
        if (document.activeElement !== p.el.val) p.el.val.value = fmtP(p.value);
      }
      animating = true;
    }

    if (animating || dirty) { draw(animating); dirty = false; }
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- input: pan / zoom / hover
  const pointers = new Map();
  const localPos = (ev) => { const r = canvas.getBoundingClientRect(); return { x: ev.clientX - r.left, y: ev.clientY - r.top }; };

  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    pointers.set(ev.pointerId, localPos(ev));
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (ev) => {
    const pos = localPos(ev);
    const prev = pointers.get(ev.pointerId);
    if (prev) {
      if (pointers.size === 1) {
        state.view.cx -= (pos.x - prev.x) / state.view.sx;
        state.view.cy += (pos.y - prev.y) / state.view.sy;
      } else if (pointers.size === 2) {
        const other = [...pointers].find(([id]) => id !== ev.pointerId)[1];
        const d0 = Math.hypot(prev.x - other.x, prev.y - other.y);
        const d1 = Math.hypot(pos.x - other.x, pos.y - other.y);
        if (d0 > 0) zoomAt((pos.x + other.x) / 2, (pos.y + other.y) / 2, d1 / d0, d1 / d0);
      }
      pointers.set(ev.pointerId, pos);
      dirty = true;
    }
    if (ev.pointerType === 'mouse' || ev.pointerType === 'pen') { state.hover = { px: pos.x, py: pos.y }; dirty = true; }
  });
  const endPointer = (ev) => {
    if (!pointers.delete(ev.pointerId)) return;
    if (!pointers.size) { canvas.classList.remove('dragging'); scheduleSave(); }
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', () => { state.hover = null; dirty = true; });
  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const pos = localPos(ev);
    let delta = ev.deltaY || ev.deltaX; // Shift+wheel turns into deltaX on macOS
    if (ev.deltaMode === 1) delta *= 16;
    const f = Math.exp(-delta * 0.0015);
    zoomAt(pos.x, pos.y, ev.shiftKey ? 1 : f, ev.altKey ? 1 : f);
    scheduleSave();
  }, { passive: false });

  function resize() {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    size.w = r.width; size.h = r.height;
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    dirty = true;
  }

  // ---------------------------------------------------------------- controls
  function syncOptsUI() {
    $('#optTangent').checked = state.opts.tangent;
    $('#optDeriv').checked = state.opts.deriv;
    $('#optGhost').checked = state.opts.ghost;
    $('#thetaMax').value = state.opts.thetaMax;
  }
  const bindCheck = (id, key) => $(id).addEventListener('change', (ev) => {
    state.opts[key] = ev.target.checked; state.ghosts = []; dirty = true; scheduleSave();
  });
  bindCheck('#optTangent', 'tangent');
  bindCheck('#optDeriv', 'deriv');
  bindCheck('#optGhost', 'ghost');
  $('#thetaMax').addEventListener('input', (ev) => {
    const v = parseFloat(ev.target.value);
    if (Number.isFinite(v) && v > 0) { state.opts.thetaMax = Math.min(v, 40); state.ghosts = []; dirty = true; scheduleSave(); }
  });

  function setTimePlaying(on) {
    state.timePlaying = on;
    timePlayBtn.textContent = on ? '⏸' : '▶';
    timePlayBtn.classList.toggle('on', on);
  }
  timePlayBtn.addEventListener('click', () => setTimePlaying(!state.timePlaying));
  $('#timeReset').addEventListener('click', () => { state.time = 0; tValueEl.textContent = '0.00'; state.ghosts = []; dirty = true; });
  $('#timeSpeed').addEventListener('change', (ev) => { state.timeSpeed = parseFloat(ev.target.value); });

  $('#addExpr').addEventListener('click', () => addExpr(''));
  $('#resetView').addEventListener('click', () => { setViewRange(-10, 10); scheduleSave(); });
  $('#share').addEventListener('click', async (ev) => {
    saveNow();
    const btn = ev.currentTarget;
    try { await navigator.clipboard.writeText(location.href); btn.textContent = 'Link copied!'; }
    catch { btn.textContent = 'Copy the URL from the address bar'; }
    setTimeout(() => { btn.textContent = 'Copy share link'; }, 1800);
  });

  const presetSel = $('#presets');
  PRESETS.forEach((p, i) => presetSel.append(new Option(p.name, String(i))));
  presetSel.addEventListener('change', () => {
    if (presetSel.value !== '') loadPreset(PRESETS[+presetSel.value]);
    presetSel.value = '';
  });

  // ---------------------------------------------------------------- loading / saving
  function loadState({ exprs, params = {}, view, opts = {} }) {
    state.params.clear();
    paramMemory.clear();
    for (const [k, s] of Object.entries(params)) {
      paramMemory.set(k, { value: s[0], min: s[1], max: s[2], step: typeof s[3] === 'number' ? s[3] : 'any' });
    }
    state.exprs = [];
    for (const item of exprs) {
      const [text, visible] = Array.isArray(item) ? [item[0], item[1] !== 0] : [item, true];
      state.exprs.push(makeExpr(text, visible));
    }
    if (!state.exprs.length) state.exprs.push(makeExpr(''));
    state.opts = { ...DEFAULT_OPTS, ...opts };
    syncOptsUI();
    state.time = 0;
    tValueEl.textContent = '0.00';
    state.activeId = state.exprs[0].id;
    renderExprList();
    afterExprChange(true);
    if (view) setViewRange(...view); else setViewRange(-10, 10);
  }
  const loadPreset = (p) => { loadState(p); scheduleSave(); };

  function saveNow() {
    clearTimeout(saveTimer);
    const r6 = (v) => +v.toPrecision(6);
    const data = {
      exprs: state.exprs.map((e) => [e.text, e.visible ? 1 : 0]),
      params: Object.fromEntries([...state.params].map(([k, p]) => [k, [r6(p.value), p.min, p.max, p.step]])),
      view: currentRange().map(r6),
      opts: state.opts,
    };
    history.replaceState(null, '', '#' + encodeURIComponent(JSON.stringify(data)));
  }
  function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 400); }

  function loadFromHash() {
    if (location.hash.length < 2) return false;
    try { loadState(JSON.parse(decodeURIComponent(location.hash.slice(1)))); return true; }
    catch { return false; }
  }

  // ---------------------------------------------------------------- boot
  if (!window.math) {
    document.body.innerHTML = '<p style="padding:24px">Could not load math.js. Check your internet connection and reload.</p>';
    return;
  }
  readTheme();
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);
  resize();
  new ResizeObserver(resize).observe(stage);
  setTimePlaying(true);
  if (!loadFromHash()) loadState(PRESETS[0]);
  requestAnimationFrame(frame);
})();
