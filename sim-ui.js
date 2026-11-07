/* Small UI + colour helpers shared by the simulations. */
window.SimUI = (() => {
  'use strict';

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function tex(target, src, display = false) {
    if (window.katex) katex.render(src, target, { throwOnError: false, displayMode: display });
    else target.textContent = src;
    return target;
  }

  function section(title) {
    const s = el('section');
    s.append(el('h2', null, title));
    return s;
  }

  function slider({ label, min, max, step, value, format = (v) => String(v), onInput }) {
    const wrap = el('label', 'sim-slider');
    const top = el('div', 'sim-slider-top');
    const out = el('span', 'sim-val', format(value));
    top.append(el('span', null, label), out);
    const input = el('input');
    Object.assign(input, { type: 'range', min, max, step, value });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      out.textContent = format(v);
      onInput(v);
    });
    wrap.append(top, input);
    wrap.set = (v) => { input.value = v; out.textContent = format(v); };
    return wrap;
  }

  function check({ label, checked, onChange }) {
    const wrap = el('label', 'check');
    const input = el('input');
    Object.assign(input, { type: 'checkbox', checked });
    input.addEventListener('change', () => onChange(input.checked));
    wrap.append(input, document.createTextNode(' ' + label));
    return wrap;
  }

  function segmented({ options, value, onChange }) {
    const g = el('div', 'seg');
    const buttons = options.map((opt) => {
      const b = el('button', opt.value === value ? 'on' : '', opt.label);
      if (opt.title) b.title = opt.title;
      b.addEventListener('click', () => {
        buttons.forEach((x) => x.classList.toggle('on', x === b));
        onChange(opt.value);
      });
      return b;
    });
    g.append(...buttons);
    return g;
  }

  function button(label, onClick, cls = 'ghost-btn') {
    const b = el('button', cls, label);
    b.addEventListener('click', onClick);
    return b;
  }

  function termCard({ color, tex: t, title, body, controls = [] }) {
    const c = el('div', 'term');
    c.style.setProperty('--tc', color);
    const head = el('div', 'term-head');
    head.append(el('span', 'term-title', title), tex(el('span', 'term-tex'), t));
    const p = el('p', 'term-body');
    p.innerHTML = body;
    c.append(head, p, ...controls);
    return c;
  }

  // ---- colour maps: list of [r,g,b] stops -> 256-entry lookup table
  function lut(stops) {
    const out = new Uint8ClampedArray(256 * 3), n = stops.length - 1;
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * n, k = Math.min(n - 1, Math.floor(t)), f = t - k;
      for (let ch = 0; ch < 3; ch++) out[i * 3 + ch] = stops[k][ch] + (stops[k + 1][ch] - stops[k][ch]) * f;
    }
    return out;
  }
  const gradientCSS = (stops) =>
    `linear-gradient(90deg, ${stops.map((s, i) => `rgb(${s.join(',')}) ${(i / (stops.length - 1)) * 100}%`).join(', ')})`;

  function legend(stops, left, right) {
    return `<div class="legend-bar" style="background:${gradientCSS(stops)}"></div>` +
      `<div class="legend-labels"><span>${left}</span><span>${right}</span></div>`;
  }

  function hsl(h, s, l) { // -> [r,g,b] in 0..1
    const a = s * Math.min(l, 1 - l);
    const f = (n) => { const k = (n + h / 30) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
    return [f(0), f(8), f(4)];
  }

  // Sequential (dark → bright) and diverging (blue ← dark → red) maps, tuned for a dark stage.
  const SEQ_STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  const DIV_STOPS = [[150, 205, 255], [45, 110, 230], [20, 30, 70], [11, 14, 20], [80, 20, 25], [225, 60, 45], [255, 200, 140]];

  return { el, tex, section, slider, check, segmented, button, termCard, lut, legend, hsl, SEQ_STOPS, DIV_STOPS };
})();
