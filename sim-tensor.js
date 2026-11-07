/* Tensors: n-dimensional arrays, how they sit in memory, operations as loops, and shapes in a neural network. */
(() => {
  'use strict';
  const UI = window.SimUI;
  const { el } = UI;

  const COL = { read: '#f5c451', write: '#2fa36b', axis: '#9aa3b2' };
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const prod = (a) => a.reduce((p, v) => p * v, 1);
  const contiguousStrides = (shape) => { const s = new Array(shape.length); let acc = 1; for (let a = shape.length - 1; a >= 0; a--) { s[a] = acc; acc *= shape[a]; } return s; };
  const hue = (off, n) => `hsl(${200 + (off / Math.max(1, n)) * 150}, 65%, ${48}%)`;
  const fmtShape = (s) => `[${s.join(', ')}]`;

  // every index tuple of a shape, in row-major (reading) order
  function indices(shape) {
    const out = [], n = prod(shape), idx = new Array(shape.length).fill(0);
    for (let k = 0; k < n; k++) {
      out.push(idx.slice());
      for (let a = shape.length - 1; a >= 0; a--) { if (++idx[a] < shape[a]) break; idx[a] = 0; }
    }
    return out;
  }

  const PRESETS = {
    scalar: { label: 'Scalar (rank 0)', shape: [] },
    vector: { label: 'Vector (rank 1)', shape: [6] },
    matrix: { label: 'Matrix (rank 2)', shape: [3, 4] },
    cube: { label: '3D tensor (rank 3)', shape: [2, 3, 4] },
    batch: { label: '4D tensor (rank 4)', shape: [2, 2, 2, 3] },
  };

  // ------------------------------------------------------------------ operations
  const A = [[1, 2, 0, 1], [0, 1, 3, 2], [2, 0, 1, 1]];
  const OPS = {
    add: {
      label: 'Add', einsum: 'ij,ij->ij', tex: String.raw`C_{ij} = A_{ij} + B_{ij}`,
      inputs: { A, B: [[1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 0, 1]] }, out: [3, 4],
      code: ['for (let i = 0; i < 3; i++)', '  for (let j = 0; j < 4; j++)', '    C[i][j] = A[i][j] + B[i][j];'],
      steps() { const s = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) s.push({ v: { i, j }, reads: [['A', [i, j]], ['B', [i, j]]], write: [i, j], f: (C, I) => { C[i][j] = I.A[i][j] + I.B[i][j]; } }); return s; },
      about: 'Element-wise: the same position in both tensors. Shapes must match. One loop level per axis.',
    },
    broadcast: {
      label: 'Broadcast add', einsum: 'ij,j->ij', tex: String.raw`C_{ij} = A_{ij} + b_{j}`,
      inputs: { A, b: [[10, 20, 30, 40]] }, out: [3, 4],
      code: ['for (let i = 0; i < 3; i++)', '  for (let j = 0; j < 4; j++)', '    C[i][j] = A[i][j] + b[j];   // b has no i: reused for every row'],
      steps() { const s = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) s.push({ v: { i, j }, reads: [['A', [i, j]], ['b', [0, j]]], write: [i, j], f: (C, I) => { C[i][j] = I.A[i][j] + I.b[0][j]; } }); return s; },
      about: '<b>Broadcasting</b>: the smaller tensor is reused along the missing axis, with no copies made. This is how a neural network adds one bias vector to every example in a batch.',
    },
    sum: {
      label: 'Sum over an axis', einsum: 'ij->i', tex: String.raw`c_{i} = \sum_j A_{ij}`,
      inputs: { A }, out: [3, 1],
      code: ['for (let i = 0; i < 3; i++) {', '  c[i] = 0;', '  for (let j = 0; j < 4; j++)', '    c[i] += A[i][j];', '}'], line: 3,
      steps() { const s = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) s.push({ v: { i, j }, reads: [['A', [i, j]]], write: [i, 0], f: (C, I) => { if (j === 0) C[i][0] = 0; C[i][0] += I.A[i][j]; } }); return s; },
      about: 'A <b>reduction</b>: the summed axis (j) disappears from the result, so rank 2 becomes rank 1. Means, softmax denominators and loss averages are reductions.',
    },
    matmul: {
      label: 'Matrix multiply', einsum: 'ij,jk->ik', tex: String.raw`C_{ik} = \sum_j A_{ij}\,B_{jk}`,
      inputs: { A, B: [[1, 0], [0, 1], [1, 1], [2, 0]] }, out: [3, 2],
      code: ['for (let i = 0; i < 3; i++)', '  for (let k = 0; k < 2; k++) {', '    C[i][k] = 0;', '    for (let j = 0; j < 4; j++)', '      C[i][k] += A[i][j] * B[j][k];', '  }'], line: 4,
      steps() { const s = []; for (let i = 0; i < 3; i++) for (let k = 0; k < 2; k++) for (let j = 0; j < 4; j++) s.push({ v: { i, k, j }, reads: [['A', [i, j]], ['B', [j, k]]], write: [i, k], f: (C, I) => { if (j === 0) C[i][k] = 0; C[i][k] += I.A[i][j] * I.B[j][k]; } }); return s; },
      about: 'A <b>contraction</b>: multiply along the shared index j and sum it away. Row i of A meets column k of B. Every dense neural-network layer is this loop; GPUs exist largely to run it fast.',
    },
    outer: {
      label: 'Outer product', einsum: 'i,j->ij', tex: String.raw`C_{ij} = a_{i}\,b_{j}`,
      inputs: { a: [[1], [2], [3]], b: [[1, 0, 2, 1]] }, out: [3, 4],
      code: ['for (let i = 0; i < 3; i++)', '  for (let j = 0; j < 4; j++)', '    C[i][j] = a[i] * b[j];'],
      steps() { const s = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) s.push({ v: { i, j }, reads: [['a', [i, 0]], ['b', [0, j]]], write: [i, j], f: (C, I) => { C[i][j] = I.a[i][0] * I.b[0][j]; } }); return s; },
      about: 'The opposite of a contraction: no shared index, so the rank <b>grows</b> (two rank-1 tensors give a rank-2 one). Attention scores and covariance matrices are built from products like this.',
    },
  };

  const MODES = [
    { value: 'memory', label: 'Shapes & memory', hint: 'Hover a cell or a memory box to see how its index becomes a memory position' },
    { value: 'ops', label: 'Operations = loops', hint: 'Step through the loop: yellow cells are read, the green cell is written' },
    { value: 'nn', label: 'Tensors in a neural net', hint: 'Change the sizes and watch the shapes, memory and work change' },
  ];

  const sim = {
    id: 'tensor',
    get hint() { return MODES.find((m) => m.value === this.o.mode).hint; },
    o: {
      mode: 'memory', preset: 'cube',
      op: 'matmul', running: true, speed: 2,
      nn: { N: 8, HW: 32, C: 3, K: 16, classes: 10 },
    },
    W: 0, H: 0, hudTick: 0, hover: null, msg: '',

    mount(panel, hud) {
      this.panel = panel; this.hud = hud;
      if (!this.view) this.setPreset(this.o.preset);
      if (!this.opState) this.resetOp();
      this.buildPanel();
    },
    unmount() {},
    resize(W, H) { this.W = W; this.H = H; },

    // ---------------------------------------------------------------- tensor view (data + shape + strides)
    setPreset(key) {
      this.o.preset = key;
      const shape = PRESETS[key].shape, n = prod(shape);
      this.data = Int32Array.from({ length: n }, (_, k) => k); // values = original memory positions, so layout is visible
      this.view = { shape: shape.slice(), strides: contiguousStrides(shape), contiguous: true };
      this.msg = '';
    },
    offset(idx) { return idx.reduce((s, v, a) => s + v * this.view.strides[a], 0); },
    transpose() {
      const v = this.view, r = v.shape.length;
      if (r < 2) { this.msg = 'Transpose needs at least 2 axes.'; return; }
      v.shape = v.shape.slice().reverse(); v.strides = v.strides.slice().reverse();
      v.contiguous = v.strides.every((s, a) => s === contiguousStrides(v.shape)[a]);
      this.msg = `Transposed: shape ${fmtShape(v.shape)}, strides ${fmtShape(v.strides)}. <b>No data moved</b>; only the strides were swapped. ` +
        (v.contiguous ? '' : 'Reading in index order now jumps around memory (non-contiguous).');
    },
    reshapeTo(shape) {
      const v = this.view;
      let copied = false;
      if (!v.contiguous) { // the new shape must read memory in order, so first materialize the current order
        this.data = Int32Array.from(indices(v.shape).map((idx) => this.data[this.offset(idx)]));
        copied = true;
      }
      this.view = { shape: shape.slice(), strides: contiguousStrides(shape), contiguous: true };
      this.msg = `Reshaped to ${fmtShape(shape)}: same ${prod(shape)} numbers, new shape and strides. ` +
        (copied ? '<b>A copy was needed</b> because the view was non-contiguous (PyTorch: <code>.contiguous()</code>).' : '<b>Free</b>: no data moved (NumPy/PyTorch <code>.reshape</code> / <code>.view</code>).');
    },
    reshapeOptions() {
      const n = this.data.length, opts = [[n]];
      for (let a = 2; a <= n / 2; a++) if (n % a === 0) opts.push([a, n / a]);
      if (n === 24) opts.push([2, 3, 4], [4, 3, 2], [2, 2, 6]);
      if (n === 12) opts.push([2, 2, 3], [3, 2, 2]);
      return opts.filter((s) => s.join() !== this.view.shape.join()).slice(0, 8);
    },

    // ---------------------------------------------------------------- operation stepping
    resetOp() {
      const op = OPS[this.o.op];
      this.opState = { steps: op.steps(), k: 0, C: Array.from({ length: op.out[0] }, () => new Array(op.out[1]).fill(null)), acc: 0 };
    },
    stepOp() {
      const s = this.opState, op = OPS[this.o.op];
      if (s.k >= s.steps.length) return;
      s.steps[s.k].f(s.C, op.inputs);
      s.k++;
    },

    // ---------------------------------------------------------------- panel
    buildPanel() {
      const o = this.o, p = this.panel;
      if (!p) return;
      p.replaceChildren();
      const s0 = UI.section('Explore');
      s0.append(UI.segmented({ options: MODES, value: o.mode, onChange: (v) => { o.mode = v; this.buildPanel(); document.getElementById('stageHint').textContent = this.hint; } }));
      p.append(s0);
      if (o.mode === 'memory') this.panelMemory(p);
      else if (o.mode === 'ops') this.panelOps(p);
      else this.panelNN(p);
    },

    panelMemory(p) {
      const s1 = UI.section('What is a tensor?');
      const eq = el('div', 'eq');
      UI.tex(eq, String.raw`\text{offset}(i_0, i_1, \dots) = \sum_a i_a \cdot \text{stride}_a`, true);
      const n1 = el('p', 'note');
      n1.innerHTML = 'In programming and ML, a <b>tensor</b> is an n-dimensional array: rank 0 is a number, rank 1 a list, rank 2 a grid, rank 3 a stack of grids (like a colour image: height × width × RGB), rank 4 a stack of those (a batch of images). ' +
        'In memory it is <b>always one flat array</b> plus two small lists: the <b>shape</b> and the <b>strides</b> (how far to jump in memory for +1 along each axis). ' +
        'Each cell shows the memory position it lives at.';
      s1.append(eq, n1);
      s1.append(UI.segmented({ options: Object.entries(PRESETS).map(([value, q]) => ({ value, label: q.label })), value: this.o.preset, onChange: (v) => { this.setPreset(v); this.buildPanel(); } }));
      const s2 = UI.section('Change the view, not the data');
      const row = el('div', 'controls-row');
      row.append(UI.button('⇄ Transpose (reverse axes)', () => { this.transpose(); this.buildPanel(); }), UI.button('↺ Reset', () => { this.setPreset(this.o.preset); this.buildPanel(); }));
      s2.append(row);
      if (this.data.length > 1) {
        s2.append(el('p', 'note', 'Reshape to:'));
        s2.append(UI.segmented({ options: this.reshapeOptions().map((s) => ({ value: s.join('x'), label: fmtShape(s) })), value: null, onChange: (v) => { this.reshapeTo(v.split('x').map(Number)); this.buildPanel(); } }));
      }
      const m = el('p', 'note'); m.innerHTML = this.msg || 'Try <b>Transpose</b>, then a reshape: the numbers stay in the same memory boxes; only the way we index them changes.';
      s2.append(m);
      s2.append(UI.termCard({ color: COL.read, tex: String.raw`\text{"math is the design, loops are the construction"}`, title: 'Why this matters',
        body: 'Every tensor operation, however fancy, ends up as loops that compute offsets with this formula and read or write one flat array. ' +
          'Libraries make reshape and transpose almost free by changing only shape and strides; contiguous memory is what lets CPUs and GPUs stream data fast.' }));
      s2.append(UI.termCard({ color: '#9aa3b2', tex: String.raw`T'_{ij} = R_{ia} R_{jb} T_{ab}`, title: 'Physics uses the word too',
        body: 'In physics a tensor (stress, inertia, the electromagnetic field tensor) is the same kind of array <i>plus a rule for how its numbers change when you rotate your axes</i>. ' +
          'ML mostly uses the array part; the index notation is shared.' }));
      p.append(s1, s2);
    },

    panelOps(p) {
      const o = this.o, op = OPS[o.op];
      const s1 = UI.section('Operations are loops');
      s1.append(UI.segmented({ options: Object.entries(OPS).map(([value, q]) => ({ value, label: q.label })), value: o.op, onChange: (v) => { o.op = v; this.resetOp(); this.buildPanel(); } }));
      const eq = el('div', 'eq'); UI.tex(eq, op.tex, true);
      const ab = el('p', 'note'); ab.innerHTML = op.about;
      s1.append(eq, ab);
      const row = el('div', 'controls-row');
      const play = UI.button(o.running ? '⏸ Pause' : '▶ Play', () => { o.running = !o.running; play.textContent = o.running ? '⏸ Pause' : '▶ Play'; });
      row.append(play, UI.button('Step ›', () => { o.running = false; play.textContent = '▶ Play'; this.stepOp(); }), UI.button('↺ Restart', () => this.resetOp()));
      s1.append(row, UI.slider({ label: 'Steps per second', min: 0.5, max: 12, step: 0.5, value: o.speed, format: (v) => `${v}`, onInput: (v) => { o.speed = v; } }));
      const s2 = UI.section('Einstein notation (einsum)');
      const n2 = el('p', 'note');
      n2.innerHTML = `<code>np.einsum('${op.einsum}', …)</code> describes the whole loop in one string: each letter is a loop variable; a letter that appears in the inputs but not after <code>-&gt;</code> is <b>summed over</b>. ` +
        'NumPy, PyTorch and TensorFlow all accept this notation, and their fast kernels generate exactly the loops shown on the right.';
      s2.append(n2);
      p.append(s1, s2);
    },

    panelNN(p) {
      const q = this.o.nn;
      const s1 = UI.section('Shapes flowing through a network');
      const n1 = el('p', 'note');
      n1.innerHTML = 'A small image classifier. Data moves as <b>4D tensors</b> [batch, height, width, channels]; the learned weights are tensors too. ' +
        'Watch how the batch size multiplies everything, and how the convolution does far more work than its few weights suggest.';
      s1.append(n1);
      const sl = (label, key, min, max, step = 1) => UI.slider({ label, min, max, step, value: q[key], format: (v) => `${v}`, onInput: (v) => { q[key] = v; } });
      s1.append(sl('Batch size N (images at once)', 'N', 1, 128), sl('Image size H = W (pixels)', 'HW', 8, 224, 8), sl('Input channels C (1 = grey, 3 = RGB)', 'C', 1, 3),
        sl('Filters K in the conv layer', 'K', 1, 256), sl('Number of classes', 'classes', 2, 1000));
      s1.append(UI.termCard({ color: COL.write, tex: String.raw`Y_{n,h,w,k} = \sum_{dy,dx,c} X_{n,h+dy,w+dx,c}\,W_{dy,dx,c,k}`, title: 'The conv layer as a tensor sum',
        body: 'The 3×3 kernels from the Convolution tab, now with channels and a batch: for every image n, pixel (h, w) and filter k, sum over the window and input channels. ' +
          'Seven nested loops, all of them arrays, multiply and add.' }));
      p.append(s1);
    },

    // ---------------------------------------------------------------- loop
    frame(ctx, W, H) {
      const o = this.o;
      if (o.mode === 'ops' && o.running) {
        this.acc = (this.acc || 0) + o.speed / 60;
        while (this.acc >= 1) { this.acc--; if (this.opState.k >= this.opState.steps.length) { this.pause = (this.pause || 0) + 1; if (this.pause > 2) { this.resetOp(); this.pause = 0; } } else this.stepOp(); }
      }
      if (o.mode === 'memory') this.drawMemory(ctx, W, H);
      else if (o.mode === 'ops') this.drawOps(ctx, W, H);
      else this.drawNN(ctx, W, H);
      if (++this.hudTick % 6 === 0) this.updateHud();
    },

    cellRect(ctx, x, y, s, fill, txt, opts = {}) {
      ctx.fillStyle = fill; ctx.fillRect(x, y, s - 3, s - 3);
      if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = opts.lw || 3; ctx.strokeRect(x - 1, y - 1, s - 1, s - 1); }
      if (txt != null) { ctx.fillStyle = opts.color || '#ffffff'; ctx.font = `600 ${Math.max(10, Math.min(15, s * 0.36))}px ui-monospace, Menlo, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(txt, x + (s - 3) / 2, y + (s - 3) / 2 + 1); }
    },

    drawMemory(ctx, W, H) {
      const v = this.view, sh = v.shape, r = sh.length, n = this.data.length, hits = [];
      const hoverOff = this.hover ? this.hover.off : null;
      // layout: each 2D slice ("sheet") is a rows × cols grid; sheets sit side by side (so nothing is hidden),
      // and for rank 4 the groups of sheets sit side by side with a wider gap
      const rows = r >= 2 ? sh[r - 2] : 1, cols = r >= 1 ? sh[r - 1] : 1, sheets = r >= 3 ? sh[r - 3] : 1, groups = r >= 4 ? sh[0] : 1;
      const sheetGap = 0.8, groupGap = 2;
      const unitsW = groups * sheets * cols + groups * (sheets - 1) * sheetGap + (groups - 1) * groupGap;
      const cs = Math.floor(clamp(Math.min((W - 80) / unitsW, (H - 360) / (rows + 1)), 22, 58));
      let x0 = (W - unitsW * cs) / 2;
      const top = 170;
      for (let g = 0; g < groups; g++) {
        const gx0 = x0;
        for (let s = 0; s < sheets; s++) {
          const ox = x0, oy = top;
          if (sheets > 1 || groups > 1) { ctx.fillStyle = 'rgba(255,255,255,0.035)'; ctx.fillRect(ox - 5, oy - 5, cols * cs + 7, rows * cs + 7); }
          for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
            const idx = [...(r >= 4 ? [g] : []), ...(r >= 3 ? [s] : []), ...(r >= 2 ? [i] : []), ...(r >= 1 ? [j] : [])];
            const off = this.offset(idx), val = this.data[off], x = ox + j * cs, y = oy + i * cs;
            this.cellRect(ctx, x, y, cs, hue(val, n), String(val), hoverOff === off ? { stroke: COL.read } : {});
            hits.push({ x, y, s: cs, idx, off });
          }
          if (r >= 3) { // label which slice this sheet is
            const lead = [...(r >= 4 ? [g] : []), s];
            ctx.fillStyle = COL.axis; ctx.font = '11.5px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
            ctx.fillText(`[${lead.join(', ')}, :, :]`, ox + (cols * cs) / 2, oy + rows * cs + 8);
          }
          x0 += cols * cs + (s < sheets - 1 ? sheetGap * cs : 0);
        }
        if (groups > 1) { // bracket under each group
          ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1.5;
          const by = top + rows * cs + 30;
          ctx.beginPath(); ctx.moveTo(gx0, by); ctx.lineTo(gx0, by + 6); ctx.lineTo(x0, by + 6); ctx.lineTo(x0, by); ctx.stroke();
          ctx.fillStyle = COL.axis; ctx.font = '11.5px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillText(`group [${g}, :, :, :]`, (gx0 + x0) / 2, by + 10);
        }
        x0 += groupGap * cs;
      }
      // shape / stride summary and what each axis means in the picture
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillStyle = '#c9cfda'; ctx.font = '600 12.5px system-ui';
      ctx.fillText(r ? `shape ${fmtShape(sh)} · strides ${fmtShape(v.strides)} · ${v.contiguous ? 'contiguous' : 'NOT contiguous (transposed view)'}` : 'shape [] (a single number)', 40, 112);
      const roles = ['columns', 'rows', 'sheets', 'groups'];
      ctx.font = '11.5px system-ui'; ctx.fillStyle = COL.axis;
      ctx.fillText(sh.map((d, a) => `axis ${a} (${roles[r - 1 - a]}): size ${d}, stride ${v.strides[a]}`).join('   ·   '), 40, 130);
      // memory strip
      const my = H - 150, mcs = Math.min(40, Math.floor((W - 80) / Math.min(n, 24)));
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 12.5px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`Memory: one flat array of ${n} number${n > 1 ? 's' : ''} (position = offset)`, 40, my - 8);
      const perRow = Math.max(1, Math.floor((W - 80) / mcs));
      for (let k = 0; k < n; k++) {
        const x = 40 + (k % perRow) * mcs, y = my + Math.floor(k / perRow) * (mcs + 16);
        this.cellRect(ctx, x, y, mcs, hue(this.data[k], n), String(this.data[k]), hoverOff === k ? { stroke: COL.read } : {});
        ctx.fillStyle = '#6c7586'; ctx.font = '10px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(String(k), x + mcs / 2 - 1, y + mcs);
        hits.push({ x, y, s: mcs, mem: k });
      }
      this.hits = hits;
      // offset arithmetic for the hovered cell
      if (this.hover && this.hover.idx && r) {
        const terms = this.hover.idx.map((i, a) => `${i}·${v.strides[a]}`).join(' + ');
        ctx.fillStyle = COL.read; ctx.font = '600 13px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`index ${fmtShape(this.hover.idx)} → offset = ${terms} = ${this.hover.off}`, 40, my - 52);
      }
    },

    drawOps(ctx, W, H) {
      const o = this.o, op = OPS[o.op], st = this.opState, cur = st.steps[Math.max(0, st.k - 1)], done = st.k;
      const grids = Object.entries(op.inputs).map(([name, m]) => ({ name, m })).concat([{ name: op.out[1] === 1 && o.op === 'sum' ? 'c' : 'C', m: st.C, out: true }]);
      const cs = 44;
      let x = 40;
      const y = 140, readSet = new Set(done ? cur.reads.map(([t, i]) => `${t}:${i.join(',')}`) : []);
      for (const g of grids) {
        const rows = g.m.length, cols = g.m[0].length;
        ctx.fillStyle = g.out ? COL.write : '#c9cfda'; ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`${g.name}  ${fmtShape(g.out ? (o.op === 'sum' ? [rows] : [rows, cols]) : (rows === 1 || cols === 1 ? [rows * cols] : [rows, cols]))}`, x, y - 8);
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
          const val = g.m[i][j], key = `${g.name}:${i},${j}`;
          const isRead = readSet.has(key), isWrite = g.out && done && cur.write[0] === i && cur.write[1] === j;
          const fill = g.out ? (val === null ? 'rgba(255,255,255,0.05)' : 'rgba(47,163,107,0.35)') : 'rgba(61,139,253,0.3)';
          this.cellRect(ctx, x + j * cs, y + i * cs, cs, fill, val === null ? '' : String(val), isRead ? { stroke: COL.read } : isWrite ? { stroke: COL.write } : {});
        }
        x += cols * cs + 50;
        if (g !== grids[grids.length - 1]) {
          ctx.fillStyle = '#e8ebf1'; ctx.font = '600 22px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          const sym = g === grids[grids.length - 2] ? '=' : ({ add: '+', broadcast: '+', matmul: '×', outer: '⊗', sum: '' }[o.op] || '');
          ctx.fillText(sym, x - 25, y + 1.5 * cs);
        }
      }
      // the code, with the executing line highlighted and current loop variables
      const cy = y + 4 * cs + 40, lh = 24, codeX = 40;
      ctx.fillStyle = '#c9cfda'; ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(`The loop that computes it  ·  einsum '${op.einsum}'`, codeX, cy - 8);
      const hl = op.line != null ? op.line : op.code.length - 1;
      ctx.fillStyle = 'rgba(255,255,255,0.04)'; ctx.fillRect(codeX - 8, cy - 4, Math.min(W - 80, 640), op.code.length * lh + 12);
      op.code.forEach((line, k) => {
        if (k === hl && done) { ctx.fillStyle = 'rgba(245,196,81,0.18)'; ctx.fillRect(codeX - 8, cy + k * lh, Math.min(W - 80, 640), lh); }
        ctx.fillStyle = k === hl ? '#f5c451' : '#e8ebf1'; ctx.font = '14px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(line, codeX, cy + k * lh + 4);
      });
      if (done) {
        const vars = Object.entries(cur.v).map(([k, val]) => `${k} = ${val}`).join('   ');
        ctx.fillStyle = COL.read; ctx.font = '600 14px ui-monospace, Menlo, monospace';
        ctx.fillText(vars, codeX, cy + op.code.length * lh + 22);
      }
    },

    drawNN(ctx, W, H) {
      const q = this.o.nn, Kp = 9 * q.C * q.K + q.K, Dp = q.K * q.classes + q.classes;
      const L = [
        { name: 'Input images X', shape: [q.N, q.HW, q.HW, q.C], kind: 'data' },
        { name: 'Conv 3×3 → Y', shape: [q.N, q.HW, q.HW, q.K], kind: 'data', w: { name: 'kernels W', shape: [3, 3, q.C, q.K], params: Kp }, macs: q.N * q.HW * q.HW * q.K * 9 * q.C },
        { name: 'ReLU (element-wise)', shape: [q.N, q.HW, q.HW, q.K], kind: 'data', macs: 0 },
        { name: 'Average over H, W', shape: [q.N, q.K], kind: 'data', note: 'reduction: 2 axes summed away' },
        { name: 'Dense → class scores', shape: [q.N, q.classes], kind: 'data', w: { name: 'weights', shape: [q.K, q.classes], params: Dp }, macs: q.N * q.K * q.classes, einsum: "nk,kc->nc" },
      ];
      const n = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(2)} G` : v >= 1e6 ? `${(v / 1e6).toFixed(2)} M` : v >= 1e3 ? `${(v / 1e3).toFixed(1)} k` : String(v));
      const x0 = 50, y0 = 110, rowH = Math.min(120, (H - 180) / L.length), maxLog = Math.log10(Math.max(...L.map((l) => prod(l.shape))));
      let totalMacs = 0;
      L.forEach((l, k) => {
        const y = y0 + k * rowH, size = prod(l.shape), bw = Math.max(18, (Math.log10(size + 1) / maxLog) * (W * 0.42));
        ctx.fillStyle = 'rgba(61,139,253,0.35)'; ctx.fillRect(x0, y, bw, rowH * 0.5);
        ctx.strokeStyle = '#5b9bff'; ctx.lineWidth = 1.5; ctx.strokeRect(x0 + 0.5, y + 0.5, bw, rowH * 0.5);
        ctx.fillStyle = '#e8ebf1'; ctx.font = '600 13px system-ui'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(`${l.name}   ${fmtShape(l.shape)}`, x0, y + rowH * 0.5 + 6);
        ctx.fillStyle = '#9aa3b2'; ctx.font = '12px system-ui';
        ctx.fillText(`${n(size)} numbers · ${n(size * 4)}B in float32${l.note ? ' · ' + l.note : ''}`, x0 + bw + 14, y + 4);
        if (l.w) {
          ctx.fillStyle = '#e09b14'; ctx.fillText(`learned ${l.w.name} ${fmtShape(l.w.shape)} + bias: ${n(l.w.params)} parameters`, x0 + bw + 14, y + 22);
          ctx.fillStyle = '#f5c451'; ctx.fillText(`${n(l.macs)} multiply-adds${l.einsum ? `  ·  einsum '${l.einsum}'` : ''}`, x0 + bw + 14, y + 40);
          totalMacs += l.macs;
        }
        if (k < L.length - 1) { ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0 + 9, y + rowH * 0.5 + 24); ctx.lineTo(x0 + 9, y + rowH - 4); ctx.stroke(); }
      });
      this.nnTotals = { macs: totalMacs, params: Kp + Dp };
    },

    updateHud() {
      const o = this.o;
      let html;
      if (o.mode === 'memory') {
        const v = this.view;
        html = `rank ${v.shape.length} · shape ${fmtShape(v.shape)} · ${this.data.length} numbers in one flat array · strides ${fmtShape(v.strides)}`;
      } else if (o.mode === 'ops') {
        const st = this.opState, op = OPS[o.op], mults = o.op === 'matmul' || o.op === 'outer';
        html = `step ${st.k} of ${st.steps.length} (${st.steps.length} ${mults ? 'multiplications' : 'additions'}) · einsum '<b>${op.einsum}</b>'`;
      } else {
        const t = this.nnTotals || { macs: 0, params: 0 };
        const f = (v) => (v >= 1e9 ? `${(v / 1e9).toFixed(2)} billion` : v >= 1e6 ? `${(v / 1e6).toFixed(2)} million` : v.toLocaleString());
        html = `${f(t.params)} learned parameters · <b>${f(t.macs)}</b> multiply-adds per batch, every one a line inside a loop`;
      }
      if (html !== this.lastHud) { this.hud.innerHTML = html; this.lastHud = html; }
    },

    pointer(type, x, y) {
      if (this.o.mode !== 'memory' || !this.hits) return;
      if (type === 'leave') { this.hover = null; return; }
      const h = this.hits.find((q) => x >= q.x && x < q.x + q.s && y >= q.y && y < q.y + q.s);
      if (!h) { this.hover = null; return; }
      if (h.mem != null) { // memory box: find the index that maps there
        const idx = indices(this.view.shape).find((ix) => this.offset(ix) === h.mem);
        this.hover = { off: h.mem, idx };
      } else this.hover = { off: h.off, idx: h.idx };
    },
  };

  sim._internals = { OPS, indices, contiguousStrides, prod };
  (window.Sims = window.Sims || {}).tensor = sim;
})();
