/* Shell for sim.html: tabs, canvas sizing, pointer routing, animation loop. */
(() => {
  'use strict';
  const canvas = document.getElementById('sim'), ctx = canvas.getContext('2d');
  const stage = document.querySelector('.stage');
  const panel = document.getElementById('simPanel'), hud = document.getElementById('hud');
  const hint = document.getElementById('stageHint');
  const tabs = [...document.querySelectorAll('.tab')];
  const size = { w: 0, h: 0 };
  let current = null, gridTimer = null;

  function resizeCanvas() {
    const r = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    size.w = r.width; size.h = r.height;
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  // Re-gridding restarts a simulation, so wait until resizing settles.
  new ResizeObserver(() => {
    resizeCanvas();
    clearTimeout(gridTimer);
    gridTimer = setTimeout(() => current && current.resize(size.w, size.h), 200);
  }).observe(stage);

  function select(id) {
    if (!window.Sims[id]) id = 'fluid';
    if (current && current.id === id) return;
    if (current) current.unmount();
    current = window.Sims[id];
    tabs.forEach((t) => t.classList.toggle('active', t.dataset.sim === id));
    hud.innerHTML = '';
    if (size.w) current.resize(size.w, size.h);
    current.mount(panel, hud);
    hint.textContent = current.hint;
    history.replaceState(null, '', '#' + id);
  }
  tabs.forEach((t) => t.addEventListener('click', () => select(t.dataset.sim)));

  const local = (ev) => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  canvas.addEventListener('pointerdown', (ev) => { canvas.setPointerCapture(ev.pointerId); current?.pointer('down', ...local(ev)); });
  canvas.addEventListener('pointermove', (ev) => current?.pointer('move', ...local(ev)));
  canvas.addEventListener('pointerup', (ev) => current?.pointer('up', ...local(ev)));
  canvas.addEventListener('pointercancel', (ev) => current?.pointer('up', ...local(ev)));
  canvas.addEventListener('pointerleave', (ev) => current?.pointer('leave', ...local(ev)));

  function loop() {
    if (current && size.w) {
      ctx.fillStyle = '#0b0e14';
      ctx.fillRect(0, 0, size.w, size.h);
      current.frame(ctx, size.w, size.h);
    }
    requestAnimationFrame(loop);
  }

  resizeCanvas();
  select(location.hash.slice(1) || 'fluid');
  requestAnimationFrame(loop);
})();
