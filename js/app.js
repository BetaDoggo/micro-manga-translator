import { webgpuInfo, missingBytes, pruneCache, clearCache, LOCAL_MODELS, MODEL_FILES } from './runtime.js';
import { PagePipeline } from './pipeline.js';
import { loadImage, imageDataToCanvas } from './image.js';

const $ = s => document.querySelector(s);
const params = new URLSearchParams(location.search);
let pipeline = null, image = null, result = null, view = 'final', imageName = 'page';

function showError(e) {
  console.error(e);
  const el = $('#error');
  el.textContent = String(e && e.message || e);
  el.style.display = 'block';
}

function setStage(name, ms, state = 'done') {
  const li = document.querySelector(`#steps li[data-s="${name}"]`);
  if (!li) return;
  li.className = state;
  li.lastElementChild.textContent = ms == null ? '' : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function updateRun() { $('#run').disabled = !(pipeline && image); }

// Sugoi beam size: 1..10, default 5; ?beam= overrides, the last choice is remembered
const BEAM_KEY = 'mt.beam';
function beamSize() {
  const v = Math.round(+$('#beam').value);
  return Number.isFinite(v) ? Math.min(10, Math.max(1, v)) : 5;
}
{
  let initial = params.get('beam');
  if (initial == null) { try { initial = localStorage.getItem(BEAM_KEY); } catch { /* storage blocked */ } }
  if (initial != null && Number.isFinite(+initial)) $('#beam').value = initial;
  $('#beam').value = beamSize();
  $('#beam').addEventListener('change', () => {
    $('#beam').value = beamSize();
    try { localStorage.setItem(BEAM_KEY, String(beamSize())); } catch { /* storage blocked */ }
  });
}

async function init() {
  const gpu = await webgpuInfo();
  const chip = $('#gpu');
  if (!gpu.ok) {
    chip.textContent = 'WebGPU unavailable';
    chip.className = 'chip err';
    showError(`WebGPU is required: ${gpu.reason}. Use a recent Chrome/Edge.`);
    return;
  }
  chip.textContent = `WebGPU · ${gpu.vendor}${gpu.f16 ? ' · fp16' : ''}`;
  chip.className = 'chip ok';

  // Nothing is downloaded until the user asks; already-cached models load straight away.
  await pruneCache();
  const missing = await missingBytes();
  if (missing === 0) return loadModels();
  const btn = $('#fetchmodels');
  btn.textContent = `Download models (${mb(missing)})`;
  btn.hidden = false;
  $('#loadmsg').textContent = 'models not downloaded yet';
  window.__mt = { status: 'idle', missingBytes: missing };
}

const mb = bytes => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(2)} GB` : `${Math.round(bytes / 2 ** 20)} MB`;

const ONNX_BYTES = Object.entries(MODEL_FILES)   // the progress-reported files
  .filter(([p]) => p.endsWith('.onnx')).reduce((a, [, n]) => a + n, 0);

async function loadModels() {
  $('#fetchmodels').hidden = true;
  $('#loadmsg').textContent = 'loading models…';
  const totals = {}, loaded = {};
  const t0 = performance.now();
  try {
    pipeline = await PagePipeline.load((model, l, t) => {
      loaded[model] = l; totals[model] = t || l;
      const L = Object.values(loaded).reduce((a, b) => a + b, 0);
      const T = Math.max(ONNX_BYTES, Object.values(totals).reduce((a, b) => a + b, 0));
      $('#loadbar').style.width = `${Math.min(100, 100 * L / Math.max(T, 1))}%`;
      $('#loadmsg').textContent = `loading ${model}… ${mb(L)} of ${mb(T)}`;
    });
  } catch (e) {
    $('#loadmsg').textContent = 'model loading failed';
    $('#fetchmodels').textContent = 'Retry download';
    $('#fetchmodels').hidden = false;
    showError(e);
    window.__mt = { status: 'error', error: String(e && e.stack || e) };
    return;
  }
  $('#loadbar').style.width = '100%';
  $('#loadmsg').textContent = `models ready in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  $('#clearmodels').hidden = LOCAL_MODELS;
  updateRun();
  window.__mt = { status: 'ready' };
  if (params.get('src')) {
    await setImage(params.get('src'), params.get('src').split('/').pop().replace(/\.\w+$/, ''));
    if (params.has('run')) await run();
  }
}

async function setImage(src, name) {
  try {
    image = await loadImage(src);
    imageName = name || 'page';
    result = null;
    for (const li of document.querySelectorAll('#steps li')) setStage(li.dataset.s, null, '');
    $('#blocks').hidden = true;
    $('#download').disabled = true;
    $('#drop').querySelector('strong').textContent = `${imageName} · ${image.width}×${image.height}`;
    view = 'original';
    show();
    updateRun();
  } catch (e) { showError(e); }
}

function show() {
  document.querySelectorAll('#tabs button[data-v]').forEach(b => b.classList.toggle('on', b.dataset.v === view));
  const holder = $('#view');
  let src = null;
  if (view === 'original' && image) src = imageDataToCanvas(image.data);
  else if (result) src = { final: result.final, clean: imageDataToCanvas(result.clean), detections: result.detections }[view];
  holder.innerHTML = '';
  if (!src) { holder.innerHTML = '<div id="placeholder">The translated page will appear here.</div>'; return; }
  const c = document.createElement('canvas');
  c.width = src.width; c.height = src.height;
  c.getContext('2d').drawImage(src, 0, 0);
  holder.appendChild(c);
}

function fillTable(blocks) {
  const tb = $('#blocks tbody');
  tb.innerHTML = '';
  for (const b of blocks) {
    const tr = document.createElement('tr');
    const td = (cls, text) => { const e = document.createElement('td'); e.className = cls; e.textContent = text; tr.appendChild(e); };
    td('id', b.id);
    td('ja', b.text || '');
    if (b.skip) td('skip', 'skipped (1-char fragment)');
    else td('en', b.translation || '');
    tb.appendChild(tr);
  }
  $('#blocks').hidden = !blocks.length;
}

async function run() {
  if (!pipeline || !image) return;
  $('#run').disabled = true;
  $('#error').style.display = 'none';
  const order = ['detect', 'ocr', 'translate', 'inpaint', 'render'];
  for (const s of [...order, 'total']) setStage(s, null, '');
  setStage('detect', null, 'active');
  const t0 = performance.now();
  try {
    result = await pipeline.run(image, {
      beam: beamSize(),
      onStage: (name, ms) => {
        setStage(name, ms);
        const next = order[order.indexOf(name) + 1];
        if (next) setStage(next, null, 'active');
      },
    });
    const total = performance.now() - t0;
    setStage('total', total);
    fillTable(result.blocks);
    view = 'final';
    show();
    $('#download').disabled = false;
    window.__mtResult = result;   // for debugging from the console
    window.__mt = {
      status: 'done', totalMs: total, timings: result.timings, size: [image.width, image.height],
      lines: result.lines.length, furigana: result.lines.filter(l => l.furigana).length,
      maskPixels: result.mask.reduce((a, b) => a + b, 0),
      blocks: result.blocks.map(b => ({ id: b.id, xyxy: b.xyxy.map(Math.round), vertical: b.vertical,
        lines: b.lines.length, text: b.text, skip: b.skip, translation: b.translation,
        render: b.render && { fontSize: b.render.fontSize, lines: b.render.lines, rotated: !!b.render.rotated,
                              fallback: !!b.render.fallback, shape: b.render.shape && b.render.shape.kind } })),
    };
  } catch (e) {
    showError(e);
    window.__mt = { status: 'error', error: String(e && e.stack || e) };
  } finally {
    updateRun();
  }
}

$('#drop').addEventListener('click', () => $('#file').click());
$('#drop').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') $('#file').click(); });
$('#file').addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) setImage(f, f.name.replace(/\.\w+$/, ''));
});
for (const ev of ['dragenter', 'dragover']) $('#drop').addEventListener(ev, e => { e.preventDefault(); $('#drop').classList.add('over'); });
for (const ev of ['dragleave', 'drop']) $('#drop').addEventListener(ev, e => { e.preventDefault(); $('#drop').classList.remove('over'); });
$('#drop').addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) setImage(f, f.name.replace(/\.\w+$/, ''));
});
$('#run').addEventListener('click', run);
$('#fetchmodels').addEventListener('click', async () => {
  $('#error').style.display = 'none';
  try { await navigator.storage?.persist?.(); } catch { /* the browser may still evict under pressure */ }
  loadModels().catch(showError);
});
$('#clearmodels').addEventListener('click', async () => {
  await clearCache();
  $('#clearmodels').hidden = true;
  $('#loadmsg').textContent = 'downloaded models deleted (still loaded until you leave this page)';
});
$('#tabs').addEventListener('click', e => {
  const v = e.target.dataset && e.target.dataset.v;
  if (v) { view = v; show(); }
});
$('#download').addEventListener('click', async () => {
  if (!result) return;
  const blob = await result.final.convertToBlob({ type: 'image/png' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${imageName}_en.png`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

init().catch(showError);
