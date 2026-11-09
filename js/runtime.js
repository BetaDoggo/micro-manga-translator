// ONNX Runtime Web (WebGPU) loading + model fetching with progress and caching.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.logLevel = 'error';

export { ort };

// Model files come from one Hugging Face repo, pinned to a commit so cached
// copies never go stale. ?models=local uses ./models/ instead (filled by
// tools/setup_models.py) for development; local files are never cached.
export const MODEL_REPO = 'HDiffusion/micro-manga-translator';
export const MODEL_REV = 'main';
export const LOCAL_MODELS = new URLSearchParams(location.search).get('models') === 'local';
export const MODELS_URL = LOCAL_MODELS
  ? new URL('../models/', import.meta.url).href
  : `https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/`;

/** Every file the page loads, with its size in bytes. */
export const MODEL_FILES = {
  'ppocr-manga/manga_det.onnx': 1_816_958,
  'ppocr-manga/manga_rec.onnx': 21_334_702,
  'ppocr-manga/ppocrv6_dict.txt': 76_327,
  'migan/migan_pipeline.onnx': 28_079_181,
  'sugoi-v4/tokenizer.json': 2_429_674,
  'sugoi-v4/encoder.onnx': 218_825_183,
  'sugoi-v4/decoder.onnx': 400_275_832,
};

const CACHE = 'micro-manga-translator-models';
const urlOf = path => new URL(path, MODELS_URL).href;

async function openCache() {
  if (LOCAL_MODELS) return null;
  try { return await caches.open(CACHE); } catch { return null; }   // no Cache API (e.g. private mode)
}

/** Bytes of the model files that still need downloading (0 = all cached). */
export async function missingBytes() {
  if (LOCAL_MODELS) return 0;
  const cache = await openCache();
  let missing = 0;
  for (const [path, size] of Object.entries(MODEL_FILES)) {
    let hit = null;
    try { hit = cache && await cache.match(urlOf(path)); } catch { /* treat as missing */ }
    if (!hit) missing += size;
  }
  return missing;
}

/** Drop cached files that aren't part of the current model set (older revisions). */
export async function pruneCache() {
  const cache = await openCache();
  if (!cache) return;
  try {
    const keep = new Set(Object.keys(MODEL_FILES).map(urlOf));
    for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req);
  } catch { /* best effort */ }
}

/** Delete every downloaded model file. */
export async function clearCache() {
  try { await caches.delete(CACHE); } catch { /* nothing to clear */ }
}

/** Fetch a model file into a Uint8Array, reporting (loadedBytes, totalBytes);
 * served from the Cache API when present, stored there after downloading. */
export async function fetchModel(path, onProgress = () => {}) {
  const url = urlOf(path);
  const cache = await openCache();
  let cached = null;
  try { cached = cache && await cache.match(url); } catch { /* fall through to network */ }
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    onProgress(bytes.length, bytes.length);
    return bytes;
  }
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${path}: HTTP ${res.status}` +
      (LOCAL_MODELS ? ' (run python tools/setup_models.py)' : ''));
  }
  const total = +res.headers.get('content-length') || MODEL_FILES[path] || 0;
  let out;
  if (!res.body) {
    out = new Uint8Array(await res.arrayBuffer());
  } else {
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress(loaded, total);
    }
    out = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
  }
  onProgress(out.length, out.length);
  if (cache) {
    try {
      await cache.put(url, new Response(out, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } catch (e) { console.warn(`not cached (${path}):`, e); }   // quota: still usable this session
  }
  return out;
}

export async function fetchText(path) {
  return new TextDecoder().decode(await fetchModel(path));
}

export async function fetchJSON(path) {
  return JSON.parse(await fetchText(path));
}

export async function webgpuInfo() {
  if (!navigator.gpu) return { ok: false, reason: 'navigator.gpu is missing (WebGPU not supported/enabled)' };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ok: false, reason: 'no WebGPU adapter available' };
  const info = adapter.info || {};
  return { ok: true, vendor: info.vendor || '?', arch: info.architecture || '?',
           f16: adapter.features.has('shader-f16') };
}

/** InferenceSession on WebGPU (ORT falls back to WASM for unsupported ops). */
export async function createSession(bytes, opts = {}) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,   // hide the "shape ops assigned to CPU" notice
    ...opts,
  });
}
