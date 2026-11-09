// Small image helpers on top of canvas/ImageData (no OpenCV).

export function canvas(w, h) {
  const c = new OffscreenCanvas(Math.max(1, Math.round(w)), Math.max(1, Math.round(h)));
  return c;
}

export function ctx2d(c, opts = {}) {
  return c.getContext('2d', { willReadFrequently: true, ...opts });
}

/** Decode a Blob/File/URL into an ImageBitmap + its RGBA ImageData. */
export async function loadImage(src) {
  const blob = typeof src === 'string' ? await (await fetch(src)).blob() : src;
  const bmp = await createImageBitmap(blob);
  const c = canvas(bmp.width, bmp.height);
  const g = ctx2d(c);
  g.drawImage(bmp, 0, 0);
  return { bitmap: bmp, width: bmp.width, height: bmp.height,
           data: g.getImageData(0, 0, bmp.width, bmp.height) };
}

export function imageDataToCanvas(img) {
  const c = canvas(img.width, img.height);
  ctx2d(c).putImageData(img, 0, 0);
  return c;
}

/** Luma (BT.601, like cv2 BGR2GRAY) of RGBA ImageData -> Uint8Array. */
export function gray(img) {
  const { data, width, height } = img;
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (data[p] * 299 + data[p + 1] * 587 + data[p + 2] * 114 + 500) / 1000;
  }
  return out;
}

/** Summed-area table of a 0/1 Uint8Array mask: rect sums in O(1). */
export class Integral {
  constructor(mask, w, h) {
    this.w = w; this.h = h;
    const W = w + 1;
    this.s = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
      let row = 0;
      for (let x = 0; x < w; x++) {
        row += mask[y * w + x];
        this.s[(y + 1) * W + x + 1] = this.s[y * W + x + 1] + row;
      }
    }
  }
  /** sum over [x0,x1) x [y0,y1), clipped */
  sum(x0, y0, x1, y1) {
    x0 = Math.max(0, Math.min(this.w, x0 | 0)); x1 = Math.max(0, Math.min(this.w, x1 | 0));
    y0 = Math.max(0, Math.min(this.h, y0 | 0)); y1 = Math.max(0, Math.min(this.h, y1 | 0));
    if (x1 <= x0 || y1 <= y0) return 0;
    const W = this.w + 1, s = this.s;
    return s[y1 * W + x1] - s[y0 * W + x1] - s[y1 * W + x0] + s[y0 * W + x0];
  }
  mean(x0, y0, x1, y1) {
    const a = (Math.min(this.w, x1 | 0) - Math.max(0, x0 | 0)) * (Math.min(this.h, y1 | 0) - Math.max(0, y0 | 0));
    return a > 0 ? this.sum(x0, y0, x1, y1) / a : 0;
  }
}

/** Connected components of a 0/1 mask (w x h). Returns { labels (Int32Array,
 * 0 = background), count, stats: [{area, x0, y0, x1, y1, sx, sy}] (index = label) }. */
export function connectedComponents(mask, w, h, eight = true) {
  const labels = new Int32Array(w * h);
  const stats = [null];
  const stack = new Int32Array(w * h);
  let count = 0;
  for (let start = 0; start < w * h; start++) {
    if (!mask[start] || labels[start]) continue;
    const lab = ++count;
    const st = { area: 0, x0: w, y0: h, x1: -1, y1: -1, sx: 0, sy: 0 };
    let sp = 0;
    stack[sp++] = start;
    labels[start] = lab;
    while (sp) {
      const p = stack[--sp];
      const x = p % w, y = (p - x) / w;
      st.area++; st.sx += x; st.sy += y;
      if (x < st.x0) st.x0 = x; if (x > st.x1) st.x1 = x;
      if (y < st.y0) st.y0 = y; if (y > st.y1) st.y1 = y;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if ((!dx && !dy) || (!eight && dx && dy)) continue;
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const q = yy * w + xx;
          if (mask[q] && !labels[q]) { labels[q] = lab; stack[sp++] = q; }
        }
      }
    }
    stats.push(st);
  }
  return { labels, count, stats };
}

/** Fill the holes of a 0/1 mask (pixels not reachable from the border through
 * zeros), like drawing its external contours filled. */
export function fillHoles(mask, w, h) {
  const outside = new Uint8Array(w * h);
  const stack = [];
  const push = p => { if (!mask[p] && !outside[p]) { outside[p] = 1; stack.push(p); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  while (stack.length) {
    const p = stack.pop();
    const x = p % w, y = (p - x) / w;
    if (x > 0) push(p - 1);
    if (x < w - 1) push(p + 1);
    if (y > 0) push(p - w);
    if (y < h - 1) push(p + w);
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = outside[i] ? 0 : 1;
  return out;
}
