// MI-GAN inpainting (Picsart MI-GAN "pipeline" ONNX: uint8 image [1,3,H,W] +
// uint8 mask [1,1,H,W] with 0 = erase / 255 = keep -> uint8 image; only the
// erased pixels change). It crops to the mask and works at 512 px internally,
// so each text block is inpainted in its own window instead of one page-wide
// call (which would downscale the whole page to 512).
import { ort, fetchModel, createSession } from './runtime.js';

export class MiGan {
  static async load(onProgress = () => {}) {
    const self = new MiGan();
    const bytes = await fetchModel('migan/migan_pipeline.onnx', (l, t) => onProgress('inpainter', l, t));
    self.session = await createSession(bytes);
    return self;
  }

  /** Erase `mask` (Uint8Array W*H, 1 = text) from RGBA ImageData `img`.
   * windows: [[x0,y0,x1,y1]] regions to process (each grown by context).
   * Returns a new ImageData. */
  async inpaint(img, mask, windows) {
    const { width: W, height: H } = img;
    const out = new ImageData(new Uint8ClampedArray(img.data), W, H);
    for (const [wx0, wy0, wx1, wy1] of mergeWindows(windows, W, H)) {
      const w = wx1 - wx0, h = wy1 - wy0;
      let any = false;
      const n = w * h;
      const im = new Uint8Array(3 * n), mk = new Uint8Array(n);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const gi = (wy0 + y) * W + wx0 + x, p = gi * 4, i = y * w + x;
          im[i] = out.data[p]; im[n + i] = out.data[p + 1]; im[2 * n + i] = out.data[p + 2];
          mk[i] = mask[gi] ? 0 : 255;
          if (mask[gi]) any = true;
        }
      }
      if (!any) continue;
      const res = await this.session.run({
        image: new ort.Tensor('uint8', im, [1, 3, h, w]),
        mask: new ort.Tensor('uint8', mk, [1, 1, h, w]),
      });
      const r = res[this.session.outputNames[0]].data;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (mk[i]) continue;
          const p = ((wy0 + y) * W + wx0 + x) * 4;
          out.data[p] = r[i]; out.data[p + 1] = r[n + i]; out.data[p + 2] = r[2 * n + i];
        }
      }
    }
    return out;
  }
}

/** Grow each window by context and merge overlapping ones. */
function mergeWindows(windows, W, H) {
  let ws = windows.map(([x0, y0, x1, y1]) => {
    const m = Math.max(48, 0.5 * Math.max(x1 - x0, y1 - y0));
    return [Math.max(0, Math.floor(x0 - m)), Math.max(0, Math.floor(y0 - m)),
            Math.min(W, Math.ceil(x1 + m)), Math.min(H, Math.ceil(y1 + m))];
  });
  for (let changed = true; changed;) {
    changed = false;
    outer: for (let i = 0; i < ws.length; i++) {
      for (let j = i + 1; j < ws.length; j++) {
        const a = ws[i], b = ws[j];
        if (a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]) {
          ws[i] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
          ws.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return ws;
}

/** Text mask from the detected line boxes (all of them, furigana included).
 * Boxes on a mostly white background (bubbles) mask only their dark ink,
 * dilated by `grow` px (anti-aliased edges included), so white bubble
 * interiors and outlines touching the box stay untouched; boxes over art /
 * dark backgrounds (white or outlined text) are masked whole. */
export function textMask(lines, grayImg, W, H, { pad = 3, ink = 215, grow = 3, whiteFrac = 0.55 } = {}) {
  const m = new Uint8Array(W * H);
  for (const l of lines) {
    const x0 = Math.max(0, l.x0 - pad), x1 = Math.min(W, l.x1 + pad);
    const y0 = Math.max(0, l.y0 - pad), y1 = Math.min(H, l.y1 + pad);
    let bright = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (grayImg[y * W + x] > 225) bright++;
    const area = Math.max(1, (x1 - x0) * (y1 - y0));
    if (bright / area < whiteFrac) {
      for (let y = y0; y < y1; y++) m.fill(1, y * W + x0, y * W + x1);
      continue;
    }
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        if (grayImg[y * W + x] >= ink) continue;
        const ya = Math.max(y0, y - grow), yb = Math.min(y1, y + grow + 1);
        const xa = Math.max(x0, x - grow), xb = Math.min(x1, x + grow + 1);
        for (let yy = ya; yy < yb; yy++) m.fill(1, yy * W + xa, yy * W + xb);
      }
    }
  }
  return m;
}
