// PP-OCRv6_manga (Kellenok/PP-OCRv6_manga v0.3) in the browser: DB line
// detector + CTC recognizer, preprocessing/post-processing following the
// model's reference script (examples/onnx_openvino.py).
import { ort, fetchModel, fetchText, createSession } from './runtime.js';
import { canvas, ctx2d, connectedComponents } from './image.js';

const DET = { thresh: 0.15, boxThresh: 0.25, unclip: 1.5, longSide: 960 };
const REC_H = 48;

export class PPOcr {
  static async load(onProgress = () => {}) {
    const self = new PPOcr();
    const det = await fetchModel('ppocr-manga/manga_det.onnx', (l, t) => onProgress('detector', l, t));
    self.det = await createSession(det);
    const rec = await fetchModel('ppocr-manga/manga_rec.onnx', (l, t) => onProgress('recognizer', l, t));
    self.rec = await createSession(rec);
    const dict = await fetchText('ppocr-manga/ppocrv6_dict.txt');
    self.vocab = ['<blank>', ...dict.replace(/\r/g, '').split('\n').filter((l, i, a) => i < a.length - 1 || l), ' '];
    return self;
  }

  /** Detect text lines. Returns [{x0,y0,x1,y1 (unclipped box), core:{x0,y0,x1,y1}, score}]
   * in image pixels. Boxes are axis-aligned: DB components are unclipped by
   * area*ratio/perimeter like the reference's polygon offset. */
  async detect(image) {
    const { width: W, height: H, bitmap } = image;
    let ratio = Math.max(H, W) / Math.min(H, W) > 2
      ? Math.min(1, Math.sqrt(0.75 * DET.longSide * DET.longSide / (H * W)))
      : DET.longSide / Math.max(H, W);
    const rh = Math.max(Math.round(H * ratio / 32) * 32, 32);
    const rw = Math.max(Math.round(W * ratio / 32) * 32, 32);
    const c = canvas(rw, rh);
    const g = ctx2d(c);
    g.imageSmoothingQuality = 'high';
    g.drawImage(bitmap, 0, 0, rw, rh);
    const px = g.getImageData(0, 0, rw, rh).data;
    const n = rw * rh;
    const x = new Float32Array(3 * n);
    // reference feeds cv2 BGR with ImageNet mean/std listed in BGR-agnostic order
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      x[i] = (px[p + 2] / 255 - mean[0]) / std[0];          // B
      x[n + i] = (px[p + 1] / 255 - mean[1]) / std[1];      // G
      x[2 * n + i] = (px[p] / 255 - mean[2]) / std[2];      // R
    }
    const out = await this.det.run({ [this.det.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, rh, rw]) });
    const pred = out[this.det.outputNames[0]].data;   // [1,1,rh,rw]
    const bin = new Uint8Array(n);
    for (let i = 0; i < n; i++) bin[i] = pred[i] > DET.thresh ? 1 : 0;
    const { labels, stats } = connectedComponents(bin, rw, rh, true);
    const sums = new Float64Array(stats.length);
    for (let i = 0; i < n; i++) if (labels[i]) sums[labels[i]] += pred[i];
    const sx = W / rw, sy = H / rh;
    const lines = [];
    for (let l = 1; l < stats.length; l++) {
      const s = stats[l];
      if (s.area < 4 || sums[l] / s.area < DET.boxThresh) continue;
      const bw = s.x1 - s.x0 + 1, bh = s.y1 - s.y0 + 1;
      if (Math.min(bw, bh) < 2) continue;
      const d = (s.area * DET.unclip) / (2 * (bw + bh));
      const box = {
        x0: Math.max(0, Math.round((s.x0 - d) * sx)), y0: Math.max(0, Math.round((s.y0 - d) * sy)),
        x1: Math.min(W, Math.round((s.x1 + 1 + d) * sx)), y1: Math.min(H, Math.round((s.y1 + 1 + d) * sy)),
      };
      if (Math.min(box.x1 - box.x0, box.y1 - box.y0) < 3) continue;
      box.core = { x0: s.x0 * sx, y0: s.y0 * sy, x1: (s.x1 + 1) * sx, y1: (s.y1 + 1) * sy };
      box.score = sums[l] / s.area;
      lines.push(box);
    }
    return lines;
  }

  /** Recognize line crops (tall crops = vertical text, rotated 90° CCW).
   * Batched: crops resized to height 48, right-padded with 0 (= mid-gray
   * after normalization) to the batch's widest crop, as PaddleOCR does. */
  async recognize(image, boxes, batch = 16) {
    const crops = boxes.map(b => {
      const w = b.x1 - b.x0, h = b.y1 - b.y0;
      const vertical = h > w;
      const cw = vertical ? h : w, ch = vertical ? w : h;    // after rotation
      const tw = Math.max(16, Math.round(REC_H * cw / Math.max(1, ch)));
      return { b, vertical, tw };
    });
    const order = crops.map((_, i) => i).sort((a, c) => crops[a].tw - crops[c].tw);
    const texts = new Array(boxes.length).fill('');
    for (let s = 0; s < order.length; s += batch) {
      const idx = order.slice(s, s + batch);
      const maxW = Math.max(...idx.map(i => crops[i].tw));
      const n = idx.length;
      const x = new Float32Array(n * 3 * REC_H * maxW);
      const plane = REC_H * maxW;
      idx.forEach((ci, k) => {
        const { b, vertical, tw } = crops[ci];
        const c = canvas(tw, REC_H);
        const g = ctx2d(c);
        g.imageSmoothingQuality = 'high';
        if (vertical) {
          // rotate 90° CCW: the crop's top edge becomes its left edge
          g.translate(0, REC_H);
          g.rotate(-Math.PI / 2);
          g.drawImage(image.bitmap, b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0, 0, 0, REC_H, tw);
        } else {
          g.drawImage(image.bitmap, b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0, 0, 0, tw, REC_H);
        }
        const px = g.getImageData(0, 0, tw, REC_H).data;
        const base = k * 3 * plane;
        for (let yy = 0; yy < REC_H; yy++) {
          for (let xx = 0; xx < tw; xx++) {
            const p = (yy * tw + xx) * 4, o = yy * maxW + xx;
            x[base + o] = px[p + 2] / 127.5 - 1;               // B
            x[base + plane + o] = px[p + 1] / 127.5 - 1;       // G
            x[base + 2 * plane + o] = px[p] / 127.5 - 1;       // R
          }
        }
      });
      const out = await this.rec.run({ [this.rec.inputNames[0]]: new ort.Tensor('float32', x, [n, 3, REC_H, maxW]) });
      const t = out[this.rec.outputNames[0]];
      const [, T, V] = t.dims;
      const data = t.data;
      idx.forEach((ci, k) => {
        let prev = -1, s = '';
        // only the frames covering this crop's own width (padding frames can emit junk)
        const frames = Math.min(T, Math.ceil(T * crops[ci].tw / maxW) + 1);
        for (let f = 0; f < frames; f++) {
          const off = (k * T + f) * V;
          let best = 0, bv = data[off];
          for (let v = 1; v < V; v++) if (data[off + v] > bv) { bv = data[off + v]; best = v; }
          if (best !== 0 && best !== prev) s += this.vocab[best] ?? '';
          prev = best;
        }
        texts[ci] = s.trim();
      });
    }
    return texts;
  }
}
