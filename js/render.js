// Stage 5: render translations into the bubbles. Port of the manga-translate
// Python renderer (render.py) to canvas: per-line width limits measured from
// the white runs of the inpainted page, layouts validated against the
// paintable (white) mask and other blocks' text, rotated labels for tall boxes,
// stroked over-art fallback so no translation is dropped.
import { canvas, ctx2d, connectedComponents, fillHoles, gray, Integral } from './image.js';

const FONT_URL = new URL('../fonts/ComicNeue-Bold.ttf', import.meta.url).href;
let fontReady = null;

/** Comic Sans MS Bold when the system has it (like the Python default on
 * Windows), else the bundled Comic Neue Bold. */
export function loadFont() {
  fontReady ??= (async () => {
    const face = new FontFace('ComicNeueBundled', `url(${FONT_URL})`, { weight: '700' });
    await face.load();
    (self.fonts || document.fonts).add(face);
  })();
  return fontReady;
}

export class Renderer {
  constructor({ minFont = 9, maxFont = 64, lineHeight = 1.22, textPad = 0.92 } = {}) {
    this.minFont = minFont; this.maxFont = maxFont;
    this.lineH = lineHeight; this.textPad = textPad;
    this.measureCanvas = ctx2d(canvas(8, 8));
    this.cache = new Map();
  }

  fontCss(size) { return `bold ${size}px "Comic Sans MS", ComicNeueBundled, sans-serif`; }

  /** {w (advance), l, r, a, d}: ink extents relative to a center/middle-anchored draw */
  measure(text, size) {
    const key = size + '\u0000' + text;
    let m = this.cache.get(key);
    if (!m) {
      const g = this.measureCanvas;
      g.font = this.fontCss(size);
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      const t = g.measureText(text);
      m = { w: t.width, l: t.actualBoundingBoxLeft, r: t.actualBoundingBoxRight,
            a: t.actualBoundingBoxAscent, d: t.actualBoundingBoxDescent };
      this.cache.set(key, m);
    }
    return m;
  }

  // ------------------------------------------------------------ bubble shapes
  findBubbleShape(origGray, W, H, xyxy) {
    const [x1, y1, x2, y2] = xyxy.map(Math.round);
    const pad = Math.round(Math.max(x2 - x1, y2 - y1) * 0.4) + 10;
    const rx1 = Math.max(0, x1 - pad), ry1 = Math.max(0, y1 - pad);
    const rx2 = Math.min(W, x2 + pad), ry2 = Math.min(H, y2 + pad);
    const rw = rx2 - rx1, rh = ry2 - ry1;
    if (rw <= 0 || rh <= 0) return null;
    const white = new Uint8Array(rw * rh);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++)
      white[y * rw + x] = origGray[(ry1 + y) * W + rx1 + x] > 240 ? 1 : 0;
    const { labels, stats } = connectedComponents(white, rw, rh, true);
    const blkArea = Math.max(1, (x2 - x1) * (y2 - y1));
    const qualifies = i => {
      const s = stats[i], bw = s.x1 - s.x0 + 1, bh = s.y1 - s.y0 + 1;
      if (s.area < 0.5 * blkArea) return false;
      if (bw > Math.max(3 * (x2 - x1), 0.9 * (y2 - y1) + 120)) return false;
      if (bh > Math.max(3 * (y2 - y1), 0.9 * (x2 - x1) + 120)) return false;
      return true;
    };
    const votes = new Map();
    for (const fy of [0.3, 0.5, 0.7]) for (const fx of [0.3, 0.5, 0.7]) {
      const px = Math.floor(x1 + fx * (x2 - x1)) - rx1, py = Math.floor(y1 + fy * (y2 - y1)) - ry1;
      if (px < 0 || py < 0 || px >= rw || py >= rh) continue;
      const lab = labels[py * rw + px];
      if (lab > 0 && qualifies(lab)) votes.set(lab, (votes.get(lab) || 0) + 1);
    }
    if (!votes.size) return null;
    const best = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const st = stats[best];
    if (st.x0 > x1 - rx1 + 4 || st.y0 > y1 - ry1 + 4 || st.x1 + 1 < x2 - rx1 - 4 || st.y1 + 1 < y2 - ry1 - 4) return null;
    let comp = new Uint8Array(rw * rh);
    let compSum = 0;
    for (let i = 0; i < comp.length; i++) if (labels[i] === best) { comp[i] = 1; compSum++; }
    // restrict to a local window around the block (stacked bubbles keep their own area)
    const lpad = Math.round(0.6 * Math.max(x2 - x1, y2 - y1)) + 20;
    const ly1 = Math.max(0, y1 - ry1 - lpad), ly2 = Math.min(rh, y2 - ry1 + lpad);
    const lx1 = Math.max(0, x1 - rx1 - lpad), lx2 = Math.min(rw, x2 - rx1 + lpad);
    const local = new Uint8Array(rw * rh);
    let localSum = 0;
    for (let y = ly1; y < ly2; y++) for (let x = lx1; x < lx2; x++) {
      const i = y * rw + x;
      if (comp[i]) { local[i] = 1; localSum++; }
    }
    if (localSum >= 0.25 * compSum) { comp = local; compSum = localSum; }
    const filled = fillHoles(comp, rw, rh);
    let bx0 = rw, by0 = rh, bx1 = -1, by1 = -1, m00 = 0, m10 = 0, m01 = 0;
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      if (!filled[y * rw + x]) continue;
      m00++; m10 += x; m01 += y;
      if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y;
    }
    if (!m00) return null;
    const bw = bx1 - bx0 + 1, bh = by1 - by0 + 1;
    const fill = compSum / Math.max(1e-6, Math.PI / 4 * bw * bh);
    return { kind: fill > 0.85 ? 'ellipse' : 'rect', cx: m10 / m00 + rx1, cy: m01 / m00 + ry1,
             bbox: [bx0 + rx1, by0 + ry1, bw, bh] };
  }

  // ------------------------------------------------------------ fitting
  layoutLines(words, size, limits) {
    if (limits.some(l => l <= 0)) return null;
    const lines = [];
    let cur = '';
    for (const w of words) {
      const trial = (cur + ' ' + w).trim();
      const limit = limits[lines.length];
      if (this.measure(trial, size).w <= limit) cur = trial;
      else if (!cur) return null;
      else {
        lines.push(cur);
        cur = w;
        if (lines.length >= limits.length) return null;
      }
    }
    if (cur) lines.push(cur);
    return lines;
  }

  rowRun(ok, W, H, yc, x0, x1, cx) {
    yc = Math.floor(yc);
    if (yc < 0 || yc >= H) return 0;
    x0 = Math.max(0, Math.floor(x0)); x1 = Math.min(W, Math.floor(x1));
    if (x1 <= x0) return 0;
    const row = yc * W;
    let c = Math.min(Math.max(Math.floor(cx), x0), x1 - 1);
    if (!ok[row + c]) {
      let bestD = Infinity, j = -1;
      for (let x = x0; x < x1; x++) if (ok[row + x] && Math.abs(x - c) < bestD) { bestD = Math.abs(x - c); j = x; }
      if (j < 0) return 0;
      c = j;
    }
    let l = c, r = c;
    while (l > x0 && ok[row + l - 1]) l--;
    while (r < x1 - 1 && ok[row + r + 1]) r++;
    return r - l + 1;
  }

  rowRunLimits(page, cx, cy, bbox, n, lineH, runPad = 5, bboxPad = 10) {
    const [bx, , bw] = bbox;
    const x0 = Math.floor(bx - bboxPad), x1 = Math.floor(bx + bw + bboxPad);
    const totalH = n * lineH;
    const limits = [];
    for (let i = 0; i < n; i++) {
      const yc = cy - totalH / 2 + (i + 0.5) * lineH;
      const top = Math.floor(yc - lineH / 2 + 2), bot = Math.floor(yc + lineH / 2 - 2);
      const samples = bot > top ? [0, 1, 2, 3, 4].map(f => Math.floor(top + (bot - top) * f / 4)) : [Math.floor(yc)];
      const run = Math.min(...samples.map(s => this.rowRun(page.ok, page.W, page.H, s, x0, x1, cx)));
      limits.push(Math.max(0, run - 2 * runPad));
    }
    return limits;
  }

  inkRect(line, size, cx, y, margin) {
    const m = this.measure(line, size);
    return [Math.floor(cx - m.l - margin), Math.floor(y - m.a - margin),
            Math.ceil(cx + m.r + margin), Math.ceil(y + m.d + margin)];
  }

  occMean(occ, W, H, [xa, ya, xb, yb]) {
    xa = Math.max(0, xa); ya = Math.max(0, ya); xb = Math.min(W, xb); yb = Math.min(H, yb);
    if (xb <= xa || yb <= ya) return 0;
    let s = 0;
    for (let y = ya; y < yb; y++) for (let x = xa; x < xb; x++) s += occ[y * W + x];
    return s / ((xb - xa) * (yb - ya));
  }

  fitTextInShape(text, shape, page, occ) {
    let words = text.split(/\s+/).filter(Boolean);
    if (!words.length && text) words = [text];
    const { W, H } = page;
    const passes = (lines, size, xc, y0, lineH, minMean) => {
      const margin = Math.max(2, Math.floor(size / 8));
      for (let i = 0; i < lines.length; i++) {
        const y = y0 + i * lineH + lineH / 2;
        const r = this.inkRect(lines[i], size, xc, y, margin);
        if (r[0] < 0 || r[1] < 0 || r[2] > W || r[3] > H) return false;
        if (page.okInt.mean(...r) < minMean) return false;
        if (occ && this.occMean(occ, W, H, r) > 0.02) return false;
      }
      return true;
    };
    const [, , , bh] = shape.bbox;
    const longest = Math.max(...words.map(w => this.measure(w, this.minFont).w));
    for (const minMean of [0.995, 0.98]) {
      for (let size = this.maxFont; size >= this.minFont; size--) {
        const lineH = size * this.lineH;
        const nMax = Math.max(1, Math.floor(bh * 0.98 / lineH));
        for (let n = 1; n <= nMax; n++) {
          const limits = this.rowRunLimits(page, shape.cx, shape.cy, shape.bbox, n, lineH);
          if (!limits.length || Math.max(...limits) < longest) continue;
          const lines = this.layoutLines(words, size, limits);
          if (!lines || lines.length > n) continue;
          const y0 = shape.cy - lines.length * lineH / 2;
          if (passes(lines, size, shape.cx, y0, lineH, minMean)) return { size, lines, cx: shape.cx, y0 };
        }
      }
    }
    return null;
  }

  fitTextRotated(text, bbox, page, occ) {
    const [bx, by, bw, bh] = bbox;
    const maxW = bh * this.textPad, maxH = bw * this.textPad;
    if (maxH < this.minFont * this.lineH) return null;
    const words = text.split(/\s+/).filter(Boolean);
    const { W, H } = page;
    for (let size = this.maxFont; size >= this.minFont; size--) {
      const lineH = size * this.lineH;
      const n = Math.max(1, Math.floor(maxH / lineH));
      const lines = this.layoutLines(words, size, new Array(n).fill(maxW));
      if (!lines || lines.length * lineH > maxH) continue;
      const blockW = Math.max(...lines.map(l => this.measure(l, size).w));
      const blockH = lines.length * lineH;
      const margin = Math.max(2, Math.floor(size / 8));
      const cx = bx + bw / 2, cy = by + bh / 2;
      // rotated 90° CW (reads top-to-bottom): line i is the i-th column from
      // the RIGHT, like a turned page
      const x0 = cx - blockH / 2, y0 = cy - blockW / 2;
      let ok = true;
      for (let i = 0; i < lines.length && ok; i++) {
        const lw = this.measure(lines[i], size).w;
        const colR = x0 + blockH - i * lineH;
        const r = [Math.floor(colR - lineH) - margin, Math.floor(y0 + (blockW - lw) / 2) - margin,
                   Math.floor(colR) + margin, Math.floor(y0 + (blockW + lw) / 2) + margin];
        const rc = [Math.max(0, r[0]), Math.max(0, r[1]), Math.min(W, r[2]), Math.min(H, r[3])];
        if (rc[2] <= rc[0] || rc[3] <= rc[1] || page.okInt.mean(...rc) < 0.995) ok = false;
        else if (occ && this.occMean(occ, W, H, rc) > 0.02) ok = false;
      }
      if (ok) return { size, lines, x0, y0, blockW, blockH };
    }
    return null;
  }

  markOccupied(occ, W, H, r) {
    const xa = Math.max(0, r[0]), ya = Math.max(0, r[1]), xb = Math.min(W, r[2]), yb = Math.min(H, r[3]);
    for (let y = ya; y < yb; y++) occ.fill(1, y * W + xa, y * W + Math.max(xa, xb));
  }

  drawLines(g, lines, size, cx, y0, stroke = 0) {
    const lineH = size * this.lineH;
    g.font = this.fontCss(size);
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.lineJoin = 'round';
    lines.forEach((line, i) => {
      const y = y0 + i * lineH + lineH / 2;
      if (stroke) {
        g.strokeStyle = '#fff';
        g.lineWidth = 2 * stroke;
        g.strokeText(line, cx, y);
      }
      g.fillStyle = '#000';
      g.fillText(line, cx, y);
    });
  }

  markLines(occ, W, H, lines, size, cx, y0) {
    const lineH = size * this.lineH;
    lines.forEach((line, i) => this.markOccupied(occ, W, H, this.inkRect(line, size, cx, y0 + i * lineH + lineH / 2, 2)));
  }

  renderFallback(g, text, bbox, page, occ) {
    const [bx, by, bw, bh] = bbox;
    let words = text.split(/\s+/).filter(Boolean);
    if (!words.length && text) words = [text];
    const { W, H } = page;
    const cx0 = bx + bw / 2, cy0 = by + bh / 2;
    const wrap0 = Math.max(Math.max(bw, Math.min(this.rowRun(page.ok, W, H, cy0, bx - bw, bx + bw * 2, bx + bw / 2), bw * 3)), 24);
    const wrapAt = (cx, cy) => Math.max(bw, Math.min(this.rowRun(page.ok, W, H, cy, cx - bw * 1.5, cx + bw * 1.5, cx), bw * 3), 24);
    const layoutAt = (size, wrapW) => {
      const n = Math.max(1, Math.floor(bh * 1.15 / (size * this.lineH)));
      return this.layoutLines(words, size, new Array(n).fill(wrapW * 0.95));
    };
    const collision = (lines, size, cx, cy) => {
      const lineH = size * this.lineH, y0 = cy - lines.length * lineH / 2;
      let score = 0;
      lines.forEach((line, i) => {
        const r = this.inkRect(line, size, cx, y0 + i * lineH + lineH / 2, 2);
        if (r[0] < 0 || r[1] < 0 || r[2] > W || r[3] > H) score += 1;
        else if (occ && this.occMean(occ, W, H, r) > 0.02) score += 1;
      });
      return score;
    };
    const offsets = () => {
      const step = Math.max(bw, 24), offs = [[0, 0]];
      for (const r of [1, 2, 3]) for (const dy of [-r, 0, r]) for (const dx of [-r, 0, r])
        if (dx || dy) offs.push([dx * step, dy * step]);
      return offs;
    };
    let chosen = null;
    for (let size = this.maxFont; size >= this.minFont && !chosen; size--) {
      const lines = layoutAt(size, wrap0);
      if (!lines || lines.length * size * this.lineH > bh * 1.25) continue;
      if (collision(lines, size, cx0, cy0) === 0) chosen = { size, lines, cx: cx0, cy: cy0 };
    }
    if (!chosen) {
      outer: for (const [dx, dy] of offsets()) {
        const cx = cx0 + dx, cy = cy0 + dy, ww = wrapAt(cx, cy);
        for (let size = this.maxFont; size >= this.minFont; size--) {
          const lines = layoutAt(size, ww);
          if (!lines || lines.length * size * this.lineH > bh * 1.25) continue;
          if (collision(lines, size, cx, cy) === 0) { chosen = { size, lines, cx, cy }; break outer; }
        }
      }
    }
    if (!chosen) {
      const size = this.minFont, lines = layoutAt(size, wrap0) || [text];
      let best = null;
      for (const [dx, dy] of offsets()) {
        const s = collision(lines, size, cx0 + dx, cy0 + dy);
        if (!best || s < best[0]) best = [s, cx0 + dx, cy0 + dy];
        if (best[0] === 0) break;
      }
      chosen = { size, lines, cx: best[1], cy: best[2] };
    }
    const { size, lines, cx, cy } = chosen;
    const y0 = cy - lines.length * size * this.lineH / 2;
    this.drawLines(g, lines, size, cx, y0, Math.max(3, Math.floor(size / 6)));
    this.markLines(occ, W, H, lines, size, cx, y0);
    return { fontSize: size, lines, fallback: true };
  }

  renderBlock(g, text, shape, page, bbox, occ) {
    let layout = this.fitTextInShape(text, shape, page, occ);
    if (!layout && bbox) {
      const [x1, y1, x2, y2] = bbox;
      const bs = { kind: 'rect', cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, bbox: [x1, y1, x2 - x1, y2 - y1] };
      layout = this.fitTextInShape(text, bs, page, occ);
      if (layout) shape = bs;
    }
    if ((!layout || layout.size < 14) && shape.kind === 'rect') {
      const [, , bw, bh] = shape.bbox;
      if (bh > bw * 1.3) {
        const rot = this.fitTextRotated(text, shape.bbox, page, occ);
        if (rot && (!layout || rot.size > layout.size * 1.25)) {
          const { size, lines, x0, y0, blockW, blockH } = rot;
          const lineH = size * this.lineH;
          g.save();
          // horizontal layout (blockW x blockH) rotated 90° CW into (x0, y0)
          g.translate(x0 + blockH, y0);
          g.rotate(Math.PI / 2);
          g.font = this.fontCss(size);
          g.textAlign = 'center';
          g.textBaseline = 'middle';
          g.fillStyle = '#000';
          // local y grows leftward after the rotation: line 0 lands in the rightmost column
          lines.forEach((line, i) => g.fillText(line, blockW / 2, i * lineH + lineH / 2));
          g.restore();
          this.markOccupied(occ, page.W, page.H, [Math.floor(x0 - 2), Math.floor(y0 - 2),
                                                  Math.ceil(x0 + blockH + 2), Math.ceil(y0 + blockW + 2)]);
          return { shape, fontSize: size, lines, rotated: true };
        }
      }
    }
    if (!layout) {
      const [x1, y1, x2, y2] = bbox || [shape.bbox[0], shape.bbox[1], shape.bbox[0] + shape.bbox[2], shape.bbox[1] + shape.bbox[3]];
      return { shape, ...this.renderFallback(g, text, [x1, y1, x2 - x1, y2 - y1], page, occ) };
    }
    const { size, lines, cx, y0 } = layout;
    this.drawLines(g, lines, size, cx, y0);
    this.markLines(occ, page.W, page.H, lines, size, cx, y0);
    return { shape, fontSize: size, lines, rotated: false };
  }

  /** original/clean: RGBA ImageData; blocks: [{id, xyxy, translation, skip}].
   * Returns { canvas (OffscreenCanvas with the final page), info: {id: ...} }. */
  async renderPage(original, clean, blocks) {
    await loadFont();
    const W = clean.width, H = clean.height;
    const out = canvas(W, H);
    const g = ctx2d(out);
    g.putImageData(clean, 0, 0);
    const cleanGray = gray(clean), origGray = gray(original);
    const ok = new Uint8Array(W * H);
    for (let i = 0; i < ok.length; i++) ok[i] = cleanGray[i] > 225 ? 1 : 0;
    const page = { W, H, ok, okInt: new Integral(ok, W, H) };
    let occupied = new Uint8Array(W * H);

    const shapes = new Map();
    for (const b of blocks) {
      const [x1, y1, x2, y2] = b.xyxy;
      shapes.set(b.id, this.findBubbleShape(origGray, W, H, b.xyxy) ||
        { kind: 'rect', cx: (x1 + x2) / 2, cy: (y1 + y2) / 2, bbox: [x1, y1, x2 - x1, y2 - y1] });
    }
    // stacked/overlapping bubbles -> block-local placement
    const contested = new Set();
    const ids = [...shapes.keys()];
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
      const a = shapes.get(ids[i]).bbox, c = shapes.get(ids[j]).bbox;
      const ox = Math.min(a[0] + a[2], c[0] + c[2]) - Math.max(a[0], c[0]);
      const oy = Math.min(a[1] + a[3], c[1] + c[3]) - Math.max(a[1], c[1]);
      if (ox > 0 && oy > 0 && ox / Math.min(a[2], c[2]) > 0.4 && oy / Math.min(a[3], c[3]) > 0.4) {
        contested.add(ids[i]); contested.add(ids[j]);
      }
    }
    const toRender = blocks.filter(b => b.translation && !b.skip);
    const zonePad = 4;
    const zones = new Uint8Array(W * H);
    for (const b of toRender) {
      const [x1, y1, x2, y2] = b.xyxy.map(Math.round);
      this.markOccupied(zones, W, H, [x1 - zonePad, y1 - zonePad, x2 + zonePad, y2 + zonePad]);
    }
    const info = {};
    for (const b of toRender) {
      let shape = shapes.get(b.id);
      const [x1, y1, x2, y2] = b.xyxy;
      if (contested.has(b.id)) {
        const bw = x2 - x1, bh = y2 - y1;
        const gw = Math.max(bw * 1.6, bw + 24), gh = Math.max(bh * 1.15, bh + 20);
        const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
        shape = { kind: 'rect', cx, cy, bbox: [cx - gw / 2, cy - gh / 2, gw, gh] };
      }
      const myOcc = new Uint8Array(W * H);
      for (let i = 0; i < myOcc.length; i++) myOcc[i] = occupied[i] | zones[i];
      const r = [x1, y1, x2, y2].map(Math.round);
      for (let y = Math.max(0, r[1] - zonePad); y < Math.min(H, r[3] + zonePad); y++)
        myOcc.fill(0, y * W + Math.max(0, r[0] - zonePad), y * W + Math.min(W, r[2] + zonePad));
      info[b.id] = this.renderBlock(g, b.translation, shape, page, b.xyxy, myOcc);
      occupied = myOcc;
    }
    return { canvas: out, info };
  }
}
