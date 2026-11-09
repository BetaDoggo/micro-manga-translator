// detect -> group -> OCR -> translate -> inpaint -> render, all in the browser.
import { PPOcr } from './ocr.js';
import { groupLines } from './group.js';
import { Sugoi } from './sugoi.js';
import { MiGan, textMask } from './inpaint.js';
import { Renderer, loadFont } from './render.js';
import { canvas, ctx2d, gray } from './image.js';

export class PagePipeline {
  /** onProgress(model, loadedBytes, totalBytes) while downloading weights */
  static async load(onProgress) {
    const p = new PagePipeline();
    p.ocr = await PPOcr.load(onProgress);
    p.inpainter = await MiGan.load(onProgress);
    p.sugoi = await Sugoi.load(onProgress);
    await loadFont();
    p.renderer = new Renderer();
    return p;
  }

  /** image: {bitmap, width, height, data(ImageData)} from image.js loadImage.
   * onStage(name, ms) after each stage. */
  async run(image, { beam = 5, onStage = () => {} } = {}) {
    const timings = {};
    let t = performance.now();
    const stage = name => {
      const now = performance.now();
      timings[name] = now - t;
      onStage(name, timings[name]);
      t = now;
    };
    const W = image.width, H = image.height;

    const lines = await this.ocr.detect(image);
    const blocks = groupLines(lines);
    stage('detect');

    const textLines = blocks.flatMap(b => b.lines);
    const texts = await this.ocr.recognize(image, textLines);
    textLines.forEach((l, i) => { l.text = texts[i]; });
    for (const b of blocks) {
      b.text = b.lines.map(l => l.text).join(b.vertical ? '' : ' ').trim();
      b.skip = [...b.text].length <= 1;       // mis-split sfx / stray kana
    }
    stage('ocr');

    const todo = blocks.filter(b => !b.skip);
    const out = await this.sugoi.translate(todo.map(b => b.text), { beam });
    todo.forEach((b, i) => { b.translation = out[i]; });
    stage('translate');

    const g = gray(image.data);
    const mask = textMask(lines, g, W, H);
    const clean = await this.inpainter.inpaint(image.data, mask,
      lines.map(l => [l.x0, l.y0, l.x1, l.y1]));
    stage('inpaint');

    const { canvas: final, info } = await this.renderer.renderPage(image.data, clean, blocks);
    for (const b of blocks) b.render = info[b.id] || null;
    stage('render');

    return { final, clean, mask, lines, blocks, timings,
             detections: drawDetections(image, lines, blocks) };
  }
}

const PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#808000'];

export function drawDetections(image, lines, blocks) {
  const c = canvas(image.width, image.height);
  const g = ctx2d(c);
  g.drawImage(image.bitmap, 0, 0);
  const lw = Math.max(2, image.width / 500);
  g.lineWidth = lw;
  g.setLineDash([lw * 2, lw * 2]);
  g.strokeStyle = '#888';
  for (const l of lines.filter(l => l.furigana)) g.strokeRect(l.x0, l.y0, l.x1 - l.x0, l.y1 - l.y0);
  g.setLineDash([]);
  for (const b of blocks) {
    const col = PALETTE[b.id % PALETTE.length];
    g.strokeStyle = col;
    g.lineWidth = lw / 2;
    for (const l of b.lines) g.strokeRect(l.x0, l.y0, l.x1 - l.x0, l.y1 - l.y0);
    g.lineWidth = lw;
    const [x0, y0, x1, y1] = b.xyxy;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const fs = Math.max(14, Math.round(image.width / 60));
    g.font = `bold ${fs}px system-ui, sans-serif`;
    const label = String(b.id);
    const tw = g.measureText(label).width + 8;
    g.fillStyle = col;
    g.fillRect(x0, Math.max(0, y0 - fs - 6), tw, fs + 6);
    g.fillStyle = '#fff';
    g.textBaseline = 'top';
    g.fillText(label, x0 + 4, Math.max(0, y0 - fs - 6) + 3);
  }
  return c;
}
