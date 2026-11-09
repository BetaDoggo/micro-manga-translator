// Group PP-OCR text lines into blocks (one per bubble/caption), the unit that
// gets translated and rendered. comic-text-detector did this in the Python
// pipeline; PP-OCR's detector only finds lines, so this is a geometric
// heuristic on the (unclipped) line boxes, which approximate the glyph
// extents (the DB cores are much thinner than the text):
//  - orientation: a line taller than wide is vertical (Japanese columns)
//  - furigana: a thin line hugging a line >= 1.7x its size is dropped from
//    the text (it is still erased by the inpainting mask)
//  - same-orientation lines of similar size that sit side by side (vertical)
//    or stacked (horizontal) with a small gap and overlapping extents merge.

function size(l) { return l.vertical ? l.x1 - l.x0 : l.y1 - l.y0; }

function gapAndOverlap(a, b, vertical) {
  // vertical lines: gap along x, overlap along y; horizontal: the reverse
  const [p0, p1, q0, q1] = vertical ? ['x0', 'x1', 'y0', 'y1'] : ['y0', 'y1', 'x0', 'x1'];
  const gap = Math.max(a[p0], b[p0]) - Math.min(a[p1], b[p1]);
  const ov = Math.min(a[q1], b[q1]) - Math.max(a[q0], b[q0]);
  const minLen = Math.min(a[q1] - a[q0], b[q1] - b[q0]);
  return { gap, overlap: ov / Math.max(1, minLen) };
}

export function groupLines(lines) {
  for (const l of lines) {
    const w = l.x1 - l.x0, h = l.y1 - l.y0;
    l.vertical = h >= w;
    l.furigana = false;
  }
  // furigana / ruby next to a much larger line of the same orientation
  for (const a of lines) {
    for (const b of lines) {
      if (a === b || a.vertical !== b.vertical) continue;
      const sa = size(a), sb = size(b);
      if (sb < 1.7 * sa) continue;
      const { gap, overlap } = gapAndOverlap(a, b, a.vertical);
      if (gap < 0.3 * sb && overlap > 0.5) { a.furigana = true; break; }
    }
  }
  const text = lines.filter(l => !l.furigana);
  const parent = text.map((_, i) => i);
  const find = i => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j < text.length; j++) {
      const a = text[i], b = text[j];
      if (a.vertical !== b.vertical) continue;
      const sa = size(a), sb = size(b);
      if (Math.max(sa, sb) > 1.5 * Math.min(sa, sb)) continue;
      const { gap, overlap } = gapAndOverlap(a, b, a.vertical);
      if (gap < 0.6 * (sa + sb) / 2 && overlap > 0.3) parent[find(i)] = find(j);
    }
  }
  const groups = new Map();
  text.forEach((l, i) => {
    const r = find(i);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(l);
  });
  const blocks = [];
  for (const ls of groups.values()) {
    const vertical = ls[0].vertical;
    // reading order inside a block: columns right-to-left / rows top-to-bottom
    ls.sort((a, b) => vertical ? (b.x0 + b.x1) - (a.x0 + a.x1) : (a.y0 + a.y1) - (b.y0 + b.y1));
    const xyxy = [Math.min(...ls.map(l => l.x0)), Math.min(...ls.map(l => l.y0)),
                  Math.max(...ls.map(l => l.x1)), Math.max(...ls.map(l => l.y1))];
    blocks.push({ lines: ls, vertical, xyxy,
                  fontSize: ls.reduce((s, l) => s + size(l), 0) / ls.length });
  }
  // page reading order (manga): right-to-left, then top-to-bottom, by block top-right
  // corner, bucketed so blocks on roughly the same row compare by x
  const rowH = Math.max(40, ...blocks.map(b => b.fontSize)) * 3;
  blocks.sort((a, b) => {
    const ra = Math.floor(a.xyxy[1] / rowH), rb = Math.floor(b.xyxy[1] / rowH);
    return ra !== rb ? ra - rb : b.xyxy[2] - a.xyxy[2];
  });
  blocks.forEach((b, i) => { b.id = i; });
  return blocks;
}
