// Sugoi v4 ja->en NMT in the browser: sentencepiece-unigram tokenizer +
// encoder/decoder ONNX (converted by tools/convert_sugoi.py) + batched
// beam search (beam 1 = greedy). The decoder has no KV cache: it re-runs the
// prefix each step (manga bubbles are short).
import { ort, fetchModel, fetchJSON, createSession } from './runtime.js';

const SPACE = '▁';

/** Unigram Viterbi segmentation mirroring sentencepiece (nmt_nfkc
 * normalization, whitespace collapse, dummy prefix); unknown characters
 * become unk pieces, consecutive ones merged. Python reference + check:
 * ref_encode() in tools/convert_sugoi.py. */
export class UnigramTokenizer {
  constructor(pieces) {
    this.vocab = new Map();
    let minScore = Infinity;
    this.maxLen = 1;
    for (const [piece, score, kind] of pieces) {
      if (kind !== 'n') continue;
      this.vocab.set(piece, score);
      if (score < minScore) minScore = score;
      const len = [...piece].length;
      if (len > this.maxLen) this.maxLen = len;
    }
    this.unkScore = minScore - 10;
  }

  static normalize(text) {
    return text.normalize('NFKC').split(/\s+/u).filter(Boolean).join(' ');
  }

  encode(text) {
    text = UnigramTokenizer.normalize(text);
    if (!text) return [];
    const chars = [...(SPACE + text.replaceAll(' ', SPACE))];
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const backPos = new Int32Array(n + 1);
    const backUnk = new Uint8Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let s = '';
      for (let j = i; j < Math.min(n, i + this.maxLen); j++) {
        s += chars[j];
        const sc = this.vocab.get(s);
        if (sc !== undefined && best[i] + sc > best[j + 1]) {
          best[j + 1] = best[i] + sc; backPos[j + 1] = i; backUnk[j + 1] = 0;
        }
      }
      if (!this.vocab.has(chars[i]) && best[i] + this.unkScore > best[i + 1]) {
        best[i + 1] = best[i] + this.unkScore; backPos[i + 1] = i; backUnk[i + 1] = 1;
      }
    }
    const toks = [];
    for (let j = n; j > 0; j = backPos[j]) toks.push([chars.slice(backPos[j], j).join(''), backUnk[j]]);
    toks.reverse();
    const out = [];
    for (const [p, unk] of toks) {
      if (unk && out.length && out[out.length - 1][1]) out[out.length - 1][0] += p;
      else out.push([p, unk]);
    }
    return out.map(([p]) => p);
  }

  static decode(pieces) {
    return pieces.join('').replaceAll(SPACE, ' ').trim();
  }
}

export class Sugoi {
  static async load(onProgress = () => {}) {
    const tok = await fetchJSON('sugoi-v4/tokenizer.json');
    const self = new Sugoi();
    self.tokenizer = new UnigramTokenizer(tok.source.pieces);
    self.srcIndex = new Map(tok.source.vocab.map((p, i) => [p, i]));
    self.tgtVocab = tok.target.vocab;
    self.sp = tok.special;
    const encBytes = await fetchModel('sugoi-v4/encoder.onnx', (l, t) => onProgress('encoder', l, t));
    self.encoder = await createSession(encBytes);
    const decBytes = await fetchModel('sugoi-v4/decoder.onnx', (l, t) => onProgress('decoder', l, t));
    self.decoder = await createSession(decBytes);
    return self;
  }

  encodeIds(text) {
    const { unk, eos } = this.sp;
    return [...this.tokenizer.encode(text).map(p => this.srcIndex.get(p) ?? unk), eos];
  }

  decodeIds(ids) {
    return UnigramTokenizer.decode(ids.map(i => this.tgtVocab[i])).replaceAll('<unk>', '').trim();
  }

  /** Translate a list of Japanese strings. beam: 1 = greedy (fastest). */
  async translate(texts, { beam = 1, maxLen = 256 } = {}) {
    if (!texts.length) return [];
    const { pad, eos } = this.sp;
    const src = texts.map(t => this.encodeIds(t));
    const B = src.length, S = Math.max(...src.map(s => s.length));
    const ids = new BigInt64Array(B * S).fill(BigInt(pad));
    const mask = new BigInt64Array(B * S);
    src.forEach((s, b) => s.forEach((t, i) => { ids[b * S + i] = BigInt(t); mask[b * S + i] = 1n; }));
    const enc = await this.encoder.run({
      input_ids: new ort.Tensor('int64', ids, [B, S]),
      attention_mask: new ort.Tensor('int64', mask, [B, S]),
    });
    const hidden = enc.hidden;
    const D = hidden.dims[2];
    const K = Math.max(1, beam | 0);

    // rows = B*K hypotheses (row r belongs to sentence r/K); tile hidden + mask once
    const R = B * K;
    const hData = hidden.data;
    const hTiled = new hData.constructor(R * S * D);
    const mTiled = new BigInt64Array(R * S);
    for (let r = 0; r < R; r++) {
      const b = Math.floor(r / K);
      hTiled.set(hData.subarray(b * S * D, (b + 1) * S * D), r * S * D);
      mTiled.set(mask.subarray(b * S, (b + 1) * S), r * S);
    }
    const hT = new ort.Tensor(hidden.type, hTiled, [R, S, D]);
    const mT = new ort.Tensor('int64', mTiled, [R, S]);
    hidden.dispose?.();

    let seqs = Array.from({ length: R }, () => [eos]);
    let scores = Float64Array.from({ length: R }, (_, r) => (r % K === 0 ? 0 : -Infinity));
    const finished = Array.from({ length: B }, () => []);
    const done = new Array(B).fill(false);
    const limits = src.map(s => Math.min(maxLen, 2 * s.length + 10));

    for (let step = 1; step <= Math.max(...limits); step++) {
      const T = seqs[0].length;
      const toks = new BigInt64Array(R * T);
      seqs.forEach((s, r) => s.forEach((t, i) => { toks[r * T + i] = BigInt(t); }));
      const out = await this.decoder.run({ tokens: new ort.Tensor('int64', toks, [R, T]),
                                           hidden: hT, attention_mask: mT });
      const lp = out.logprobs.data;
      const V = out.logprobs.dims[1];
      const nextSeqs = [], nextScores = new Float64Array(R).fill(-Infinity);
      for (let b = 0; b < B; b++) {
        if (done[b]) {                     // keep rows shaped; their output is ignored
          for (let k = 0; k < K; k++) nextSeqs.push([...seqs[b * K + k], pad]);
          continue;
        }
        // top-K expansions over the sentence's K rows
        const cand = [];
        for (let k = 0; k < K; k++) {
          const r = b * K + k;
          if (scores[r] === -Infinity) continue;
          const row = lp.subarray(r * V, (r + 1) * V);
          // partial selection of the K+1 best tokens of this row
          const top = [];
          for (let v = 0; v < V; v++) {
            const s = row[v];
            if (top.length < K + 1) { top.push([s, v]); top.sort((a, c) => c[0] - a[0]); }
            else if (s > top[K][0]) { top[K] = [s, v]; top.sort((a, c) => c[0] - a[0]); }
          }
          for (const [s, v] of top) cand.push([scores[r] + s, r, v]);
        }
        cand.sort((a, c) => c[0] - a[0]);
        let kept = 0;
        for (const [s, r, v] of cand) {
          if (kept >= K) break;
          if (v === eos || step >= limits[b]) {
            const toks = v === eos ? seqs[r].slice(1) : [...seqs[r].slice(1), v];
            finished[b].push([s / Math.max(1, toks.length + 1), toks]);
            if (finished[b].length >= K) break;
            continue;
          }
          nextSeqs.push([...seqs[r], v]);
          nextScores[b * K + kept] = s;
          kept++;
        }
        while (kept < K) { nextSeqs.push([...seqs[b * K], pad]); kept++; }
        if (finished[b].length >= K || step >= limits[b] ||
            (K === 1 && finished[b].length)) done[b] = true;
      }
      seqs = nextSeqs;
      scores = nextScores;
      if (done.every(Boolean)) break;
    }
    return finished.map(f => {
      if (!f.length) return '';
      f.sort((a, c) => c[0] - a[0]);
      return this.decodeIds(f[0][1]);
    });
  }
}
