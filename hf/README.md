---
license: other
license_name: mixed
license_link: LICENSE
language:
  - ja
  - en
tags:
  - onnx
  - webgpu
  - manga
  - translation
  - ocr
  - inpainting
---

# micro-manga-translator models

The model files loaded by the micro-manga-translator web page, which detects,
recognizes, translates (Japanese → English), erases and re-letters manga text
entirely in the browser on WebGPU with onnxruntime-web.

| folder | model | source | license |
|---|---|---|---|
| `ppocr-manga/` | PP-OCRv6_manga v0.3 text detector + recognizer (unmodified) | [Kellenok/PP-OCRv6_manga](https://huggingface.co/Kellenok/PP-OCRv6_manga) | Apache-2.0 |
| `migan/` | MI-GAN inpainting, ONNX pipeline export (unmodified) | [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) via [anyisalin/migan-onnx](https://huggingface.co/anyisalin/migan-onnx) | MIT |
| `sugoi-v4/` | Sugoi v4 ja→en, converted to fp16 ONNX (encoder + decoder) | [entai2965/sugoi-v4-ja-en-ctranslate2](https://huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2) | NTT terms of use: **research use only, no commercial use** |

Each folder has its own `LICENSE`; the terms of that file apply to the files in
that folder.

## Sugoi v4 conversion

The ctranslate2 `model.bin` was parsed directly, the fairseq transformer
(6+6 layers, d=1024) rebuilt in PyTorch, checked token-for-token against
ctranslate2, and exported as two ONNX graphs:

- `encoder.onnx`: `input_ids[B,S]`, `attention_mask[B,S]` → `hidden[B,S,1024]`
- `decoder.onnx`: `tokens[B,T]`, `hidden`, `attention_mask` → log-probs of the last position `[B,V]` (no KV cache)
- `tokenizer.json`: sentencepiece unigram pieces/scores and model vocabularies

These files are derived data of NTT's model and are under the same terms: they
may be used only for research purposes involving information analysis, and not
commercially.
