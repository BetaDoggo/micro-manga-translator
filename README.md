# micro-manga-translator

A simple manga translator which is light enough to run entirely in a web page on your own hardware, nothing is uploaded. Quality is far from the best, but it's usable enough. Speed depends on your hardware. At time of writing firefox webgpu is bugged with fp16, so some parts will fall back to cpu. Use a chromium based browser if you can.

| stage | model | notes |
|---|---|---|
| detect | PP-OCRv6_manga `manga_det` (1.8 MB) | DB line detector; lines are grouped into bubble blocks by a geometric heuristic (`js/group.js`), furigana dropped from the text |
| OCR | PP-OCRv6_manga `manga_rec` (21 MB) | CTC recognizer, batched line crops (vertical lines rotated 90° CCW) |
| translate | Sugoi v4 ja→en (620 MB, fp16) | encoder/decoder ONNX converted from the ctranslate2 model; JS sentencepiece tokenizer + batched beam search; beam size 1–10 on the page, default 5 (1 = greedy, fastest) |
| inpaint | MI-GAN (28 MB) | per-block windows; ink-only mask inside white bubbles, whole line box over art |
| render | — | canvas port of the manga-translate Python renderer (bubble-shape fitting, rotated labels, stroked fallback) |

Japanese only (Sugoi is ja→en). Needs a WebGPU browser (recent Chrome/Edge).
The first page after loading is slower (shader compilation); later pages take
around a second on a desktop GPU.

## Models

The page loads its models from one Hugging Face repo,
[HDiffusion/micro-manga-translator](https://huggingface.co/HDiffusion/micro-manga-translator),
pinned to a commit (`MODEL_REPO` / `MODEL_REV` in `js/runtime.js`). Nothing is
downloaded until the user clicks **Download models** (about 640 MB); the files
are then kept in the browser's Cache Storage, so later visits load them
straight from disk. **Delete downloaded models** on the page removes them.
Files from an older `MODEL_REV` are pruned automatically.

## Running locally

```bash
python serve.py                 # http://localhost:8000/
```

## Building and publishing the models

```bash
pip install requests torch ctranslate2 sentencepiece onnx onnxscript onnxruntime huggingface_hub
python tools/setup_models.py    # fills models/ (downloads + Sugoi conversion)
```

`setup_models.py` downloads PP-OCRv6_manga and MI-GAN (pinned, SHA-256
verified) and the Sugoi v4 ctranslate2 weights (1.1 GB, to `build/`), then runs
`tools/convert_sugoi.py`. The converter parses `model.bin` directly (a flat
tensor file; no pickle is loaded), rebuilds the fairseq transformer in PyTorch,
checks it token-for-token against ctranslate2 and the tokenizer port against
sentencepiece, then exports fp16 ONNX (`--fp32-sugoi` for fp32).

## Files

- `index.html`, `js/app.js`: the page
- `js/runtime.js`: onnxruntime-web setup, model URLs, download and caching
- `js/pipeline.js`: stage orchestration
- `js/ocr.js`, `js/group.js`, `js/sugoi.js`, `js/inpaint.js`, `js/render.js`, `js/image.js`
- `fonts/`: Comic Neue Bold (SIL OFL 1.1, `fonts/OFL.txt`)
- `tools/`: model build, conversion and upload scripts
- `hf/`: model card and license files for the Hugging Face repo

## Licenses

Code: GPL-3.0 (`LICENSE`). PP-OCRv6_manga: Apache-2.0. MI-GAN: MIT (Picsart AI
Research). Sugoi v4: NTT terms of use, research use only, no commercial use;
the converted ONNX files are derived data under the same terms. Comic Neue:
SIL OFL 1.1. onnxruntime-web is loaded from jsDelivr (MIT).
