"""Build models/ (the files the page loads) from the original sources.

  models/ppocr-manga/  manga_det.onnx, manga_rec.onnx, ppocrv6_dict.txt
                       (Kellenok/PP-OCRv6_manga v0.3, Apache-2.0)
  models/migan/        migan_pipeline.onnx (MI-GAN by Picsart AI Research, MIT;
                       ONNX pipeline export from anyisalin/migan-onnx)
  models/sugoi-v4/     encoder/decoder ONNX + tokenizer.json, converted by
                       convert_sugoi.py from entai2965/sugoi-v4-ja-en-ctranslate2
                       (NTT terms: research use only, no commercial use)

Every download is pinned to a commit and SHA-256 verified. The Sugoi
ctranslate2 weights (1.1 GB) go to build/sugoi-v4-ct2/ and are only needed
for the conversion. Then serve.py + ?models=local uses the result, and
upload_models.py publishes it to the Hugging Face repo the page reads from.

Usage: python tools/setup_models.py [--skip-sugoi] [--fp32-sugoi] [--sugoi-src DIR]
Needs: requests; for Sugoi also torch, ctranslate2, sentencepiece, onnx, onnxscript, onnxruntime
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models'
CT2 = ROOT / 'build' / 'sugoi-v4-ct2'


def hf(repo, rev, name):
    return f'https://huggingface.co/{repo}/resolve/{rev}/{name}'


PPOCR = ('Kellenok/PP-OCRv6_manga', 'd65838996a00ef6b1c73ca3a700d7a8c5fbbef8a')   # v0.3
MIGAN = ('anyisalin/migan-onnx', 'a8be381d5206abb962386d8ef4bd6e08363bb520')
SUGOI = ('entai2965/sugoi-v4-ja-en-ctranslate2', '71d67eb8e73ec2f5aaefc0689e03a4eb843d3a2b')

# destination -> (url, sha256, size)
FILES = {
    OUT / 'ppocr-manga/manga_det.onnx': (
        hf(*PPOCR, 'det/manga_det.onnx'),
        '97c434ae8deaad83513fb132b605d5085a3af8ffac2cb15e2aa096815c206172', 1_816_958),
    OUT / 'ppocr-manga/manga_rec.onnx': (
        hf(*PPOCR, 'rec/manga_rec.onnx'),
        '5288bd221a9a047c6b7d1ff56bb98de8c875fb613df98ebe8754946448bda540', 21_334_702),
    OUT / 'ppocr-manga/ppocrv6_dict.txt': (
        hf(*PPOCR, 'ppocrv6_dict.txt'),
        '4f0f8971ecb95e3a4d278985d6eaac154413e81fda0b4c941db17389580e9e9b', 76_327),
    OUT / 'migan/migan_pipeline.onnx': (
        hf(*MIGAN, 'onnx/migan_pipeline.onnx'),
        '6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b', 28_079_181),
}
SUGOI_FILES = {
    CT2 / name: (hf(*SUGOI, name), sha, size) for name, sha, size in (
        ('LICENSE', 'f028708b8b8b137b68b5196d38a76c43871945a77f041def5b9c2965ec49943a', 2_949),
        ('config.json', '3e732fbc16f7155d1354f96e746092f95f290d6b0a0cac56e9e4a1dd7ddb13d2', 222),
        ('source_vocabulary.json', '1ae20a6f46611cfd6766fc9dd289747ec835aebbc27a161c1ab4bd311912af4c', 701_538),
        ('target_vocabulary.json', 'ad53ade1465d0752842772e1e1d5c5eb870afb9b42de99a8a23442008a4a35e8', 501_642),
        ('spm/spm.ja.nopretok.model', '1bff3529a8e0bd898f00707a4e36dc16540d84112cc8a4a14462c0099e4aab9d', 785_276),
        ('spm/spm.en.nopretok.model', '183aad11f5fa48b21fdbbb7a97082d160b86cbcc4f9dc5e61d0eebd48390d36a', 751_435),
        ('model.bin', 'de9bbeeb1b74716536c08a82e44d4ce1950b3cf690c4a426f6ce44a6c1c17a3d', 1_102_425_134),
    )
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest, url, sha, size):
    if dest.is_file() and sha256_file(dest) == sha:
        print(f'ok (already present): {dest.relative_to(ROOT)}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    for attempt in range(1, 4):
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                done = 0
                with open(tmp, 'wb') as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        print(f'\r  {dest.name}: {100 * min(done, size) / size:5.1f}%', end='', flush=True)
            print()
            if sha256_file(tmp) != sha:
                raise RuntimeError('SHA-256 mismatch after download')
            tmp.replace(dest)
            print(f'verified: {dest.relative_to(ROOT)}')
            return
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f'\n  attempt {attempt} failed ({e}); retrying')
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-sugoi', action='store_true')
    ap.add_argument('--fp32-sugoi', action='store_true', help='fp32 Sugoi weights (~1.2 GB instead of 0.6)')
    ap.add_argument('--sugoi-src', type=Path, default=CT2,
                    help='existing Sugoi v4 ctranslate2 folder (skips its download)')
    args = ap.parse_args()

    for dest, spec in FILES.items():
        fetch(dest, *spec)
    if args.skip_sugoi:
        print('skipped Sugoi conversion')
        return 0
    if args.sugoi_src == CT2:
        print('Sugoi v4 ctranslate2 weights (NTT terms: research use only, no commercial use)')
        for dest, spec in SUGOI_FILES.items():
            fetch(dest, *spec)
    cmd = [sys.executable, str(ROOT / 'tools' / 'convert_sugoi.py'), '--src', str(args.sugoi_src)]
    if args.fp32_sugoi:
        cmd.append('--fp32')
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
