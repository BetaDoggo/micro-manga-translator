"""Publish models/ to the Hugging Face repo the page loads from.

Uploads the files listed in js/runtime.js (MODEL_FILES) plus the model card
and license files from hf/, as one commit, then prints the commit hash to put
in MODEL_REV (js/runtime.js) so the page and browser caches stay pinned.

Usage: python tools/upload_models.py [--repo HDiffusion/micro-manga-translator]
Needs: huggingface_hub, and `hf auth login` with a write token.
"""
import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[1]


def model_files():
    """{path: size} parsed from MODEL_FILES in js/runtime.js (single source of truth)."""
    src = (ROOT / 'js' / 'runtime.js').read_text(encoding='utf8')
    block = src[src.index('MODEL_FILES = {'):]
    block = block[:block.index('};')]
    return {p: int(n.replace('_', '')) for p, n in re.findall(r"'([^']+)':\s*([\d_]+)", block)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='HDiffusion/micro-manga-translator')
    ap.add_argument('--private', action='store_true')
    args = ap.parse_args()

    ops = []
    for rel, size in model_files().items():
        f = ROOT / 'models' / rel
        if not f.is_file():
            sys.exit(f'missing {f}: run python tools/setup_models.py')
        if f.stat().st_size != size:
            sys.exit(f'{rel}: {f.stat().st_size} bytes, js/runtime.js says {size}; update MODEL_FILES')
        ops.append(CommitOperationAdd(rel, str(f)))
    ops.append(CommitOperationAdd('sugoi-v4/LICENSE', str(ROOT / 'models' / 'sugoi-v4' / 'LICENSE')))
    for f in (ROOT / 'hf').rglob('*'):
        if f.is_file():
            ops.append(CommitOperationAdd(f.relative_to(ROOT / 'hf').as_posix(), str(f)))

    api = HfApi()
    api.create_repo(args.repo, repo_type='model', private=args.private, exist_ok=True)
    info = api.create_commit(args.repo, ops, commit_message='Upload micro-manga-translator models')
    rev = info.oid
    print(f'uploaded {len(ops)} files to https://huggingface.co/{args.repo}')
    print(f"set MODEL_REV = '{rev}' in js/runtime.js")


if __name__ == '__main__':
    main()
