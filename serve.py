"""Serve the page locally: python serve.py [--port 8000]

Serves this folder on 127.0.0.1 with explicit MIME types: Python's default map
reads the Windows registry, where .js/.mjs can come back as text/plain and
browsers then refuse to run the ES modules. Then open http://localhost:8000/
(add ?models=local to use ./models/ from tools/setup_models.py)
"""
import argparse
import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.mjs': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.json': 'application/json',
        '.onnx': 'application/octet-stream',
        '.wasm': 'application/wasm',
        '.ttf': 'font/ttf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')   # revalidate; 304s keep reloads fast
        super().end_headers()

    def log_message(self, fmt, *args):
        if not str(args[1] if len(args) > 1 else '').startswith(('2', '3')):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as srv:
        print(f'serving {ROOT} -> http://localhost:{args.port}/', flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
