#!/usr/bin/env python3
"""Sidecar HTTP pour /files/raw — process isolé d'uvicorn.

Hypothèse : les reads depuis un process *séparé* d'uvicorn (pas de threadpool
hérité, pas d'event loop commun) matérialisent Google Drive FS sans EDEADLK.
Testé 2026-04-19 : bash `cat` hors uvicorn fonctionne ; ce sidecar joue le rôle
de "bash cat" mais en HTTP pour que uvicorn puisse le proxifier sur /files/raw.

Lancement :
    nohup /Users/julien/.pyenv/versions/Artefact/bin/python \\
        /Users/julien/docfinder/scripts/file_sidecar.py > /tmp/file_sidecar.log 2>&1 &
    disown

Uvicorn (server/files_api.py) fait un httpx.get() vers 127.0.0.1:8002/raw?path=...
"""
from __future__ import annotations

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8002
ALLOWED_ROOTS = (
    Path("/Users/julien/Documents"),
    Path("/Users/julien/icloud"),
)
ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[sidecar] {fmt % args}\n")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/raw":
            self.send_error(404, "only /raw supported")
            return
        qs = parse_qs(parsed.query)
        path_param = qs.get("path", [None])[0]
        if not path_param:
            self.send_error(400, "missing ?path=")
            return
        p = Path(path_param).expanduser().resolve()
        if not any(str(p).startswith(str(r)) for r in ALLOWED_ROOTS):
            self.send_error(400, f"path not under allowed roots: {p}")
            return
        if p.suffix.lower() not in ALLOWED_EXTS:
            self.send_error(400, f"extension not allowed: {p.suffix}")
            return
        if not p.is_file():
            self.send_error(404, f"not a file: {p}")
            return
        try:
            data = p.read_bytes()
        except OSError as e:
            self.send_error(503, f"read failed: {type(e).__name__}: {e}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    # Bind localhost uniquement — jamais exposé au réseau.
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[sidecar] listening on 127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
