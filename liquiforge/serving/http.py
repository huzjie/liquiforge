# -*- coding: utf-8 -*-
"""Minimal HTTP API on top of the stdlib http.server (no third-party deps)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..train import SkillModel


class _Handler(BaseHTTPRequestHandler):
    model = None

    def _send(self, code: int, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._send(200, {"models": ["liquiforge-omni-0.1"], "modalities": ["text", "vision", "audio", "omni", "decision"]})
        if self.path == "/v1/bench":
            from ..bench import run_all
            res = run_all()
            out = {k: {"baseline": b, "trained": t} for k, (b, t) in res.items()}
            return self._send(200, out)
        if self.path == "/v1/metrics":
            return self._send(200, self.model.metrics())
        return self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def serve(cfg: dict, host: str = "127.0.0.1", port: int = 8787):
    model = SkillModel(cfg)
    _Handler.model = model
    httpd = HTTPServer((host, port), _Handler)
    print("liquiforge serving on http://%s:%d" % (host, port))
    print("  GET /health /v1/models /v1/metrics /v1/bench")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
