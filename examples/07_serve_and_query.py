# -*- coding: utf-8 -*-
"""Start the stdlib HTTP server and query it (run in two terminals)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.serving.http import serve
from liquiforge.config import load_config

if __name__ == "__main__":
    serve(load_config(), host="127.0.0.1", port=8787)
