# -*- coding: utf-8 -*-
"""Standalone benchmark entrypoint: python scripts/run_bench.py [--recipe path]."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.bench import run_all, summarize
from liquiforge.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=None)
    args = ap.parse_args()
    cfg = load_config(args.recipe)
    results = run_all(cfg)
    print(summarize(results))


if __name__ == "__main__":
    main()
