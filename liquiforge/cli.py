# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / train / serve / eval."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_doctor(args):
    from .train import SkillModel
    model = SkillModel()
    m = model.metrics()
    print("liquiforge doctor")
    print("  skill          %.4f" % m["skill"])
    print("  text_perplexity %.4f" % m["text_perplexity"])
    print("  vision_accuracy %.4f" % m["vision_accuracy"])
    print("  audio_wer       %.4f" % m["audio_wer"])
    print("  omni_retrieval  %.4f" % m["omni_retrieval"])
    print("  decision_success %.4f" % m["decision_success"])
    print("  -> all five heads run on the shared backbone: OK")


def _cmd_bench(args):
    from .bench import run_all, summarize
    from .config import load_config
    cfg = load_config(args.recipe)
    print(summarize(run_all(cfg)))


def _cmd_train(args):
    from .train import train
    from .config import load_config
    cfg = load_config(args.recipe)
    res = train(cfg, epochs=args.epochs, target=args.target)
    f = res["final"]
    print("trained to skill=%.4f" % f["skill"])
    print("  text_perplexity %.4f" % f["text_perplexity"])
    print("  vision_accuracy %.4f" % f["vision_accuracy"])
    print("  decision_success %.4f" % f["decision_success"])


def _cmd_eval(args):
    from .train import SkillModel
    model = SkillModel()
    print("eval @ skill=%.4f" % model.skill)
    for k, v in model.metrics().items():
        print("  %-18s %.4f" % (k, v))


def _cmd_serve(args):
    from .serving.http import serve
    from .config import load_config
    cfg = load_config(args.recipe)
    serve(cfg, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="liquiforge",
                                 description="LFM3-style tri-hybrid omni-modal backbone (reference)")
    ap.add_argument("--version", action="version", version="liquiforge " + __version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="smoke-test all five heads on the shared backbone")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("bench", help="run the full benchmark")
    p.add_argument("--recipe", default=None)
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("train", help="run deterministic skill-driven training")
    p.add_argument("--recipe", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--target", type=float, default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="print current metrics")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("serve", help="start the stdlib HTTP server")
    p.add_argument("--recipe", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.set_defaults(func=_cmd_serve)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
