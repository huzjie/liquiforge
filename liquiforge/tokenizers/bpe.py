# -*- coding: utf-8 -*-
"""Byte-level BPE tokenizer — zero dependencies, deterministic merge learning."""
from __future__ import annotations

import collections
import hashlib


class BPETokenizer:
    """Learns byte-pair merges from a corpus, then encodes/decodes bytes.

    The merge table is learned deterministically: pair frequencies are counted
    in a stable order so identical corpora always produce identical vocabularies.
    """

    def __init__(self, vocab_size: int = 256, seed: int = 0):
        self.vocab_size = max(vocab_size, 256)
        self.seed = seed
        self.merges: list[tuple[int, int]] = []
        self.merges_index: dict[tuple[int, int], int] = {}

    # -- training ---------------------------------------------------------
    def train(self, corpus: list[str]):
        """Learn merges from a list of raw strings (bytes)."""
        ids = [list(s.encode("utf-8")) for s in corpus if s]
        n_merges = self.vocab_size - 256
        self.merges = []
        self.merges_index = {}
        for _ in range(n_merges):
            counts: dict[tuple[int, int], int] = collections.defaultdict(int)
            for seq in ids:
                for i in range(len(seq) - 1):
                    counts[(seq[i], seq[i + 1])] += 1
            if not counts:
                break
            # stable tie-break by (freq desc, pair asc)
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0] * 1000 - kv[0][1]))
            pair = best[0]
            idx = 256 + len(self.merges)
            self.merges.append(pair)
            self.merges_index[pair] = idx
            # apply merge
            new_ids = []
            for seq in ids:
                merged = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        merged.append(idx)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_ids.append(merged)
            ids = new_ids

    # -- encode / decode --------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair in self.merges:
            idx = self.merges_index[pair]
            merged = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    merged.append(idx)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[int] = []
        for tok in ids:
            if tok < 256:
                out.append(tok)
            else:
                pair = self.merges[tok - 256]
                out.append(pair[0])
                out.append(pair[1])
        # expand any remaining multi-byte merge tokens iteratively
        changed = True
        while changed:
            changed = False
            nxt: list[int] = []
            for tok in out:
                if tok < 256:
                    nxt.append(tok)
                else:
                    pair = self.merges[tok - 256]
                    nxt.append(pair[0])
                    nxt.append(pair[1])
                    changed = True
            out = nxt
        return bytes(out).decode("utf-8", errors="replace")


def _demo_corpus() -> list[str]:
    return [
        "the quick brown fox jumps over the lazy dog",
        "attention is all you need for sequence modeling",
        "multimodal foundation models unify vision and language",
    ]


def default_tokenizer(vocab_size: int = 256) -> BPETokenizer:
    t = BPETokenizer(vocab_size=vocab_size, seed=hashlib.md5(b"liquiforge-bpe").hexdigest()[:8])
    t.train(_demo_corpus())
    return t
