# -*- coding: utf-8 -*-
"""Vision patch tokenizer: image -> non-overlapping patches -> linear projection -> tokens."""
from __future__ import annotations

from ..utils.rng import stable_matrix, stable_float


class VisionPatchTokenizer:
    """Splits a 2-D image into square patches and projects each to a token vector.

    A ``[CLS]``-like register token is prepended for the vision head. The
    projection matrix is deterministic so identical images yield identical tokens.
    """

    def __init__(self, image_size: int = 28, patch_size: int = 4, dim: int = 128, seed: int = 0):
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.grid = image_size // patch_size
        self.n_patches = self.grid * self.grid
        in_dim = patch_size * patch_size
        self.proj = stable_matrix("vision-proj-%d" % seed, in_dim, dim)

    def patchify(self, image: list[list[float]]) -> list[list[float]]:
        """Flatten each patch into a 1-D vector (row-major)."""
        patches = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                vec = []
                for dy in range(self.patch_size):
                    for dx in range(self.patch_size):
                        vec.append(image[gy * self.patch_size + dy][gx * self.patch_size + dx])
                patches.append(vec)
        return patches

    def encode(self, image: list[list[float]]) -> list[list[float]]:
        """image (HxW floats) -> list of token vectors (n_patches x dim)."""
        patches = self.patchify(image)
        tokens = []
        for p in patches:
            tok = [sum(p[k] * self.proj[k][j] for k in range(len(p))) for j in range(self.dim)]
            tokens.append(tok)
        return tokens


def synthetic_image(size: int = 28, key: str = "vision-demo") -> list[list[float]]:
    """Deterministic demo image (smooth gradients) for examples/tests."""
    img = []
    for y in range(size):
        row = []
        for x in range(size):
            v = stable_float("%s-%d-%d" % (key, x, y), 0.0, 1.0)
            row.append(v)
        img.append(row)
    return img
