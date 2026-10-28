# -*- coding: utf-8 -*-
import unittest

from liquiforge.backbone.linear_attention import LinearAttention
from liquiforge.backbone.conv import DepthwiseConv
from liquiforge.backbone.moe import SparseMoE
from liquiforge.backbone.block import TriHybridBlock
from liquiforge.backbone.model import OmniBackbone
from liquiforge.utils.rng import stable_vector


def seq(n, dim):
    return [stable_vector("t%d" % i, dim) for i in range(n)]


class TestLinearAttention(unittest.TestCase):
    def test_shape(self):
        la = LinearAttention(128, 4)
        out = la.forward(seq(8, 128))
        self.assertEqual(len(out), 8)
        self.assertEqual(len(out[0]), 128)


class TestConv(unittest.TestCase):
    def test_shape(self):
        c = DepthwiseConv(128, 5)
        out = c.forward(seq(6, 128))
        self.assertEqual(len(out), 6)
        self.assertEqual(len(out[0]), 128)


class TestMoE(unittest.TestCase):
    def test_shape_and_loss(self):
        moe = SparseMoE(128, 8, 2)
        out, lb = moe.forward(seq(5, 128))
        self.assertEqual(len(out), 5)
        self.assertEqual(len(out[0]), 128)
        self.assertGreaterEqual(lb, 0.0)


class TestBlock(unittest.TestCase):
    def test_forward(self):
        b = TriHybridBlock(128, 4, 8, 2, 5)
        out, lb = b.forward(seq(4, 128))
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[0]), 128)
        self.assertIsInstance(lb, float)


class TestModel(unittest.TestCase):
    def test_forward(self):
        m = OmniBackbone(128, 3, 4, 8, 2, 5)
        out, lbs = m.forward(seq(4, 128))
        self.assertEqual(len(out), 4)
        self.assertEqual(len(lbs), 3)


if __name__ == "__main__":
    unittest.main()
