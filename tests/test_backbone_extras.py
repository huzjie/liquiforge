# -*- coding: utf-8 -*-
import unittest

from liquiforge.backbone.feature_map import phi, elu_plus_one, relu
from liquiforge.backbone.rope import rope_tables, apply_rope


class TestFeatureMap(unittest.TestCase):
    def test_elu_positive(self):
        for v in elu_plus_one([-3.0, -0.1, 0.0, 1.0]):
            self.assertGreater(v, 0.0)

    def test_relu(self):
        self.assertEqual(relu([-1.0, 0.0, 2.0]), [0.0, 0.0, 2.0])

    def test_phi_dispatch(self):
        self.assertEqual(phi("identity", [1.0, 2.0]), [1.0, 2.0])


class TestRoPE(unittest.TestCase):
    def test_tables_shape(self):
        cos, sin = rope_tables(8, 16)
        self.assertEqual(len(cos), 8)
        self.assertEqual(len(cos[0]), 8)

    def test_apply(self):
        cos, sin = rope_tables(4, 8)
        out = apply_rope([1.0, 0.0, 0.0, 1.0], 0, cos, sin)
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
