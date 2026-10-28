# -*- coding: utf-8 -*-
import math
import unittest

from liquiforge.heads.text_head import TextHead
from liquiforge.heads.vision_head import VisionHead
from liquiforge.heads.audio_head import AudioHead
from liquiforge.heads.omni_head import OmniHead
from liquiforge.heads.decision_head import DecisionHead
from liquiforge.utils.rng import stable_vector


def seq(n, dim):
    return [stable_vector("h%d" % i, dim) for i in range(n)]


class TestTextHead(unittest.TestCase):
    def test_perplexity_finite(self):
        h = TextHead(128, 256)
        ppl = h.perplexity(seq(4, 128), [1, 2, 3, 4])
        self.assertTrue(math.isfinite(ppl))


class TestVisionHead(unittest.TestCase):
    def test_classify(self):
        h = VisionHead(128, 10)
        pred, p = h.classify(seq(3, 128))
        self.assertTrue(0 <= pred < 10)
        self.assertTrue(0.0 <= p <= 1.0)


class TestAudioHead(unittest.TestCase):
    def test_decode(self):
        h = AudioHead(128, 256)
        ids = h.decode(seq(5, 128))
        self.assertEqual(len(ids), 5)


class TestOmniHead(unittest.TestCase):
    def test_similarity_range(self):
        h = OmniHead(128, 64)
        s = h.similarity(seq(3, 128), seq(3, 128))
        self.assertTrue(-1.0 <= s <= 1.0)


class TestDecisionHead(unittest.TestCase):
    def test_policy_sums_to_one(self):
        h = DecisionHead(128, 16)
        p = h.policy(seq(3, 128))
        self.assertAlmostEqual(sum(p), 1.0, places=4)
        self.assertEqual(len(p), 16)


if __name__ == "__main__":
    unittest.main()
