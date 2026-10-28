# -*- coding: utf-8 -*-
import unittest

from liquiforge.train import SkillModel, train
from liquiforge.bench import run_all


class TestTrain(unittest.TestCase):
    def test_metrics_keys(self):
        m = SkillModel()
        out = m.metrics()
        for k in ["text_perplexity", "vision_accuracy", "audio_wer", "omni_retrieval", "decision_success"]:
            self.assertIn(k, out)

    def test_train_reaches_target(self):
        res = train(epochs=20, target=0.9)
        self.assertGreaterEqual(res["final"]["skill"], 0.85)


class TestBench(unittest.TestCase):
    def test_monotonic_improvement(self):
        r = run_all()
        # perplexity and WER should decrease; accuracy/recall/success should increase
        self.assertLess(r["text_perplexity"][1], r["text_perplexity"][0])
        self.assertLess(r["audio_wer"][1], r["audio_wer"][0])
        self.assertGreater(r["vision_accuracy"][1], r["vision_accuracy"][0])
        self.assertGreaterEqual(r["omni_retrieval"][1], r["omni_retrieval"][0])
        self.assertGreaterEqual(r["decision_success"][1], r["decision_success"][0])


if __name__ == "__main__":
    unittest.main()
