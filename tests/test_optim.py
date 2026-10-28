# -*- coding: utf-8 -*-
import math
import unittest

from liquiforge.optim.adamw import AdamW
from liquiforge.optim.sgd import SGD, CosineScheduler


class TestOptim(unittest.TestCase):
    def test_adamw_step(self):
        params = [[[1.0, 2.0]]]
        grads = [[[0.1, 0.2]]]
        opt = AdamW(params, lr=0.01)
        opt.step(grads)
        self.assertLess(params[0][0][0], 1.0)

    def test_sgd_step(self):
        params = [[[1.0]]]
        grads = [[[0.5]]]
        opt = SGD(params, lr=0.1)
        opt.step(grads)
        self.assertAlmostEqual(params[0][0][0], 0.95, places=6)

    def test_cosine_schedule(self):
        sched = CosineScheduler(0.1, 100)
        self.assertAlmostEqual(sched(0), 0.1, places=6)
        self.assertAlmostEqual(sched(100), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
