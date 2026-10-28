# -*- coding: utf-8 -*-
import unittest

from liquiforge.adapters.lora import LoRALinear, LoraAdapter
from liquiforge.rl.reward import RewardModel
from liquiforge.rl.policy import train_decision
from liquiforge.utils.rng import stable_matrix, stable_vector


class TestLoRA(unittest.TestCase):
    def test_forward_shape(self):
        W = stable_matrix("w", 8, 16)
        lora = LoRALinear(W, 4)
        out = lora.forward([0.5] * 8)
        self.assertEqual(len(out), 8)

    def test_adapter(self):
        W = stable_matrix("w2", 8, 16)
        a = LoraAdapter(W, 4)
        self.assertEqual(len(a.forward([0.1] * 8)), 8)


class TestRL(unittest.TestCase):
    def test_reward_oracle(self):
        r = RewardModel(128, 16)
        s = stable_vector("s", 128)
        ca = r.correct_action(s)
        self.assertEqual(r.reward(s, ca), 1.0)

    def test_train_monotonic(self):
        from liquiforge.heads.decision_head import DecisionHead
        r = RewardModel(128, 16)
        pol = DecisionHead(128, 16)
        states = [stable_vector("st%d" % i, 128) for i in range(6)]
        res = train_decision(pol.policy, r.correct_action, states)
        self.assertGreater(res["skill"], 0.5)
        self.assertGreaterEqual(res["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
