# -*- coding: utf-8 -*-
"""Decision: fine-tune the policy head with policy-gradient RL."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquiforge.train import SkillModel
from liquiforge.rl.policy import train_decision

m = SkillModel()
res = train_decision(m.decision_head.policy, m.reward.correct_action, m.decision_states)
print("skill        : %.4f" % res["skill"])
print("success rate : %.4f" % res["success_rate"])
print("history (first/last):", res["history"][0], "->", res["history"][-1])
