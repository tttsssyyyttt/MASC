"""Test SEA collaboration: wrapper, budget, interface."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from meirp_project.core.config import EnvConfig
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.iql.agent import IQLAgent
from meirp_project.collaboration.base import SEABase
from meirp_project.collaboration.sea_wrapper import SEAWrapper
from meirp_project.collaboration.budget import BudgetManager

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ===== BudgetManager Tests =====
print("\n=== BudgetManager Tests ===")

bm = BudgetManager(total_budget=3, cooldown=2)
check("can intervene initially", bm.can_intervene(0))
check("budget used 0", bm.used == 0)

bm.record_intervention(0)
check("budget used 1", bm.used == 1)
check("cannot intervene same step (cooldown)", not bm.can_intervene(1))
check("can intervene after cooldown", bm.can_intervene(2))

bm.record_intervention(2)
bm.record_intervention(4)
check("budget exhausted", bm.used == 3)
check("cannot intervene when exhausted", not bm.can_intervene(6))

bm.reset()
check("budget reset", bm.used == 0)
check("can intervene after reset", bm.can_intervene(0))


# ===== SEABase Interface =====
print("\n=== SEABase Interface Tests ===")

class DummySEA(SEABase):
    """Simple SEA that always intervenes and returns fixed BDQ actions."""
    def should_intervene(self, obs, step):
        return True

    def get_expert_action(self, obs):
        n = obs.shape[0]
        return [0, [0], [0]]

    def fuse_actions(self, agent_actions, expert_actions, confidence):
        if confidence > 0.5:
            return expert_actions
        return agent_actions

sea = DummySEA()
check("dummy sea created", sea is not None)
check("should_intervene returns True", sea.should_intervene(np.zeros((3, 12)), 0))

expert_actions = sea.get_expert_action(np.zeros((3, 12)))
check("expert actions length", len(expert_actions) == 3)

agent_actions = [5, [5], [5]]
fused_high = sea.fuse_actions(agent_actions, expert_actions, confidence=0.8)
check("fuse with high confidence uses expert", fused_high[0] == 0)

fused_low = sea.fuse_actions(agent_actions, expert_actions, confidence=0.3)
check("fuse with low confidence uses agent", fused_low[0] == 5)


# ===== SEAWrapper Tests =====
print("\n=== SEAWrapper Tests ===")

cfg = EnvConfig(
    num_nodes=3, edges=[(0,1),(0,2)], layers=[0,1,1],
    H=[1,2,2], B=[5,4,4],
    max_order=10, lead_time=2, episode_len=20,
    demand_mean=5.0, seed=42,
)
obs_dim = 6 + 3 * cfg.lead_time

iql = IQLAgent(cfg=cfg, obs_dim=obs_dim, hidden_dim=32, buffer_capacity=100)

# Without SEA (use_sea=False)
wrapper_off = SEAWrapper(iql, sea=DummySEA(), use_sea=False)
obs = np.random.randn(3, obs_dim)
actions_off = wrapper_off.get_actions(obs)
check("wrapper off returns agent actions", len(actions_off) == 3)

# With SEA (use_sea=True)
wrapper_on = SEAWrapper(iql, sea=DummySEA(), use_sea=True)
actions_on = wrapper_on.get_actions(obs)
check("wrapper on returns actions", len(actions_on) == 3)
# Since DummySEA always intervenes with q=0 and confidence=0.5,
# fuse_actions uses agent actions (confidence 0.5 is NOT > 0.5)
# BDQ format: root -> int, non-root -> list[int]
valid = True
for a in actions_on:
    if isinstance(a, list):
        for q in a:
            if not (0 <= q <= cfg.max_order):
                valid = False
    else:
        if not (0 <= a <= cfg.max_order):
            valid = False
check("wrapper on actions valid", valid)

# Test update passthrough
metrics = wrapper_on.update()
check("wrapper update passthrough", isinstance(metrics, dict))

# Test save/load passthrough
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "sea_test.pt")
    wrapper_on.save(save_path)
    check("wrapper save succeeded", os.path.exists(save_path))

    iql2 = IQLAgent(cfg=cfg, obs_dim=obs_dim, hidden_dim=32, buffer_capacity=100)
    wrapper2 = SEAWrapper(iql2, use_sea=False)
    wrapper2.load(save_path)
    check("wrapper load succeeded", True)


# ===== Summary =====
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
