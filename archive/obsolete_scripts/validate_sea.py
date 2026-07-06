"""SEA validation: IQL + BaseStock(algo expert) + MILP(human expert proxy).

Validates:
  1. Three-level escalation mechanism works end-to-end
  2. SPC correctly detects anomalous states
  3. BaseStock filters most alarms, MILP handles only extreme cases
  4. SEA-IQL outperforms Pure IQL and Pure BaseStock

Usage:
    set PYTHONIOENCODING=utf-8 && python meirp_project/scripts/validate_sea.py
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from meirp_project.core.config import from_yaml
from meirp_project.core.env import MEIRPEnv
from meirp_project.algorithms.registry import make_agent
from meirp_project.baselines.base_stock import BaseStockPolicy
from meirp_project.evaluation.metrics import compute_metrics
from meirp_project.collaboration.escalation import EscalationController
from meirp_project.collaboration.milp_advisor import MILPAdvisor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "configs")


# ============================================================
# 训练函数 (带SEA escalation)
# ============================================================
def train_sea(agent, cfg, escalation, n_ep=200, warmup=500, label=""):
    """Train RL agent with SEA escalation controller."""
    env = MEIRPEnv(cfg)
    rewards = []
    steps = 0

    for ep in range(n_ep):
        obs, _ = env.reset()
        escalation.reset_episode()
        ep_r = 0.0
        done = False

        while not done:
            # RL proposes actions
            rl_actions = agent.get_actions(obs)

            # SEA escalation decides final action
            final_actions, level, log_entry = escalation.step(
                obs=obs,
                rl_actions=rl_actions,
                env_inventory=env.inventory,
                env_backlog=env.backlog,
                env_pipeline=env.pipeline,
                env_demand=env.demand,
                fill_rates=np.ones(cfg.num_nodes),  # placeholder
            )

            # Step environment with SEA-chosen action
            nobs, r, term, trunc, info = env.step(final_actions)
            done = term or trunc

            # RL agent learns from the executed action (regardless of source)
            agent.store_transition(obs, final_actions, r, nobs,
                                   np.full(cfg.num_nodes, float(term)))
            steps += 1

            # Update RL (off-policy: per-step after warmup)
            if hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= warmup:
                    agent.update()

            ep_r += r.sum()
            obs = nobs

        rewards.append(ep_r)
        if (ep + 1) % 50 == 0:
            stats = escalation.get_escalation_stats()
            print(f"  [{label}] Ep {ep+1}/{n_ep} | "
                  f"R={np.mean(rewards[-50:]):.1f} | "
                  f"SPC={stats['spc_alarm_rate']:.3f} | "
                  f"Algo={stats['algo_rate']:.3f} | "
                  f"Human={stats['human_rate']:.3f}")

    return rewards


def train_pure_rl(agent, cfg, n_ep=200, warmup=500, label=""):
    """Standard RL training without SEA."""
    env = MEIRPEnv(cfg)
    rewards = []
    steps = 0

    for ep in range(n_ep):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            actions = agent.get_actions(obs)
            nobs, r, term, trunc, info = env.step(actions)
            done = term or trunc
            agent.store_transition(obs, actions, r, nobs,
                                   np.full(cfg.num_nodes, float(term)))
            steps += 1
            if hasattr(agent, 'batch_size') and len(agent.buffer) >= agent.batch_size:
                if steps >= warmup:
                    agent.update()
            ep_r += r.sum()
            obs = nobs
        rewards.append(ep_r)
        if (ep + 1) % 50 == 0:
            print(f"  [{label}] Ep {ep+1}/{n_ep} | R={np.mean(rewards[-50:]):.1f}")
    return rewards


# ============================================================
# 评估函数
# ============================================================
def evaluate_agent(agent, cfg, n_seeds=3, escalation=None, is_baseline=False):
    """Evaluate agent (optionally with SEA escalation)."""
    all_m = []
    for seed in range(n_seeds):
        env = MEIRPEnv(cfg)
        obs, _ = env.reset(seed=seed)
        if escalation:
            escalation.reset_episode()
        info_h = []
        done = False
        while not done:
            if is_baseline == "milp":
                actions = agent.get_action(
                    env.inventory, env.backlog, env.pipeline
                )
            elif is_baseline:
                # BaseStock/SSPolicy: get_actions(env)
                actions = agent.get_actions(env)
            else:
                # RL agent: get_actions(obs, deterministic=True)
                actions = agent.get_actions(obs, deterministic=True)

            if escalation and not is_baseline:
                fr = np.ones(cfg.num_nodes)
                if info_h:
                    fr = info_h[-1].get("fill_rates", np.ones(cfg.num_nodes))
                actions, level, _ = escalation.step(
                    obs, actions, env.inventory, env.backlog,
                    env.pipeline, env.demand, fr
                )
            obs, rewards, term, trunc, info = env.step(actions)
            done = term or trunc
            info_h.append(info)
        all_m.append(compute_metrics(info_h, cfg))

    result = {}
    for k in ["avg_cost_per_step", "avg_fill_rate", "avg_stockout_rate",
              "avg_bullwhip_cv", "avg_inventory_turnover",
              "total_holding", "total_backlog", "total_cost"]:
        vals = [m.get(k, 0) for m in all_m]
        result[k] = np.mean(vals)
        result[f"{k}_std"] = np.std(vals)
    return result


# ============================================================
# 主函数
# ============================================================
def main():
    # Load config
    cfg = from_yaml(os.path.join(CFG_DIR, "env_3layers.yaml"))
    obs_dim = 3 + 3 * cfg.lead_time

    print("=" * 80)
    print("  SEA Framework Validation")
    print(f"  Topology: {cfg.num_nodes} nodes, {len(cfg.edges)} edges")
    print(f"  Algo Expert: BaseStock | Human Expert (proxy): MILP")
    print("=" * 80)

    n_ep = 200  # 训练轮数

    # ================ Baselines ================
    print("\n--- Baselines ---")
    bs = BaseStockPolicy(cfg)
    bs_result = evaluate_agent(bs, cfg, n_seeds=5, escalation=None, is_baseline=True)

    milp = MILPAdvisor(cfg)
    milp_result = evaluate_agent(milp, cfg, n_seeds=5, escalation=None, is_baseline="milp")

    print(f"  BaseStock:      cost/step={bs_result['avg_cost_per_step']:.1f}  "
          f"fill={bs_result['avg_fill_rate']:.4f}  bullwhip={bs_result['avg_bullwhip_cv']:.4f}")
    print(f"  MILP (proxy human): cost/step={milp_result['avg_cost_per_step']:.1f}  "
          f"fill={milp_result['avg_fill_rate']:.4f}  bullwhip={milp_result['avg_bullwhip_cv']:.4f}")

    # ================ Pure IQL ================
    print("\n--- Pure IQL (no SEA) ---")
    iql_pure = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                          hidden_dim=128, lr=1e-3, gamma=0.99,
                          buffer_capacity=10000, batch_size=32,
                          epsilon_decay=5000, target_update_freq=50)

    t0 = time.time()
    _ = train_pure_rl(iql_pure, cfg, n_ep=n_ep, label="Pure IQL")
    pure_result = evaluate_agent(iql_pure, cfg, n_seeds=5)
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"  Pure IQL:       cost/step={pure_result['avg_cost_per_step']:.1f}  "
          f"fill={pure_result['avg_fill_rate']:.4f}  bullwhip={pure_result['avg_bullwhip_cv']:.4f}")

    # ================ SEA-IQL ================
    print("\n--- SEA-IQL (BaseStock + MILP escalation) ---")
    iql_sea = make_agent("iql", cfg, obs_dim=obs_dim, device='cuda',
                         hidden_dim=128, lr=1e-3, gamma=0.99,
                         buffer_capacity=10000, batch_size=32,
                         epsilon_decay=5000, target_update_freq=50)

    escalation = EscalationController(cfg, k_spc=3.0)

    t0 = time.time()
    _ = train_sea(iql_sea, cfg, escalation, n_ep=n_ep, label="SEA-IQL")
    sea_result = evaluate_agent(iql_sea, cfg, n_seeds=5, escalation=escalation)
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"  SEA-IQL:        cost/step={sea_result['avg_cost_per_step']:.1f}  "
          f"fill={sea_result['avg_fill_rate']:.4f}  bullwhip={sea_result['avg_bullwhip_cv']:.4f}")

    # ================ Escalation Statistics ================
    stats = escalation.get_escalation_stats()
    print(f"\n--- Escalation Statistics ({stats['total_steps']} total steps) ---")
    print(f"  SPC alarm rate:    {stats['spc_alarm_rate']:.3%}")
    print(f"  Algo (BaseStock):  {stats['algo_rate']:.3%}")
    print(f"  Human (MILP):      {stats['human_rate']:.3%}")
    print(f"  RL autonomous:     {stats['rl_rate']:.3%}")
    print(f"  Total alarms:      {stats['total_alarms']}")
    print(f"  Algo interventions: {stats['algo_interventions']}")
    print(f"  Human interventions:{stats['human_interventions']}")

    # ================ Summary ================
    print(f"\n{'='*80}")
    print("  Results Summary")
    print(f"{'='*80}")
    print(f"  {'Method':<25} {'Cost/Step':>10} {'Fill Rate':>10} {'Bullwhip':>10}")
    print(f"  {'-'*55}")
    print(f"  {'BaseStock':<25} {bs_result['avg_cost_per_step']:>10.1f} "
          f"{bs_result['avg_fill_rate']:>10.4f} {bs_result['avg_bullwhip_cv']:>10.4f}")
    print(f"  {'MILP (human proxy)':<25} {milp_result['avg_cost_per_step']:>10.1f} "
          f"{milp_result['avg_fill_rate']:>10.4f} {milp_result['avg_bullwhip_cv']:>10.4f}")
    print(f"  {'Pure IQL':<25} {pure_result['avg_cost_per_step']:>10.1f} "
          f"{pure_result['avg_fill_rate']:>10.4f} {pure_result['avg_bullwhip_cv']:>10.4f}")
    print(f"  {'SEA-IQL (ours)':<25} {sea_result['avg_cost_per_step']:>10.1f} "
          f"{sea_result['avg_fill_rate']:>10.4f} {sea_result['avg_bullwhip_cv']:>10.4f}")

    # Check if SEA beats both baselines
    better_than_pure = sea_result['avg_fill_rate'] > pure_result['avg_fill_rate']
    better_than_bs = sea_result['avg_fill_rate'] > bs_result['avg_fill_rate']
    lower_bullwhip = sea_result['avg_bullwhip_cv'] < pure_result['avg_bullwhip_cv']

    print(f"\n  SEA fill > Pure IQL: {better_than_pure}")
    print(f"  SEA fill > BaseStock: {better_than_bs}")
    print(f"  SEA bullwhip < Pure IQL: {lower_bullwhip}")

    print(f"\n{'='*80}")
    print("  Validation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
