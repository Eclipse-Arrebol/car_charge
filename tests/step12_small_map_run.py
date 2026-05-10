"""
Step 12: EMA 标准路网实验
=========================
用 Eastern Massachusetts（74节点/129边）标准路网验证模型。
BPR 参数来自 TNTP（α=0.15, β=4），与环境硬编码一致，直接复用。

用法:
    python tests/step12_small_map_run.py --mode dry-run
    python tests/step12_small_map_run.py --mode train
    python tests/step12_small_map_run.py --mode eval
"""

import argparse
import json
import os
import random
import sys
import time
from statistics import mean

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.real_env import RealTrafficEnv
from env.tntp_loader import load_tntp_network
from evaluation.metrics import Evaluator
from evaluation.strategies import FedDQNStrategy, GreedyStrategy, RandomStrategy
from training.config import TrainConfig
from training.trainer import run_training_real

NET_FILE = os.path.join(PROJECT_ROOT, "map_outputs", "ema", "EMA_net.tntp")
GRAPHML_FILE = os.path.join(PROJECT_ROOT, "map_outputs", "ema", "ema.graphml")

NUM_STATIONS = 3
NUM_EVS = 30
EPISODES = 50
STEPS_PER_EPISODE = 500
EPSILON_FINAL = 0.60
BATCH_SIZE = 64
REWARD_MODE = "voltage"
VOLTAGE_USER_WEIGHT = 0.7
VOLTAGE_GRID_WEIGHT = 0.3
VOLTAGE_GRID_NORM_SCALE = 5.0
VOLTAGE_ABANDON_PENALTY = 0.0

EXPERIMENT_SEED = 0
EVAL_EPISODES = 10
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")


def _episode_seeds(base_seed, episodes):
    rng = random.Random(base_seed)
    return [rng.randint(0, 10000) for _ in range(episodes)]


def _build_env(seed, respawn=False):
    return RealTrafficEnv(
        graphml_file=GRAPHML_FILE,
        num_stations=NUM_STATIONS,
        num_evs=NUM_EVS,
        max_nodes=9999,
        seed=seed,
        station_node_ids=None,
        respawn_after_full_charge=respawn,
    )


def _build_cfg():
    return TrainConfig(
        num_evs=NUM_EVS,
        episodes=EPISODES,
        steps_per_episode=STEPS_PER_EPISODE,
        fed_rounds_per_episode=1,
        batch_size=BATCH_SIZE,
        step_local_train_steps=1,
        step_train_interval=2,
        fed_local_steps=4,
        epsilon_final=EPSILON_FINAL,
        checkpoint_interval=20,
        reward_mode=REWARD_MODE,
        voltage_user_weight=VOLTAGE_USER_WEIGHT,
        voltage_grid_weight=VOLTAGE_GRID_WEIGHT,
        voltage_grid_norm_scale=VOLTAGE_GRID_NORM_SCALE,
        voltage_abandon_penalty=VOLTAGE_ABANDON_PENALTY,
        num_stations=NUM_STATIONS,
        max_nodes=9999,
        graphml_file=GRAPHML_FILE,
        station_config_file=None,
        station_id_key="",
        graph_group="l0",
        train_scale="step12_ema",
        base_seed=EXPERIMENT_SEED,
        output_dir="runs/step12_ema_voltage",
        checkpoint_basename=f"step12_ema_voltage_seed{EXPERIMENT_SEED}",
    )


# ── dry-run ──────────────────────────────────────────────────
def run_dry_run():
    print("=" * 62)
    print(f"  Step 12 dry-run: EMA (74 nodes, 3 stations) [reward={REWARD_MODE}]")
    print("=" * 62)

    G, stations, _ = load_tntp_network(NET_FILE, num_stations=NUM_STATIONS, seed=EXPERIMENT_SEED)
    e0 = list(G.edges(data=True))[0]
    print(f"\n  [Edge sample] {e0[0]}->{e0[1]}: "
          f"length={e0[2].get('length', '?'):.0f}m, "
          f"speed={e0[2].get('speed_kph', '?'):.1f}kph, "
          f"cap={e0[2].get('capacity', '?'):.0f}veh/h")
    print(f"  [BPR] env uses α={0.15}, β={4.0} (matches TNTP)")

    env = _build_env(seed=EXPERIMENT_SEED)
    print(f"  [Env] nodes={env.num_nodes}, stations={env.station_node_ids}")

    reward_components = {
        "user_cost": [],
        "queue_cost": [],
        "grid_cost": [],
        "fluct_cost": [],
        "voltage_penalty": [],
    }
    total_decisions = 0

    for step_idx in range(50):
        urgent_evs = env.get_pending_decision_evs()
        actions = {}
        for ev in urgent_evs:
            mask = env.get_action_mask(ev)
            valid = [i for i in range(env.num_stations) if mask[0, i].item()]
            if valid:
                actions[ev.id] = random.choice(valid)
                total_decisions += 1

        _, reward, _, info = env.step(actions)
        terms = info.get("objective_terms", {})
        for k in reward_components:
            reward_components[k].append(terms.get(k, 0.0))

        if step_idx in (0, 10, 25, 49):
            bv = info.get("bus_voltages", {})
            min_v = min(bv.values()) if bv else 1.0
            print(f"  step={step_idx:>3}  reward={reward:>8.2f}  "
                  f"min_v={min_v:.4f}  decisions={total_decisions}")

    print(f"\n  [Reward breakdown] (50-step avg)")
    for k, vals in reward_components.items():
        print(f"    {k:<18} {mean(vals):>+10.2f}")

    print(f"\n  [Dry-run OK] 50 steps, {total_decisions} decisions")


# ── train ────────────────────────────────────────────────────
def run_train():
    cfg = _build_cfg()
    print("=" * 62)
    print(f"  Step 12 train: EMA, {EPISODES}ep × {STEPS_PER_EPISODE}step")
    print("=" * 62)

    t0 = time.time()
    run_training_real(
        num_evs=cfg.num_evs,
        episodes=cfg.episodes,
        steps_per_episode=cfg.steps_per_episode,
        fed_rounds_per_episode=cfg.fed_rounds_per_episode,
        batch_size=cfg.batch_size,
        step_local_train_steps=cfg.step_local_train_steps,
        step_train_interval=cfg.step_train_interval,
        proximal_mu=cfg.proximal_mu,
        use_dp=cfg.use_dp,
        dp_noise_multiplier=cfg.dp_noise_multiplier,
        dp_clip_C=cfg.dp_clip_C,
        epsilon_final=cfg.epsilon_final,
        checkpoint_interval=cfg.checkpoint_interval,
        mixed_reward_scale=cfg.mixed_reward_scale,
        mixed_reward_min=cfg.mixed_reward_min,
        mixed_reward_max=cfg.mixed_reward_max,
        reward_mode=cfg.reward_mode,
        voltage_user_weight=cfg.voltage_user_weight,
        voltage_grid_weight=cfg.voltage_grid_weight,
        voltage_grid_norm_scale=cfg.voltage_grid_norm_scale,
        voltage_abandon_penalty=cfg.voltage_abandon_penalty,
        graphml_file=cfg.graphml_file,
        station_config_file=cfg.station_config_file,
        station_id_key=cfg.station_id_key,
        max_nodes=cfg.max_nodes,
        graph_group=cfg.graph_group,
        train_scale=cfg.train_scale,
        base_seed=cfg.base_seed,
        output_dir=cfg.output_dir,
        checkpoint_basename=cfg.checkpoint_basename,
    )
    elapsed = time.time() - t0
    print(f"\n[Train done] {elapsed:.0f}s ({elapsed / 60:.1f}min)")


# ── eval ─────────────────────────────────────────────────────
def _evaluate_policy(policy, model_basename=None):
    episode_seeds = _episode_seeds(EXPERIMENT_SEED, EVAL_EPISODES)
    evaluator = Evaluator()
    reports = []

    for ep_seed in episode_seeds:
        env = _build_env(seed=ep_seed, respawn=False)

        if policy == "fed":
            model_path = os.path.join(CHECKPOINT_DIR, f"{model_basename}.pth")
            strategy = FedDQNStrategy(model_path, env)
        elif policy == "greedy":
            strategy = GreedyStrategy()
        else:
            strategy = RandomStrategy()

        evaluator.reset()
        per_step_queue = []
        per_step_trip = []
        reward_sum = 0.0

        for _ in range(STEPS_PER_EPISODE):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}
            for ev in urgent_evs:
                action = strategy.select_action(env, ev, pending_counts)
                actions[ev.id] = action
                pending_counts[action] += 1

            _, reward, _, info = env.step(actions)
            reward_sum += reward
            evaluator.update(info, env.power_grid, env.stations)

            for m in info.get("decision_costs", {}).values():
                per_step_queue.append(m.get("queue_time_h", 0.0))
                per_step_trip.append(m.get("trip_time_h", 0.0))

        report = evaluator.report(env.evs, env.stations, verbose=False)
        report["queue_time_h_mean"] = mean(per_step_queue) if per_step_queue else 0.0
        report["trip_time_h_mean"] = mean(per_step_trip) if per_step_trip else 0.0
        report["reward_sum"] = reward_sum
        reports.append(report)

    avg = {}
    for key in reports[0]:
        avg[key] = sum(r[key] for r in reports) / len(reports)
    return avg


def run_eval():
    model_basename = f"step12_ema_voltage_seed{EXPERIMENT_SEED}"
    print("=" * 62)
    print("  Step 12 eval: EMA [reward=voltage]")
    print("=" * 62)

    results = {}
    for policy in ["random", "greedy", "fed"]:
        print(f"\n  Evaluating {policy}...")
        results[policy] = _evaluate_policy(
            policy, model_basename if policy == "fed" else None
        )

    rows = [
        "avg_wait_time_min", "abandoned_evs",
        "trip_time_h_mean", "queue_time_h_mean",
        "accumulated_voltage_excursion_pu", "distribution_network_cost_cny",
        "reward_sum",
    ]

    print(f"\n{'=' * 78}")
    print(f"  Step 12 Results: EMA (74 nodes, 3 stations)")
    print(f"{'=' * 78}")
    print(f"{'metric':<45} {'Random':>10} {'Greedy':>10} {'FedDQN':>10}")
    print("-" * 78)
    for metric in rows:
        vals = [float(results[p].get(metric, 0.0)) for p in ["random", "greedy", "fed"]]
        print(f"{metric:<45} {vals[0]:>10.2f} {vals[1]:>10.2f} {vals[2]:>10.2f}")

    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "step12_ema_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "map": "Eastern Massachusetts",
            "nodes": 74,
            "stations": NUM_STATIONS,
            "evs": NUM_EVS,
            "voltage_user_weight": VOLTAGE_USER_WEIGHT,
            "voltage_grid_weight": VOLTAGE_GRID_WEIGHT,
            "seed": EXPERIMENT_SEED,
            "eval_episodes": EVAL_EPISODES,
            "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {path}")


# ── main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Step 12: EMA 标准路网实验")
    parser.add_argument("--mode", choices=["dry-run", "train", "eval"], required=True)
    args = parser.parse_args()

    if args.mode == "dry-run":
        run_dry_run()
    elif args.mode == "train":
        run_train()
    elif args.mode == "eval":
        run_eval()


if __name__ == "__main__":
    main()
