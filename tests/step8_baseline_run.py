import argparse
import json
import os
import random
import sys
from statistics import mean

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from evaluation.metrics import Evaluator
from evaluation.strategies import FedDQNStrategy
from training.config import EvalConfig, TrainConfig
from training.trainer import run_training_real


TRAIN_NUM_EVS = 40
TRAIN_EPISODES = 50
TRAIN_STEPS = 1200
EVAL_EPISODES = 10
EVAL_STEPS = 600
EXPERIMENT_SEED = 0
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
BASELINE_CHECKPOINT = "step8_baseline_50ep_seed0"


def _episode_seeds(base_seed: int, episodes: int):
    rng = random.Random(base_seed)
    return [rng.randint(0, 10000) for _ in range(episodes)]


def _build_train_cfg():
    cfg = TrainConfig.ablation_l0()
    scale = TrainConfig.ablation()
    for attr in [
        "num_evs",
        "episodes",
        "steps_per_episode",
        "fed_rounds_per_episode",
        "batch_size",
        "step_local_train_steps",
        "step_train_interval",
        "fed_local_steps",
        "epsilon_final",
        "checkpoint_interval",
    ]:
        setattr(cfg, attr, getattr(scale, attr))
    cfg.enable_queue_timeout_mask = False  # baseline 不用 mask
    cfg.num_evs = TRAIN_NUM_EVS
    cfg.episodes = TRAIN_EPISODES
    cfg.steps_per_episode = TRAIN_STEPS
    cfg.epsilon_final = 0.05
    cfg.reward_mode = "baseline"
    cfg.base_seed = EXPERIMENT_SEED
    cfg.train_scale = "step8_baseline_50ep_1200step"
    cfg.output_dir = os.path.join("runs", BASELINE_CHECKPOINT)
    cfg.checkpoint_basename = BASELINE_CHECKPOINT
    return cfg


def _build_eval_cfg(eval_seed: int):
    cfg = EvalConfig.ablation_l0()
    cfg.episodes = EVAL_EPISODES
    cfg.steps_per_episode = EVAL_STEPS
    cfg.num_evs = TRAIN_NUM_EVS
    cfg.base_seed = eval_seed
    return cfg


def _build_env(cfg, seed):
    station_node_ids = resolve_station_nodes(cfg.station_config_file, cfg.station_id_key)
    env = RealTrafficEnv(
        graphml_file=cfg.graphml_file,
        num_stations=cfg.num_stations,
        num_evs=cfg.num_evs,
        max_nodes=cfg.max_nodes,
        seed=seed,
        station_node_ids=station_node_ids,
        respawn_after_full_charge=getattr(cfg, "respawn_after_full_charge", True),
    )
    env.enable_queue_timeout_mask = False
    return env


def train_baseline():
    cfg = _build_train_cfg()
    print("\n=== Train baseline ===")
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
        cheat_grid_cost_scale=cfg.cheat_grid_cost_scale,
        voltage_user_weight=cfg.voltage_user_weight,
        voltage_grid_weight=cfg.voltage_grid_weight,
        voltage_grid_norm_scale=cfg.voltage_grid_norm_scale,
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


def _evaluate_checkpoint(model_basename: str, eval_seed: int):
    eval_cfg = _build_eval_cfg(eval_seed)
    strategy = FedDQNStrategy(
        os.path.join(CHECKPOINT_DIR, f"{model_basename}.pth"),
        _build_env(eval_cfg, seed=eval_seed),
    )
    evaluator = Evaluator()
    episode_seeds = _episode_seeds(eval_seed, eval_cfg.episodes)
    reports = []

    for ep_seed in episode_seeds:
        env = _build_env(eval_cfg, seed=ep_seed)
        evaluator.reset()
        per_step_queue = []
        per_step_trip = []

        for _ in range(eval_cfg.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}
            for ev in urgent_evs:
                action = strategy.select_action(env, ev, pending_counts)
                actions[ev.id] = action
                pending_counts[action] += 1

            _, _, _, info = env.step(actions)
            evaluator.update(info, env.power_grid, env.stations)

            decision_costs = list(info.get("decision_costs", {}).values())
            if decision_costs:
                per_step_queue.append(mean(m.get("queue_time_h", 0.0) for m in decision_costs))
                per_step_trip.append(mean(m.get("trip_time_h", 0.0) for m in decision_costs))
            _count_eval_voltage_violations(info.get("bus_voltages", {}))

        report = evaluator.report(env.evs, env.stations, verbose=False)
        report["queue_time_h_mean"] = mean(per_step_queue) if per_step_queue else 0.0
        report["trip_time_h_mean"] = mean(per_step_trip) if per_step_trip else 0.0
        reports.append(report)

    avg = {}
    for key in reports[0]:
        values = [r[key] for r in reports]
        avg[key] = sum(values) / len(values)
    return avg


def run_eval():
    eval_seed = EXPERIMENT_SEED
    reports = {
        "baseline": _evaluate_checkpoint(BASELINE_CHECKPOINT, eval_seed),
    }
    rows = [
        "abandoned_evs",
        "accumulated_voltage_excursion_pu",
        "total_line_losses_kwh",
        "trip_time_h_mean",
        "queue_time_h_mean",
        "distribution_network_cost_cny",
    ]
    print("\n=== Baseline (epsilon_final=0.05, 50 ep × 1200 step) ===")
    for key in rows:
        val = float(reports["baseline"][key])
        print(f"{key:<40} {val:>12.4f}")

    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "step8_baseline_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": EXPERIMENT_SEED,
                "eval_num_evs": TRAIN_NUM_EVS,
                "eval_episodes": EVAL_EPISODES,
                "config": {
                    "epsilon_final": 0.05,
                    "episodes": TRAIN_EPISODES,
                    "steps_per_episode": TRAIN_STEPS,
                    "reward_mode": "baseline",
                    "queue_timeout_mask": False,
                },
                "reports": reports,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved] eval json: {path}")


def main():
    parser = argparse.ArgumentParser(description="Step 8 baseline runner")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="train: train baseline checkpoint; eval: evaluate baseline checkpoint",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_baseline()
        return
    if args.mode == "eval":
        run_eval()


if __name__ == "__main__":
    main()
