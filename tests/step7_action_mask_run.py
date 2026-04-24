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
TRAIN_EPISODES = 30
TRAIN_STEPS = 600
EVAL_EPISODES = 10
EVAL_STEPS = 600
EXPERIMENT_SEED = 0
GRID_COST_SCALE = 300.0
EVAL_VOLTAGE_THRESHOLD = 0.92
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

BASELINE_CHECKPOINT = "step5_baseline_seed0"
CHEAT_CHECKPOINT = "step5_cheat_seed0"
OLD_VOLTAGE_CHECKPOINT = "step6_voltage_seed0"
NEW_VOLTAGE_CHECKPOINT = "step6_voltage_penalty5_seed0"
MASK_CHECKPOINT = "step7_action_mask_seed0"


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
    cfg.num_evs = TRAIN_NUM_EVS
    cfg.episodes = TRAIN_EPISODES
    cfg.steps_per_episode = TRAIN_STEPS
    cfg.reward_mode = "voltage"
    cfg.cheat_grid_cost_scale = GRID_COST_SCALE
    cfg.voltage_user_weight = 0.3
    cfg.voltage_grid_weight = 0.7
    cfg.voltage_grid_norm_scale = 5.0
    cfg.voltage_abandon_penalty = 0.0
    cfg.base_seed = EXPERIMENT_SEED
    cfg.train_scale = "step7_action_mask"
    cfg.output_dir = os.path.join("runs", MASK_CHECKPOINT)
    cfg.checkpoint_basename = MASK_CHECKPOINT
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
    env.enable_queue_timeout_mask = True
    env.queue_timeout_mask_safety_margin_h = 3.5
    env.queue_timeout_mask_capacity_ratio = 1.5
    return env


def _count_eval_voltage_violations(bus_voltages):
    return sum(
        1
        for v_pu in bus_voltages.values()
        if float(v_pu) < EVAL_VOLTAGE_THRESHOLD
    )


def train_voltage_penalty():
    cfg = _build_train_cfg()
    print("\n=== Train voltage_action_mask ===")
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


def _pct_change(base, new):
    denom = max(abs(base), 1e-9)
    return (new - base) / denom * 100.0


def run_eval():
    eval_seed = EXPERIMENT_SEED
    reports = {
        "baseline": _evaluate_checkpoint(BASELINE_CHECKPOINT, eval_seed),
        "cheat": _evaluate_checkpoint(CHEAT_CHECKPOINT, eval_seed),
        "voltage": _evaluate_checkpoint(OLD_VOLTAGE_CHECKPOINT, eval_seed),
        "voltage_penalty5": _evaluate_checkpoint(NEW_VOLTAGE_CHECKPOINT, eval_seed),
        "voltage_mask": _evaluate_checkpoint(MASK_CHECKPOINT, eval_seed),
    }

    rows = [
        "abandoned_evs",
        "accumulated_voltage_excursion_pu",
        "total_line_losses_kwh",
        "trip_time_h_mean",
        "queue_time_h_mean",
        "distribution_network_cost_cny",
    ]
    print("\n================================================================================================================")
    print("Compare: baseline vs cheat vs voltage vs voltage_penalty5 vs voltage_mask")
    print("================================================================================================================")
    print(
        f"{'metric':<34} {'baseline':>11} {'cheat':>11} {'voltage':>11} {'penalty5':>11} {'mask':>11} "
        f"{'mask vs base':>14} {'mask vs penalty5':>18}"
    )
    print("-" * 146)
    for key in rows:
        base = float(reports["baseline"][key])
        cheat = float(reports["cheat"][key])
        voltage = float(reports["voltage"][key])
        penalty = float(reports["voltage_penalty5"][key])
        mask = float(reports["voltage_mask"][key])
        print(
            f"{key:<34} {base:>11.4f} {cheat:>11.4f} {voltage:>11.4f} {penalty:>11.4f} {mask:>11.4f} "
            f"{_pct_change(base, mask):>13.2f}% {_pct_change(penalty, mask):>17.2f}%"
        )

    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "step7_compare_with_action_mask.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": EXPERIMENT_SEED,
                "eval_num_evs": TRAIN_NUM_EVS,
                "eval_episodes": EVAL_EPISODES,
                "eval_voltage_threshold": EVAL_VOLTAGE_THRESHOLD,
                "penalty_config": {
                    "user_weight": 0.3,
                    "grid_weight": 0.7,
                    "grid_norm_scale": 5.0,
                    "abandon_penalty": 0.0,
                    "queue_timeout_mask": True,
                    "queue_timeout_mask_safety_margin_h": 0.5,
                },
                "reports": reports,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved] compare json: {path}")


def main():
    parser = argparse.ArgumentParser(description="Step 7 voltage action-mask runner")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train_voltage_penalty()
    elif args.mode == "eval":
        run_eval()


if __name__ == "__main__":
    main()
