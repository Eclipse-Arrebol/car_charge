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


TRAIN_EPISODES = 30
TRAIN_STEPS = 600
TRAIN_NUM_EVS = 40
EVAL_EPISODES = 10
EVAL_STEPS = 600
EXPERIMENT_SEED = 0
GRID_COST_SCALE = 300.0
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def _build_train_cfg(reward_mode: str):
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
    cfg.reward_mode = reward_mode
    cfg.cheat_grid_cost_scale = GRID_COST_SCALE
    cfg.base_seed = EXPERIMENT_SEED
    cfg.train_scale = "step1"
    cfg.output_dir = os.path.join("runs", f"step1_{reward_mode}_seed0")
    cfg.checkpoint_basename = f"step1_{reward_mode}_seed0"
    return cfg


def _episode_seeds(base_seed: int, episodes: int):
    rng = random.Random(base_seed)
    return [rng.randint(0, 10000) for _ in range(episodes)]


def _build_eval_env(eval_cfg, seed):
    station_node_ids = resolve_station_nodes(eval_cfg.station_config_file, eval_cfg.station_id_key)
    return RealTrafficEnv(
        graphml_file=eval_cfg.graphml_file,
        num_stations=eval_cfg.num_stations,
        num_evs=eval_cfg.num_evs,
        max_nodes=eval_cfg.max_nodes,
        seed=seed,
        station_node_ids=station_node_ids,
        respawn_after_full_charge=eval_cfg.respawn_after_full_charge,
    )


def _evaluate_checkpoint(model_basename: str, eval_seed: int):
    eval_cfg = EvalConfig.ablation_l0()
    eval_cfg.episodes = EVAL_EPISODES
    eval_cfg.steps_per_episode = EVAL_STEPS
    eval_cfg.num_evs = 100
    eval_cfg.base_seed = eval_seed

    strategy = FedDQNStrategy(
        os.path.join(CHECKPOINT_DIR, f"{model_basename}.pth"),
        _build_eval_env(eval_cfg, seed=eval_seed),
    )
    evaluator = Evaluator()
    episode_seeds = _episode_seeds(eval_seed, eval_cfg.episodes)

    queue_means = []
    trip_means = []
    violation_step_rates = []
    reports = []

    for ep_seed in episode_seeds:
        env = _build_eval_env(eval_cfg, seed=ep_seed)
        evaluator.reset()
        per_step_queue = []
        per_step_trip = []
        violation_steps = 0

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
            if info.get("voltage_violations", 0) > 0:
                violation_steps += 1

        report = evaluator.report(env.evs, env.stations, verbose=False)
        report["queue_time_h_mean"] = mean(per_step_queue) if per_step_queue else 0.0
        report["trip_time_h_mean"] = mean(per_step_trip) if per_step_trip else 0.0
        report["voltage_violation_step_ratio"] = violation_steps / max(1, eval_cfg.steps_per_episode)
        reports.append(report)

    avg = {}
    for key in reports[0]:
        values = [r[key] for r in reports]
        avg[key] = sum(values) / len(values)
    return avg


def _pct_change(base, new):
    denom = max(abs(base), 1e-9)
    return (new - base) / denom * 100.0


def _print_compare_table(baseline, cheat):
    rows = [
        ("queue_time_h_mean", "queue_time_h_mean"),
        ("trip_time_h_mean", "trip_time_h_mean"),
        ("abandoned_evs", "abandoned_evs"),
        ("distribution_network_cost_cny", "distribution_network_cost_cny"),
        ("accumulated_voltage_excursion_pu", "accumulated_voltage_excursion_pu"),
        ("voltage_violation_step_ratio", "voltage_violation_step_ratio"),
        ("total_line_losses_kwh", "total_line_losses_kwh"),
    ]

    print("\n================================================================")
    print("Step 1 Compare: baseline vs cheat")
    print("================================================================")
    print(f"{'metric':<34} {'baseline':>12} {'cheat':>12} {'cheat vs base':>16}")
    print("-" * 78)
    for label, key in rows:
        b = float(baseline[key])
        c = float(cheat[key])
        pct = _pct_change(b, c)
        print(f"{label:<34} {b:>12.4f} {c:>12.4f} {pct:>15.2f}%")


def _save_compare_json(baseline, cheat):
    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "step1_compare_seed0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "grid_cost_scale": GRID_COST_SCALE,
            "seed": EXPERIMENT_SEED,
            "baseline": baseline,
            "cheat": cheat,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] compare json: {path}")


def train_one(reward_mode: str):
    cfg = _build_train_cfg(reward_mode)
    print(f"\n=== Train {reward_mode} ===")
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


def main():
    os.makedirs(RUN_ROOT, exist_ok=True)

    train_one("baseline")
    train_one("cheat")

    eval_seed = 0
    baseline_report = _evaluate_checkpoint("step1_baseline_seed0", eval_seed)
    cheat_report = _evaluate_checkpoint("step1_cheat_seed0", eval_seed)

    _print_compare_table(baseline_report, cheat_report)
    _save_compare_json(baseline_report, cheat_report)


if __name__ == "__main__":
    main()
