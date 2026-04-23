import argparse
import json
import os
import random
import sys
from statistics import mean, median

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
from training.trainer import (
    GRID_WEIGHT,
    GRID_NORM_SCALE,
    USER_WEIGHT,
    VOLTAGE_THRESHOLD,
    run_training_real,
)


TRAIN_EPISODES = 30
TRAIN_NUM_EVS = 40
DRY_RUN_EPISODES = 3
EVAL_EPISODES = 10
EVAL_STEPS = 600
EXPERIMENT_SEED = 0
GRID_COST_SCALE = 300.0
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


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
    cfg.reward_mode = "voltage"
    cfg.cheat_grid_cost_scale = GRID_COST_SCALE
    cfg.base_seed = EXPERIMENT_SEED
    cfg.train_scale = "step2"
    cfg.output_dir = os.path.join("runs", "step2_voltage_seed0")
    cfg.checkpoint_basename = "step2_voltage_seed0"
    return cfg


def _build_eval_cfg(eval_seed: int):
    cfg = EvalConfig.ablation_l0()
    cfg.episodes = EVAL_EPISODES
    cfg.steps_per_episode = EVAL_STEPS
    cfg.num_evs = TRAIN_NUM_EVS
    cfg.base_seed = eval_seed
    return cfg


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


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summary(values):
    if not values:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": median(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _print_summary(name, values):
    s = _summary(values)
    print(
        f"{name:<18} min={s['min']:.6f} median={s['median']:.6f} "
        f"p90={s['p90']:.6f} max={s['max']:.6f}"
    )
    return s


def run_voltage_dry_run():
    cfg = _build_train_cfg()
    rng = random.Random(EXPERIMENT_SEED)
    episode_seeds = _episode_seeds(EXPERIMENT_SEED, DRY_RUN_EPISODES)
    user_norms = []
    grid_norms = []

    print("Step 2 Part 2: voltage reward dry-run with random policy")
    print(
        f"episodes={DRY_RUN_EPISODES}, steps_per_episode={cfg.steps_per_episode}, "
        f"num_evs={cfg.num_evs}, threshold={VOLTAGE_THRESHOLD}, "
        f"grid_norm_scale={GRID_NORM_SCALE}"
    )

    for ep_idx, ep_seed in enumerate(episode_seeds, start=1):
        env = RealTrafficEnv(
            graphml_file=cfg.graphml_file,
            num_stations=cfg.num_stations,
            num_evs=cfg.num_evs,
            max_nodes=cfg.max_nodes,
            seed=ep_seed,
            station_node_ids=resolve_station_nodes(cfg.station_config_file, cfg.station_id_key),
        )
        decisions = 0

        for _ in range(cfg.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                action_mask = env.get_action_mask(ev)
                valid = [i for i in range(len(env.stations)) if action_mask[0, i].item()]
                action = rng.choice(valid) if valid else 0
                station = env.stations[action]
                metrics = env.estimate_action_metrics(ev, action, pending_counts)
                pred = env._estimate_ev_station_metrics(ev, station)
                est_voltage = pred.get("estimated_voltage_pu_after", None)

                user_norm = (metrics["queue_time_h"] + metrics["trip_time_h"]) / 2.0
                grid_norm = 0.0
                if est_voltage is not None:
                    est_voltage_excursion = max(0.0, VOLTAGE_THRESHOLD - float(est_voltage))
                    grid_norm = est_voltage_excursion * GRID_NORM_SCALE

                user_norms.append(user_norm)
                grid_norms.append(grid_norm)
                actions[ev.id] = action
                pending_counts[action] += 1
                decisions += 1

            env.step(actions)

        print(f"[DryRun] episode={ep_idx} decisions={decisions}")

    print("\nReward component distributions:")
    user_summary = _print_summary("user_norm", user_norms)
    grid_summary = _print_summary("grid_norm_scaled", grid_norms)
    user_weighted_p90 = USER_WEIGHT * user_summary["p90"]
    grid_weighted_p90 = GRID_WEIGHT * grid_summary["p90"]
    print(
        f"weighted p90: {USER_WEIGHT:.1f}*user_norm={user_weighted_p90:.6f}, "
        f"{GRID_WEIGHT:.1f}*grid_norm={grid_weighted_p90:.6f}"
    )

    larger = max(user_weighted_p90, grid_weighted_p90)
    smaller = min(user_weighted_p90, grid_weighted_p90)
    ratio = larger / max(smaller, 1e-9)
    print(f"weighted p90 ratio={ratio:.3f}x")

    if ratio <= 2.0:
        print("[Decision] weighted p90 values are aligned within 2x; proceed to Part 3.")
    elif ratio > 3.0:
        print(
            "[Decision] weighted p90 ratio > 3x; stop before Part 3 and tune GRID_NORM_SCALE."
        )
    else:
        print("[Decision] weighted p90 ratio is between 2x and 3x; please confirm before Part 3.")


def train_voltage():
    cfg = _build_train_cfg()
    print("\n=== Train voltage ===")
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


def _evaluate_checkpoint(model_basename: str, eval_seed: int):
    eval_cfg = _build_eval_cfg(eval_seed)
    strategy = FedDQNStrategy(
        os.path.join(CHECKPOINT_DIR, f"{model_basename}.pth"),
        _build_eval_env(eval_cfg, seed=eval_seed),
    )
    episode_seeds = _episode_seeds(eval_seed, eval_cfg.episodes)
    evaluator = Evaluator()
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
    return (new - base) / max(abs(base), 1e-9) * 100.0


def _ceiling_capture(base, cheat, voltage):
    denom = base - cheat
    if abs(denom) < 1e-9:
        return None
    return (base - voltage) / denom * 100.0


def _print_three_way_table(reports):
    rows = [
        "queue_time_h_mean",
        "trip_time_h_mean",
        "abandoned_evs",
        "distribution_network_cost_cny",
        "accumulated_voltage_excursion_pu",
        "voltage_violation_step_ratio",
        "total_line_losses_kwh",
    ]
    print("\n================================================================================================")
    print("Step 2 Compare: baseline vs cheat vs voltage")
    print("================================================================================================")
    print(
        f"{'metric':<36} {'baseline':>11} {'cheat':>11} {'voltage':>11} "
        f"{'voltage vs base':>16} {'voltage vs cheat':>17}"
    )
    print("-" * 96)
    for key in rows:
        base = float(reports["baseline"][key])
        cheat = float(reports["cheat"][key])
        voltage = float(reports["voltage"][key])
        print(
            f"{key:<36} {base:>11.4f} {cheat:>11.4f} {voltage:>11.4f} "
            f"{_pct_change(base, voltage):>15.2f}% {_pct_change(cheat, voltage):>16.2f}%"
        )


def _build_capture_report(reports):
    capture = {}
    for key in [
        "distribution_network_cost_cny",
        "accumulated_voltage_excursion_pu",
        "voltage_violation_step_ratio",
        "total_line_losses_kwh",
    ]:
        capture[key] = _ceiling_capture(
            float(reports["baseline"][key]),
            float(reports["cheat"][key]),
            float(reports["voltage"][key]),
        )
    return capture


def run_eval():
    eval_seed = EXPERIMENT_SEED
    reports = {
        "baseline": _evaluate_checkpoint("step1_baseline_seed0", eval_seed),
        "cheat": _evaluate_checkpoint("step1_cheat_seed0", eval_seed),
        "voltage": _evaluate_checkpoint("step2_voltage_seed0", eval_seed),
    }
    _print_three_way_table(reports)

    capture = _build_capture_report(reports)
    print("\nVoltage-to-cheat ceiling capture on grid metrics:")
    for key, value in capture.items():
        text = "n/a" if value is None else f"{value:.2f}%"
        print(f"{key:<36} {text:>10}")

    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "step2_compare_seed0.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": EXPERIMENT_SEED,
                "eval_num_evs": TRAIN_NUM_EVS,
                "reward_weights": {
                    "user_weight": USER_WEIGHT,
                    "grid_weight": GRID_WEIGHT,
                    "voltage_threshold": VOLTAGE_THRESHOLD,
                    "grid_norm_scale": GRID_NORM_SCALE,
                },
                "reports": reports,
                "voltage_to_cheat_ceiling_capture_pct": capture,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved] compare json: {path}")


def main():
    parser = argparse.ArgumentParser(description="Step 2 voltage reward experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Run Part 2 random-policy signal check.")
    group.add_argument("--train-voltage", action="store_true", help="Run Part 3 voltage training.")
    group.add_argument("--eval", action="store_true", help="Run Part 4 three-way evaluation.")
    args = parser.parse_args()

    if args.dry_run:
        run_voltage_dry_run()
    elif args.train_voltage:
        train_voltage()
    elif args.eval:
        run_eval()


if __name__ == "__main__":
    main()
