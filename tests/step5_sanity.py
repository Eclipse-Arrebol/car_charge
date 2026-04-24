import argparse
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.graph_mapping import resolve_station_nodes
from env.power_grid_pp import PPPowerGrid33
from env.real_env import RealTrafficEnv
from evaluation.strategies import FedDQNStrategy, GreedyStrategy, RandomStrategy
from training.config import EvalConfig


DEFAULT_MODEL = "step5_baseline_seed0.pth"
DEFAULT_EPISODES = 3
DEFAULT_STEPS = 600
DEFAULT_NUM_EVS = 40
EXPERIMENT_SEED = 0
VOLTAGE_THRESHOLD = 0.95
LINE_LOSS_SNIPPET_START = 180
LINE_LOSS_SNIPPET_END = 196


def _episode_seeds(base_seed: int, episodes: int):
    rng = random.Random(base_seed)
    return [rng.randint(0, 10000) for _ in range(episodes)]


def _build_eval_cfg(num_evs: int):
    cfg = EvalConfig.ablation_l0()
    cfg.episodes = DEFAULT_EPISODES
    cfg.steps_per_episode = DEFAULT_STEPS
    cfg.num_evs = num_evs
    cfg.base_seed = EXPERIMENT_SEED
    return cfg


def _build_env(cfg, seed):
    station_node_ids = resolve_station_nodes(cfg.station_config_file, cfg.station_id_key)
    return RealTrafficEnv(
        graphml_file=cfg.graphml_file,
        num_stations=cfg.num_stations,
        num_evs=cfg.num_evs,
        max_nodes=cfg.max_nodes,
        seed=seed,
        station_node_ids=station_node_ids,
        respawn_after_full_charge=cfg.respawn_after_full_charge,
    )


def _build_strategy(policy: str, model_path: str, env):
    if policy == "baseline":
        return FedDQNStrategy(model_path, env)
    if policy == "greedy":
        return GreedyStrategy()
    if policy == "random":
        return RandomStrategy()
    raise ValueError(f"Unsupported policy: {policy}")


def _step_voltage_summary(bus_voltages):
    if not bus_voltages:
        return 1.0, 0, 0.0, []
    items = [(bus, float(v)) for bus, v in bus_voltages.items()]
    min_bus, min_v = min(items, key=lambda item: item[1])
    violating = [(bus, v) for bus, v in items if v < VOLTAGE_THRESHOLD]
    excursion_sum = sum(max(0.0, VOLTAGE_THRESHOLD - v) for _, v in items)
    return min_bus, min_v, len(violating), excursion_sum, violating


def _print_empty_load_reference():
    print("\n================================================================")
    print("Empty-Load Reference (No EV load added)")
    print("================================================================")
    grid = PPPowerGrid33()
    bus_voltages = grid.run_power_flow({})
    min_bus, min_v, violating_count, excursion_sum, violating = _step_voltage_summary(bus_voltages)
    print(f"min_v_pu={min_v:.6f} at {min_bus}")
    print(f"violating_bus_count={violating_count}")
    print(f"sum_voltage_excursions={excursion_sum:.6f}")
    if violating:
        formatted = ", ".join(f"{bus}={v:.6f}" for bus, v in violating)
        print(f"violating_buses={formatted}")
    else:
        print("violating_buses=NONE")


def _print_line_loss_snippet():
    print("\n================================================================")
    print("Evaluation Line-Loss Accumulation Code")
    print("================================================================")
    path = os.path.join(PROJECT_ROOT, "evaluation", "metrics.py")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for lineno in range(LINE_LOSS_SNIPPET_START, LINE_LOSS_SNIPPET_END + 1):
        text = lines[lineno - 1].rstrip("\n")
        print(f"{lineno:>4}: {text}")


def _print_step_line_loss_diagnosis(env, info):
    print("\n[LineLoss Diagnosis]")
    print("pandapower net.res_line.pl_mw head:")
    head = env.power_grid.net.res_line[["pl_mw"]].head()
    for idx, row in head.iterrows():
        line_name = env.power_grid.net.line.at[idx, "name"]
        print(f"  {line_name:<12} pl_mw={float(row.pl_mw):.6f} MW")

    pl_mw_total = float(env.power_grid.net.res_line.pl_mw.sum())
    info_line_losses = float(info.get("line_losses", 0.0))
    step_duration_h = float(getattr(env, "step_duration_h", 0.0))
    print(f"sum(res_line.pl_mw)={pl_mw_total:.6f} MW")
    print(f"info['line_losses']={info_line_losses:.6f}")
    print(f"env.step_duration_h={step_duration_h:.6f} h")
    print(f"expected_step_loss_kW_from_pp={pl_mw_total * 1000.0:.6f} kW")
    print(f"expected_step_loss_kWh_if_integrated={pl_mw_total * 1000.0 * step_duration_h:.6f} kWh")
    print(
        "note: evaluation currently accumulates info['line_losses'] directly into "
        "total_line_losses_kwh without multiplying by step_duration_h"
    )


def run_sanity(policy: str, model_name: str, episodes: int, steps: int, num_evs: int):
    cfg = _build_eval_cfg(num_evs)
    cfg.episodes = episodes
    cfg.steps_per_episode = steps
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    episode_seeds = _episode_seeds(EXPERIMENT_SEED, episodes)

    _print_empty_load_reference()
    _print_line_loss_snippet()

    print("\n================================================================")
    print("Step 5 Sanity Run")
    print("================================================================")
    print(
        f"policy={policy}, model={model_name}, episodes={episodes}, "
        f"steps_per_episode={steps}, num_evs={num_evs}"
    )

    for ep_idx, ep_seed in enumerate(episode_seeds, start=1):
        env = _build_env(cfg, seed=ep_seed)
        strategy = _build_strategy(policy, model_path, env)
        violation_steps = 0
        no_violation_steps = 0
        min_voltage_seen = 1.0
        min_voltage_bus_seen = None
        line_loss_printed = False

        print("\n----------------------------------------------------------------")
        print(f"Episode {ep_idx} seed={ep_seed}")
        print("----------------------------------------------------------------")
        for step_idx in range(1, steps + 1):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}
            for ev in urgent_evs:
                action = strategy.select_action(env, ev, pending_counts)
                actions[ev.id] = action
                pending_counts[action] += 1

            _, _, _, info = env.step(actions)
            min_bus, min_v, violating_count, excursion_sum, violating = _step_voltage_summary(
                info.get("bus_voltages", {})
            )

            if violating_count > 0:
                violation_steps += 1
            else:
                no_violation_steps += 1

            if min_v < min_voltage_seen:
                min_voltage_seen = min_v
                min_voltage_bus_seen = min_bus

            if step_idx <= 10:
                print(
                    f"[step {step_idx:03d}] min_v_pu={min_v:.6f} at {min_bus} | "
                    f"violating_bus_count={violating_count} | "
                    f"sum_voltage_excursions={excursion_sum:.6f}"
                )

            if not line_loss_printed:
                _print_step_line_loss_diagnosis(env, info)
                line_loss_printed = True

        print("\n[Episode Summary]")
        print(f"violation_step_ratio={violation_steps / max(1, steps):.6f}")
        print(f"min_v_pu_across_all_steps={min_voltage_seen:.6f} at {min_voltage_bus_seen}")
        print(f"steps_with_no_bus_violating={no_violation_steps}")


def main():
    parser = argparse.ArgumentParser(description="Step 5 sanity diagnostics for violation and line-loss metrics")
    parser.add_argument("--policy", choices=["baseline", "greedy", "random"], default="baseline")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-evs", type=int, default=DEFAULT_NUM_EVS)
    args = parser.parse_args()
    run_sanity(
        policy=args.policy,
        model_name=args.model_name,
        episodes=args.episodes,
        steps=args.steps,
        num_evs=args.num_evs,
    )


if __name__ == "__main__":
    main()
