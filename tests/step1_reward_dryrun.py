import math
import os
import random
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from training.config import TrainConfig


def _fmt(value):
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _print_distribution(name, values):
    print(f"[{name}] n={len(values)}")
    if not values:
        print("  no data\n")
        return
    print(f"  min:    {_fmt(min(values))}")
    print(f"  median: {_fmt(statistics.median(values))}")
    print(f"  max:    {_fmt(max(values))}")
    print(f"  p90:    {_fmt(_percentile(values, 0.90))}")
    print()


def _build_env(seed):
    cfg = TrainConfig.ablation_l0()
    station_node_ids = resolve_station_nodes(cfg.station_config_file, cfg.station_id_key)
    env = RealTrafficEnv(
        graphml_file=cfg.graphml_file,
        num_stations=cfg.num_stations,
        num_evs=40,
        max_nodes=cfg.max_nodes,
        seed=seed,
        station_node_ids=station_node_ids,
        respawn_after_full_charge=False,
    )
    env.reset()
    return env


def main():
    rng = random.Random(20260422)
    user_raw_values = []
    voltage_excursion_values = []
    true_grid_cost_values = []

    print("================================================================")
    print("Step 1 / Part 2: reward dry-run raw value distributions")
    print("================================================================")

    for episode_idx in range(1, 4):
        env = _build_env(20260422 + episode_idx)
        print(f"--- Episode {episode_idx} ---")

        for step_idx in range(200):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            metrics_by_ev = {}

            for ev in urgent_evs:
                mask = env.get_action_mask(ev)
                valid = [idx for idx in range(env.num_stations) if mask[0, idx].item()]
                if not valid:
                    continue
                action = rng.choice(valid)
                actions[ev.id] = action
                metrics_by_ev[ev.id] = env.estimate_action_metrics(ev, action)

            _, _, _, info = env.step(actions)

            for ev in urgent_evs:
                action = actions.get(ev.id)
                metrics = metrics_by_ev.get(ev.id)
                if action is None or metrics is None:
                    continue

                station = env.stations[action]
                true_voltage_pu = info.get("bus_voltages", {}).get(station.power_node_id)
                true_grid_cost = info.get("objective_terms", {}).get("grid_cost")

                user_raw = float(metrics["queue_time_h"] + metrics["trip_time_h"])
                voltage_excursion = 0.0 if true_voltage_pu is None else max(0.0, 0.95 - float(true_voltage_pu))

                user_raw_values.append(user_raw)
                voltage_excursion_values.append(voltage_excursion)
                if true_grid_cost is not None:
                    true_grid_cost_values.append(float(true_grid_cost))

            if step_idx in (0, 50, 100, 150, 199):
                print(
                    f"step={step_idx:>3} urgent={len(urgent_evs):>2} "
                    f"grid_cost={_fmt(info.get('objective_terms', {}).get('grid_cost'))} "
                    f"total_decisions={len(user_raw_values)}"
                )

        print()

    print("================================================================")
    print("Aggregated distributions")
    print("================================================================")
    _print_distribution("user_raw", user_raw_values)
    _print_distribution("voltage_excursion", voltage_excursion_values)
    _print_distribution("true_grid_cost", true_grid_cost_values)


if __name__ == "__main__":
    main()
