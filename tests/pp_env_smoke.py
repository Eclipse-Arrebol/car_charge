import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from training.config import TrainConfig


EPISODE_STEPS = 600
NUM_EVS = 40
SEED = 0


def _random_actions(env, rng):
    actions = {}
    pending_counts = {station.id: 0 for station in env.stations}
    for ev in env.get_pending_decision_evs():
        mask = env.get_action_mask(ev)
        valid = [idx for idx in range(env.num_stations) if mask[0, idx].item()]
        action = rng.choice(valid) if valid else 0
        actions[ev.id] = action
        pending_counts[action] += 1
    return actions


def _voltage_excursion(env):
    return sum(
        max(0.0, env.power_grid.v_min - float(v_pu))
        for v_pu in env.power_grid.bus_voltages.values()
    )


def main():
    random.seed(SEED)
    rng = random.Random(SEED)
    cfg = TrainConfig.ablation_l0()
    station_nodes = resolve_station_nodes(cfg.station_config_file, cfg.station_id_key)
    env = RealTrafficEnv(
        graphml_file=cfg.graphml_file,
        num_stations=cfg.num_stations,
        num_evs=NUM_EVS,
        max_nodes=cfg.max_nodes,
        seed=SEED,
        station_node_ids=station_nodes,
    )

    start_calls = getattr(env.power_grid, "runpp_call_count", 0)
    start_time_s = getattr(env.power_grid, "runpp_total_time_s", 0.0)
    min_voltage_seen = 1.0
    min_voltage_bus_seen = None

    wall_t0 = time.perf_counter()
    for step in range(1, EPISODE_STEPS + 1):
        actions = _random_actions(env, rng)
        _, _, _, info = env.step(actions)
        min_bus = info.get("min_voltage_bus")
        min_v = float(info.get("min_voltage_pu", 1.0))
        if min_v < min_voltage_seen:
            min_voltage_seen = min_v
            min_voltage_bus_seen = min_bus

        if step % 100 == 0:
            print(
                f"[Step {step:03d}] "
                f"min_v={min_v:.6f} bus={min_bus} "
                f"voltage_excursion={info.get('voltage_excursion', _voltage_excursion(env)):.6f} "
                f"line_losses_kw={float(info.get('line_losses', 0.0)):.6f}"
            )
    wall_s = time.perf_counter() - wall_t0

    abandoned = sum(getattr(ev, "abandoned_charge_count", 0) for ev in env.evs)
    incomplete = sum(1 for ev in env.evs if getattr(ev, "status", "IDLE") != "IDLE")
    runpp_calls = getattr(env.power_grid, "runpp_call_count", 0) - start_calls
    runpp_time_s = getattr(env.power_grid, "runpp_total_time_s", 0.0) - start_time_s
    avg_runpp_s = runpp_time_s / max(1, runpp_calls)

    print("\nEpisode summary:")
    print(f"  abandoned_evs={abandoned}")
    print(f"  incomplete_evs={incomplete}")
    print(f"  min_voltage_seen={min_voltage_seen:.6f} bus={min_voltage_bus_seen}")
    print(f"  runpp_calls={runpp_calls}")
    print(f"  runpp_total_time_s={runpp_time_s:.3f}")
    print(f"  runpp_avg_time_s={avg_runpp_s:.6f}")
    print(f"  wall_time_s={wall_s:.3f}")

    print("\nSmoke criteria:")
    print(f"  runpp_total_time_s < 60: {runpp_time_s < 60.0}")
    print(f"  min_voltage_seen < 0.90: {min_voltage_seen < 0.90}")
    print(f"  5 <= abandoned_evs <= 15: {5 <= abandoned <= 15}")


if __name__ == "__main__":
    main()
