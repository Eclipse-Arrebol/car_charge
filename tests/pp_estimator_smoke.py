import os
import random
import sys
from statistics import median

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from training.config import TrainConfig


WARMUP_STEPS = 20
MAX_EXTRA_STEPS = 80
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


def _percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _print_stats(values):
    print(
        f"delta_v_pu stats: min={min(values):.6f} "
        f"median={median(values):.6f} "
        f"p90={_percentile(values, 0.90):.6f} "
        f"max={max(values):.6f}"
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

    for _ in range(WARMUP_STEPS):
        env.step(_random_actions(env, rng))

    pending = env.get_pending_decision_evs()
    extra_steps = 0
    while not pending and extra_steps < MAX_EXTRA_STEPS:
        env.step(_random_actions(env, rng))
        pending = env.get_pending_decision_evs()
        extra_steps += 1

    if not pending:
        print("No pending EVs after warmup; estimator smoke cannot collect samples.")
        return

    rows = []
    for ev in pending:
        for station in env.stations:
            metrics = env._estimate_ev_station_metrics(ev, station)
            delta = metrics.get("estimated_voltage_delta_pu")
            pred_v = metrics.get("estimated_voltage_pu_after")
            current_v = metrics.get("current_voltage_pu")
            if delta is None or pred_v is None or current_v is None:
                continue
            rows.append({
                "ev_id": ev.id,
                "station_id": station.id,
                "bus": getattr(station, "power_bus_idx", None),
                "current_v": float(current_v),
                "pred_v": float(pred_v),
                "delta": float(delta),
                "r_th": float(metrics.get("thevenin_r_ohm") or 0.0),
            })

    if not rows:
        print("Estimator returned no voltage predictions.")
        return

    print(
        f"Collected {len(rows)} EV-station predictions "
        f"after {WARMUP_STEPS + extra_steps} warmup steps."
    )
    print("Sample predictions:")
    print(f"{'ev':>3} {'station':>7} {'bus':>4} {'current_v':>11} {'pred_v':>11} {'delta_pu':>10} {'r_th':>10}")
    for row in rows[:20]:
        print(
            f"{row['ev_id']:>3} {row['station_id']:>7} {row['bus']:>4} "
            f"{row['current_v']:>11.6f} {row['pred_v']:>11.6f} "
            f"{row['delta']:>10.6f} {row['r_th']:>10.4f}"
        )

    deltas = [row["delta"] for row in rows]
    print()
    _print_stats(deltas)

    by_bus = {}
    for row in rows:
        by_bus.setdefault(row["bus"], []).append(row["delta"])
    print("\nMean delta_v_pu by station bus:")
    for bus in sorted(by_bus):
        vals = by_bus[bus]
        print(f"  bus={bus:>2} mean_delta={sum(vals) / len(vals):.6f}")

    if 6 in by_bus and 18 in by_bus:
        bus6 = sum(by_bus[6]) / len(by_bus[6])
        bus18 = sum(by_bus[18]) / len(by_bus[18])
        ratio = bus18 / max(bus6, 1e-9)
        print(f"\nBus 18 / Bus 6 mean delta ratio: {ratio:.3f}x")

    print("\nSmoke criteria:")
    print(f"  typical delta in [0.001, 0.03]: {0.001 <= median(deltas) <= 0.03}")
    if 6 in by_bus and 18 in by_bus:
        print(f"  bus18 delta > bus6 delta: {bus18 > bus6}")


if __name__ == "__main__":
    main()
