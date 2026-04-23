import math
import os
import random
import statistics
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.real_env import RealTrafficEnv


L0_GRAPHML = os.path.join(
    PROJECT_ROOT,
    "map_outputs",
    "baseline_eps40_artifacts",
    "G_L0_indexed.graphml",
)
STATION_IDS = [394, 147, 182, 708, 764, 964, 1496, 200]
NUM_EPISODES = 3
WARMUP_STEPS = 20
MAX_STEPS = 600
NUM_EVS = 60
SATURATION_THRESHOLD = 0.99


def _build_env(seed: int) -> RealTrafficEnv:
    env = RealTrafficEnv(
        graphml_file=L0_GRAPHML,
        num_stations=len(STATION_IDS),
        num_evs=NUM_EVS,
        max_nodes=9999,
        seed=seed,
        station_node_ids=STATION_IDS,
        respawn_after_full_charge=False,
    )
    env.reset()
    return env


def _random_actions(env, rng: random.Random) -> Dict[int, int]:
    actions = {}
    for ev in env.get_pending_decision_evs():
        mask = env.get_action_mask(ev)
        valid = [idx for idx in range(env.num_stations) if mask[0, idx].item()]
        if valid:
            actions[ev.id] = rng.choice(valid)
    return actions


def _current_station_load_ratio(station) -> float:
    return float(station.last_total_load) / max(1.0, float(station.max_grid_power))


def _current_station_voltage(env, station) -> Optional[float]:
    if hasattr(env.power_grid, "get_last_bus_voltage") and hasattr(station, "power_bus_idx"):
        return env.power_grid.get_last_bus_voltage(station.power_bus_idx)
    return env.power_grid.bus_voltages.get(station.power_node_id)


def _record_phase_row(env, episode_idx: int, step_idx: int, phase_rows: List[dict]) -> None:
    ratios = [_current_station_load_ratio(station) for station in env.stations]
    voltages = [
        float(v) for v in
        (_current_station_voltage(env, station) for station in env.stations)
        if v is not None
    ]
    ratio_sorted = sorted(ratios)
    volt_sorted = sorted(voltages)
    phase_rows.append({
        "episode": episode_idx,
        "step": step_idx,
        "ratio_min": min(ratio_sorted) if ratio_sorted else None,
        "ratio_median": statistics.median(ratio_sorted) if ratio_sorted else None,
        "ratio_max": max(ratio_sorted) if ratio_sorted else None,
        "volt_min": min(volt_sorted) if volt_sorted else None,
        "volt_median": statistics.median(volt_sorted) if volt_sorted else None,
        "volt_max": max(volt_sorted) if volt_sorted else None,
        "saturated_count": sum(1 for ratio in ratios if ratio >= SATURATION_THRESHOLD),
    })


def _warmup(env, rng: random.Random, steps: int) -> None:
    for _ in range(steps):
        env.step(_random_actions(env, rng))


def _safe_fmt(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def _mae(preds: List[float], trues: List[float]) -> float:
    return sum(abs(p - t) for p, t in zip(preds, trues)) / max(1, len(preds))


def _rmse(preds: List[float], trues: List[float]) -> float:
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, trues)) / max(1, len(preds)))


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def _print_pair_sample_table(records: List[dict], title: str, limit: int = 24) -> None:
    print(title)
    print(
        f"{'ep':>2} {'ev':>3} {'st':>2} {'t_est':>5} {'t_true':>6} "
        f"{'pred_ratio':>10} {'true_ratio':>10} "
        f"{'pred_load':>10} {'true_load':>10} "
        f"{'pred_v':>8} {'true_v':>8}"
    )
    print("-" * 92)
    for row in records[:limit]:
        print(
            f"{row['episode']:>2} {row['ev_id']:>3} {row['station_id']:>2} "
            f"{row['estimate_step']:>5} {row['charge_start_step']:>6} "
            f"{_safe_fmt(row['pred_ratio_after']):>10} {_safe_fmt(row['true_ratio_after']):>10} "
            f"{_safe_fmt(row['pred_load_after']):>10} {_safe_fmt(row['true_load_after']):>10} "
            f"{_safe_fmt(row['pred_voltage_after']):>8} {_safe_fmt(row['true_voltage_after']):>8}"
        )
    print()


def _metric_values(records: List[dict], pred_key: str, true_key: str):
    usable = [
        row for row in records
        if row.get(pred_key) is not None and row.get(true_key) is not None
    ]
    preds = [float(row[pred_key]) for row in usable]
    trues = [float(row[true_key]) for row in usable]
    return usable, preds, trues


def _print_metric_summary(records: List[dict], field_name: str, pred_key: str, true_key: str) -> None:
    usable, preds, trues = _metric_values(records, pred_key, true_key)
    corr = _pearson(preds, trues)
    print(f"[{field_name}] samples={len(usable)}")
    print(f"  MAE:   {_mae(preds, trues):.4f}")
    print(f"  RMSE:  {_rmse(preds, trues):.4f}")
    print(f"  Corr:  {'None' if corr is None else f'{corr:.4f}'}")

    sat = [
        row for row in usable
        if float(row["pred_ratio_after"]) >= SATURATION_THRESHOLD
    ]
    unsat = [
        row for row in usable
        if float(row["pred_ratio_after"]) < SATURATION_THRESHOLD
    ]
    for label, subset in [("pred_ratio_after < 0.99", unsat), ("pred_ratio_after >= 0.99", sat)]:
        if not subset:
            print(f"  [{label}] samples=0")
            continue
        sub_preds = [float(row[pred_key]) for row in subset]
        sub_trues = [float(row[true_key]) for row in subset]
        sub_corr = _pearson(sub_preds, sub_trues)
        print(
            f"  [{label}] samples={len(subset)} "
            f"MAE={_mae(sub_preds, sub_trues):.4f} "
            f"RMSE={_rmse(sub_preds, sub_trues):.4f} "
            f"Corr={'None' if sub_corr is None else f'{sub_corr:.4f}'}"
        )
    print()


def _print_phase_table(rows: List[dict], episode_idx: int, limit: int = 80) -> None:
    print(f"=== Part B / Episode {episode_idx} Time Series ===")
    print(
        f"{'step':>4} {'ratio_min':>10} {'ratio_med':>10} {'ratio_max':>10} "
        f"{'volt_min':>9} {'volt_med':>9} {'volt_max':>9} {'sat_cnt':>7}"
    )
    print("-" * 78)
    for row in rows[:limit]:
        print(
            f"{row['step']:>4} "
            f"{_safe_fmt(row['ratio_min']):>10} {_safe_fmt(row['ratio_median']):>10} {_safe_fmt(row['ratio_max']):>10} "
            f"{_safe_fmt(row['volt_min']):>9} {_safe_fmt(row['volt_median']):>9} {_safe_fmt(row['volt_max']):>9} "
            f"{row['saturated_count']:>7}"
        )
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows omitted)")
    print()


def _print_phase_summary(rows: List[dict], episode_idx: int) -> None:
    total = max(1, len(rows))
    bins = {
        "0": 0,
        "1~3": 0,
        "4~6": 0,
        "7~8": 0,
    }
    for row in rows:
        cnt = row["saturated_count"]
        if cnt == 0:
            bins["0"] += 1
        elif 1 <= cnt <= 3:
            bins["1~3"] += 1
        elif 4 <= cnt <= 6:
            bins["4~6"] += 1
        else:
            bins["7~8"] += 1
    print(f"=== Part B / Episode {episode_idx} Summary ===")
    for label in ["0", "1~3", "4~6", "7~8"]:
        print(f"  saturated_count={label}: {bins[label]}/{total} ({bins[label] / total:.1%})")
    print()


def run_episode(env, rng: random.Random, episode_idx: int):
    env.reset()
    _warmup(env, rng, WARMUP_STEPS)

    pending_predictions: Dict[int, dict] = {}
    completed_pairs: List[dict] = []
    phase_rows: List[dict] = []
    already_started = set()

    for step_idx in range(MAX_STEPS):
        _record_phase_row(env, episode_idx, step_idx, phase_rows)

        actions = {}
        urgent_evs = env.get_pending_decision_evs()
        for ev in urgent_evs:
            mask = env.get_action_mask(ev)
            valid = [idx for idx in range(env.num_stations) if mask[0, idx].item()]
            if not valid:
                continue

            choice = rng.choice(valid)
            actions[ev.id] = choice
            metrics = env.estimate_action_metrics(ev, choice)
            pending_predictions[ev.id] = {
                "episode": episode_idx,
                "ev_id": ev.id,
                "station_id": choice,
                "estimate_step": step_idx,
                "pred_ratio_after": metrics.get("estimated_grid_load_ratio_after"),
                "pred_voltage_after": metrics.get("estimated_voltage_pu_after"),
                "pred_load_after": metrics.get("estimated_station_load_kw_after"),
            }

        env.step(actions)

        for station in env.stations:
            true_ratio = _current_station_load_ratio(station)
            true_voltage = _current_station_voltage(env, station)
            true_load = float(station.last_total_load)
            for ev in station.connected_evs:
                if ev.id in already_started:
                    continue
                pred = pending_predictions.get(ev.id)
                if pred is None:
                    continue
                if pred["station_id"] != station.id:
                    continue
                if ev.status != "CHARGING":
                    continue
                completed_pairs.append({
                    **pred,
                    "charge_start_step": step_idx,
                    "true_ratio_after": true_ratio,
                    "true_voltage_after": true_voltage,
                    "true_load_after": true_load,
                })
                already_started.add(ev.id)

    return completed_pairs, phase_rows


def _print_voltage_gate(records: List[dict]) -> None:
    usable, preds, trues = _metric_values(
        records,
        pred_key="pred_voltage_after",
        true_key="true_voltage_after",
    )
    if not usable:
        print("[Gate] No voltage samples; stop before Step 5.")
        return
    mae = _mae(preds, trues)
    corr = _pearson(preds, trues)
    corr_text = "None" if corr is None else f"{corr:.4f}"
    print("================================================================")
    print("Step 4 gate")
    print("================================================================")
    print(f"estimated_voltage_pu_after Pearson: {corr_text}")
    print(f"estimated_voltage_pu_after MAE:     {mae:.4f}")
    if corr is not None and corr < 0.5:
        print("[Gate] Pearson < 0.5; stop and inspect estimator.")
    if mae > 0.05:
        print("[Gate] MAE > 0.05; stop and inspect estimator.")
    if (corr is None or corr >= 0.5) and mae <= 0.05:
        print("[Gate] PASS for Step 5 threshold.")


def main():
    print("================================================================")
    print("Part A: estimate vs true consistency test (pandapower grid)")
    print("================================================================")

    rng = random.Random(20260422)
    all_pairs: List[dict] = []
    all_phase_rows: Dict[int, List[dict]] = {}

    for episode_idx in range(1, NUM_EPISODES + 1):
        env = _build_env(seed=20260422 + episode_idx)
        pairs, phase_rows = run_episode(env, rng, episode_idx)
        all_pairs.extend(pairs)
        all_phase_rows[episode_idx] = phase_rows
        print(
            f"[Episode {episode_idx}] matched pred/true pairs: {len(pairs)}, "
            f"phase rows: {len(phase_rows)}"
        )

    print()
    _print_pair_sample_table(all_pairs, "=== Part A / Sample Pair Table ===")
    _print_metric_summary(
        all_pairs,
        field_name="estimated_grid_load_ratio_after",
        pred_key="pred_ratio_after",
        true_key="true_ratio_after",
    )
    _print_metric_summary(
        all_pairs,
        field_name="estimated_voltage_pu_after",
        pred_key="pred_voltage_after",
        true_key="true_voltage_after",
    )
    _print_metric_summary(
        all_pairs,
        field_name="estimated_station_load_kw_after",
        pred_key="pred_load_after",
        true_key="true_load_after",
    )
    _print_voltage_gate(all_pairs)

    print("================================================================")
    print("Part B: episode phase distribution statistics")
    print("================================================================")
    for episode_idx in range(1, NUM_EPISODES + 1):
        rows = all_phase_rows[episode_idx]
        _print_phase_table(rows, episode_idx)
        _print_phase_summary(rows, episode_idx)


if __name__ == "__main__":
    main()
