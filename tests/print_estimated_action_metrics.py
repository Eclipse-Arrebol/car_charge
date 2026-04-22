import os
import sys
from typing import Iterable, List, Optional, Tuple
from statistics import median

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from env.base_env import TrafficPowerEnv
from env.real_env import RealTrafficEnv


L0_GRAPHML = os.path.join(
    PROJECT_ROOT,
    "map_outputs",
    "baseline_eps40_artifacts",
    "G_L0_indexed.graphml",
)
STATION_IDS = [394, 147, 182, 708, 764, 964, 1496, 200]


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def _print_sample(title: str, ev, station, metrics: dict) -> None:
    print(title)
    print(f"EV id: {ev.id}")
    print(f"Station id: {station.id}")
    print(f"trip_dist_km: {metrics['trip_dist_km']:.3f}")
    print(f"grid_load_ratio (current):     {_format_metric(metrics.get('grid_load_ratio'))}")
    print(
        "estimated_grid_load_ratio_after: "
        f"{_format_metric(metrics.get('estimated_grid_load_ratio_after'))}"
    )
    print(
        "estimated_station_load_kw_after: "
        f"{_format_metric(metrics.get('estimated_station_load_kw_after'))}"
    )
    print(
        "estimated_voltage_pu_after:    "
        f"{_format_metric(metrics.get('estimated_voltage_pu_after'))}"
    )
    print()


def _print_station_scan_table(title: str, ev, station_metrics: List[Tuple[object, dict]]) -> None:
    print(title)
    print(f"EV id: {ev.id}")
    print(
        f"{'station_id':>10}  {'trip_dist_km':>12}  {'grid_ratio':>12}  "
        f"{'ratio_after':>12}  {'voltage_after':>13}"
    )
    print("-" * 70)
    for station, metrics in station_metrics:
        print(
            f"{station.id:>10}  "
            f"{metrics['trip_dist_km']:>12.3f}  "
            f"{_format_metric(metrics.get('grid_load_ratio')):>12}  "
            f"{_format_metric(metrics.get('estimated_grid_load_ratio_after')):>12}  "
            f"{_format_metric(metrics.get('estimated_voltage_pu_after')):>13}"
        )
    print()


def _print_station_scan_summary(station_metrics: List[Tuple[object, dict]]) -> None:
    ratios = [
        float(metrics["estimated_grid_load_ratio_after"])
        for _, metrics in station_metrics
        if metrics.get("estimated_grid_load_ratio_after") is not None
    ]
    voltages = [
        float(metrics["estimated_voltage_pu_after"])
        for _, metrics in station_metrics
        if metrics.get("estimated_voltage_pu_after") is not None
    ]
    saturated = sum(1 for value in ratios if value >= 0.99)
    print(
        f"[Summary] saturated_ratio_after(>=0.99): {saturated}/{len(ratios)}"
    )
    if voltages:
        print(
            f"[Summary] voltage_pu_after min/max/median: "
            f"{min(voltages):.3f} / {max(voltages):.3f} / {median(voltages):.3f}"
        )
    print()


def _build_env():
    if os.path.exists(L0_GRAPHML):
        env = RealTrafficEnv(
            graphml_file=L0_GRAPHML,
            num_stations=len(STATION_IDS),
            num_evs=60,
            max_nodes=9999,
            seed=20260422,
            station_node_ids=STATION_IDS,
            respawn_after_full_charge=False,
        )
        env.reset()
        return env

    print("[Warn] Real graph not found, fallback to TrafficPowerEnv. Scenario 2 may be weak.")
    env = TrafficPowerEnv(num_evs=20, respawn_after_full_charge=False)
    env.reset()
    return env


def _warm_up_env(env, warm_steps: int = 18) -> None:
    for _ in range(warm_steps):
        actions = {}
        for ev in env.get_pending_decision_evs():
            mask = env.get_action_mask(ev)
            valid_ids = [idx for idx in range(env.num_stations) if mask[0, idx].item()]
            if valid_ids:
                actions[ev.id] = valid_ids[ev.id % len(valid_ids)]
        env.step(actions)


def _iter_candidate_pairs(env) -> Iterable[Tuple[object, object, dict]]:
    for ev in env.evs:
        for station in env.stations:
            metrics = env.estimate_action_metrics(ev, station.id)
            yield ev, station, metrics


def _scenario_1(env) -> None:
    print("=" * 80)
    print("Scenario 1: 站点已有明显负载")
    print("=" * 80)
    loaded_stations = [
        station for station in env.stations
        if getattr(station, "last_total_load", 0.0) > 1e-6
    ]
    if not loaded_stations:
        print("No loaded stations found.\n")
        return

    sample_count = 0
    for station in sorted(loaded_stations, key=lambda s: s.last_total_load, reverse=True):
        for ev in env.evs:
            metrics = env.estimate_action_metrics(ev, station.id)
            current_ratio = metrics.get("grid_load_ratio")
            est_ratio = metrics.get("estimated_grid_load_ratio_after")
            if (
                current_ratio is not None
                and est_ratio is not None
                and est_ratio > current_ratio + 1e-6
            ):
                sample_count += 1
                _print_sample(
                    f"--- Scenario 1, Sample {sample_count} ---",
                    ev,
                    station,
                    metrics,
                )
                if sample_count >= 3:
                    return

    print("Could not find 3 increasing-load samples.\n")


def _pick_far_and_near_same_station(env):
    best = None
    for station in env.stations:
        near = None
        far = None
        for ev in env.evs:
            metrics = env.estimate_action_metrics(ev, station.id)
            dist = metrics["trip_dist_km"]
            if dist < 1.0 and near is None:
                near = (ev, metrics)
            if dist > 5.0 and far is None:
                far = (ev, metrics)
            if near and far:
                return station, near, far

        if near or far:
            best = (station, near, far)
    return best


def _scenario_2(env) -> None:
    print("=" * 80)
    print("Scenario 2: 不同距离的 EV 对同一站点")
    print("=" * 80)
    picked = _pick_far_and_near_same_station(env)
    if not picked:
        print("No near/far EV pair found.\n")
        return

    station, near, far = picked
    sample_idx = 1
    if near is not None:
        ev, metrics = near
        if metrics["trip_dist_km"] > 1e-6:
            _print_sample(f"--- Scenario 2, Sample {sample_idx} ---", ev, station, metrics)
            sample_idx += 1
    if far is not None:
        ev, metrics = far
        _print_sample(f"--- Scenario 2, Sample {sample_idx} ---", ev, station, metrics)
        sample_idx += 1

    extra_candidates: List[Tuple[object, dict]] = []
    for ev in env.evs:
        metrics = env.estimate_action_metrics(ev, station.id)
        if 1.0 <= metrics["trip_dist_km"] <= 5.0:
            extra_candidates.append((ev, metrics))
        if len(extra_candidates) >= 1:
            break
    for ev, metrics in extra_candidates[:1]:
        _print_sample(f"--- Scenario 2, Sample {sample_idx} ---", ev, station, metrics)


def _select_ev_for_full_station_scan(env):
    preferred_id = 46
    candidates = list(env.evs)
    preferred = next((ev for ev in candidates if ev.id == preferred_id), None)
    if preferred is not None:
        return preferred

    chosen = None
    best_spread = -1.0
    for ev in candidates:
        ratios = []
        for station in env.stations:
            metrics = env.estimate_action_metrics(ev, station.id)
            ratio_after = metrics.get("estimated_grid_load_ratio_after")
            if ratio_after is not None:
                ratios.append(float(ratio_after))
        if len(ratios) < 2:
            continue
        spread = max(ratios) - min(ratios)
        if spread > best_spread:
            best_spread = spread
            chosen = ev
    return chosen


def _scan_all_stations_for_ev(env, ev):
    rows = []
    for station in env.stations:
        metrics = env.estimate_action_metrics(ev, station.id)
        rows.append((station, metrics))
    return rows


def _scenario_3(env) -> None:
    print("=" * 80)
    print("Scenario 3: 同一辆 EV 对不同充电站（全站扫描，两个时相）")
    print("=" * 80)

    ev = _select_ev_for_full_station_scan(env)
    if ev is None:
        print("No EV with multiple distinguishable station estimates found.\n")
        return

    peak_station_metrics = sorted(
        _scan_all_stations_for_ev(env, ev),
        key=lambda item: item[0].id,
    )
    _print_station_scan_table(
        "--- Scenario 3, Peak-ish Phase Full Scan ---",
        ev,
        peak_station_metrics,
    )
    _print_station_scan_summary(peak_station_metrics)

    _warm_up_env(env, warm_steps=30)
    post_station_metrics = sorted(
        _scan_all_stations_for_ev(env, ev),
        key=lambda item: item[0].id,
    )
    _print_station_scan_table(
        "--- Scenario 3, Post-Peak Full Scan ---",
        ev,
        post_station_metrics,
    )
    _print_station_scan_summary(post_station_metrics)

    peak_ratios = [
        float(metrics["estimated_grid_load_ratio_after"])
        for _, metrics in peak_station_metrics
        if metrics.get("estimated_grid_load_ratio_after") is not None
    ]
    post_ratios = [
        float(metrics["estimated_grid_load_ratio_after"])
        for _, metrics in post_station_metrics
        if metrics.get("estimated_grid_load_ratio_after") is not None
    ]
    peak_voltages = [
        float(metrics["estimated_voltage_pu_after"])
        for _, metrics in peak_station_metrics
        if metrics.get("estimated_voltage_pu_after") is not None
    ]
    post_voltages = [
        float(metrics["estimated_voltage_pu_after"])
        for _, metrics in post_station_metrics
        if metrics.get("estimated_voltage_pu_after") is not None
    ]
    print(
        f"[Scenario 3 Compare] ratio spread peak/post: "
        f"{max(peak_ratios) - min(peak_ratios):.3f} / {max(post_ratios) - min(post_ratios):.3f}"
    )
    print(
        f"[Scenario 3 Compare] voltage spread peak/post: "
        f"{max(peak_voltages) - min(peak_voltages):.3f} / {max(post_voltages) - min(post_voltages):.3f}"
    )
    print()


def main():
    env = _build_env()
    _warm_up_env(env)
    _scenario_1(env)
    _scenario_2(env)
    _scenario_3(env)


if __name__ == "__main__":
    main()
