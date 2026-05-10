"""
模型评估脚本 - 支持不同车辆数量
================================
测试不同模型在不同车辆数量下的性能表现

用法:
  python tests/evaluate_models_by_evs.py --num_evs 80
  python tests/evaluate_models_by_evs.py --num_evs 120
  python tests/evaluate_models_by_evs.py --num_evs 160
"""

import argparse
import json
import os
import random as rng
import sys
from datetime import datetime

# 确保项目根目录在 Python 搜索路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.graph_mapping import resolve_station_nodes
from env.real_env import RealTrafficEnv
from evaluation.metrics import Evaluator
from evaluation.strategies import DQNStrategy, GreedyStrategy, RandomStrategy
from training.config import EvalConfig

# 真实路网评估默认使用全图
REAL_MAP_MAX_NODES = 9999

# 模型配置 - 映射模型名称到checkpoint文件
MODEL_CONFIGS = {
    "voltage_buggy": {
        "name": "Voltage Buggy (原始电压奖励)",
        "checkpoint": "step10_voltage_50ep_seed0.pth",
        "description": "原始电压奖励函数（有bug）"
    },
    "voltage_fixed": {
        "name": "Voltage Fixed (修复bug)",
        "checkpoint": "step12_voltage_fixed_50ep_seed0.pth",
        "description": "修复abandon penalty bug"
    },
    "voltage_new": {
        "name": "Voltage New (调整权重)",
        "checkpoint": "step14_voltage_new_50ep_seed0.pth",
        "description": "调整奖励权重 user:grid = 0.7:0.3"
    },
    "mask_new": {
        "name": "Mask New (添加mask)",
        "checkpoint": "step15_mask_new_50ep_seed0.pth",
        "description": "添加action mask机制"
    }
}


def _build_eval_env(eval_cfg, seed):
    """构建评估环境"""
    graphml_path = os.path.join(
        PROJECT_ROOT, "map_outputs", "baseline_eps40_artifacts", "G_L0_indexed.graphml"
    )

    if not os.path.exists(graphml_path):
        print(f"[错误] 未找到真实路网文件: {graphml_path}")
        return None

    station_node_ids = None
    station_cfg = getattr(eval_cfg, "station_config_file", None)
    station_key = getattr(eval_cfg, "station_id_key", None)
    if station_cfg and station_key:
        station_node_ids = resolve_station_nodes(station_cfg, station_key)

    env = RealTrafficEnv(
        graphml_file=graphml_path,
        num_stations=eval_cfg.num_stations,
        num_evs=eval_cfg.num_evs,
        max_nodes=eval_cfg.max_nodes,
        seed=seed,
        station_node_ids=station_node_ids,
        respawn_after_full_charge=getattr(eval_cfg, "respawn_after_full_charge", False),
    )

    # 使用与训练一致的mask配置
    env.enable_queue_timeout_mask = False
    return env


def _load_model_strategy(model_name, env):
    """加载模型策略"""
    config = MODEL_CONFIGS[model_name]
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", config["checkpoint"])

    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        return None

    strategy = DQNStrategy(model_path, env)
    print(f"[模型] 加载 {config['name']}: {config['checkpoint']}")
    return strategy


def evaluate_strategy(strategy, eval_cfg, episode_seeds):
    """评估单个策略"""
    evaluator = Evaluator()
    all_reports = []

    for ep_idx, ep_seed in enumerate(episode_seeds):
        env = _build_eval_env(eval_cfg, ep_seed)
        if env is None:
            return None

        evaluator.reset()
        per_step_queue = []
        per_step_trip = []

        for _ in range(eval_cfg.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                action = strategy.select_action(env, ev, pending_counts)
                if hasattr(action, 'item'):
                    action_int = int(action.item())
                else:
                    action_int = int(action)

                actions[ev.id] = action_int
                if 0 <= action_int < len(env.stations):
                    selected_station = env.stations[action_int]
                    pending_counts[selected_station.id] = pending_counts.get(selected_station.id, 0) + 1

            _, _, _, info = env.step(actions)
            evaluator.update(info, env.power_grid, env.stations)

            # 记录排队时间和行程时间
            decision_costs = list(info.get("decision_costs", {}).values())
            if decision_costs:
                per_step_queue.append(sum(m.get("queue_time_h", 0.0) for m in decision_costs) / len(decision_costs))
                per_step_trip.append(sum(m.get("trip_time_h", 0.0) for m in decision_costs) / len(decision_costs))

        report = evaluator.report(env.evs, env.stations, verbose=False)
        report["queue_time_h_mean"] = sum(per_step_queue) / len(per_step_queue) if per_step_queue else 0.0
        report["trip_time_h_mean"] = sum(per_step_trip) / len(per_step_trip) if per_step_trip else 0.0
        all_reports.append(report)

        if ep_idx == 0:
            print(f"  Episode {ep_idx + 1}/{len(episode_seeds)} 完成 - "
                  f"等待时间: {report['avg_wait_time_min']:.2f}分钟, "
                  f"弃单EV: {report['abandoned_evs']:.2f}")

    # 计算平均值
    avg_report = {}
    for key in all_reports[0]:
        values = [r[key] for r in all_reports]
        avg_report[key] = round(float(sum(values) / len(values)), 4)

    return avg_report


def run_evaluation_for_evs(num_evs, episodes=10, steps_per_episode=600):
    """运行指定车辆数量的评估"""
    print(f"\n{'='*70}")
    print(f"  Evaluation Config: {num_evs} EVs, {episodes} episodes, {steps_per_episode} steps/episode")
    print(f"{'='*70}")

    # 生成稳定的 episode seed
    base_seed = 0
    rng_instance = rng.Random(base_seed)
    episode_seeds = [rng_instance.randint(0, 10000) for _ in range(episodes)]

    # 构建评估配置
    graphml_path = os.path.join(
        PROJECT_ROOT, "map_outputs", "baseline_eps40_artifacts", "G_L0_indexed.graphml"
    )
    eval_cfg = EvalConfig(
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        num_evs=num_evs,
        num_stations=8,  # 保持8个充电站
        max_nodes=REAL_MAP_MAX_NODES,
        graphml_file=graphml_path,
    )
    eval_cfg.use_real_map = True

    results = {}

    # 评估 Random 策略
    print(f"\n--- Evaluating Random Strategy ---")
    random_strategy = RandomStrategy()
    random_report = evaluate_strategy(random_strategy, eval_cfg, episode_seeds)
    if random_report:
        results["Random"] = random_report
        print(f"  Avg Wait Time: {random_report['avg_wait_time_min']:.2f} min")
        print(f"  Abandoned EVs: {random_report['abandoned_evs']:.2f}")

    # 评估 Greedy 策略
    print(f"\n--- Evaluating Greedy Strategy ---")
    greedy_strategy = GreedyStrategy()
    greedy_report = evaluate_strategy(greedy_strategy, eval_cfg, episode_seeds)
    if greedy_report:
        results["Greedy"] = greedy_report
        print(f"  Avg Wait Time: {greedy_report['avg_wait_time_min']:.2f} min")
        print(f"  Abandoned EVs: {greedy_report['abandoned_evs']:.2f}")

    # 评估各个模型
    for model_key in ["voltage_buggy", "voltage_fixed", "voltage_new", "mask_new"]:
        print(f"\n--- Evaluating {MODEL_CONFIGS[model_key]['name']} ---")

        # 创建临时环境用于加载模型
        temp_env = _build_eval_env(eval_cfg, episode_seeds[0])
        if temp_env is None:
            continue

        strategy = _load_model_strategy(model_key, temp_env)
        if strategy is None:
            continue

        model_report = evaluate_strategy(strategy, eval_cfg, episode_seeds)
        if model_report:
            model_name = MODEL_CONFIGS[model_key]["name"]
            results[model_name] = model_report
            print(f"  Avg Wait Time: {model_report['avg_wait_time_min']:.2f} min")
            print(f"  Abandoned EVs: {model_report['abandoned_evs']:.2f}")

    return results


def print_comparison_table(results, num_evs, baseline_name="Random"):
    """Print comparison table"""
    if baseline_name not in results:
        baseline_name = next(iter(results.keys()))

    baseline = results[baseline_name]

    print(f"\n{'='*80}")
    print(f"  {num_evs} EVs Performance Comparison (Baseline: {baseline_name})")
    print(f"{'='*80}")

    # Key metrics
    key_metrics = [
        ("avg_wait_time_min", "Avg Wait Time (min)"),
        ("abandoned_evs", "Abandoned EVs"),
        ("queue_time_h_mean", "Queue Time Mean (h)"),
        ("total_charging_cost_per_veh", "Total Cost (CNY/veh)"),
        ("accumulated_voltage_excursion_pu", "Voltage Excursion (pu)"),
    ]

    # Table header
    header = f"  {'Metric':<25}"
    for name in results.keys():
        header += f"  {name:>15}"
    print(header)
    print("-" * 80)

    # Data rows
    for metric_key, metric_name in key_metrics:
        if metric_key not in baseline:
            continue

        row = f"  {metric_name:<25}"
        base_val = baseline[metric_key]

        for name, report in results.items():
            if metric_key not in report:
                row += f"  {'N/A':>15}"
                continue

            val = report[metric_key]
            if name == baseline_name:
                row += f"  {val:>15.2f}"
            else:
                diff = val - base_val
                if base_val != 0:
                    pct = diff / abs(base_val) * 100
                else:
                    pct = 0

                # Improvement direction: lower is better for these metrics
                if metric_key in ["avg_wait_time_min", "abandoned_evs", "queue_time_h_mean",
                                 "total_charging_cost_per_veh", "accumulated_voltage_excursion_pu"]:
                    tag = "↓" if diff < 0 else ("↑" if diff > 0 else "=")
                else:
                    tag = "↑" if diff < 0 else ("↓" if diff > 0 else "=")

                row += f"  {val:>8.2f}{tag}{pct:+.1f}%"

        print(row)

    print("-" * 80)


def save_results(results, num_evs):
    """Save results to JSON file"""
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"runs/models_evaluation_{num_evs}evs_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "num_evs": num_evs,
            "episodes": 10,
            "steps_per_episode": 600,
            "num_stations": 8,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="评估不同模型在不同车辆数量下的性能")
    parser.add_argument("--num_evs", type=int, default=80, help="车辆数量 (80, 120, 160)")
    parser.add_argument("--episodes", type=int, default=10, help="评估轮数")
    parser.add_argument("--steps", type=int, default=600, help="每轮步数")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  Model Performance Evaluation Script")
    print(f"{'='*70}")
    print(f"  Number of EVs: {args.num_evs}")
    print(f"  Evaluation Episodes: {args.episodes}")
    print(f"  Steps per Episode: {args.steps}")
    print(f"{'='*70}")

    # 运行评估
    results = run_evaluation_for_evs(
        num_evs=args.num_evs,
        episodes=args.episodes,
        steps_per_episode=args.steps
    )

    # 打印对比表格
    if results:
        print_comparison_table(results, args.num_evs, baseline_name="Random")

        # 保存结果
        save_results(results, args.num_evs)
    else:
        print("\n[Error] Evaluation failed, no valid results obtained")


if __name__ == "__main__":
    main()
