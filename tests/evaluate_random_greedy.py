"""
重新评估 Random 和 Greedy 策略
使用当前环境参数，与训练好的模型进行公平对比
"""

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
from evaluation.strategies import RandomStrategy, GreedyStrategy
from training.config import EvalConfig, TrainConfig


# 使用与 step8/step9 相同的参数
TRAIN_NUM_EVS = 40
EVAL_EPISODES = 10
EVAL_STEPS = 600
EXPERIMENT_SEED = 0
RUN_ROOT = os.path.join(PROJECT_ROOT, "runs")


def _episode_seeds(base_seed: int, episodes: int):
    """生成稳定的 episode seed 序列"""
    rng = random.Random(base_seed)
    return [rng.randint(0, 10000) for _ in range(episodes)]


def _build_eval_cfg(eval_seed: int):
    """构建评估配置"""
    cfg = EvalConfig.ablation_l0()
    cfg.episodes = EVAL_EPISODES
    cfg.steps_per_episode = EVAL_STEPS
    cfg.num_evs = TRAIN_NUM_EVS
    cfg.base_seed = eval_seed
    return cfg


def _build_env(cfg, seed):
    """构建环境"""
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
    # 使用与训练一致的 mask 配置
    env.enable_queue_timeout_mask = False
    return env


def evaluate_strategy(strategy, eval_seed: int):
    """评估单个策略"""
    eval_cfg = _build_eval_cfg(eval_seed)
    env = _build_env(eval_cfg, seed=eval_seed)
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

        report = evaluator.report(env.evs, env.stations, verbose=False)
        report["queue_time_h_mean"] = mean(per_step_queue) if per_step_queue else 0.0
        report["trip_time_h_mean"] = mean(per_step_trip) if per_step_trip else 0.0
        reports.append(report)

    # 计算平均值
    avg = {}
    for key in reports[0]:
        values = [r[key] for r in reports]
        avg[key] = sum(values) / len(values)
    return avg


def run_evaluation():
    """运行 Random 和 Greedy 策略评估"""
    eval_seed = EXPERIMENT_SEED

    # 创建策略实例
    random_strategy = RandomStrategy()
    greedy_strategy = GreedyStrategy()

    print("\n=== Evaluating Random and Greedy Strategies ===")
    print(f"Environment: {TRAIN_NUM_EVS} EVs, {EVAL_EPISODES} episodes, {EVAL_STEPS} steps/episode")
    print(f"Seed: {eval_seed}")
    print(f"Mask: disabled (consistent with training)")

    # 评估 Random 策略
    print("\n--- Evaluating Random Strategy ---")
    random_report = evaluate_strategy(random_strategy, eval_seed)

    # 评估 Greedy 策略
    print("\n--- Evaluating Greedy Strategy ---")
    greedy_report = evaluate_strategy(greedy_strategy, eval_seed)

    # 打印结果
    rows = [
        "abandoned_evs",
        "accumulated_voltage_excursion_pu",
        "total_line_losses_kwh",
        "trip_time_h_mean",
        "queue_time_h_mean",
        "distribution_network_cost_cny",
        "avg_wait_time_min",
        "evs_in_line",
        "end_evs_in_line",
        "started_charge_sessions",
        "completed_charge_sessions",
        "total_energy_charged_kwh",
        "incomplete_evs",
    ]

    print("\n=== Results Summary ===")
    print(f"{'Metric':<40} {'Random':>15} {'Greedy':>15}")
    print("-" * 70)

    for key in rows:
        if key in random_report and key in greedy_report:
            random_val = float(random_report[key])
            greedy_val = float(greedy_report[key])
            print(f"{key:<40} {random_val:>15.4f} {greedy_val:>15.4f}")

    # 计算改进百分比
    print("\n=== Improvement Percentage (vs Random) ===")
    for key in ["avg_wait_time_min", "abandoned_evs", "queue_time_h_mean"]:
        if key in random_report and key in greedy_report:
            random_val = float(random_report[key])
            greedy_val = float(greedy_report[key])
            improvement = (random_val - greedy_val) / random_val * 100
            print(f"{key:<40} {improvement:>15.1f}%")

    # 保存结果到 JSON
    os.makedirs(RUN_ROOT, exist_ok=True)
    path = os.path.join(RUN_ROOT, "random_greedy_evaluation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": EXPERIMENT_SEED,
                "eval_num_evs": TRAIN_NUM_EVS,
                "eval_episodes": EVAL_EPISODES,
                "eval_steps": EVAL_STEPS,
                "config": {
                    "reward_mode": "baseline",
                    "queue_timeout_mask": False,
                    "num_stations": 8,  # 当前配置
                },
                "reports": {
                    "random": random_report,
                    "greedy": greedy_report,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[Saved] evaluation json: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Random and Greedy strategies")
    parser.add_argument(
        "--seed",
        type=int,
        default=EXPERIMENT_SEED,
        help="Random seed for evaluation",
    )
    args = parser.parse_args()

    # 使用局部变量而不是全局变量
    run_evaluation_with_seed(args.seed)


def run_evaluation_with_seed(seed: int):
    """使用指定seed运行评估"""
    global EXPERIMENT_SEED
    EXPERIMENT_SEED = seed
    run_evaluation()


if __name__ == "__main__":
    main()
