"""
模型评估入口脚本
================
训练完成后运行此脚本，用当前策略跑评估 episode，
输出用户侧 + 电网侧的完整评价指标报告。

用法:
  python evaluation/run_evaluation.py
"""

import os
import sys
import json
import random as rng

# 确保项目根目录在 Python 搜索路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.base_env import TrafficPowerEnv
from env.real_env import RealTrafficEnv
from evaluation.metrics import Evaluator
from evaluation.strategies import (
    BaseStrategy,
    DQNStrategy,
    FedDQNStrategy,
    GreedyStrategy,
    RandomStrategy,
)
from training.config import EvalConfig


# 真实路网评估默认使用全图（仅当节点数超过该值时才会精简）
REAL_MAP_MAX_NODES = 9999


def _build_eval_env(eval_cfg, seed):
    use_real_map = getattr(eval_cfg, "use_real_map", True)
    if use_real_map:
        graphml_path = getattr(
            eval_cfg, "graphml_file", os.path.join(project_root, "zhujiang_new_town.graphml")
        )
        if not os.path.exists(graphml_path):
            print(f"[错误] 未找到真实路网文件: {graphml_path}")
            print("自动回退到 3x3 基础网格环境。")
            return TrafficPowerEnv(num_evs=eval_cfg.num_evs)
        return RealTrafficEnv(
            graphml_file=graphml_path,
            num_stations=eval_cfg.num_stations,
            num_evs=eval_cfg.num_evs,
            max_nodes=eval_cfg.max_nodes,
            seed=seed,
        )
    return TrafficPowerEnv(num_evs=eval_cfg.num_evs)


def _resolve_strategy(use_random, use_greedy, model_file, env):
    if use_random:
        print("[策略] 使用随机策略 (Random Baseline)")
        return RandomStrategy()
    if use_greedy:
        print("[策略] 使用贪心策略 (Greedy Baseline)")
        return GreedyStrategy()

    print("[策略] 使用训练后的 DQN 策略")
    if model_file is None:
        model_name = "trained_dqn_real.pth" if env.__class__.__name__ == "RealTrafficEnv" else "trained_dqn.pth"
    else:
        model_name = model_file
    model_path = os.path.join(project_root, "checkpoints", model_name)
    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        print("请先运行训练脚本生成模型文件。")
        return None
    if model_file and "federated" in model_file.lower():
        strategy = FedDQNStrategy(model_path, env)
    else:
        strategy = DQNStrategy(model_path, env)
    print(f"[模型] 成功加载权重: {model_path}")
    return strategy


def evaluate(
    strategy: BaseStrategy,
    eval_cfg,
    episode_seeds: list,
) -> dict:
    evaluator = Evaluator()
    all_reports = []

    for ep in range(eval_cfg.episodes):
        env = _build_eval_env(eval_cfg, episode_seeds[ep])
        evaluator.reset()

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

        report = evaluator.report(env.evs, env.stations, verbose=(eval_cfg.episodes == 1))
        all_reports.append(report)

    report = all_reports[-1]
    if eval_cfg.episodes > 1:
        avg_report = {}
        for key in all_reports[0]:
            vals = [r[key] for r in all_reports]
            avg_report[key] = round(float(sum(vals) / len(vals)), 4)
        print(f"\n{'='*62}")
        print(f"  {eval_cfg.episodes} 轮评估均值:")
        print(f"{'='*62}")
        evaluator._print_report(avg_report)
        report = avg_report

    save_dir = os.path.join(project_root, "evaluation", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"evaluation_report_{strategy.name()}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n评估结果已保存: {save_path}")

    return report


def run_evaluation(episodes=50, steps_per_episode=1000, use_random=False, use_greedy=False,
                   use_real_map=True, model_file=None, num_evs=100, num_stations=4,
                   episode_seeds=None):
    """
    运行评估。

    Args:
        episodes:          评估轮数 (多轮取均值更稳定)
        steps_per_episode: 每轮步数
        use_random:        True -> 随机策略基线;  False -> 加载已训练模型
        use_greedy:        True -> 贪心策略基线 (最短队列归一化)
        use_real_map:      True -> 使用真实路网 (如珠江新城); False -> 使用 3x3 网格
        model_file:        指定 checkpoints/ 下的权重文件名，None 时自动选择
    """
    graphml_path = os.path.join(project_root, "zhujiang_new_town.graphml")
    eval_cfg = EvalConfig(
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        num_evs=num_evs,
        num_stations=num_stations,
        max_nodes=REAL_MAP_MAX_NODES,
        graphml_file=graphml_path,
    )
    eval_cfg.use_real_map = use_real_map

    preview_env = _build_eval_env(eval_cfg, seed=42)
    if use_real_map and isinstance(preview_env, RealTrafficEnv):
        print(f"[环境] 加载真实路网: {graphml_path}")
    elif use_real_map:
        print("[环境] 使用 3x3 基础网格")
        eval_cfg.use_real_map = False
    else:
        print("[环境] 使用 3x3 基础网格")

    strategy = _resolve_strategy(use_random, use_greedy, model_file, preview_env)
    if strategy is None:
        return None

    if episode_seeds is None:
        episode_seeds = [rng.randint(0, 10000) for _ in range(episodes)]
    elif len(episode_seeds) != episodes:
        raise ValueError(f"episode_seeds length {len(episode_seeds)} != episodes {episodes}")
    return evaluate(strategy, eval_cfg, episode_seeds)


def _compare_table(reports: dict):
    """打印多策略对比表，reports = {策略名: report_dict}"""
    names = list(reports.keys())
    valid = {n: r for n, r in reports.items() if r is not None}
    if len(valid) < 2:
        return
    baseline_name, baseline = next(iter(valid.items()))
    keys = list(baseline.items())
    print("\n" + "=" * 80)
    print(f"  多策略对比（基线: {baseline_name}）")
    print("=" * 80)
    header = f"  {'指标':<33}" + "".join(f"  {n:>12}" for n in valid)
    print(header)
    print("-" * 80)
    for k, base_val in keys:
        row = f"  {k:<33}"
        for n, r in valid.items():
            val = r[k]
            if n == baseline_name:
                row += f"  {val:>12.2f}"
            else:
                diff = val - base_val
                pct = diff / max(abs(base_val), 0.01) * 100
                tag = "↑" if diff < 0 else ("↓" if diff > 0 else "=")
                row += f"  {val:>8.2f}{tag}{pct:+.0f}%"
        print(row)


if __name__ == "__main__":
    # 配置是否使用真实路网(珠江新城.graphml) 或 3x3网格
    USE_REAL_MAP = True
    EPISODES = 50
    STEPS = 1000
    episode_seeds = [rng.randint(0, 10000) for _ in range(EPISODES)]

    map_str = "真实路网 (珠江新城)" if USE_REAL_MAP else "3x3 人工网格"
    print(f"\n>>>> 当前评估使用的地图环境: {map_str} <<<<\n")

    print("=" * 62)
    print("  【1/4】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                   use_random=True, use_real_map=USE_REAL_MAP,
                                   episode_seeds=episode_seeds)

    print("\n")
    print("=" * 62)
    print("  【2/4】贪心策略基线")
    print("=" * 62)
    greedy_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                   use_greedy=True, use_real_map=USE_REAL_MAP,
                                   episode_seeds=episode_seeds)

    print("\n")
    print("=" * 62)
    print("  【3/4】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_dqn_real.pth" if USE_REAL_MAP else "trained_dqn.pth",
                                episode_seeds=episode_seeds)

    print("\n")
    print("=" * 62)
    print("  【4/4】联邦 DQN 策略评估")
    print("=" * 62)
    fed_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_federated_dqn_real.pth",
                                episode_seeds=episode_seeds)

    _compare_table({"Random": random_report, "Greedy": greedy_report,
                    "DQN": dqn_report, "FedDQN": fed_report})
