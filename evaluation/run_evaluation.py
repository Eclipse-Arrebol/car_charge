"""
模型评估入口脚本
================
训练完成后运行此脚本，用当前策略跑一轮评估 episode，
输出用户侧 + 电网侧的完整评价指标报告。

用法:
  python evaluation/run_evaluation.py
"""

import os
import sys
import json
import random

# 确保项目根目录在 Python 搜索路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simEvn.Traffic import TrafficPowerEnv
from train import DQNAgent
from evaluation.metrics import Evaluator


def run_evaluation(episodes=1, steps_per_episode=100, use_random=True):
    """
    运行评估。

    Args:
        episodes:          评估轮数 (多轮取均值更稳定)
        steps_per_episode: 每轮步数
        use_random:        True → 随机策略基线;  False → 加载已训练模型
    """
    env = TrafficPowerEnv()
    evaluator = Evaluator()

    # 策略选择
    if use_random:
        print("[评估模式] 使用随机策略 (Random Baseline)")
        agent = None
    else:
        print("[评估模式] 使用训练后的 DQN 策略")
        agent = DQNAgent(num_features=8, num_actions=2)
        model_path = os.path.join(project_root, "model", "trained_dqn.pth")
        if os.path.exists(model_path):
            agent.load_model(model_path)
        else:
            print(f"  [WARNING] 未找到模型文件: {model_path}")
            print(f"  请先运行 train.py 或 visualization/run_training_with_viz.py 完成训练")
            return None
        agent.epsilon = 0.0  # 纯利用，不探索

    all_reports = []

    for ep in range(episodes):
        # 重置环境与评估器
        env = TrafficPowerEnv()
        evaluator.reset()

        for step in range(steps_per_episode):
            current_state = env.get_graph_state()

            # 决策
            actions = {}
            for ev in env.evs:
                if ev.status == "IDLE" and ev.soc < 30.0:
                    if agent is not None:
                        actions[ev.id] = agent.select_action(current_state)
                    else:
                        actions[ev.id] = random.choice([0, 1])

            # 执行
            _, reward, _, info = env.step(actions)

            # 累计电网侧指标
            evaluator.update(info, env.power_grid)

        # 生成本轮报告
        report = evaluator.report(env.evs, env.stations, verbose=(episodes == 1))
        all_reports.append(report)

    # 多轮均值
    if episodes > 1:
        avg_report = {}
        for key in all_reports[0]:
            vals = [r[key] for r in all_reports]
            avg_report[key] = round(float(sum(vals) / len(vals)), 4)
        print(f"\n{'='*62}")
        print(f"  {episodes} 轮评估均值:")
        print(f"{'='*62}")
        evaluator._print_report(avg_report)
        report = avg_report

    # 保存结果
    save_dir = os.path.join(project_root, "evaluation", "results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "evaluation_report.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n评估结果已保存: {save_path}")

    return report


if __name__ == "__main__":
    print("=" * 62)
    print("  【1/2】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(episodes=3, steps_per_episode=100, use_random=True)

    print("\n\n")
    print("=" * 62)
    print("  【2/2】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(episodes=3, steps_per_episode=100, use_random=False)
