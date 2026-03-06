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

from simEvn.Traffic import TrafficPowerEnv
from simEvn.RealTrafficEnv import RealTrafficEnv
from train import DQNAgent
from evaluation.metrics import Evaluator


# 真实路网评估默认使用全图（仅当节点数超过该值时才会精简）
REAL_MAP_MAX_NODES = 9999


def run_evaluation(episodes=3, steps_per_episode=100, use_random=True, use_real_map=True):
    """
    运行评估。

    Args:
        episodes:          评估轮数 (多轮取均值更稳定)
        steps_per_episode: 每轮步数
        use_random:        True -> 随机策略基线;  False -> 加载已训练模型
        use_real_map:      True -> 使用真实路网 (如珠江新城); False -> 使用 3x3 网格
    """
    # 1. 环境初始化
    if use_real_map:
        graphml_path = os.path.join(project_root, "zhujiang_new_town.graphml")
        if not os.path.exists(graphml_path):
            print(f"[错误] 未找到真实路网文件: {graphml_path}")
            print("自动回退到 3x3 基础网格环境。")
            use_real_map = False
            env = TrafficPowerEnv()
        else:
            print(f"[环境] 加载真实路网: {graphml_path}")
            env = RealTrafficEnv(
                graphml_file=graphml_path,
                num_stations=2,
                num_evs=20,
                max_nodes=REAL_MAP_MAX_NODES,
                seed=42
            )
    else:
        print("[环境] 使用 3x3 基础网格")
        env = TrafficPowerEnv()

    evaluator = Evaluator()

    # 2. 策略选择
    if use_random:
        print("[策略] 使用随机策略 (Random Baseline)")
        agent = None
    else:
        print("[策略] 使用训练后的 DQN 策略")
        
        # 获取环境特定的参数以初始化网络
        num_features = 10
        num_actions = getattr(env, 'num_stations', 2)
        station_node_ids = getattr(env, 'station_node_ids', [0, 8])
        num_nodes_per_graph = getattr(env, 'num_nodes', 9)

        agent = DQNAgent(
            num_features=num_features, 
            num_actions=num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph
        )
        
        # 根据是否是真实路网，尝试加载对应的权重文件
        model_name = "trained_dqn_real.pth" if use_real_map else "trained_dqn.pth"
        model_path = os.path.join(project_root, "model", model_name)
        
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"[模型] 成功加载权重: {model_path}")
        else:
            print(f"[错误] 未找到模型文件: {model_path}")
            print(f"请先运行训练脚本生成模型文件。")
            return None
            
        agent.epsilon = 0.02  # 近乎纯利用，保留微量随机打破 Q 值退化

    all_reports = []

    for ep in range(episodes):
        # 3. 每轮重置环境与评估器
        if use_real_map:
            env = RealTrafficEnv(
                graphml_file=graphml_path,
                num_stations=2,
                num_evs=20,
                max_nodes=REAL_MAP_MAX_NODES,
                seed=rng.randint(0, 10000) # 每次评估换个种子改变EV初始位置
            )
        else:
            env = TrafficPowerEnv()
        evaluator.reset()

        for step in range(steps_per_episode):

            # --- 顺序决策: 按 SOC 从低到高, 逐个选站 ---
            urgent_evs = [ev for ev in env.evs
                          if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)

            actions = {}
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                if agent is not None:
                    ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                    action = agent.select_action(ev_state)
                else:
                    action = rng.choice([0, 1])
                actions[ev.id] = action
                pending_counts[action] += 1

            # 执行
            _, reward, _, info = env.step(actions)

            # 累计电网侧指标 + 排队过程指标
            evaluator.update(info, env.power_grid, env.stations)

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
    # 配置是否使用真实路网(珠江新城.graphml) 或 3x3网格
    USE_REAL_MAP = True
    EPISODES = 5
    STEPS = 100
    
    map_str = "真实路网 (珠江新城)" if USE_REAL_MAP else "3x3 人工网格"
    print(f"\n>>>> 当前评估使用的地图环境: {map_str} <<<<\n")

    print("=" * 62)
    print("  【1/2】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS, use_random=True, use_real_map=USE_REAL_MAP)

    print("\n\n")
    print("=" * 62)
    print("  【2/2】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS, use_random=False, use_real_map=USE_REAL_MAP)

    # 对比表
    if random_report and dqn_report:
        print("\n" + "=" * 62)
        print("  DQN vs Random 对比")
        print("=" * 62)
        for k in random_report:
            r_val = random_report[k]
            d_val = dqn_report[k]
            diff = d_val - r_val
            pct = diff / max(abs(r_val), 0.01) * 100
            tag = "DQN更优" if diff < 0 else ("Random更优" if diff > 0 else "持平")
            print(f"  {k:35s}: Random={r_val:10.2f}  DQN={d_val:10.2f}  "
                  f"diff={diff:+8.2f} ({pct:+.1f}%) {tag}")
