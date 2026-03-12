"""
EV 充电调度强化学习系统 — 统一入口
=====================================
用法:
    python main.py train           # 基础训练 (合成路网)
    python main.py train-real      # 联邦训练 (真实路网 + 可视化)
    python main.py train-viz       # 基础训练 + 可视化
    python main.py evaluate        # 评估已训练模型
    python main.py download-map    # 下载真实路网地图
"""

import argparse
import sys
import os


def cmd_train(_args):
    """基础 DQN 训练（合成路网）"""
    from train import DQNAgent
    from env.Traffic import TrafficPowerEnv
    import torch

    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=10, num_actions=2)

    episodes = 800
    batch_size = 64
    print("开始训练 (合成路网)...")
    for e in range(episodes):
        env.reset()
        total_reward = 0
        for _ in range(100):
            urgent_evs = [ev for ev in env.evs if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)
            actions = {}
            pending = {s.id: 0 for s in env.stations}
            for ev in urgent_evs:
                state = env.get_graph_state_for_ev(ev, pending)
                mask = env.get_action_mask(ev)
                action = agent.select_action(state, action_mask=mask)
                actions[ev.id] = action
                pending[action] = pending.get(action, 0) + 1
            _, reward, _, _ = env.step(actions)
            total_reward += reward
        agent.replay(batch_size)
        if (e + 1) % 50 == 0:
            print(f"Episode {e+1}/{episodes}  reward={total_reward:.1f}  ε={agent.epsilon:.3f}")
    agent.save_model()
    print("训练完成，模型已保存至 checkpoints/trained_dqn.pth")


def cmd_train_real(_args):
    """联邦 DQN 训练（真实路网 + 可视化）"""
    from visualization.run_training_real_map import run_training_real
    run_training_real()


def cmd_train_viz(_args):
    """基础训练 + 可视化输出"""
    from visualization.run_training_with_viz import run_training
    run_training()


def cmd_evaluate(_args):
    """评估已训练模型（随机基线 / DQN / 联邦DQN 三方对比）"""
    from evaluation.run_evaluation import run_evaluation, _compare_table

    USE_REAL_MAP = True
    EPISODES = 5
    STEPS = 100

    map_str = "真实路网 (珠江新城)" if USE_REAL_MAP else "3x3 人工网格"
    print(f"\n>>>> 当前评估使用的地图环境: {map_str} <<<<\n")

    print("=" * 62)
    print("  【1/3】随机策略基线")
    print("=" * 62)
    random_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                   use_random=True, use_real_map=USE_REAL_MAP)

    print("\n")
    print("=" * 62)
    print("  【2/3】DQN 策略评估")
    print("=" * 62)
    dqn_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_dqn_real.pth" if USE_REAL_MAP else "trained_dqn.pth")

    print("\n")
    print("=" * 62)
    print("  【3/3】联邦 DQN 策略评估")
    print("=" * 62)
    fed_report = run_evaluation(episodes=EPISODES, steps_per_episode=STEPS,
                                use_random=False, use_real_map=USE_REAL_MAP,
                                model_file="trained_federated_dqn_real.pth")

    _compare_table({"Random": random_report, "DQN": dqn_report, "FedDQN": fed_report})


def cmd_download_map(_args):
    """下载真实城市路网"""
    import download_map  # noqa: F401


COMMANDS = {
    "train":        cmd_train,
    "train-real":   cmd_train_real,
    "train-viz":    cmd_train_viz,
    "evaluate":     cmd_evaluate,
    "download-map": cmd_download_map,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EV 充电调度强化学习系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:15s}  {v.__doc__}" for k, v in COMMANDS.items()),
    )
    parser.add_argument(
        "command",
        choices=list(COMMANDS.keys()),
        help="要执行的操作",
    )
    args = parser.parse_args()
    COMMANDS[args.command](args)

