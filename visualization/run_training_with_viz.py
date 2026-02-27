"""
Run training with visualization outputs without modifying train.py.
"""

import os
import sys

# Ensure project root is on the Python path when running from visualization/.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train import DQNAgent
from simEvn.Traffic import TrafficPowerEnv
from visualization.visualize_training import TrainingVisualizer


def run_training(episodes=200, steps_per_episode=100, batch_size=32):
    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=9, num_actions=2)
    viz = TrainingVisualizer()

    print("开始训练 (EV感知 + 顺序决策 + 目标网络 + 凸优化调度 + 可视化)...")

    for e in range(episodes):
        env.reset()
        total_reward = 0.0
        avg_queue_sum = 0.0
        overload_count = 0
        total_realized = 0.0

        for _ in range(steps_per_episode):

            # --- 顺序决策: 按 SOC 从低到高排序, 逐个选站 ---
            urgent_evs = [ev for ev in env.evs
                          if ev.status == "IDLE" and ev.soc < 30.0]
            urgent_evs.sort(key=lambda ev: ev.soc)

            actions = {}
            ev_transitions = []
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                action = agent.select_action(ev_state)
                actions[ev.id] = action
                ev_transitions.append((ev_state, action))
                pending_counts[action] += 1   # 后续EV能看到前面的分配

            next_global_state, reward, _, info = env.step(actions)

            # 存储每辆EV的独立经验
            for ev_state, act in ev_transitions:
                agent.store_transition(ev_state, act, reward, next_global_state)

            if ev_transitions:
                agent.replay(batch_size)

            total_reward += reward
            total_realized += info.get("realized_power", 0.0)

            total_queue = sum(len(s.queue) for s in env.stations)
            avg_queue_sum += total_queue / max(1, len(env.stations))

            overload_count += int(info.get("voltage_violations", 0))

        avg_queue = avg_queue_sum / steps_per_episode
        viz.add_episode_data(
            episode=e + 1,
            reward=total_reward,
            epsilon=agent.epsilon,
            avg_queue=avg_queue,
            overload_count=overload_count,
        )

        if (e + 1) % 10 == 0:
            print(
                f"Episode {e + 1}/{episodes}, Reward: {total_reward:.2f}, "
                f"RealizedPower: {total_realized:.1f}kW, "
                f"AvgQueue: {avg_queue:.2f}, "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    viz.plot_training_curves()
    viz.plot_reward_distribution()
    viz.plot_learning_progress()
    viz.save_data()
    viz.generate_summary_report()

    agent.save_model()
    print("训练结束！可视化结果已保存。")


if __name__ == "__main__":
    run_training()
