import os
import time

import torch

from env.base_env import TrafficPowerEnv
from agents.dqn_base import DQNBase
from collections import OrderedDict


# ── 进度打印工具（供 training/trainer.py 复用）────────────────────────────

def _print_training_progress(
    episode,
    episodes,
    step,
    steps_per_episode,
    total_reward,
    epsilon,
    episode_start_time,
    extra_metrics="",
    update_every=10,
):
    """在同一行渲染紧凑的训练进度，每 update_every 步刷新一次。"""
    completed_steps = step + 1
    if (
        completed_steps != 1
        and completed_steps != steps_per_episode
        and completed_steps % max(1, int(update_every)) != 0
    ):
        return
    elapsed = time.time() - episode_start_time
    avg_reward = total_reward / completed_steps if completed_steps else 0.0
    message = (
        f"\r[Train] Episode {episode + 1}/{episodes} "
        f"Step {completed_steps}/{steps_per_episode} "
        f"Reward={total_reward:.2f} AvgStepReward={avg_reward:.2f} "
        f"Epsilon={epsilon:.3f} Elapsed={elapsed:.1f}s"
    )
    if extra_metrics:
        message += f" {extra_metrics}"
    print(message, end="", flush=True)


def _finish_progress_line():
    print()


# ============================================================
# DQN 智能体（单机，用于消融对比）
# ============================================================
class DQNAgent(DQNBase):
    """
    单机 Double DQN 智能体，继承 DQNBase。
    相比 FederatedClient，额外提供模型的保存与加载。
    """

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9):
        super().__init__(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
            memory_size=50000,
        )

    def replay(self, batch_size):
        """经验回放训练（封装基类 train_on_batch）。"""
        self.train_on_batch(batch_size)

    def save_model(self, path="checkpoints/trained_dqn.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path="checkpoints/trained_dqn.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = dict(checkpoint['policy_net'])

        # station_node_ids 必须与当前环境一致，不从 checkpoint 覆盖
        ckpt_station_ids = state_dict.pop('station_node_ids', None)
        if ckpt_station_ids is not None:
            cur_station_ids = self.policy_net.state_dict().get('station_node_ids')
            if cur_station_ids is not None and not torch.equal(ckpt_station_ids, cur_station_ids):
                print(
                    "[load_model] 忽略 checkpoint 中的 station_node_ids，"
                    f"使用当前环境站点: {cur_station_ids.tolist()}"
                )

        # 兼容旧版 checkpoint：输入层维度扩展时自动补零
        cur_state = self.policy_net.state_dict()
        for key in ("conv1.lin_l.weight", "conv1.lin_r.weight"):
            if key in state_dict and key in cur_state:
                old_w, new_w = state_dict[key], cur_state[key]
                if (old_w.shape != new_w.shape and old_w.ndim == 2 and new_w.ndim == 2
                        and old_w.shape[0] == new_w.shape[0]
                        and old_w.shape[1] < new_w.shape[1]):
                    upgraded = new_w.clone().zero_()
                    upgraded[:, :old_w.shape[1]] = old_w
                    state_dict[key] = upgraded
                    print(f"[load_model] 升级旧版输入层权重: {key} "
                          f"{tuple(old_w.shape)} -> {tuple(new_w.shape)}")

        missing, unexpected = self.policy_net.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_model] 旧版 checkpoint，缺少 key（使用默认值）: {missing}")
        if unexpected:
            print(f"[load_model] checkpoint 中有多余 key（已忽略）: {unexpected}")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = checkpoint.get('epsilon', 0.05)
        print(f"模型已加载: {path} (epsilon={self.epsilon:.3f})")


# ============================================================
# 合成路网训练（保留供 __main__ 调试，正式训练用 main.py train-real）
# ============================================================
if __name__ == "__main__":
    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=18, num_actions=2)

    episodes = 800
    batch_size = 64
    steps_per_episode = 100

    print("开始训练 (EV感知 + 顺序决策 + Double DQN + 目标网络 + episodic reset)...")

    for e in range(episodes):
        env.reset()
        total_reward = 0
        episode_start_time = time.time()

        for time_step in range(steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            ev_dispatch = []
            pending_counts = {s.id: 0 for s in env.stations}

            for ev in urgent_evs:
                ev_state = env.get_graph_state_for_ev(ev, pending_counts)
                action_mask = env.get_action_mask(ev)
                action = agent.select_action(ev_state, action_mask=action_mask)
                actions[ev.id] = action

                metrics = env.estimate_action_metrics(ev, action, pending_counts)
                per_ev_r = -metrics["generalized_cost"]
                per_ev_r -= 6.0 * metrics["queue_time_h"]
                per_ev_r -= 2.0 * metrics["trip_time_h"]

                ev_dispatch.append((ev, ev_state, action, per_ev_r, action_mask))
                pending_counts[action] += 1

            _, reward, done, info = env.step(actions)

            global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
            for ev, ev_state, act, per_ev_r, mask in ev_dispatch:
                realized = info["decision_costs"].get(ev.id, {})
                realized_cost = realized.get("generalized_cost", 0.0)
                mixed_r = per_ev_r - 0.2 * realized_cost + global_bonus
                next_ev_state = env.get_graph_state_for_ev(ev)
                agent.store_transition(ev_state, act, mixed_r, next_ev_state, action_mask=mask)

            if ev_dispatch or len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            total_reward += reward
            _print_training_progress(
                episode=e, episodes=episodes, step=time_step,
                steps_per_episode=steps_per_episode, total_reward=total_reward,
                epsilon=agent.epsilon, episode_start_time=episode_start_time,
                extra_metrics=f"Buffer={len(agent.memory)} UrgentEVs={len(urgent_evs)}",
                update_every=20,
            )

        agent.decay_epsilon()
        _finish_progress_line()

        if (e + 1) % 20 == 0:
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    agent.save_model()
    print("训练结束！")
