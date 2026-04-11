# 文件路径: train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import random
import numpy as np
import copy
import time
from collections import deque
import networkx as nx

# 导入你的环境和模型
from env.Traffic import TrafficPowerEnv
from agents.GraphQNetwork import GraphQNetwork


def _clone_data_to_cpu(data):
    """Keep replay-buffer graph samples on CPU to avoid mixed-device batches."""
    return data.clone().cpu()


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
    """Render a compact single-line progress view for long-running training."""
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


# ==========================================
# 1. DQN 智能体 (带目标网络)
# ==========================================
class DQNAgent:
    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络 + 目标网络
        self.policy_net = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph
        ).to(self.device)
        self.target_net = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.default_action_mask = torch.ones(1, self.num_actions, dtype=torch.bool)

        # 记忆库 (Replay Buffer)
        self.memory = deque(maxlen=50000)

        # 探索参数 (Epsilon-Greedy) — 更慢衰减，充分探索两站的差异
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999

        # 目标网络同步间隔 (每多少步同步一次) — 缩短至 100 步，更及时更新
        self.target_update_freq = 100
        self.train_step_count = 0

    def select_action(self, state_data, action_mask=None):
        """输入当前图状态 + 可选动作掩码，输出动作(去哪个站)"""
        if random.random() < self.epsilon:
            # 探索时也遵守掩码：只从有效动作中随机选
            if action_mask is not None:
                valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
                if valid:
                    return random.choice(valid)
            return random.randint(0, self.num_actions - 1)

        with torch.inference_mode():
            state_data = state_data.to(self.device)
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
            q_values = self.policy_net(state_data, action_mask=action_mask)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, action_mask=None):
        state_cpu = _clone_data_to_cpu(state)
        next_state_cpu = _clone_data_to_cpu(next_state)
        mask_cpu = action_mask.detach().clone().cpu() if action_mask is not None else None
        self.memory.append((state_cpu, action, reward, next_state_cpu, mask_cpu))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size):
        """经验回放 (使用目标网络计算 TD 目标, 支持 Action Mask)"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        state_batch = Batch.from_data_list([m[0] for m in minibatch]).to(self.device)
        next_state_batch = Batch.from_data_list([m[3] for m in minibatch]).to(self.device)

        action_batch = torch.tensor([m[1] for m in minibatch], device=self.device)
        reward_batch = torch.tensor([m[2] for m in minibatch], dtype=torch.float, device=self.device)

        # 拼装 action mask batch (兼容旧版无 mask 的经验)
        mask_list = []
        for m in minibatch:
            if len(m) > 4 and m[4] is not None:
                mask_list.append(m[4])
            else:
                mask_list.append(self.default_action_mask)
        mask_batch = torch.cat(mask_list, dim=0).to(self.device)  # [B, num_actions]

        # Reward 截断：per-EV 路由奖励范围约 [-120, +40]，截断过大异常值
        reward_batch = torch.clamp(reward_batch, min=-500.0, max=50.0)

        # Q(s, a) — 训练时不施加 mask（mask 只影响动作选择，不影响 Q 值回归）
        q_values = self.policy_net(state_batch)
        curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Double DQN: 用策略网络选动作(带 mask)，目标网络估值
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch, action_mask=mask_batch)
            next_actions = next_q_policy.argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q = reward_batch + 0.99 * next_q

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 定期同步目标网络
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="checkpoints/trained_dqn.pth"):
        """保存训练好的模型权重"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path="checkpoints/trained_dqn.pth"):
        """加载训练好的模型权重（兼容旧版不含 station_node_ids buffer 的 checkpoint）"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = dict(checkpoint['policy_net'])

        # station_node_ids 必须与当前环境一致；不从 checkpoint 覆盖，避免跨路网越界。
        ckpt_station_ids = state_dict.pop('station_node_ids', None)
        if ckpt_station_ids is not None:
            cur_station_ids = self.policy_net.state_dict().get('station_node_ids', None)
            if cur_station_ids is not None and not torch.equal(ckpt_station_ids, cur_station_ids):
                print(
                    "[load_model] 忽略 checkpoint 中的 station_node_ids，"
                    f"使用当前环境站点: {cur_station_ids.tolist()}"
                )

        # 兼容旧版 checkpoint：若输入特征进一步扩展，
        # 自动扩展第一层输入权重，保留原有 10 维参数，新增维度置 0。
        cur_state = self.policy_net.state_dict()
        for key in ("conv1.lin_l.weight", "conv1.lin_r.weight"):
            if key in state_dict and key in cur_state:
                old_w = state_dict[key]
                new_w = cur_state[key]
                if old_w.shape != new_w.shape and old_w.ndim == 2 and new_w.ndim == 2:
                    if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] < new_w.shape[1]:
                        upgraded = new_w.clone()
                        upgraded.zero_()
                        upgraded[:, :old_w.shape[1]] = old_w
                        state_dict[key] = upgraded
                        print(
                            f"[load_model] 升级旧版输入层权重: {key} "
                            f"{tuple(old_w.shape)} -> {tuple(new_w.shape)}"
                        )

        # strict=False：旧 checkpoint 缺少 station_node_ids buffer 时不报错，
        # 该 buffer 会保留构造函数中设定的默认值
        missing, unexpected = self.policy_net.load_state_dict(
            state_dict, strict=False
        )
        if missing:
            print(f"[load_model] 旧版 checkpoint，缺少 key（使用默认值）: {missing}")
        if unexpected:
            print(f"[load_model] checkpoint 中有多余 key（已忽略）: {unexpected}")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = checkpoint.get('epsilon', 0.05)
        print(f"模型已加载: {path} (epsilon={self.epsilon:.3f})")


# ==========================================
# 2. 主训练循环
# ==========================================
if __name__ == "__main__":
    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=18, num_actions=2)  # 18个特征, 2个充电站

    episodes = 800
    batch_size = 64
    steps_per_episode = 100

    print("开始训练 (EV感知 + 顺序决策 + Double DQN + 目标网络 + episodic reset)...")

    for e in range(episodes):
        env.reset()               # 每个 episode 重置环境
        total_reward = 0
        episode_start_time = time.time()

        for time_step in range(steps_per_episode):

            # --- A. 顺序决策：利用率导向奖励 + per-EV 状态 ---
            urgent_evs = env.get_pending_decision_evs()

            actions = {}
            ev_dispatch = []       # (ev, ev_state, action, shaped_reward)
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

            # --- B. 环境执行 ---
            _, reward, done, info = env.step(actions)

            # --- C. 经验存储：动作成本 + 全局联合目标混合 ---
            global_bonus = reward / max(1, len(ev_dispatch)) * 0.3
            for ev, ev_state, act, per_ev_r, mask in ev_dispatch:
                realized = info["decision_costs"].get(ev.id, {})
                realized_cost = realized.get("generalized_cost", 0.0)
                mixed_r = per_ev_r - 0.2 * realized_cost + global_bonus
                next_ev_state = env.get_graph_state_for_ev(ev)
                agent.store_transition(ev_state, act, mixed_r, next_ev_state, action_mask=mask)

            # --- D. 经验回放 ---
            if ev_dispatch or len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            total_reward += reward
            _print_training_progress(
                episode=e,
                episodes=episodes,
                step=time_step,
                steps_per_episode=steps_per_episode,
                total_reward=total_reward,
                epsilon=agent.epsilon,
                episode_start_time=episode_start_time,
                extra_metrics=f"Buffer={len(agent.memory)} UrgentEVs={len(urgent_evs)}",
                update_every=20,
            )

        agent.decay_epsilon()
        _finish_progress_line()

        if (e + 1) % 20 == 0:
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    agent.save_model()
    print("训练结束！")
