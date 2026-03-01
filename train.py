# 文件路径: train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import random
import numpy as np
import copy
from collections import deque
import networkx as nx

# 导入你的环境和模型
from simEvn.Traffic import TrafficPowerEnv
from model.GraphQNetwork import GraphQNetwork


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

        # 记忆库 (Replay Buffer) — 扩大至 20000 条，保留更多历史经验
        self.memory = deque(maxlen=20000)

        # 探索参数 (Epsilon-Greedy) — 减慢衰减，匹配更长训练周期
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        # 目标网络同步间隔 (每多少步同步一次) — 缩短至 100 步，更及时更新
        self.target_update_freq = 100
        self.train_step_count = 0

    def select_action(self, state_data):
        """输入当前图状态，输出动作(去哪个站)"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_data = state_data.to(self.device)
            q_values = self.policy_net(state_data)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        """经验回放 (使用目标网络计算 TD 目标)"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        state_batch = Batch.from_data_list([m[0] for m in minibatch]).to(self.device)
        next_state_batch = Batch.from_data_list([m[3] for m in minibatch]).to(self.device)

        action_batch = torch.tensor([m[1] for m in minibatch], device=self.device)
        reward_batch = torch.tensor([m[2] for m in minibatch], dtype=torch.float, device=self.device)

        # Reward 截断：per-EV 路由奖励范围约 [-120, +40]，截断过大异常值
        reward_batch = torch.clamp(reward_batch, min=-120.0, max=40.0)

        # Q(s, a)
        q_values = self.policy_net(state_batch)
        curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Double DQN: 用策略网络选动作，目标网络估值，缓解 Q 值高估
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q = reward_batch + 0.99 * next_q

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # epsilon 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 定期同步目标网络
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="model/trained_dqn.pth"):
        """保存训练好的模型权重"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path="model/trained_dqn.pth"):
        """加载训练好的模型权重（兼容旧版不含 station_node_ids buffer 的 checkpoint）"""
        checkpoint = torch.load(path, map_location=self.device)
        # strict=False：旧 checkpoint 缺少 station_node_ids buffer 时不报错，
        # 该 buffer 会保留构造函数中设定的默认值
        missing, unexpected = self.policy_net.load_state_dict(
            checkpoint['policy_net'], strict=False
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
    agent = DQNAgent(num_features=10, num_actions=2)  # 10个特征(含SOC+距离), 2个充电站

    episodes = 800
    batch_size = 64

    print("开始训练 (EV感知 + 顺序决策 + Double DQN + 目标网络 + episodic reset)...")

    for e in range(episodes):
        env.reset()               # 每个 episode 重置环境
        total_reward = 0

        for time_step in range(100):

            # --- A. 顺序决策 (按 SOC 从低到高，逐个选站) ---
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

                # Per-EV 奖励：综合行驶距离 + 等待时间 + 电价
                target_st = env.stations[action]
                other_st  = env.stations[1 - action]
                eff_target = (len(target_st.queue) + len(target_st.connected_evs)
                              + pending_counts.get(action, 0))
                eff_other  = (len(other_st.queue) + len(other_st.connected_evs)
                              + pending_counts.get(1 - action, 0))
                # ① 行驶距离
                try:
                    dist_target = nx.shortest_path_length(
                        env.traffic_graph, ev.curr_node, target_st.traffic_node_id)
                except nx.NetworkXNoPath:
                    dist_target = 5
                try:
                    dist_other = nx.shortest_path_length(
                        env.traffic_graph, ev.curr_node, other_st.traffic_node_id)
                except nx.NetworkXNoPath:
                    dist_other = 5
                # ② 预估等待 (超出充电桩容量的排队车辆)
                excess_target = max(0, eff_target - target_st.num_chargers)
                excess_other  = max(0, eff_other  - other_st.num_chargers)
                # ③ 总预估代价 = 行驶时间 + 等待时间
                cost_target = dist_target + excess_target * 3.0
                cost_other  = dist_other  + excess_other  * 3.0
                # 反事实优势：选了更优方案则奖，否则罚
                per_ev_r = (cost_other - cost_target) * 4.0
                # 绝对代价惩罚
                per_ev_r -= cost_target * 2.0
                # 电价偏好：选低价站有奖励
                per_ev_r += (other_st.current_price - target_st.current_price) * 1.5

                ev_transitions.append((ev_state, action, per_ev_r))
                pending_counts[action] += 1   # 后续EV能看到前面的分配

            # --- B. 环境执行 ---
            next_global_state, reward, done, info = env.step(actions)

            # --- C. 存储每辆EV的独立经验（纯 per-EV 路由奖励，不混入全局奖励）---
            for ev_state, act, per_ev_r in ev_transitions:
                agent.store_transition(ev_state, act, per_ev_r, next_global_state)

            # --- D. 经验回放 ---
            if ev_transitions:
                agent.replay(batch_size)

            total_reward += reward

        if (e + 1) % 20 == 0:
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    agent.save_model()
    print("训练结束！")