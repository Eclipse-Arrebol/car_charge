# 文件路径: train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import random
import numpy as np
import copy
from collections import deque

# 导入你的环境和模型
from simEvn.Traffic import TrafficPowerEnv
from model.GraphQNetwork import GraphQNetwork


# ==========================================
# 1. DQN 智能体 (带目标网络)
# ==========================================
class DQNAgent:
    def __init__(self, num_features, num_actions):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 策略网络 + 目标网络
        self.policy_net = GraphQNetwork(num_features, num_actions).to(self.device)
        self.target_net = GraphQNetwork(num_features, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 记忆库 (Replay Buffer)
        self.memory = deque(maxlen=5000)

        # 探索参数 (Epsilon-Greedy)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # 目标网络同步间隔 (每多少步同步一次)
        self.target_update_freq = 200
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

        # Q(s, a)
        q_values = self.policy_net(state_batch)
        curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 用 目标网络 计算 TD 目标: R + gamma * max Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + 0.99 * next_q

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
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
        """加载训练好的模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        print(f"模型已加载: {path} (epsilon={self.epsilon:.3f})")


# ==========================================
# 2. 主训练循环
# ==========================================
if __name__ == "__main__":
    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=9, num_actions=2)  # 9个特征(含EV指示器), 2个充电站

    episodes = 200
    batch_size = 32

    print("开始训练 (EV感知 + 顺序决策 + 目标网络 + episodic reset)...")

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
                ev_transitions.append((ev_state, action))
                pending_counts[action] += 1   # 后续EV能看到前面的分配

            # --- B. 环境执行 ---
            next_global_state, reward, done, info = env.step(actions)

            # --- C. 存储每辆EV的独立经验 ---
            for ev_state, act in ev_transitions:
                agent.store_transition(ev_state, act, reward, next_global_state)

            # --- D. 经验回放 ---
            if ev_transitions:
                agent.replay(batch_size)

            total_reward += reward

        if (e + 1) % 10 == 0:
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    agent.save_model()
    print("训练结束！")