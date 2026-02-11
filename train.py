# 文件路径: train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import random
import numpy as np
from collections import deque

# 导入你的环境和模型
from simEvn.Traffic import TrafficPowerEnv
from model.GraphQNetwork import GraphQNetwork


# ==========================================
# 1. DQN 智能体 (Teacher)
# ==========================================
class DQNAgent:
    def __init__(self, num_features, num_actions):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 大脑 (Q-Network)
        self.policy_net = GraphQNetwork(num_features, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # 记忆库 (Replay Buffer)
        self.memory = deque(maxlen=2000)

        # 探索参数 (Epsilon-Greedy)
        self.epsilon = 1.0  # 初始完全随机
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state_data):
        """输入当前图状态，输出动作(去哪个站)"""
        # 1. 探索: 随机乱选
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # 2. 利用: 听模型的
        with torch.no_grad():
            state_data = state_data.to(self.device)
            q_values = self.policy_net(state_data)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        """经验回放: 从记忆库里抽取数据训练模型"""
        if len(self.memory) < batch_size:
            return

        # 随机抽取 batch
        minibatch = random.sample(self.memory, batch_size)

        # 整理数据 (PyG 特有的 Batch 方式)
        state_batch = Batch.from_data_list([m[0] for m in minibatch]).to(self.device)
        next_state_batch = Batch.from_data_list([m[3] for m in minibatch]).to(self.device)

        action_batch = torch.tensor([m[1] for m in minibatch], device=self.device)
        reward_batch = torch.tensor([m[2] for m in minibatch], dtype=torch.float, device=self.device)

        # --- 核心训练逻辑 ---
        # 1. 预测值: Q(s, a)
        q_values = self.policy_net(state_batch)
        # 提取出实际执行那个动作对应的 Q 值
        curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # 2. 目标值: R + gamma * max(Q(s', a'))
        with torch.no_grad():
            next_q = self.policy_net(next_state_batch).max(1)[0]
            target_q = reward_batch + 0.99 * next_q  # gamma = 0.99

        # 3. 梯度下降
        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. 减少探索
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        self.epsilon = checkpoint.get('epsilon', 0.05)
        print(f"模型已加载: {path} (epsilon={self.epsilon:.3f})")


# ==========================================
# 2. 主训练循环 (Main Loop)
# ==========================================
if __name__ == "__main__":
    # 初始化
    env = TrafficPowerEnv()
    agent = DQNAgent(num_features=8, num_actions=2)  # 8个特征, 2个充电站

    episodes = 50
    batch_size = 32

    print("开始训练...")

    for e in range(episodes):
        # 每次训练开始前可以重置一些统计变量，但这里环境是持续运行的
        # 也就是 'Continual Learning'，不需要 env.reset()

        total_reward = 0
        step_count = 0

        # 每一轮跑 100 步
        for time_step in range(100):

            # --- A. 获取当前状态 ---
            current_state = env.get_graph_state()  # PyG Data 对象

            # --- B. 决策 ---
            # 这里的逻辑是：所有需要充电的车共用这一个大脑
            # 如果有多个车同时请求，我们简单地多次调用模型(或做成batch)
            actions = {}
            active_requests = False  # 标记这一步有没有车需要决策

            # 找出所有需要充电且还没出发的车
            for ev in env.evs:
                if ev.status == "IDLE" and ev.soc < 30.0:
                    # 询问大脑: "我现在该去哪?"
                    action = agent.select_action(current_state)
                    actions[ev.id] = action
                    active_requests = True

            # --- C. 环境执行 ---
            next_state, reward, done, info = env.step(actions)

            # --- D. 存储经验 & 学习 ---
            # 只有当这一步确实做出了决策（有车要充电），我们才把它作为一条经验存下来
            # 如果这一步大家都在瞎跑，没有决策发生，就没有"因果关系"，不存。
            if active_requests:
                # 注意：这里我们把"整张图的状态"和"这一步的全局奖励"存下来
                # Action 存的是其中一个决策（简化处理，假设 parameter sharing）
                # 更严谨的做法是把每个车的决策都单独存一条
                for ev_id, act in actions.items():
                    agent.store_transition(current_state, act, reward, next_state)
                    agent.replay(batch_size)

            total_reward += reward
            step_count += 1

        print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    agent.save_model()
    print("训练结束！")