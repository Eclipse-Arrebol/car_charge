"""
联邦学习 DQN (Federated DQN)
==============================
基于 FedAvg 算法的联邦强化学习架构。

架构设计:
  - FederatedServer: 全局服务器，维护全局模型，执行 FedAvg 聚合
  - FederatedClient: 本地客户端（每个充电站区域一个），
                     在本区域 EV 数据上本地训练，定期上传模型更新

工作流程 (每个联邦轮次):
  1. Server → Clients:  下发全局模型参数
  2. Clients:            各自在本地 EV 经验上训练若干步
  3. Clients → Server:   上传本地模型参数
  4. Server:             按 FedAvg 加权聚合 → 更新全局模型
"""

import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import random
from collections import deque, OrderedDict

from agents.GraphQNetwork import GraphQNetwork


# ============================================================
# 1. 联邦客户端 (Federated Client)
# ============================================================
class FederatedClient:
    """
    联邦学习客户端，对应一个充电站区域。
    拥有独立的策略网络、目标网络、经验回放池。
    """

    def __init__(self, client_id, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9,
                 lr=0.0003, memory_size=20000):
        self.client_id = client_id
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 本地策略网络 + 目标网络
        self.policy_net = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
        ).to(self.device)
        self.target_net = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 本地经验回放池
        self.memory = deque(maxlen=memory_size)

        # 探索参数
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

        self.target_update_freq = 100
        self.train_step_count = 0

        # 本轮训练的样本数（用于 FedAvg 加权）
        self.num_samples_this_round = 0

    def load_global_model(self, global_state_dict):
        """从全局服务器加载模型参数"""
        # 保留本地的 station_node_ids buffer（可能与全局不同）
        local_dict = self.policy_net.state_dict()
        filtered = OrderedDict()
        for k, v in global_state_dict.items():
            if k == 'station_node_ids':
                filtered[k] = local_dict[k]
            else:
                filtered[k] = v
        self.policy_net.load_state_dict(filtered)
        self.target_net.load_state_dict(filtered)

    def get_model_params(self):
        """返回本地模型参数（用于上传给服务器聚合）"""
        return copy.deepcopy(self.policy_net.state_dict())

    def select_action(self, state_data, action_mask=None):
        """输入图状态 + 动作掩码，输出动作"""
        if random.random() < self.epsilon:
            if action_mask is not None:
                valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
                if valid:
                    return random.choice(valid)
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state_data = state_data.to(self.device)
            if action_mask is not None:
                action_mask = action_mask.to(self.device)
            q_values = self.policy_net(state_data, action_mask=action_mask)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, action_mask=None):
        self.memory.append((state, action, reward, next_state, action_mask))
        self.num_samples_this_round += 1

    def local_train(self, batch_size, num_steps=1):
        """本地训练若干步"""
        for _ in range(num_steps):
            if len(self.memory) < batch_size:
                return

            minibatch = random.sample(self.memory, batch_size)

            state_batch = Batch.from_data_list([m[0] for m in minibatch]).to(self.device)
            next_state_batch = Batch.from_data_list([m[3] for m in minibatch]).to(self.device)

            action_batch = torch.tensor([m[1] for m in minibatch], device=self.device)
            reward_batch = torch.tensor([m[2] for m in minibatch],
                                        dtype=torch.float, device=self.device)

            # 拼装 action mask batch
            mask_list = []
            for m in minibatch:
                if len(m) > 4 and m[4] is not None:
                    mask_list.append(m[4])
                else:
                    mask_list.append(torch.ones(1, self.num_actions, dtype=torch.bool))
            mask_batch = torch.cat(mask_list, dim=0).to(self.device)

            reward_batch = torch.clamp(reward_batch, min=-120.0, max=40.0)

            # Q(s, a)
            q_values = self.policy_net(state_batch)
            curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            # Double DQN + Action Mask
            with torch.no_grad():
                next_q_policy = self.policy_net(next_state_batch, action_mask=mask_batch)
                next_actions = next_q_policy.argmax(1, keepdim=True)
                next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
                target_q = reward_batch + 0.99 * next_q

            loss = F.mse_loss(curr_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()

            # epsilon 衰减
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # 目标网络同步
            self.train_step_count += 1
            if self.train_step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_round_counter(self):
        """每个联邦轮次开始时重置样本计数"""
        self.num_samples_this_round = 0


# ============================================================
# 2. 联邦服务器 (Federated Server) — FedAvg
# ============================================================
class FederatedServer:
    """
    联邦学习服务器，维护全局模型，执行 FedAvg 聚合。

    FedAvg 公式:
        θ_global = Σ (n_k / n_total) · θ_k
    其中 n_k 为客户端 k 的本地训练样本数。
    """

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 全局模型（仅用于聚合和下发，不直接训练）
        self.global_model = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
        ).to(self.device)

        self.num_actions = num_actions
        self.clients = []

    def register_client(self, client):
        """注册联邦客户端"""
        self.clients.append(client)

    def distribute_global_model(self):
        """将全局模型参数下发给所有客户端"""
        global_params = copy.deepcopy(self.global_model.state_dict())
        for client in self.clients:
            client.load_global_model(global_params)

    def aggregate(self):
        """
        FedAvg 聚合：按各客户端本轮样本数加权平均模型参数。
        """
        # 收集各客户端的模型参数和样本数
        client_params = []
        client_weights = []
        for client in self.clients:
            params = client.get_model_params()
            n_samples = max(1, client.num_samples_this_round)
            client_params.append(params)
            client_weights.append(n_samples)

        total_weight = sum(client_weights)
        if total_weight == 0:
            return

        # 加权平均
        aggregated = OrderedDict()
        for key in client_params[0]:
            if key == 'station_node_ids':
                # buffer 不参与聚合，保留全局模型的值
                aggregated[key] = self.global_model.state_dict()[key]
                continue
            aggregated[key] = sum(
                client_params[i][key].float() * (client_weights[i] / total_weight)
                for i in range(len(client_params))
            ).to(client_params[0][key].dtype)

        self.global_model.load_state_dict(aggregated)

        # 重置各客户端的轮次样本计数
        for client in self.clients:
            client.reset_round_counter()

    def save_global_model(self, path="checkpoints/trained_federated_dqn.pth"):
        """保存全局模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 取所有客户端最小的 epsilon 作为代表
        min_eps = min((c.epsilon for c in self.clients), default=0.05)
        torch.save({
            'policy_net': self.global_model.state_dict(),
            'epsilon': min_eps,
        }, path)
        print(f"[FedServer] 全局模型已保存: {path}")

    def load_global_model(self, path="checkpoints/trained_federated_dqn.pth"):
        """加载全局模型并下发给客户端"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = dict(checkpoint['policy_net'])
        state_dict.pop('station_node_ids', None)
        self.global_model.load_state_dict(state_dict, strict=False)
        eps = checkpoint.get('epsilon', 0.05)
        self.distribute_global_model()
        for client in self.clients:
            client.epsilon = eps
        print(f"[FedServer] 全局模型已加载: {path} (epsilon={eps:.3f})")
