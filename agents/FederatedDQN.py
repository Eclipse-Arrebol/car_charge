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


def _clone_data_to_cpu(data):
    """Keep replay-buffer graph samples on CPU to avoid mixed-device batches."""
    return data.clone().cpu()


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
                 lr=0.0003, memory_size=20000, proximal_mu=1e-4):
        self.client_id = client_id
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proximal_mu = proximal_mu
        self.lr = lr

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
        self.default_action_mask = torch.ones(1, self.num_actions, dtype=torch.bool)

        # 本地经验回放池
        self.memory = deque(maxlen=memory_size)

        # 探索参数
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999

        self.target_update_freq = 100
        self.train_step_count = 0
        self.local_train_call_count = 0

        # 本轮训练的样本数（用于 FedAvg 加权）
        self.num_samples_this_round = 0
        self.global_anchor_params = None
        self.verbose = False

    def _parameter_mean(self):
        means = []
        for tensor in self.policy_net.state_dict().values():
            if torch.is_floating_point(tensor):
                means.append(tensor.detach().float().mean().item())
        return float(sum(means) / max(1, len(means)))

    def optimizer_debug_state(self):
        state_entries = sum(len(state) for state in self.optimizer.state.values())
        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "lr": float(lr),
            "state_entries": int(state_entries),
        }

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
        self.global_anchor_params = {
            name: tensor.detach().clone().to(self.device)
            for name, tensor in self.policy_net.state_dict().items()
            if torch.is_floating_point(tensor)
        }
        if self.verbose:
            opt_state = self.optimizer_debug_state()
            print(
                f"[FedClient {self.client_id}] load_global_model "
                f"param_mean={self._parameter_mean():.6f} "
                f"lr={opt_state['lr']:.6f} optimizer_state_entries={opt_state['state_entries']}"
            )

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
        self.num_samples_this_round += 1

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
                    mask_list.append(self.default_action_mask)
            mask_batch = torch.cat(mask_list, dim=0).to(self.device)

            reward_batch = torch.clamp(reward_batch, min=-500.0, max=50.0)

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

            # FedProx: 约束本地模型不要偏离最近一次下发的全局模型过远，
            # 缓解非IID样本导致的客户端漂移。
            if self.global_anchor_params is not None and self.proximal_mu > 0:
                prox_term = torch.zeros(1, device=self.device)
                for name, param in self.policy_net.named_parameters():
                    anchor = self.global_anchor_params.get(name, None)
                    if anchor is not None:
                        prox_term = prox_term + torch.sum((param - anchor) ** 2)
                loss = loss + 0.5 * self.proximal_mu * prox_term

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()
            self.local_train_call_count += 1
            if self.verbose and self.local_train_call_count % 10 == 0:
                print(
                    f"[FedClient {self.client_id}] local_train "
                    f"step={self.local_train_call_count} loss={loss.item():.6f}"
                )

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
                 station_node_ids=None, num_nodes_per_graph=9,
                 aggregation_momentum=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregation_momentum = aggregation_momentum
        self.verbose = False

        # 全局模型（仅用于聚合和下发，不直接训练）
        self.global_model = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
        ).to(self.device)

        self.num_actions = num_actions
        self.clients = []

    def _state_param_mean(self, state_dict):
        means = []
        for tensor in state_dict.values():
            if torch.is_floating_point(tensor):
                means.append(tensor.detach().float().mean().item())
        return float(sum(means) / max(1, len(means)))

    def register_client(self, client):
        """注册联邦客户端"""
        self.clients.append(client)

    def distribute_global_model(self):
        """将全局模型参数下发给所有客户端"""
        global_params = copy.deepcopy(self.global_model.state_dict())
        if self.verbose:
            global_mean = self._state_param_mean(global_params)
            print(f"[FedServer] distribute_global_model global_param_mean={global_mean:.6f}")
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

        if self.verbose:
            before_mean = self._state_param_mean(self.global_model.state_dict())
            client_mean_parts = []
            for client, params, weight in zip(self.clients, client_params, client_weights):
                client_mean_parts.append(
                    f"client{client.client_id}:mean={self._state_param_mean(params):.6f},samples={weight}"
                )
            print(
                f"[FedServer] aggregate_before global_param_mean={before_mean:.6f} | "
                + " | ".join(client_mean_parts)
            )

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
        if self.aggregation_momentum < 1.0:
            smoothed = OrderedDict()
            current_global = self.global_model.state_dict()
            for key, value in aggregated.items():
                if key == 'station_node_ids' or not torch.is_floating_point(value):
                    smoothed[key] = value
                    continue
                smoothed[key] = (
                    (1.0 - self.aggregation_momentum) * current_global[key].float()
                    + self.aggregation_momentum * value.float()
                ).to(current_global[key].dtype)
            aggregated = smoothed

        if self.verbose:
            after_mean = self._state_param_mean(aggregated)
            print(f"[FedServer] aggregate_after global_param_mean={after_mean:.6f}")
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
