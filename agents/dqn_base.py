"""
DQN 基类 — DQNAgent 和 FederatedClient 的公共基础。

封装所有子类共享的逻辑：
  - 策略网络 + 目标网络构建
  - ε-greedy 动作选择（支持动作掩码）
  - 经验存储到回放缓冲区
  - 批次准备（图数据拼批、张量化）
  - Double DQN TD 误差计算
  - 梯度更新 + 目标网络定期同步

子类只需覆盖 _apply_gradients 即可注入 FedProx / DP-SGD 等差异逻辑。
"""

import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
from collections import deque

from agents.network import GraphQNetwork


def _clone_data_to_cpu(data):
    """将图数据复制到 CPU，避免回放缓冲区出现混合设备问题。"""
    return data.clone().cpu()


class DQNBase:
    def __init__(
        self,
        num_features,
        num_actions,
        station_node_ids=None,
        num_nodes_per_graph=9,
        lr=0.0003,
        memory_size=20000,
        epsilon_decay=0.9999,
        epsilon_min=0.05,
        target_update_freq=100,
    ):
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.memory = deque(maxlen=memory_size)

        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self.train_step_count = 0

    # ------------------------------------------------------------------
    # 动作选择
    # ------------------------------------------------------------------

    def select_action(self, state_data, action_mask=None):
        """ε-greedy 动作选择，探索时同样遵守动作掩码。"""
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

    # ------------------------------------------------------------------
    # 经验存储
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, action_mask=None):
        state_cpu = _clone_data_to_cpu(state)
        next_state_cpu = _clone_data_to_cpu(next_state)
        mask_cpu = action_mask.detach().clone().cpu() if action_mask is not None else None
        self.memory.append((state_cpu, action, reward, next_state_cpu, mask_cpu))

    # ------------------------------------------------------------------
    # ε 衰减
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # 训练核心
    # ------------------------------------------------------------------

    def _prepare_batch(self, minibatch):
        """将采样的经验列表转换为训练所需的张量批次。"""
        state_batch = Batch.from_data_list([m[0] for m in minibatch]).to(self.device)
        next_state_batch = Batch.from_data_list([m[3] for m in minibatch]).to(self.device)
        action_batch = torch.tensor([m[1] for m in minibatch], device=self.device)
        reward_batch = torch.tensor(
            [m[2] for m in minibatch], dtype=torch.float, device=self.device
        )
        mask_list = [
            m[4] if len(m) > 4 and m[4] is not None else self.default_action_mask
            for m in minibatch
        ]
        mask_batch = torch.cat(mask_list, dim=0).to(self.device)
        return state_batch, next_state_batch, action_batch, reward_batch, mask_batch

    def _compute_td_loss(self, state_batch, next_state_batch, action_batch,
                         reward_batch, mask_batch):
        """Double DQN TD 误差（MSE Loss）。"""
        reward_batch = torch.clamp(reward_batch, min=-500.0, max=50.0)

        q_values = self.policy_net(state_batch)
        curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_batch, action_mask=mask_batch)
            next_actions = next_q_policy.argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q = reward_batch + 0.99 * next_q

        return F.mse_loss(curr_q, target_q)

    def _apply_gradients(self, loss, batch_size=None):
        """反向传播 + 梯度裁剪 + 参数更新。
        子类可覆盖此方法以注入 FedProx 正则项或 DP-SGD。
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

    def _sync_target(self):
        """按频率将策略网络参数同步到目标网络。"""
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_on_batch(self, batch_size):
        """采样一个批次并执行完整的一步训练。"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        batches = self._prepare_batch(minibatch)
        loss = self._compute_td_loss(*batches)
        self._apply_gradients(loss, batch_size=len(minibatch))
        self._sync_target()
