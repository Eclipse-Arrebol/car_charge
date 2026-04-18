"""
联邦学习 DQN (Federated DQN)
==============================
基于 FedAvg 算法的联邦强化学习架构。

  - FederatedClient: 继承 DQNBase，覆盖 _apply_gradients 以注入 FedProx + DP-SGD
  - FederatedServer:  维护全局模型，执行 FedAvg 加权聚合

工作流程 (每个联邦轮次):
  1. Server → Clients:  下发全局模型参数
  2. Clients:            各自在本地 EV 经验上训练若干步
  3. Clients → Server:   上传本地模型参数
  4. Server:             按 FedAvg 加权聚合 → 更新全局模型
"""

import copy
import torch
from collections import OrderedDict

from agents.dqn_base import DQNBase
from agents.network import GraphQNetwork

try:
    from opacus.accountants import RDPAccountant
    _OPACUS_AVAILABLE = True
except ImportError:
    _OPACUS_AVAILABLE = False


# ============================================================
# 1. 联邦客户端 (Federated Client)
# ============================================================
class FederatedClient(DQNBase):
    """
    联邦学习客户端，对应一个充电站区域。

    在 DQNBase 基础上新增：
      - FedProx 近端正则项（_apply_gradients 中注入）
      - DP-SGD 差分隐私梯度噪声（_apply_gradients 中注入）
      - 联邦相关：load_global_model / get_model_params / reset_round_counter
    """

    def __init__(self, client_id, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9,
                 lr=0.0003, memory_size=20000, proximal_mu=1e-4,
                 use_dp=False, dp_noise_multiplier=1.0, dp_clip_C=1.0,
                 dp_delta=1e-5, dp_sample_rate=0.01):
        super().__init__(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
            lr=lr,
            memory_size=memory_size,
        )
        self.client_id = client_id
        self.proximal_mu = proximal_mu

        # 差分隐私参数
        self.use_dp = use_dp and _OPACUS_AVAILABLE
        self.dp_noise_multiplier = dp_noise_multiplier  # σ，越大隐私越强
        self.dp_clip_C = dp_clip_C                      # 梯度裁剪范数 C
        self.dp_delta = dp_delta                        # (ε,δ)-DP 中的 δ
        self.dp_sample_rate = dp_sample_rate            # q = batch_size / 数据集大小
        if self.use_dp:
            self._privacy_accountant = RDPAccountant()
        elif use_dp and not _OPACUS_AVAILABLE:
            print(f"[FedClient {client_id}] 警告: opacus 未安装，DP 已跳过")

        # 联邦专用状态
        self.num_samples_this_round = 0   # 本轮样本数（用于 FedAvg 加权）
        self.global_anchor_params = None  # 最近一次下发的全局参数（FedProx 锚点）
        self.local_train_call_count = 0
        self.verbose = False

    # ------------------------------------------------------------------
    # 覆盖：经验存储时顺带计数（供 FedAvg 加权）
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, action_mask=None):
        super().store_transition(state, action, reward, next_state, action_mask)
        self.num_samples_this_round += 1

    # ------------------------------------------------------------------
    # 覆盖：梯度更新 — 注入 FedProx + DP-SGD
    # ------------------------------------------------------------------

    def _apply_gradients(self, loss, batch_size=None):
        # FedProx：约束本地模型不偏离全局锚点
        if self.global_anchor_params is not None and self.proximal_mu > 0:
            prox_term = torch.zeros(1, device=self.device)
            for name, param in self.policy_net.named_parameters():
                anchor = self.global_anchor_params.get(name)
                if anchor is not None:
                    prox_term = prox_term + torch.sum((param - anchor) ** 2)
            loss = loss + 0.5 * self.proximal_mu * prox_term

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.use_dp:
            # DP-SGD：裁剪到 C，再注入高斯噪声 N(0, (σC/batch)²)
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), max_norm=self.dp_clip_C
            )
            n = batch_size or 1
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * (
                        self.dp_noise_multiplier * self.dp_clip_C / n
                    )
                    param.grad.add_(noise)
            self._privacy_accountant.step(
                noise_multiplier=self.dp_noise_multiplier,
                sample_rate=self.dp_sample_rate,
            )
        else:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

    # ------------------------------------------------------------------
    # 联邦专用：批量本地训练
    # ------------------------------------------------------------------

    def local_train(self, batch_size, num_steps=1):
        """执行 num_steps 步本地训练。"""
        for _ in range(num_steps):
            self.train_on_batch(batch_size)
            self.local_train_call_count += 1
            if self.verbose and self.local_train_call_count % 10 == 0:
                print(f"[FedClient {self.client_id}] local_train "
                      f"step={self.local_train_call_count}")

    # ------------------------------------------------------------------
    # 联邦专用：模型下发 / 上传 / 隐私预算
    # ------------------------------------------------------------------

    def load_global_model(self, global_state_dict):
        """从全局服务器加载模型参数，保留本地 station_node_ids buffer。"""
        local_dict = self.policy_net.state_dict()
        filtered = OrderedDict(
            (k, local_dict[k] if k == 'station_node_ids' else v)
            for k, v in global_state_dict.items()
        )
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
        """返回本地模型参数（用于上传给服务器聚合）。"""
        return copy.deepcopy(self.policy_net.state_dict())

    def get_privacy_spent(self):
        """返回当前累积的 (ε, δ) 隐私预算；未启用 DP 时返回 None。"""
        if not self.use_dp:
            return None
        eps = self._privacy_accountant.get_epsilon(delta=self.dp_delta)
        return {"epsilon": round(eps, 4), "delta": self.dp_delta}

    def reset_round_counter(self):
        """每个联邦轮次开始时重置样本计数。"""
        self.num_samples_this_round = 0

    # ------------------------------------------------------------------
    # 调试工具
    # ------------------------------------------------------------------

    def _parameter_mean(self):
        means = [
            t.detach().float().mean().item()
            for t in self.policy_net.state_dict().values()
            if torch.is_floating_point(t)
        ]
        return float(sum(means) / max(1, len(means)))

    def optimizer_debug_state(self):
        return {
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "state_entries": int(sum(len(s) for s in self.optimizer.state.values())),
        }


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

        self.global_model = GraphQNetwork(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
        ).to(self.device)

        self.num_actions = num_actions
        self.clients = []

    def _state_param_mean(self, state_dict):
        means = [
            t.detach().float().mean().item()
            for t in state_dict.values()
            if torch.is_floating_point(t)
        ]
        return float(sum(means) / max(1, len(means)))

    def register_client(self, client):
        self.clients.append(client)

    def distribute_global_model(self):
        """将全局模型参数下发给所有客户端。"""
        global_params = copy.deepcopy(self.global_model.state_dict())
        if self.verbose:
            print(f"[FedServer] distribute global_param_mean="
                  f"{self._state_param_mean(global_params):.6f}")
        for client in self.clients:
            client.load_global_model(global_params)

    def aggregate(self):
        """FedAvg 聚合：按各客户端本轮样本数加权平均模型参数。"""
        client_params = [c.get_model_params() for c in self.clients]
        client_weights = [max(1, c.num_samples_this_round) for c in self.clients]
        total_weight = sum(client_weights)
        if total_weight == 0:
            return

        if self.verbose:
            before_mean = self._state_param_mean(self.global_model.state_dict())
            parts = [
                f"client{c.client_id}:mean={self._state_param_mean(p):.6f},samples={w}"
                for c, p, w in zip(self.clients, client_params, client_weights)
            ]
            print(f"[FedServer] aggregate_before global_param_mean={before_mean:.6f} | "
                  + " | ".join(parts))

        aggregated = OrderedDict()
        for key in client_params[0]:
            if key == 'station_node_ids':
                aggregated[key] = self.global_model.state_dict()[key]
                continue
            aggregated[key] = sum(
                client_params[i][key].float() * (client_weights[i] / total_weight)
                for i in range(len(client_params))
            ).to(client_params[0][key].dtype)

        if self.aggregation_momentum < 1.0:
            current_global = self.global_model.state_dict()
            smoothed = OrderedDict()
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
            print(f"[FedServer] aggregate_after global_param_mean="
                  f"{self._state_param_mean(aggregated):.6f}")
        self.global_model.load_state_dict(aggregated)

        for client in self.clients:
            client.reset_round_counter()

    def save_global_model(self, path="checkpoints/trained_federated_dqn.pth"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        min_eps = min((c.epsilon for c in self.clients), default=0.05)
        torch.save({
            'policy_net': self.global_model.state_dict(),
            'epsilon': min_eps,
        }, path)
        print(f"[FedServer] 全局模型已保存: {path}")

    def load_global_model(self, path="checkpoints/trained_federated_dqn.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = dict(checkpoint['policy_net'])
        state_dict.pop('station_node_ids', None)
        self.global_model.load_state_dict(state_dict, strict=False)
        eps = checkpoint.get('epsilon', 0.05)
        self.distribute_global_model()
        for client in self.clients:
            client.epsilon = eps
        print(f"[FedServer] 全局模型已加载: {path} (epsilon={eps:.3f})")
