import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


# ── 特征分组索引（对应 Traffic.py get_graph_state 的 18 维布局）──────────
# 队列/容量: is_station(1), queue(2), connected(4), load(5), pred_arrivals(14)
_IDX_QUEUE   = [1, 2, 4, 5, 14, 15, 16, 17]
# 价格/时间: price(3), tou(7), travel_time(10), service_time(11), gen_cost(12), price_noise(13)
_IDX_COST    = [3, 7, 10, 11, 12, 13]
# 空间/电网: ev_count(0), voltage(6), inv_dist(9)
_IDX_SPATIAL = [0, 6, 9]
# EV 状态: SOC(8)
_IDX_EV      = [8]


class FeatureEncoder(nn.Module):
    """
    分组特征编码器 — 将 18 维异质节点特征按语义分组独立编码后融合。

    四个语义组:
      队列/容量组  [1,2,4,5,14]      → 站点拥堵状态（5维）
      价格/时间组  [3,7,10,11,12,13] → 成本与时间信号（6维）
      空间/电网组  [0,6,9]           → 流量、电压、距离（3维）
      EV 状态组   [8]               → 请求车辆 SOC（1维）

    融合后经 LayerNorm 输出 out_dim 维向量，作为 GAT 的输入节点特征。
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.queue_enc   = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.cost_enc    = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        self.spatial_enc = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.ev_enc      = nn.Sequential(nn.Linear(1, 32), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(32 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 18]  →  [N, out_dim]"""
        q = self.queue_enc(x[:, _IDX_QUEUE])
        c = self.cost_enc(x[:, _IDX_COST])
        s = self.spatial_enc(x[:, _IDX_SPATIAL])
        e = self.ev_enc(x[:, _IDX_EV])
        return self.fusion(torch.cat([q, c, s, e], dim=-1))


class GraphQNetwork(nn.Module):
    """
    改进版 GNN Q-Network:
      FeatureEncoder（分组编码）→ GATv2 × 2 → Dueling Head → Q 值

    相比原版的改进:
      1. FeatureEncoder: 18 维原始特征 → 按语义分 4 组编码融合，避免异质特征直接混入 GAT
      2. Dueling Network: V(s) + A(s,a) − mean(A)，Q 值估计更稳定
    """

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9,
                 num_edge_features=2):
        """
        Args:
            num_features:        节点特征维度（当前固定=18，由 FeatureEncoder 处理）
            num_actions:         动作数 = 充电站数量
            station_node_ids:    充电站对应的节点索引列表
            num_nodes_per_graph: 每张图的节点数
            num_edge_features:   边特征维度（1:长度, 2:速度/限速）
        """
        super(GraphQNetwork, self).__init__()

        if station_node_ids is None:
            station_node_ids = [0, 8]

        self.register_buffer(
            'station_node_ids',
            torch.tensor(station_node_ids, dtype=torch.long)
        )
        self.num_nodes_per_graph = num_nodes_per_graph
        self.num_actions = num_actions

        # 1. 分组特征编码器 (18 → 64)
        self.feature_encoder = FeatureEncoder(out_dim=64)

        # 2. GATv2 卷积层 (64 → 32 → 64)
        self.conv1 = GATv2Conv(
            in_channels=64,
            out_channels=32,
            heads=4,
            concat=False,
            edge_dim=num_edge_features
        )
        self.conv2 = GATv2Conv(
            in_channels=32,
            out_channels=64,
            heads=4,
            concat=False,
            edge_dim=num_edge_features
        )

        # 3. Dueling Head
        #    输入: station_emb(64) + global_ctx(64) = 128
        #    Value stream:     V(s)    — 与具体站点无关的状态价值
        #    Advantage stream: A(s,a)  — 各站相对优势
        self.value_fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_fc = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data, action_mask=None):
        """
        Args:
            data:        PyG Data/Batch 对象
            action_mask: 可选，shape [B, num_actions] 的 bool 张量，
                         False 的动作 Q 值设为 -1e8
        """
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # --- 分组特征编码 ---
        x = self.feature_encoder(x)                      # [N, 64]

        # --- GATv2 消息传递 ---
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))

        # --- 全局 context ---
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        global_ctx = global_mean_pool(x, batch)           # [B, 64]
        batch_size = global_ctx.shape[0]

        # --- Dueling readout：对每个充电站计算 Advantage ---
        advantages = []
        for node_id in self.station_node_ids:
            indices = (torch.arange(batch_size, device=x.device)
                       * self.num_nodes_per_graph + node_id)
            station_emb = x[indices]                      # [B, 64]
            combined = torch.cat([station_emb, global_ctx], dim=1)  # [B, 128]
            adv = self.advantage_fc(combined)             # [B, 1]
            advantages.append(adv)

        advantages = torch.cat(advantages, dim=1)         # [B, num_actions]

        # Value 由全局 context 决定（与具体站无关）
        global_combined = torch.cat(
            [global_ctx, global_ctx], dim=1               # [B, 128]
        )
        value = self.value_fc(global_combined)            # [B, 1]

        # Q = V(s) + A(s,a) − mean_a(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)

        # --- 无效动作掩码 ---
        if action_mask is not None:
            action_mask = action_mask.to(q_values.device)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e8)

        return q_values
