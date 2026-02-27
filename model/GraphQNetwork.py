import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphQNetwork(nn.Module):
    """
    GNN Q-Network: 图卷积 → 站节点 embedding readout → Q 值

    改进: 不再用 global_mean_pool 把整张图压成一个向量，
    而是提取每个充电站节点的 embedding，然后通过 MLP 各自输出 Q 值。

    这样 Q(station_0) 和 Q(station_1) 分别基于各站的局部信息
    (排队、电价、负荷率、电压等)，DQN 能真正"对比"两个站的优劣。
    """

    # 充电站所在的交通节点 ID (与 TrafficPowerEnv 一致)
    STATION_NODE_IDS = [0, 8]

    def __init__(self, num_features, num_actions):
        super(GraphQNetwork, self).__init__()
        # 1. 图卷积层 — 捕获空间邻域信息
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)

        # 2. 站节点 readout MLP — 每个站节点 embedding → 该站 Q 值
        #    输入: 站节点 embedding(64) + 全局 context(64)
        self.fc1 = nn.Linear(64 + 64, 64)
        self.fc2 = nn.Linear(64, 1)        # 每个站输出 1 个 Q 值

        self.num_actions = num_actions

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- GCN 卷积 ---
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # --- 提取全局 context (辅助信息) ---
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        global_ctx = global_mean_pool(x, batch)  # [B, 64]

        # --- 提取每个充电站节点的 embedding ---
        num_nodes_per_graph = len(self.STATION_NODE_IDS) + 7  # 3x3=9 nodes
        batch_size = global_ctx.shape[0]

        q_values_list = []
        for station_idx, node_id in enumerate(self.STATION_NODE_IDS):
            # 对于 Batch 中的每张图，找到该站节点的 embedding
            indices = torch.arange(batch_size, device=x.device) * num_nodes_per_graph + node_id
            station_emb = x[indices]  # [B, 64]

            # 拼接: 站节点 embedding + 全局 context
            combined = torch.cat([station_emb, global_ctx], dim=1)  # [B, 128]

            # MLP 输出该站的 Q 值
            q = F.relu(self.fc1(combined))
            q = self.fc2(q)            # [B, 1]
            q_values_list.append(q)

        # 拼成 [B, num_actions]
        q_values = torch.cat(q_values_list, dim=1)
        return q_values