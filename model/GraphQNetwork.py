import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphQNetwork(nn.Module):
    """
    GNN Q-Network: 图卷积 → 站节点 embedding readout → Q 值

    支持任意路网规模和充电站位置，不再硬编码节点数。
    通过构造参数传入充电站节点 ID 和每图节点数，适配 3×3 网格或真实 OSMnx 路网。
    """

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9):
        """
        Args:
            num_features:       节点特征维度 (当前=9)
            num_actions:        动作数 = 充电站数量
            station_node_ids:   充电站对应的节点索引列表，默认 [0, 8]（3×3网格）
            num_nodes_per_graph: 每张图的节点数，默认 9（3×3网格）
        """
        super(GraphQNetwork, self).__init__()

        if station_node_ids is None:
            station_node_ids = [0, 8]

        # 注册为 buffer，保存/加载模型时自动携带，不参与梯度计算
        self.register_buffer(
            'station_node_ids',
            torch.tensor(station_node_ids, dtype=torch.long)
        )
        self.num_nodes_per_graph = num_nodes_per_graph
        self.num_actions = num_actions

        # 1. 图卷积层 — 捕获空间邻域信息
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)

        # 2. 站节点 readout MLP — 每个站节点 embedding → 该站 Q 值
        #    输入: 站节点 embedding(64) + 全局 context(64)
        self.fc1 = nn.Linear(64 + 64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- GCN 卷积 ---
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # --- 提取全局 context ---
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        global_ctx = global_mean_pool(x, batch)  # [B, 64]
        batch_size = global_ctx.shape[0]

        # --- 提取每个充电站节点的 embedding ---
        q_values_list = []
        for node_id in self.station_node_ids:
            # 对 Batch 中每张图，定位该站节点行号
            indices = (torch.arange(batch_size, device=x.device)
                       * self.num_nodes_per_graph + node_id)
            station_emb = x[indices]                          # [B, 64]
            combined = torch.cat([station_emb, global_ctx], dim=1)  # [B, 128]
            q = F.relu(self.fc1(combined))
            q = self.fc2(q)                                   # [B, 1]
            q_values_list.append(q)

        return torch.cat(q_values_list, dim=1)                # [B, num_actions]