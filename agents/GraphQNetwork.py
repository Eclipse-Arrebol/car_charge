import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GraphQNetwork(nn.Module):
    """
    GNN Q-Network: 考虑边特征的图注意力机制 (GATv2) → 站节点 embedding readout → Q 值

    支持任意路网规模和充电站位置。
    """

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9,
                 num_edge_features=2):
        """
        Args:
            num_features:       节点特征维度 (当前=10)
            num_actions:        动作数 = 充电站数量
            station_node_ids:   充电站对应的节点索引列表
            num_nodes_per_graph: 每张图的节点数
            num_edge_features:  边特征维度 (1:长度, 2:速度/限速)
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

        # 1. 图注意力层 (GATv2) — 捕获微观拓扑并吸收道路信息
        # 设置多头注意力机制 (heads=4)，通过 concat=False 聚合各头输出(平均)
        self.conv1 = GATv2Conv(
            in_channels=num_features,
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

        # 2. 站节点 readout MLP — 每个站节点 embedding → 该站 Q 值
        #    输入: 站节点 embedding(64) + 全局 context(64)
        self.fc1 = nn.Linear(64 + 64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data, action_mask=None):
        """
        Args:
            data:        PyG Data/Batch 对象
            action_mask: 可选，shape [B, num_actions] 的 bool/float 张量，
                         True/1.0 表示该动作有效，False/0.0 表示无效。
                         无效动作的 Q 值会被设为 -1e8 以阻止选取。
        """
        # 提取节点特征、边索引结构、边特征
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)

        # --- GAT 卷积 ---
        # 传入 edge_attr，使网络可以“看到”边长度、限速等物理路网属性
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))

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

        q_values = torch.cat(q_values_list, dim=1)            # [B, num_actions]

        # --- 无效动作掩码 (Action Mask) ---
        if action_mask is not None:
            action_mask = action_mask.to(q_values.device)
            # 将无效动作的 Q 值设为极大负值，使其被 argmax/softmax 忽略
            q_values = q_values.masked_fill(~action_mask.bool(), -1e8)

        return q_values