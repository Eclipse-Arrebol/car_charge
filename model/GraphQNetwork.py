import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphQNetwork(nn.Module):
    def __init__(self, num_features, num_actions):
        super(GraphQNetwork, self).__init__()
        # 1. 图卷积层
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)

        # 2. 决策层
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN 卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # --- 关键修改点开始 ---
        # 自动识别是"单张图"还是"Batch图"
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch  # 如果是 Batch训练，使用自带的 batch 向量
        else:
            # 如果是单张图测试，手动生成全0 batch
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        x = global_mean_pool(x, batch)
        # --- 关键修改点结束 ---

        # 决策
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)