"""GAT模型训练与预测

    加载GAT模型, 生成训练必要组件实例

    Input:
    ------
    params: dict, 模型参数和超参数, 格式为:
            {
                'sparse': False,
                'random_state' 42,
                'model': {
                    'input_dim': 1433,
                    'hidden_dim': 8,
                    'output_dim': 7,
                    'num_heads': 8,
                    'dropout': 0.6,
                    'alpha': 0.2
                },
                'hyper': {
                    'lr': 3e-3,
                    'epochs': 10,
                    'patience': 100,
                    'weight_decay': 5e-4
                }
            }

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

from .layers import GraphAttentionLayer
from .layers import SparseGraphAttentionLayer

# dgl自带的GAT
class GATBlock(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_heads=8):
        super(GATBlock, self).__init__()
        self.gat1 = dglnn.GATConv(in_feats=in_feats, out_feats=hid_feats, num_heads=num_heads)
        self.gat2 = dglnn.GATConv(in_feats=hid_feats, out_feats=out_feats, num_heads=num_heads)

    def forward(self, graph, inputs):
        h = self.gat1(graph, inputs)
        h = torch.mean(F.relu(h), dim=1)  # 多头的参数
        h = self.gat2(graph, h)
        return torch.mean(h, dim=1)


# 自实现的GAT
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout, alpha, sparse=False):
        """定义GAT网络

            Inputs:
            -------
            input_dim: int, 输入维度
            hidden_dim: int, 隐层维度
            outut_dim: int, 输出维度
            num_heads: int, 多头注意力个数
            dropout: float, dropout比例
            alpha: float, LeakyReLU负数部分斜率
            sparse: boolean, 是否使用稀疏数据

        """

        super(GAT, self).__init__()

        if sparse:  # 使用稀疏数据的attention层
            attention_layer = SparseGraphAttentionLayer
        else:  # 使用稠密数据的attention层
            attention_layer = GraphAttentionLayer

        # 多头注意力层
        self.attentions = nn.ModuleList()
        for _ in range(num_heads):
            self.attentions.append(attention_layer(input_dim, hidden_dim, dropout, alpha, True))

        self.elu = nn.ELU(inplace=True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.output = attention_layer(num_heads * hidden_dim, output_dim, dropout, alpha, False)

        return

    def forward(self, edges, X):
        """GAT网络前馈

            Inputs:
            -------
            X: tensor, 节点特征
            edges: tensor, 边的源节点与目标节点索引

            Output:
            -------
            output: tensor, 输出

        """
        # 拼接多头注意力层输出
        out = torch.cat([attention(X, edges) for attention in self.attentions], dim=1)

        # 计算输出
        output = self.output(self.elu(out), edges)
        return output
