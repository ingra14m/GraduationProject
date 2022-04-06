"""GCN模型训练与预测

    加载GCN模型, 生成训练必要组件实例

    Input:
    ------
    params: dict, 模型参数和超参数, 格式为:
            {
                'random_state': 42,
                'model': {
                    'input_dim': 1433,
                    'output_dim': 7,
                    'hidden_dim': 16,
                    'use_bias': True,
                    'dropout': 0.5
                },
                'hyper': {
                    'lr': 1e-2,
                    'epochs': 100,
                    'weight_decay': 5e-4
                }
            }

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


# dgl需要保证degree没有为0的情况，dgl没有写degree不为0的分支
class GCNBlock2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCNBlock2, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats)
        self.conv2 = dglnn.GraphConv(in_feats=hid_feats, out_feats=out_feats)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积层

            Inputs:
            -------
            input_dim: int, 输入特征维度
            output_dim: int, 输出特征维度
            use_bias: boolean, 是否使用偏置

        """

        super(GCN, self).__init__()

        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.__init_parameters()

        return

    def __init_parameters(self):
        """初始化权重和偏置
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, adjacency, x):
        """图卷积层前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            output: tensor in shape [num_nodes, output_dim], 输出

        """

        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias

        return output


# 可以允许有degree为0的情况，只要不对度矩阵归一化就可以了
class GCNBlock(nn.Module):
    """简单图卷积网络

        定义包含两层图卷积的简单网络。

    """

    def __init__(self, input_dim, output_dim, hidden_dim, dropout, use_bias=True):
        """简单图卷积网络

            Inputs:
            -------
            input_dim: int, 节点特征维度
            output_dim: int, 节点类别数
            hidden_dim: int, 第一层图卷积输出维度
            dropout: float, dropout比例
            use_bias: boolean, 是否使用偏置

        """

        super(GCNBlock, self).__init__()

        self.gcn1 = GCN(input_dim, hidden_dim, use_bias)
        self.gcn2 = GCN(hidden_dim, output_dim, use_bias)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        return

    def forward(self, graph, x):
        """简单图卷积网络前馈

            Inputs:
            -------
            adjacency: tensor in shape [num_nodes, num_nodes], 邻接矩阵
            X: tensor in shape [num_nodes, input_dim], 节点特征

            Output:
            -------
            logits: tensor in shape [num_nodes, output_dim], 输出

        """
        adjacency = graph.adjacency
        out = self.gcn1(adjacency, x)
        out = self.dropout(self.act(out))
        logits = self.gcn2(adjacency, out)
        return logits
