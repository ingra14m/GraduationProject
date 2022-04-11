import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


# 这个显然是不会使用的
class DotProductPredictor(nn.Module):
    def __init__(self, g):
        super(DotProductPredictor, self).__init__()
        self.g = g

    def forward(self, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with self.g.local_scope():
            self.g.ndata['h'] = h
            self.g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            score = self.g.edata['score']  # 这里正确的做法是，把 score的值归一化到0-1之间
            s1 = score.detach().numpy()
            max_num = max(s1)
            min_num = min(s1)
            score = (score - min_num) / (max_num - min_num)
            score = score
            # a=0
            # b=0
            for i in range(len(score)):
                if score[i] == 0:
                    score[i] = 0
                else:
                    score[i] = 1

            return score


# 返回的是边的score，按照顺序
class MLPPredictor(nn.Module):
    def __init__(self, g, h_feats):
        super().__init__()
        self.g = g
        self.W1 = nn.Linear(h_feats, 16)
        self.W2 = nn.Linear(32, 8)
        self.W3 = nn.Linear(8, 1)

    def apply_edges(self, edges):
        s1 = torch.cat([edges.src['h'], edges.dst['h']], 1)
        # score = torch.exp(self.W2(F.leaky_relu(self.W1(h))))
        # score = F.sigmoid(self.W2(F.leaky_relu(self.W1(h))))
        s2 = self.W2(s1)
        score = self.W3(F.relu(s2))

        s1 = score.detach().cpu().numpy()  # 不计算梯度了，得到边的一个最终得分，成为了一个数组
        max_num = max(s1)
        min_num = min(s1)
        # 归一化
        score = (torch.tensor(s1) - torch.tensor([min_num])) / (torch.tensor([max_num]) - torch.tensor([min_num]))
        # 由于是apply_edges，因此将这一部分数据存储在edata['score']中

        try:
            score = score.to("cuda:0")
        finally:
            pass
        return {'score': score}

    def forward(self, h):
        with self.g.local_scope():
            h = self.W1(h)  # 将node特征映射到16维
            self.g.ndata['h'] = h
            self.g.apply_edges(self.apply_edges)
            # x=self.g.edata['score']
            # print('score:', self.g.edata['score'][-10:])

            return self.g.edata['score']


class GateGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GateGATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        # print('a:', a[-20])
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e'], 'score': edges.data['score']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        # 这里，就是刘老师说的那种，把没有的边，就不要进 softmax 了
        alpha = F.softmax(nodes.mailbox['e'] * nodes.mailbox['score'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        # ----------------------------分割线--------------------------
        # alpha1 = F.softmax(nodes.mailbox['e'], dim=1)
        # h = torch.sum(alpha1 * nodes.mailbox['z']* nodes.mailbox['score'], dim=1)

        # print('alpha:', alpha1[-1])
        # b=nodes.mailbox['score']
        # equation (4)
        return {'h': h}

    def forward(self, h, gate):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.edata['score'] = gate
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GateGATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h, gate):
        head_outs = [attn_head(h, gate) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class MLPEdgePredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['final']
        h_v = edges.dst['final']
        score = self.W(torch.cat([h_u, h_v], 1))  # (74528， 1024)
        return {'edge_score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['final'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['edge_score']


class GateGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, out_classes, dot=False):
        super(GateGAT, self).__init__()
        if dot:
            self.layer0 = DotProductPredictor(g)
        else:
            self.layer0 = MLPPredictor(g, in_dim)

        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        self.pred = MLPEdgePredictor(out_dim, out_classes)

    def forward(self, g, h):
        gate = self.layer0(h)
        h = self.layer1(h, gate)
        h = F.leaky_relu(h)
        h = self.layer2(h, gate)
        return self.pred(g, h), gate


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        # print('a:', a[-20])
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        # equation (4)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class GATMultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(GATMultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, out_classes):
        super(GAT, self).__init__()
        self.layer1 = GATMultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = GATMultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

        self.pred = MLPEdgePredictor(out_dim, out_classes)

    def forward(self, g, h, delete_eids):
        g.remove_edges(delete_eids)
        h = self.layer1(h)
        h = F.leaky_relu(h)
        h = self.layer2(h)
        g.add_edges(delete_eids)
        return self.pred(g, h), None
