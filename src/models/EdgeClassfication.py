import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from .GAT.script.model import GAT
from .GCN.script.model import GCNBlock, GCNBlock2
from .GraphSAGE.script.model import GraphSAGEBlock


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes, softmax=False):
        super().__init__()
        self.softmax = softmax
        self.W1 = nn.Linear(in_features * 2, 256)
        self.W2 = nn.Linear(256, 128)
        self.W3 = nn.Linear(128, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = F.relu(self.W1(torch.cat([h_u, h_v], 1)))  # (74528， 1024)
        score = F.relu(self.W2(score))
        if self.softmax:
            score = nn.functional.softmax(self.W3(score), dim=1)
        else:
            score = self.W3(score)
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class SAGEModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes, aggregator):
        super().__init__()
        self.sage = GraphSAGEBlock(in_features, hidden_features, out_features, aggregator)
        self.sage2 = dglnn.SAGEConv(
            in_feats=out_features, out_feats=out_features, aggregator_type=aggregator)
        # self.gat = GAT(input_dim=in_features, hidden_dim=hidden_features, output_dim=out_features, num_heads=8,
        #                dropout=0.6, alpha=0.4)
        # self.pred = DotProductPredictor()  # 边回归问题
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, g, x):
        # for SAGE
        h = F.relu(self.sage(g, x))  # (572, out_features)
        h = self.sage2(g, h)
        # h = self.gat(x, g.edges())
        return self.pred(g, h)


class GATModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        self.gat = GAT(in_features, hidden_features, out_features, num_heads=8, dropout=0.6, alpha=0.2)
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, g, x):
        h = self.gat(g.edges(), x)
        return self.pred(g, h)


class GCNModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes, norm=False):
        super(GCNModel, self).__init__()
        if norm:
            self.gcn = GCNBlock2(in_features, hidden_features, out_features)
        else:
            self.gcn = GCNBlock(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_classes, softmax=True)

    def forward(self, g, x):
        h = self.gcn(g, x)
        return self.pred(g, h)
