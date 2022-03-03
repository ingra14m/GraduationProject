import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from .GAT.script.model import GAT
from .GCN.script.model import GCN
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
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))  # (74528， 1024)
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class SAGEModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        self.sage = GraphSAGEBlock(in_features, hidden_features, out_features)
        # self.gat = GAT(input_dim=in_features, hidden_dim=hidden_features, output_dim=out_features, num_heads=8,
        #                dropout=0.6, alpha=0.4)
        # self.pred = DotProductPredictor()  # 边回归问题
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, g, x):
        # for SAGE
        h = self.sage(g, x)  # (572, out_features)
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
