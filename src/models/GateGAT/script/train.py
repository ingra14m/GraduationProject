import dgl
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dgl.data import citation_graph as citegrh

import time
import numpy as np
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import copy
import argparse

from .model import GAT
from .model import GateGAT

# from GateGAT import GATE_GAT
final_gate = None


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W1 = nn.Linear(in_features * 2, 256)
        self.W2 = nn.Linear(256, 128)
        self.W3 = nn.Linear(128, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = F.relu(self.W1(torch.cat([h_u, h_v], 1)))  # (74528， 1024)
        score = F.relu(self.W2(score))
        score = self.W3(score)
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class GraphSAGEBlock(nn.Module):

    # Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).

    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')

        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


class SAGEModel(nn.Module):
    def __init__(self, g_origin, g_delete, in_features, hidden_features, out_features, out_classes, delete_id):
        super().__init__()
        self.delete_id = delete_id
        # self.delete_start = g_origin.edges()[0][delete_id]
        # self.delete_end = g_origin.edges()[1][delete_id]
        self.g_origin = g_origin
        g_delete.remove_edges(self.delete_id)
        self.g_delete = dgl.add_self_loop(g_delete)

        self.sage = GraphSAGEBlock(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, x):
        # for SAGE
        h = self.sage(self.g_delete, x)  # (572, out_features)
        # h = self.gat(x, g.edges())
        return self.pred(self.g_origin, h)


def plot_embeddings(embeddings, X, Y):
    print(Y)
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)
    # 降维
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.savefig(str(int(time.time())) + '.png', dpi=300)
    plt.show()


def train(g, net, output, device, search=True, eid=None):
    net.to(device)
    logits = 0
    gate = 0

    features = g.ndata['feature']
    labels = g.edata['label']
    train_mask = g.edata['train_mask']
    val_mask = g.edata['val_mask']
    test_mask = g.edata['test_mask']

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    best_val_acc = 0
    best_test_acc = 0
    dur = []
    # fp = open("logGAT.txt", "a+", encoding="utf-8")
    if search:
        EPOCH = 5000  # previous 400
        # fp.write("Search Stage:\n")
    else:
        EPOCH = 50000  # previous 200
        # if isreTrain:
        #     fp.write("reTrain GAT Stage:\n")
        # else:
        #     fp.write("GAT Stage:\n")
    with open("{}.txt".format(output), 'a') as f:
        for epoch in range(EPOCH):
            if epoch % 5 == 0:
                t0 = time.time()

            if search:
                logits, gate = net(g, features)
            else:
                logits = net(features)
            pred = logits.argmax(1)

            if search:
                loss = F.cross_entropy(logits, labels)
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            else:
                loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            if best_val_acc < val_acc:
                best_val_acc = val_acc
            if best_test_acc < test_acc:
                best_test_acc = test_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                dur.append(time.time() - t0)
                # if search:
                #     print("Epoch {:05d} | Loss {:.4f} | train acc: {:.3f}| Time(s) {:.4f}".format(
                #         epoch, loss.item(), train_acc, np.mean(dur)))
                # else:
                expLog = 'Epoch {:05d} | Loss: {:.3f} | train acc: {:.3f} | val acc: {:.3f} (best {:.3f}) | test acc: {:.3f} (best {:.3f}) | Time(s) {:.4f}'.format(
                    epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc, np.mean(dur))
                print(expLog)
                f.write(expLog + '\n')

        f.close()

        # fp.write(expLog + '\n')

    # fp.close()
    # ------------打印训练参数-----------
    # for name, param in net.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Grad_requires: {param.requires_grad} | Grad: {param.grad} | Values : {param[:2]} \n")

    return logits, gate


def main(g, event_num, output, device):
    for delEdge in [5]:
        # 载入数据
        # data = citegrh.load_cora()
        # g = data[0]
        # # 加入自环
        # g.add_edges(g.nodes(), g.nodes())

        # 第一阶段：search stage ：search = True，返回所有边的 gate 值；
        print('------------------------search stage--------------------------')
        net = GateGAT(g,
                      in_dim=g.ndata['feature'].shape[1],
                      hidden_dim=512,
                      out_dim=64,
                      num_heads=2, out_classes=event_num, dot=False)

        _, gate = train(g, net, output, device, search=True)  # 得到的是所有边的得分

        # 第二阶段：retrain stage ：在 gate 的基础上，得出预测结果，验证模型
        print('------------------------retrain stage--------------------------')
        # 1.根据gate的结果，删除对应的边

        gate_np = gate.squeeze()
        _, indices = torch.sort(gate_np)
        position = int(len(gate_np) / delEdge)
        delete_eids = indices[0:position]
        # g.remove_edges(delete_eids)

        net = GAT(g, g,
                  in_features=g.ndata['feature'].shape[1],
                  hidden_features=1024,
                  out_features=128,
                  out_classes=event_num, delete_id=delete_eids)

        net = SAGEModel(g, g,
                        in_features=g.ndata['feature'].shape[1],
                        hidden_features=1024,
                        out_features=128,
                        out_classes=event_num, delete_id=delete_eids)


        # 2.训练，报告结果
        retrain_start = time.time()
        h, _ = train(g, net, output, device, search=False, eid=delete_eids)
        # h, _ = train(g, net, output, search=False)
        retrain_end = time.time()
        restrainGat = "retrain gat time : {}".format(retrain_end - retrain_start)
        print(restrainGat)
