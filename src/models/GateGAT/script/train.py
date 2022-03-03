import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dgl.data import citation_graph as citegrh

import time
import numpy as np
import torch.nn as nn
import copy
import argparse

from .model import GAT
from .model import GateGAT


# from GateGAT import GATE_GAT
final_gate = None

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


def train(g, net, search=True, isreTrain=False):
    logits = 0
    gate = 0
    features = g.ndata['feature']
    labels = g.edata['label']
    train_mask = g.edata['train_mask']
    val_mask = g.edata['val_mask']
    test_mask = g.edata['test_mask']

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=5e-4)
    lossFunction = nn.CrossEntropyLoss(reduction='mean')
    best_val_acc = 0
    best_test_acc = 0
    dur = []
    # fp = open("logGAT.txt", "a+", encoding="utf-8")
    if search:
        EPOCH = 10  # previous 400
        # fp.write("Search Stage:\n")
    else:
        EPOCH = 20  # previous 200
        # if isreTrain:
        #     fp.write("reTrain GAT Stage:\n")
        # else:
        #     fp.write("GAT Stage:\n")

    for epoch in range(EPOCH):
        if epoch % 5 == 0:
            t0 = time.time()

        logits, gate = net(g, features)
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

            # fp.write(expLog + '\n')

    # fp.close()
    # ------------打印训练参数-----------
    # for name, param in net.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Grad_requires: {param.requires_grad} | Grad: {param.grad} | Values : {param[:2]} \n")

    return logits, gate


def main(g, event_num):
    for delEdge in [20]:
        # 载入数据
        # data = citegrh.load_cora()
        # g = data[0]
        # # 加入自环
        # g.add_edges(g.nodes(), g.nodes())

        # 第一阶段：search stage ：search = True，返回所有边的 gate 值；
        print('------------------------search stage--------------------------')
        net = GateGAT(g,
                      in_dim=g.ndata['feature'].shape[1],
                      hidden_dim=8,
                      out_dim=7,
                      num_heads=8, out_classes=event_num, dot=False)

        _, gate = train(g, net, search=True)

        print('------------------------original GAT--------------------------')

        # 第二阶段：retrain stage ：在 gate 的基础上，得出预测结果，验证模型
        print('------------------------retrain stage--------------------------')
        net = GAT(g,
                  in_dim=g.ndata['feature'].shape[1],
                  hidden_dim=8,
                  out_dim=7,
                  num_heads=8, out_classes=event_num)
        # 1.根据gate的结果，删除对应的边

        gate_np = gate.squeeze()
        _, indices = torch.sort(gate_np)
        position = int(len(gate_np) / delEdge)
        delete_eids = indices[0:position]
        g.remove_edges(delete_eids)

        # 2.训练，报告结果
        retrain_start = time.time()
        h, _ = train(g, net, search=False, isreTrain=True)
        retrain_end = time.time()
        restrainGat = "retrain gat time : {}".format(retrain_end - retrain_start)
        print(restrainGat)
