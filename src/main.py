import torch
import torch.nn as nn
import argparse
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
from utils.PreProcessing import *
from utils import Function as MyF

'''
    model for edge classfication
'''

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


class SAGE(nn.Module):
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


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        # self.pred = DotProductPredictor()  # 边回归问题
        self.pred = MLPPredictor(out_features, out_classes)

    def forward(self, g, x):
        h = self.sage(g, x)  # (572, out_features)

        return self.pred(g, h)


if __name__ == "__main__":
    # 启用这个之后，命令行似乎就不接受没有名字的参数了
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='en')  # ch均可
    parser.add_argument('-p', '--path', default='.')
    parser.add_argument('-o', '--output', default='ocr_result')
    parser.add_argument('-m', '--model', default='gcn')
    args = parser.parse_args()

    best_val_acc = 0
    best_test_acc = 0

    df_drug, mechanism, action, drugA, drugB = data_import()
    node_feature, edge_label, event_num, edge_src, edge_dst = feature_extraction(df_drug, mechanism, action, drugA,
                                                                                 drugB)

    # 无向图，需要两边连接
    # graph = dgl.graph((np.concatenate([edge_src, edge_dst]), np.concatenate([edge_dst, edge_src])),
    #                   num_nodes=node_feature.shape[0])

    graph = dgl.graph((edge_src, edge_dst), num_nodes=node_feature.shape[0])

    # 输入图数据的属性
    # synthetic node and edge features, as well as edge labels
    graph.ndata['feature'] = torch.from_numpy(node_feature).float()
    # graph.edata['feature'] = torch.randn(1000, 10)
    graph.edata['label'] = torch.from_numpy(edge_label)
    # synthetic train-validation-test splits

    train_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    train_mask[:10000] = True
    val_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    val_mask[10000:25000] = True
    test_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    test_mask[25000:] = True

    graph.edata['train_mask'] = train_mask
    graph.edata['val_mask'] = val_mask
    graph.edata['test_mask'] = val_mask

    ndata_features = graph.ndata['feature']
    edata_label = graph.edata['label']
    model = Model(ndata_features.shape[1], 1024, 128, event_num)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1000):
        pred = model(graph, ndata_features)
        # loss = ((pred[train_mask] - edata_label[train_mask]) ** 2).mean()
        loss = F.cross_entropy(pred[train_mask], edata_label[train_mask])
        # if epoch == 30:
        #     result1 = np.array(edata_label[train_mask])
        #     np.savetxt('npresult1.txt', result1)
        #     result2 = np.array(pred[train_mask].argmax(1))
        #     np.savetxt('npresult2.txt', result2)

        train_acc = MyF.Accuracy(pred[train_mask], edata_label[train_mask])
        val_acc = MyF.Accuracy(pred[val_mask], edata_label[val_mask])
        test_acc = MyF.Accuracy(pred[test_mask], edata_label[test_mask])

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(
                'In epoch {}, loss: {:.3f},train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
        # print(loss.item())

    # if args.model.upper() == 'GCN':
    #     from models.GCN import main as GCN_main
    #     GCN_main.main()
    # elif args.model.upper() == 'GAT':
    #     from models.GAT import main as GAT_main
    #     GAT_main.main()
    #
    # elif args.model.upper() == 'GRAPHSAGE':
    #     from models.GraphSAGE import main as GRAPHSAGE_main
    #     GRAPHSAGE_main.main()
    # elif args.model.upper() == 'GRAND':
    #     from models.GRAND import main as GRAND_main
    #     GRAND_main.main()
    # elif args.model.upper() == 'FASTGCN':
    #     from models.FastGCN import main as FASTGCN_main
    #     FASTGCN_main.main()
