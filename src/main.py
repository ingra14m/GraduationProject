import torch
import torch.nn as nn
import argparse
import dgl
import torch.nn.functional as F
from utils.PreProcessing import *
from utils import Function as MyF
import models.EdgeClassfication as mynn

'''
    model for edge classfication
'''
GNN_MODEL = {
    'GCN': None,
    'GAT': mynn.GATModel,
    'GRAPHSAGE': mynn.SAGEModel,
    'GATEGAT': None,
    'FASTGCN': None
}


def train(model, graph, optimizer):
    best_val_acc = 0
    best_test_acc = 0

    ndata_features = graph.ndata['feature']
    edata_label = graph.edata['label']
    train_mask = graph.edata['train_mask']
    val_mask = graph.edata['val_mask']
    test_mask = graph.edata['test_mask']

    with open("result.txt") as f:
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
            if best_test_acc < test_acc:
                best_test_acc = test_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                content = 'In epoch {}, loss: {:.3f},train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                    epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc)
                print(content)
                f.write(content + '\n')
        f.close()


if __name__ == "__main__":
    # 启用这个之后，命令行似乎就不接受没有名字的参数了
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='en')  # ch均可
    parser.add_argument('-p', '--path', default='.')
    parser.add_argument('-o', '--output', default='ocr_result')
    parser.add_argument('-m', '--model', default='gategat')
    args = parser.parse_args()

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
    graph.edata['label'] = torch.from_numpy(edge_label)

    graph.edata['train_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['train_mask'][:10000] = True
    graph.edata['val_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['val_mask'][10000:25000] = True
    graph.edata['test_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['test_mask'][25000:] = True

    # model = GNN_MODEL[args.model.upper()](graph.ndata['feature'].shape[1], 1024, 128, event_num)
    model = None
    optimizer = None

    if args.model.upper() == 'GCN':
        model = mynn.GCNModel(graph.ndata['feature'].shape[1], 1024, 128, event_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif args.model.upper() == 'GAT':
        model = mynn.GATModel(graph.ndata['feature'].shape[1], 1024, 128, event_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    elif args.model.upper() == 'GRAPHSAGE':
        model = mynn.SAGEModel(graph.ndata['feature'].shape[1], 1024, 128, event_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    elif args.model.upper() == 'GATEGAT':
        from models.GateGAT.script import train

        train.main(graph, event_num)

    elif args.model.upper() == 'FASTGCN':
        pass

    if args.model.upper() != 'GATEGAT':
        train(model=model, optimizer=optimizer, graph=graph)
