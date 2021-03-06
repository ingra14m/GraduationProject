import time

import torch
import torch.nn as nn
import argparse
import dgl
import torch.nn.functional as F
from utils.PreProcessing import *
from utils import Function as MyF
from utils.radam import RAdam
import models.EdgeClassfication as mynn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from utils.FocalLoss import FocalLoss

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

SET_SPLIT = {
    "GATEGAT": (33000, 35000),
    "GAT": (30000, 34000),
    "GRAPHSAGE": (33000, 35000),
    "GCN": (33000, 35000)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, graph, optimizer, output, add_self_loop=False):
    model.to(device)

    best_val_acc = 0
    best_test_acc = 0

    ndata_features = graph.ndata['feature']
    edata_label = graph.edata['label']
    train_mask = graph.edata['train_mask'].cpu()
    val_mask = graph.edata['val_mask'].cpu()
    test_mask = graph.edata['test_mask'].cpu()

    loss_function = FocalLoss(65)

    # if add_self_loop:
    #     graph = dgl.add_self_loop(graph)
    #     graph.ndata['adj'] = graph.adjacency_matrix()
    #     graph = dgl.remove_self_loop(graph)

    with open("{}.txt".format(output), 'w') as f:
        for epoch in range(5000):
            pred = model(graph, ndata_features)
            # loss = ((pred[train_mask] - edata_label[train_mask]) ** 2).mean()

            # loss = F.cross_entropy(pred[graph.edata['train_mask']], edata_label[graph.edata['train_mask']])
            loss = loss_function(pred[graph.edata['train_mask']], edata_label[graph.edata['train_mask']])
            # if epoch == 30:
            #     result1 = np.array(edata_label[train_mask])
            #     np.savetxt('npresult1.txt', result1)
            #     result2 = np.array(pred[train_mask].argmax(1))
            #     np.savetxt('npresult2.txt', result2)

            result_label, result_pred = MyF.GetLabel(pred, edata_label)

            # train_acc = MyF.Accuracy(pred[train_mask], edata_label[train_mask])
            # val_acc = MyF.Accuracy(pred[val_mask], edata_label[val_mask])
            # test_acc = MyF.Accuracy(pred[test_mask], edata_label[test_mask])
            train_acc = accuracy_score(result_label[train_mask], result_pred[train_mask])
            val_acc = accuracy_score(result_label[val_mask], result_pred[val_mask])
            test_acc = accuracy_score(result_label[test_mask], result_pred[test_mask])

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
            if best_test_acc < test_acc:
                best_test_acc = test_acc

            t0 = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                auc_input = nn.functional.softmax(pred, dim=1)
                content = 'In epoch {}, loss: {:.3f},train acc: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f}), {}'.format(
                    epoch, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc, time.time() - t0)
                quality = 'recall: {:.4f}, {:.4f}, {:.4f}\nprecision: {:.4f}, {:.4f}, {:.4f}\nf1: {:.4f}, {:.4f}, {:.4f}\nauc: {:.4f}\n'.format(
                    recall_score(result_label[train_mask], result_pred[train_mask], average='weighted'),
                    recall_score(result_label[val_mask], result_pred[val_mask], average='weighted'),
                    recall_score(result_label[test_mask], result_pred[test_mask], average='weighted'),
                    precision_score(result_label[train_mask], result_pred[train_mask], average='weighted'),
                    precision_score(result_label[val_mask], result_pred[val_mask], average='weighted'),
                    precision_score(result_label[test_mask], result_pred[test_mask], average='weighted'),
                    f1_score(result_label[train_mask], result_pred[train_mask], average='weighted'),
                    f1_score(result_label[val_mask], result_pred[val_mask], average='weighted'),
                    f1_score(result_label[test_mask], result_pred[test_mask], average='weighted'),
                    roc_auc_score(result_label, auc_input.cpu().data.numpy(), multi_class='ovr')
                    # roc_auc_score(result_label[val_mask], auc_input.cpu().data.numpy()[val_mask], multi_class='ovr'),
                    # roc_auc_score(result_label[test_mask], auc_input.cpu().data.numpy()[test_mask], multi_class='ovr'),
                )
                print(content)
                print(quality)
                f.write(content + '\n')
                f.write(quality + '\n')
                t0 = time.time()
        f.close()


if __name__ == "__main__":
    # ????????????????????????????????????????????????????????????????????????
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='en')  # ch??????
    parser.add_argument('-g', '--gpu', default=True)
    parser.add_argument('-o', '--output', default='ocr_result')
    parser.add_argument('-m', '--model', default='graphsage')
    args = parser.parse_args()

    df_drug, mechanism, action, drugA, drugB = data_import()
    node_feature, edge_label, event_num, edge_src, edge_dst = feature_extraction(df_drug, mechanism, action, drugA,
                                                                                 drugB)

    # ??????????????????????????????
    # graph = dgl.graph((np.concatenate([edge_src, edge_dst]), np.concatenate([edge_dst, edge_src])),
    #                   num_nodes=node_feature.shape[0])

    graph = dgl.graph((edge_src, edge_dst), num_nodes=node_feature.shape[0])

    # ????????????????????????
    # synthetic node and edge features, as well as edge labels
    graph.ndata['feature'] = torch.from_numpy(node_feature).float()
    graph.edata['label'] = torch.from_numpy(edge_label)

    # ?????????????????????????????????????????????
    graph.edata['train_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['train_mask'][:SET_SPLIT[args.model.upper()][0]] = True
    graph.edata['val_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['val_mask'][SET_SPLIT[args.model.upper()][0]:SET_SPLIT[args.model.upper()][1]] = True
    graph.edata['test_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool)
    graph.edata['test_mask'][SET_SPLIT[args.model.upper()][1]:] = True

    # model = GNN_MODEL[args.model.upper()](graph.ndata['feature'].shape[1], 1024, 128, event_num)
    model = None
    optimizer = None

    if args.gpu == False:
        device = torch.device("cpu")

    graph = graph.to(device)

    if args.model.upper() == 'GCN':
        # ?????????LeakyRelu
        model = mynn.GCNModel(graph.ndata['feature'].shape[1], 1024, 128, event_num, norm=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    elif args.model.upper() == 'GAT':
        model = mynn.GATModel(graph.ndata['feature'].shape[1], 1024, 128, event_num)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # GraphSage???????????????????????????????????????degree???0?????????
    elif args.model.upper() == 'GRAPHSAGE':
        # ?????????Relu
        model = mynn.SAGEModel(graph.ndata['feature'].shape[1], 1024, 128, event_num, 'mean')
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer = RAdam(model.parameters(), lr=1e-4)

    elif args.model.upper() == 'GATEGAT':
        from models.GateGAT.script import train as gate_train

        gate_train.main(graph, event_num, args.output, device)

    elif args.model.upper() == 'FASTGCN':
        pass

    if args.model.upper() == 'GCN':
        train(model=model, optimizer=optimizer, graph=graph, output=args.output, add_self_loop=True)

    else:
        train(model=model, optimizer=optimizer, graph=graph, output=args.output)
