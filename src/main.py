import torch
import torch.nn as nn
import taichi as ti
import argparse
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
import os
import sys
import sqlite3
import numpy as np
import pandas as pd

GRADUATION_SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

if GRADUATION_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, GRADUATION_SCRIPTS_PATH)


# sys.path.remove(CURRENT_DIR_PATH)

def data_import(path="data/event.db"):
    conn = sqlite3.connect(GRADUATION_SCRIPTS_PATH + '/' + path)

    '''
        drug information
        - index
        - id
        - target
        - enzyme
        - pathway  # 之前的都没啥用，通过name来索引的，但是在这个实验中，似乎将target和enzyme集成到了feature中
        - smile 
        - name
        drug shape (572, 7)
    '''
    df_drug = pd.read_sql('select * from drug;', conn)

    '''
        extraction information 这个就是边的信息
        - index
        - mechanism
        - action   # ---- 之前的都是描述信息，没啥用
        - drugA
        - drugB
        extraction shape (37264, 5)
    '''
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    return df_drug, mechanism, action, drugA, drugB


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = pd.DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1

    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))

    print(feature_name + " len is:" + str(len(sim_matrix[0])))
    return sim_matrix


def feature_extraction(df_drug, mechanism, action, drugA, drugB, feature_list=("smile", "target", "enzyme")):
    d_label = {}

    d_event = []  # mechanism + action
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])

    count = {}  # 记录event出现的次数，可以用哈希表优化
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    event_num = len(count)
    # 按照event出现的次数进行排序，出来的是一个65的list，元素是tuple（"mechanism + action", count）
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)

    # 恢复成一个dict，key为mechanism + action，value为index
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    # node特征的dict(572, 1716)
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  # vector=[]
    for i in feature_list:
        # vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec = feature_vector(i, df_drug)
        vector = np.hstack((vector, tempvec))

    # Transfrom the drug ID to feature vector
    drug_name = np.array(df_drug['name'])
    drug_index_table = {}
    for idx, value in enumerate(drug_name):
        drug_index_table[value] = idx

    edge_src = [drug_index_table[item] for item in drugA]
    edge_dst = [drug_index_table[item] for item in drugB]

    # Use the dictionary to obtain feature vector and label
    new_label = []

    # 统计每一条边的情况
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])

    new_label = np.concatenate([new_label, new_label])

    return vector, new_label, event_num, edge_src, edge_dst


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
        score = self.W(torch.cat([h_u, h_v], 1))
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
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


if __name__ == "__main__":
    # 启用这个之后，命令行似乎就不接受没有名字的参数了
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language', default='en')  # ch均可
    parser.add_argument('-p', '--path', default='.')
    parser.add_argument('-o', '--output', default='ocr_result')
    parser.add_argument('-m', '--model', default='gcn')
    args = parser.parse_args()

    df_drug, mechanism, action, drugA, drugB = data_import()
    node_feature, edge_label, event_num, edge_src, edge_dst = feature_extraction(df_drug, mechanism, action, drugA,
                                                                                 drugB)

    # 无向图，需要两边连接
    graph = dgl.graph((np.concatenate([edge_src, edge_dst]), np.concatenate([edge_dst, edge_src])),
                      num_nodes=node_feature.shape[0])

    # synthetic node and edge features, as well as edge labels
    graph.ndata['feature'] = torch.from_numpy(node_feature).float()
    # graph.edata['feature'] = torch.randn(1000, 10)
    graph.edata['label'] = torch.from_numpy(edge_label)
    # synthetic train-validation-test splits
    graph.edata['train_mask'] = torch.zeros(graph.num_edges(), dtype=torch.bool).bernoulli(0.6)

    ndata_features = graph.ndata['feature']
    edata_label = graph.edata['label']
    train_mask = graph.edata['train_mask']
    model = Model(ndata_features.shape[1], 100, len(edata_label))
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(graph, ndata_features)
        # loss = F.cross_entropy(pred[train_mask], edata_label[train_mask])
        loss = ((pred[train_mask] - edata_label[train_mask]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

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
