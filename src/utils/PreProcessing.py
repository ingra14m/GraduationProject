import sqlite3
import sys
import os
import numpy as np
import pandas as pd

GRADUATION_SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
GRADUATION_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
CURRENT_DIR_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

if GRADUATION_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, GRADUATION_SCRIPTS_PATH)

def data_import(path="data/event.db"):
    conn = sqlite3.connect(GRADUATION_PROJECT_PATH + '/' + path)

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

    # new_label = np.concatenate([new_label, new_label])
    new_label = np.array(new_label)

    return vector, new_label, event_num, edge_src, edge_dst