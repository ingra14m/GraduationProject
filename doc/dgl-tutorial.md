# dgl-tutorials

## Dgl-Dataset

DGL Dataset的实例可以包含**一张或多张图**。

A DGL graph 包含的属性：

- node features  `ndata` 。其具有如下的属性
  - `train_mask`: A boolean tensor indicating whether the node is in the training set.
  - `val_mask`: A boolean tensor indicating whether the node is in the validation set.
  - `test_mask`: A boolean tensor indicating whether the node is in the test set.
  - `label`: The ground truth node category.
  - `feat`: The node features.

- edge features  `edata`。以上两个都是dict-like的结构
  - 存边的属性
  - 在一半的图分类任务中不需要这个

## 图分类任务的流程



## 创建自己的Graph

### 初始化

用edge初始化，传入的第一个参数是边，第二个参数是节点个数。不指定的话，会用edge的index最大值 + 1表示node个数

```python
g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
```

### 传入边和节点属性

edata和ndata都是dict，没有什么key的限制

- ndata
  - 指定feat和label
  - 指定mask

### 图函数

#### 图属性

```python
print(g.num_nodes())
print(g.num_edges())
# Out degrees of the center node
print(g.out_degrees(0))
# In degrees of the center node - note that the graph is directed so the in degree should be 0.
print(g.in_degrees(0))
```

#### 图迁移

```python
# Induce a subgraph from node 0, node 1 and node 3 from the original graph.
sg1 = g.subgraph([0, 1, 3])
# Induce a subgraph from edge 0, edge 1 and edge 3 from the original graph.
sg2 = g.edge_subgraph([0, 1, 3])
```