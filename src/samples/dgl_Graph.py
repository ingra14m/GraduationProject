import dgl
import numpy as np
import torch

#TODO 用edge初始化，传入的第一个参数是边，第二个参数是节点个数。不指定的话，会用edge的index最大值 + 1表示node个数
g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)

# Equivalently, PyTorch LongTensors also work.
g = dgl.graph((torch.LongTensor([0, 0, 0, 0, 0]), torch.LongTensor([1, 2, 3, 4, 5])), num_nodes=6)

# You can omit the number of nodes argument if you can tell the number of nodes from the edge list alone.
g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]))

# Assign a 3-dimensional node feature vector for each node.
g.ndata['x'] = torch.randn(6, 3)
# Assign a 4-dimensional edge feature vector for each edge.
g.edata['a'] = torch.randn(5, 4)
# Assign a 5x4 node feature matrix for each node.  Node and edge features in DGL can be multi-dimensional.
g.ndata['y'] = torch.randn(6, 5, 4)

# print(g.edata)
# print(g.ndata)

print(g.num_nodes())
print(g.num_edges())
# Out degrees of the center node
print(g.out_degrees(0))
# In degrees of the center node - note that the graph is directed so the in degree should be 0.
print(g.in_degrees(0))
# print(g.ndata)


# SubGraph 节点信息等都会被处理
# Induce a subgraph from node 0, node 1 and node 3 from the original graph.
sg1 = g.subgraph([0, 1, 3])
# Induce a subgraph from edge 0, edge 1 and edge 3 from the original graph.
sg2 = g.edge_subgraph([0, 1, 3])

# The original IDs of each node in sg1
print(sg1.ndata[dgl.NID])
# The original IDs of each edge in sg1
print(sg1.edata[dgl.EID])
# The original IDs of each node in sg2
print(sg2.ndata[dgl.NID])
# The original IDs of each edge in sg2
print(sg2.edata[dgl.EID])

# The original node feature of each node in sg1
print(sg1.ndata['x'])
# The original edge feature of each node in sg1
print(sg1.edata['a'])
# The original node feature of each node in sg2
print(sg2.ndata['x'])
# The original edge feature of each node in sg2
print(sg2.edata['a'])

newg = dgl.add_reverse_edges(g)
print(newg.edges())

# Save graphs
dgl.save_graphs('graph.dgl', g)
dgl.save_graphs('graphs.dgl', [g, sg1, sg2])

# Load graphs
(g,), _ = dgl.load_graphs('graph.dgl')
print(g)
(g, sg1, sg2), _ = dgl.load_graphs('graphs.dgl')
print(g)
print(sg1)
print(sg2)