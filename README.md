# My Graduation Project

通过图神经网络预测DDI（Drug-Drug Interaction）问题。

原始数据集位于`data/event.db`中



## 执行

```shell
cd src
python main [……]
```

- `-m`：指定训练使用的模型
  - gat
  - graphsage
  - gcn
  - gategat
- -o：指定输出的文件名，记录训练的准确率
- 
