# GNN-Based DDI Prediction

> 本工程作为作者的本科毕业设计，只是作为记录代码的更新迭代与同步服务器的工具
>
> 对于代码没有达到成熟的简洁程度我表示抱歉，由于研究生阶段不会接着炼丹，我对于整理工程代码没有很强烈的欲望，在此再次抱歉



## 1. 数据集

通过图神经网络预测DDI（Drug-Drug Interaction）问题。

原始数据集位于`data/event.db`中



## 2. 目录结构

本工程的目录结构按照工业界中的工程存放逻辑

- src：工程源文件，核心功能函数
  - models：存放图神经网络的不同模型，基于dgl搭建
    - script：主要是存放模型文件
  - utils：用于模型评估、预处理等功能函数
    - 在这里由于在实验过程中方便测试，很多模型的测试工作并没有严格按照这个逻辑，例如GateGAT就冗余在一起。某些评估指标如Precision、Recall、F1也没有严格放置在utils.Function中
  - EdgeClassification
    - main通过该接口调用不同的模型
- data：存放本次工程的数据集
- doc：放置文档说明
- log：记录模型产生的结果
- compare.py & ModelCompare.py都是用来测试用MacOS训练神经网络速度的脚本。其中使用到了taichi与pytorch刚推出的mps device训练框架



## 执行

```shell
cd src
python main.py -m graphsage -o graph-output
```

- `-m`：指定训练使用的模型
  - gat
  - graphsage
  - gcn
  - gategat
- -o：指定输出的文件名，记录训练的准确率



以上面的命令为例，会执行graphSAGE，输出到graph-output.txt文件中

由于作者 **精力有限**，目前只暴露了这两个命令行接口。
