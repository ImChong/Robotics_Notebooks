# GCN：图卷积网络的半监督分类（arXiv:1609.02907）

> 论文来源归档（ingest）

- **标题：** Semi-Supervised Classification with Graph Convolutional Networks
- **作者：** Thomas N. Kipf, Max Welling
- **类型：** paper / graph-neural-network / semi-supervised
- **arXiv：** <https://arxiv.org/abs/1609.02907> · PDF：<https://arxiv.org/pdf/1609.02907.pdf>
- **会议：** ICLR 2017
- **官方代码：** <https://github.com/tkipf/gcn>
- **入库日期：** 2026-09-21
- **一句话说明：** 用一阶切比雪夫近似得到 **对称归一化邻接矩阵** 上的层间传播 \(H'=\sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW)\)，在引用网络等图上做半监督节点分类。

## 核心摘录（面向 wiki 编译）

### 1) 图上的「卷积」是邻域聚合

- **要点：** 每个节点用邻居特征的归一化加权和再做线性变换与非线性。归纳偏置是 **关系局部性**：标签沿边平滑。
- **对 wiki 的映射：** [`wiki/concepts/graph-neural-network.md`](../../wiki/concepts/graph-neural-network.md)

### 2) 重归一化技巧稳住深层

- **要点：** \(\tilde A=A+I\) 加自环，再对称归一化，缓解数值尺度与过平滑早期形式。后续 GIN/GAT/GraphSAGE 都在改聚合器，而不是改「图=数据几何」这一前提。
- **对 wiki 的映射：** [`wiki/concepts/graph-neural-network.md`](../../wiki/concepts/graph-neural-network.md)、[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)

### 3) 机器人图：运动学树、场景图、接触图

- **要点：** 关节树、物体关系、多机器人通信天然是图。GNN 适合编码 **可变拓扑**，不适合替代高频向量策略。常见用法是图编码器 + MLP/Transformer 头。
- **对 wiki 的映射：** [`wiki/concepts/humanoid-policy-network-architecture.md`](../../wiki/concepts/humanoid-policy-network-architecture.md)

## 开源状态（步骤 2.5）

- `tkipf/gcn` **已开源**（TensorFlow 1 参考实现）。

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
