---
type: concept
tags: [gnn, graph, architecture, relational]
status: complete
updated: 2026-09-21
summary: "图神经网络在节点与边上做邻域聚合，适合运动学树、场景图与接触关系等可变拓扑；它是结构编码器，通常不替代高频向量策略。"
related:
  - ./mlp.md
  - ./convolutional-neural-network.md
  - ./humanoid-policy-network-architecture.md
  - ../overview/ai-architecture-map.md
sources:
  - ../../sources/papers/kipf_gcn_arxiv_1609_02907.md
  - ../../sources/papers/ai_architecture_foundations.md
---

# GNN（Graph Neural Network，图神经网络）

**GNN**：把数据写成图 \(G=(V,E)\)，每层让节点用 **邻居聚合** 更新自身表示。图卷积网络（GCN）是最常用的一阶线性聚合特例。

## 一句话定义

当「谁和谁有关系」比「谁在网格第几格」更重要时，用边而不是像素邻域来定义局部性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GNN | Graph Neural Network | 图上邻域聚合网络族 |
| GCN | Graph Convolutional Network | 对称归一化邻接的一阶图卷积 |
| GAT | Graph Attention Network | 用注意力加权邻居 |
| MPNN | Message Passing Neural Network | 消息传递统一视角 |
| Homophily | Homophily | 相邻节点标签相似的假设 |

## 为什么重要

- [Kipf & Welling, 2017](../../sources/papers/kipf_gcn_arxiv_1609_02907.md) 把谱图卷积收成可训练的一阶传播，成为后续 GAT/GraphSAGE 的基线。
- 机器人里 **关节树、物体关系、多机通信、接触图** 都是可变大小图，硬做成固定向量会丢拓扑。
- 正确用法是 **图编码器 → MLP/Transformer 策略头**，而不是整条控制环都 GNN。

## 核心原理

GCN 一层：

\[
H'=\sigma\big(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}HW\big),\quad \tilde A=A+I
\]

每个节点看到一阶邻居的归一化均值，再做共享线性变换。堆叠 \(L\) 层约等于 \(L\) 跳邻域。过深会 **过平滑**：节点特征趋同。

```mermaid
flowchart LR
  g["图 G=(V,E)"] --> agg["邻域聚合"]
  agg --> upd["节点更新"]
  upd --> pool["可选读出"]
  pool --> head["任务头"]
```

## 工程实践

| 场景 | 倾向 |
|------|------|
| 固定人形拓扑 | 先问 MLP + 结构化观测是否够用 |
| 物体/关系变化 | 场景图 GNN 或关系 Transformer |
| 点云 | 动态 kNN 图；注意构图开销 |
| 训练 | 监控过平滑（节点特征余弦↑）与度分布偏差 |

## 局限与风险

- 聚合假设标签沿边平滑；异配图（相邻节点语义相反）会失效。
- 构图本身可能比网络还贵。
- 不要把注意力 Transformer 在「完全图」上的应用误称为 GNN——有边稀疏性才是图归纳偏置。

## 关联页面

- [MLP](./mlp.md)
- [CNN](./convolutional-neural-network.md)
- [人形策略网络架构](./humanoid-policy-network-architecture.md)
- [AI 架构地图](../overview/ai-architecture-map.md)

## 参考来源

- [GCN（arXiv:1609.02907）](../../sources/papers/kipf_gcn_arxiv_1609_02907.md)
- [AI 架构地图一手论文簇](../../sources/papers/ai_architecture_foundations.md)

## 推荐继续阅读

- 官方实现：<https://github.com/tkipf/gcn>
