---
type: concept
tags: [mlp, feedforward, deep-learning, policy, architecture]
status: complete
updated: 2026-09-21
summary: "多层感知机是逐层仿射+非线性的向量函数逼近器，也是机器人低频观测与高频关节策略最常见的默认骨干；加深靠残差，扩容靠 MoE，而不是无限加宽一层。"
related:
  - ./mixture-of-experts.md
  - ./deep-learning-foundations.md
  - ./backpropagation.md
  - ./humanoid-policy-network-architecture.md
  - ../overview/ai-architecture-map.md
  - ../entities/paper-resnet-deep-residual-learning.md
sources:
  - ../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md
  - ../../sources/papers/ai_architecture_foundations.md
  - ../../sources/personal/ai-architecture-map.md
---

# MLP / Feedforward NN（多层感知机）

**多层感知机（MLP）**：把输入向量交替通过 **仿射变换** 与 **逐点非线性**，组成 \(f(x)=W_L\sigma(W_{L-1}\cdots\sigma(W_1x+b_1)\cdots)+b_L\) 的前馈网络，不含卷积、循环或注意力。

## 一句话定义

在 **固定维向量** 上做可微函数逼近的最小完备积木：能拟合连续映射，推理延迟极低，因此至今仍是腿式 / 人形 **高频策略** 的默认骨干。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLP | Multi-Layer Perceptron | 多层全连接前馈网络 |
| FFN | Feed-Forward Network | Transformer 块里的逐位置 MLP |
| ReLU | Rectified Linear Unit | 常用分段线性激活 |
| PPO | Proximal Policy Optimization | 真机 locomotion 里最常训 MLP 策略的算法 |
| Sim2Real | Simulation to Real | 小 MLP 策略能落地，往往因为延迟与吞吐而不是因为表达力最大 |

## 为什么重要

- **万能逼近**只保证「够深/够宽时能表示」，不保证好训、好泛化；机器人真正用的是 **2–3 层、256–1024 宽** 的可部署逼近器。
- Transformer、ViT、DiT 内部仍有大量 **逐 token MLP**；读不懂 MLP 就读不懂现代块结构。
- [人形策略网络架构](./humanoid-policy-network-architecture.md) 的经验事实：真机最强 locomotion 经常不是最大模型，而是 **小 MLP + 好观测/奖励/sim2real**。

## 核心原理

### 1. 一层在做什么

一层计算 \(h=\sigma(Wx+b)\)。\(W\) 混合同一时刻的全部坐标；\(\sigma\) 提供非线性，否则整网塌成一次仿射。输出头按任务换：分类 logit、连续动作均值、价值标量。

### 2. 深度与残差

朴素加深会出现 **退化**：更深训练误差反而升。残差把映射改成「学扰动」，见 [ResNet](../entities/paper-resnet-deep-residual-learning.md)。策略网很少堆到百层，但 actor/critic 里的 skip 与 LayerNorm 同源。

### 3. 与 MoE 的分工

稠密 MLP 每步用全部参数。需要多技能容量时，把门控接到多个专家 MLP 上，见 [MoE](./mixture-of-experts.md)。低层力矩环通常 **不必** 上 MoE。

```mermaid
flowchart LR
  x["向量观测 x"] --> a1["仿射 W1"]
  a1 --> n1["非线性 σ"]
  n1 --> a2["仿射 W2"]
  a2 --> y["动作 / 价值"]
```

## 工程实践

| 项 | 建议 |
|----|------|
| 腿式默认 | 2–3 层，宽 256–512，`ELU`/`tanh`；先测延迟再加宽 |
| 观测 | 本体 + 指令；视觉应先经 CNN/ViT 压成向量再进 MLP |
| 调试 | 梯度范数、动作饱和、价值尺度；激活别在输出层乱加 `tanh` 除非动作本身有界 |
| 并行仿真 | 小 MLP 才能吃满 Isaac/MJX 的 env 吞吐 |

## 局限与风险

- **无结构归纳偏置**：像素、点云、图直接拉平会丢几何，应用专用骨干。
- **无记忆**：单步 MLP 看不见历史，除非你把历史堆进观测或外接 RNN/SSM。
- 误区：「换更大 MLP 就能补奖励设计」——真机瓶颈很少是隐藏单元个数。

## 关联页面

- [MoE](./mixture-of-experts.md)
- [深度学习基础](./deep-learning-foundations.md)
- [反向传播](./backpropagation.md)
- [人形策略网络架构](./humanoid-policy-network-architecture.md)
- [AI 架构地图](../overview/ai-architecture-map.md)

## 参考来源

- [Rumelhart et al. 反向传播（Nature 1986）](../../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md)
- [AI 架构地图一手论文簇](../../sources/papers/ai_architecture_foundations.md)
- [AI 架构地图 taxonomy](../../sources/personal/ai-architecture-map.md)

## 推荐继续阅读

- [Deep Learning Book Ch. 6 — MLP](https://www.deeplearningbook.org/contents/mlp.html)
