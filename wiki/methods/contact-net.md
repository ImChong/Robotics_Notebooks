---
type: method
tags: [perception, contact, deep-learning, point-cloud]
status: complete
updated: 2026-04-21
related:
  - ../concepts/contact-estimation.md
  - ../concepts/terrain-adaptation.md
sources:
  - ../../sources/papers/perception.md
summary: "ContactNet 是一种基于点云的深度学习模型，专门用于预测物体的接触面与抓取概率，是实现凌乱场景下稳健操作的核心感知技术。"
---

# ContactNet

**ContactNet** 解决了“在杂乱无章的堆叠物中，手手该按在哪”的问题。它直接输入原始点云，输出稠密的接触成功概率图。

## 主要技术路线

1. **点云表征**：使用 PointNet++ 或 Transformer 提取空间特征。
2. **接触平面预测**：为每个点预测一个局部的 6D 接触框架（Contact Frame）。
3. **抓取过滤**：利用几何约束剔除会导致碰撞或力矩不平衡的备选点。

## 关联页面
- [Contact Estimation (接触估计)](../concepts/contact-estimation.md)
- [Terrain Adaptation](../concepts/terrain-adaptation.md)

## 参考来源
- Qin, H., et al. (2020). *ContactNet: Weakly Supervised Learning of Grasping Surfaces*.
