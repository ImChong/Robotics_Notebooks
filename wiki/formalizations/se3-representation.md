---
type: formalization
tags: [kinematics, math, deep-learning, rotation]
status: complete
updated: 2026-04-21
related:
  - ../concepts/whole-body-control.md
  - ../methods/visual-servoing.md
  - ../formalizations/mdp.md
sources:
  - ../../sources/papers/perception.md
summary: "SE(3) 位姿表示形式化：探讨了欧拉角、四元数、旋转矩阵及 6D 连续表示在机器人学习中的优劣对比，重点关注其在神经网络训练中的连续性与独特性。"
---

# SE(3) Representation (位姿表示形式化)

在机器人学与具身智能中，如何表示物体的**位姿（Pose）**——即位置与姿态的组合，是感知与控制的基础。**SE(3)** (Special Euclidean Group) 描述了三维空间中的刚体运动。

## 数学定义

一个 SE(3) 元素通常由两部分组成：
- **位置 (Translation)**：$t \in \mathbb{R}^3$。
- **姿态 (Rotation)**：属于 SO(3) 组，$R \in SO(3)$。

$$ T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4} $$

## 主流表示法对比（面向神经网络）

在深度学习模型（如 VLA 或 Pose Estimation）中，选择姿态表示法至关重要，因为它直接影响梯度的平滑性和损失函数的收敛。

| 表示法 | 维度 | 优势 | 劣势 | 推荐场景 |
|------|-----|-----|-----|---------|
| **欧拉角 (Euler Angles)** | 3 | 直观，最省空间 | 存在万向节死锁 (Gimbal Lock)；不连续 | 简单的 UI 显示 |
| **四元数 (Quaternions)** | 4 | 紧凑，无死锁，插值平滑 | 存在双倍覆盖 ($q = -q$)；单位化约束 | 控制器内部状态 |
| **旋转矩阵 (Rotation Matrix)** | 9 | 线性，无死锁，唯一 | 自由度冗余 (9D 表示 3D)；需要正交化 | 坐标变换计算 |
| **6D 连续表示 (6D Rep)** | 6 | **姿态空间连续**；适合神经网络回归 | 需要格拉姆-施密特正交化 | **Deep Learning 姿态估计** |

### 1. 为什么 6D 表示法更适合 DL？
传统的四元数和欧拉角在 $\mathbb{R}^n$ 到 $SO(3)$ 的映射过程中存在**不连续点**。这意味着当网络预测值发生微小连续变化时，映射出的旋转可能发生突变。
6D 表示法通过取旋转矩阵的前两列 $(a_1, a_2)$，并利用正交化重建第三列，实现了全空间的连续映射。

## 损失函数 (Loss Functions)

对于 SE(3) 位姿回归，常见的损失函数包括：
- **L2 距离**：分别计算位置和姿态分量的均方误差。
- **测地线距离 (Geodesic Distance)**：姿态误差最真实的物理描述：
  $$ \mathcal{L}_{rot} = \arccos\left( \frac{\text{Tr}(R_{pred} R_{target}^T) - 1}{2} \right) $$

## 关联页面
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Visual Servoing](../methods/visual-servoing.md)
- [Action Tokenization](./vla-tokenization.md)

## 参考来源
- Zhou, Y., et al. (2019). *On the continuity of rotation representations in neural networks*. (CVPR 最佳论文候选，提出了 6D 表示)
- [sources/papers/perception.md](../../sources/papers/perception.md)
