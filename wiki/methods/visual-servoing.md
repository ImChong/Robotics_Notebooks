---
type: method
tags: [perception, control, visual-servoing, manipulation, camera]
status: complete
updated: 2026-04-21
related:
  - ../concepts/tactile-sensing.md
  - ../tasks/manipulation.md
  - ../concepts/state-estimation.md
  - ../formalizations/behavior-cloning-loss.md
sources:
  - ../../sources/papers/perception.md
summary: "视觉伺服（Visual Servoing）将相机图像特征的误差直接反馈到机器人运动控制器中，形成闭环视觉控制，有效克服了传统“查表式”抓取的开环误差。"
---

# Visual Servoing（视觉伺服控制）

**视觉伺服 (Visual Servoing)** 是一门将计算机视觉（Computer Vision）与经典控制理论（Control Theory）深度融合的技术。它不依赖于将图像构建为复杂的 3D 静态世界模型，而是直接**利用相机画面中的图像特征（Image Features）计算误差，并将其作为反馈信号实时计算机器人的运动指令**。

这使得机器人在面对目标物体移动、环境扰动以及自身运动学标定不准确时，仍然能够精准地完成追踪、抓取或对齐任务。

## 核心理念：闭环视觉

在传统的“抓取流水线（Grasp Pipeline）”中：
1. 相机拍照 -> 2. 位姿估计算法算出物体 3D 坐标 -> 3. 规划一条末端到坐标的轨迹 -> 4. 盲摸（开环执行）。
如果在步骤 4 的过程中物体移动了 1 厘米，机器人就会抓空。

**视觉伺服**打破了这种开环模式，将其变为一个高频闭环（通常 30Hz-60Hz）：
在每一步，控制器都在计算“当前画面中目标的位置”与“期望画面中目标的位置”的误差，并计算出让机械臂向期望位置靠近的即时速度指令。

## 主要技术路线

根据误差是在 2D 图像平面计算，还是在 3D 三维空间计算，视觉伺服分为两类：

### 1. 基于图像的视觉伺服 (Image-Based Visual Servoing, IBVS)
- **原理**：直接在 2D 图像平面上计算特征点（如角点、边缘、孔洞中心）的像素坐标 $(u, v)$ 与期望像素坐标 $(u^*, v^*)$ 的误差。
- **数学核心**：需要计算**图像雅可比矩阵 (Image Jacobian / Interaction Matrix)** $\mathbf{L_s}$，它描述了相机的 3D 空间速度 $\mathbf{v_c}$ 如何引起图像特征点像素速度 $\mathbf{\dot{s}}$ 的变化：$\mathbf{\dot{s}} = \mathbf{L_s} \mathbf{v_c}$。控制律通常基于求伪逆来反推相机速度：$\mathbf{v_c} = -\lambda \mathbf{L_s}^+ (\mathbf{s} - \mathbf{s}^*)$。
- **优点**：极度鲁棒。不需要知道物体的精确 3D 模型，对相机的内参（焦距等）标定误差具有极强的抵抗力。
- **缺点**：容易遇到局部极小值；机器人末端可能在 3D 空间中走出极其奇怪的弧线轨迹（因为它是严格按照图像 2D 直线走），甚至可能导致手眼相机退退（Camera Retreat）问题。

### 2. 基于位置的视觉伺服 (Position-Based Visual Servoing, PBVS)
- **原理**：先从 2D 图像中提取特征，利用相机的内外参，实时估算出目标物体相对于相机的 3D 相对位姿 $({}^c\mathbf{T}_o)$，然后在 3D 三维空间（SE(3)）中计算当前位姿与期望位姿的误差，并生成运动指令。
- **优点**：末端执行器在 3D 空间中的运动轨迹是最优的（如走直线），不会发生退退问题。
- **缺点**：强烈依赖精准的相机标定（Camera Calibration）和完美的 3D CAD 模型估计算法。一旦标定出现少许偏差，控制立刻发散。

## 在现代机器人学习中的地位

尽管端到端的视觉强化学习（Vision-based RL）和基础模型（VLA）正在崛起，但传统视觉伺服的思想依然是其底层地基：

1. **混合架构 (Hybrid Architecture)**：很多前沿方案使用 VLA 输出高层的抓取策略（如“抓取杯子把手”），而在最后几厘米的接触阶段，切换为经典的 IBVS 以保证极其精准的亚毫米级对齐。
2. **密集监督 (Dense Supervision)**：视觉伺服的算法（已知图像雅可比）可以直接作为模仿学习（Imitation Learning）的自动数据采集器，廉价生成海量精准的专家演示轨迹。

## 关联页面
- [AprilTag（视觉 fiducial 库）](../entities/april-tag.md) — PBVS 与标定中常用的 6D 位姿观测源
- [Tactile Sensing (触觉感知)](../concepts/tactile-sensing.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [State Estimation](../concepts/state-estimation.md)
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md)

## 参考来源
- Chaumette, F., & Hutchinson, S. (2006). *Visual servo control. I. Basic approaches*. IEEE Robotics & Automation Magazine.
- [sources/papers/perception.md](../../sources/papers/perception.md)
