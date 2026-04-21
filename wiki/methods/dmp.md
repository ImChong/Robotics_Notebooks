---
type: method
tags: [control, manipulation, primitives, imitation-learning]
status: complete
updated: 2026-04-21
related:
  - ./imitation-learning.md
  - ./behavior-cloning.md
sources:
  - ../../sources/papers/imitation_learning.md
summary: "动态运动基元（DMP）通过二阶微分方程描述运动轨迹，具有空间伸缩不变性和时域缩放特性，是轨迹级模仿学习的经典工具。"
---

# Dynamic Movement Primitives (DMP)

**DMP** 是一种用于轨迹建模和控制的方法。它将复杂的运动路径表示为一个非线性动力学系统，其核心是一个受迫振荡器，可以通过调整参数来改变运动的速度和目标位置，而不需要重新规划。

## 主要技术路线

DMP 将轨迹建模为如下二阶微分方程：
$$ \tau^2 \ddot{y} = \alpha_y (\beta_y (g - y) - \dot{y}) + f(s) $$
其中：
- $g$ 是目标位置（Goal）。
- $y$ 是当前位置。
- $f(s)$ 是一个可学习的非线性驱动项，通常由一组基函数（如高斯核）线性组合而成。

## 核心优势

1. **目标自适应**：改变 $g$ 的值，系统会自动平滑地向新目标收敛。
2. **时域缩放**：通过改变时间常数 $\tau$，可以随心所欲地加快或放慢动作。
3. **空间不变性**：平移或缩放起始点和终点，动作的整体轮廓保持不变。

它是实现多任务[全身协调控制](../concepts/whole-body-coordination.md)的一种轻量化轨迹表示方式。

## 关联页面
- [Imitation Learning (模仿学习)](./imitation-learning.md)
- [Behavior Cloning](./behavior-cloning.md)

## 参考来源
- Ijspeert, A. J., et al. (2013). *Dynamical movement primitives: Learning attractor models for motor behaviors*.
