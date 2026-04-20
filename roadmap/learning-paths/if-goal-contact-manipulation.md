# 学习路径：如果目标是接触丰富的操作任务

> 如果你想让机器人完成装配、拧螺丝、插拔等需要精细接触的任务，从这里切入。

**这条路径怎么用：**
- 目标读者是有 RL/IL 基础、想深入操作任务的工程师
- 需要了解基础动力学和控制，Python 编程熟练
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

---

## Stage 0 操作基础

### 前置知识
- 机器人运动学基础（正逆运动学）
- 基础控制理论（PID、阻抗控制概念）
- Python + ROS 或类似框架

### 核心问题
- 什么叫"接触丰富"，和 free-space manipulation 有什么区别
- 刚性抓取和顺应性控制分别适合什么场景

### 推荐读什么
- [Manipulation](../../wiki/tasks/manipulation.md)
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- Mason, *Mechanics of Robotic Manipulation* — Chapter 1-2

### 学完输出什么
- 能区分 prehensile / non-prehensile / contact-rich 操作
- 理解阻抗控制（impedance control）的基本原理

---

## Stage 1 接触力建模与控制

### 核心问题
- 怎么建模接触力和摩擦
- 阻抗控制 vs. 导纳控制 vs. 力控有什么区别

### 推荐读什么
- [Contact Dynamics](../../wiki/concepts/contact-dynamics.md)
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
- Hogan, *Impedance Control* (1985)

### 推荐做什么
- 在 MuJoCo 或 Isaac Sim 中实现一个 impedance controller，完成推箱子任务
- 观察接触刚度参数对稳定性的影响

### 学完输出什么
- 能写出 impedance control 的数学形式
- 能解释 soft contact 模型（MuJoCo）vs. hard contact 的区别

---

## Stage 2 模仿学习用于操作

### 核心问题
- 为什么 contact-rich 任务适合 IL 而不是 RL
- ACT 和 Diffusion Policy 各适合什么场景

### 推荐读什么
- [Imitation Learning](../../wiki/methods/imitation-learning.md)
- [Behavior Cloning](../../wiki/methods/behavior-cloning.md)
- [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT, 2023)

### 推荐做什么
- 用 ACT 框架收集遥操作数据并训练一个抓取策略
- 对比 chunk size 对 contact-rich 任务成功率的影响

### 学完输出什么
- 能解释 ACT 的 action chunking 机制和 Diffusion Policy 的多模态优势
- 能设计一个 contact-rich 任务的数据采集方案

---

## Stage 3 Contact-Rich 策略进阶

### 核心问题
- 如何在 sim2real 中处理 contact-rich 任务的接触不一致问题
- 有哪些专门针对 contact-rich 的学习方法

### 推荐读什么
- [Query：接触丰富操作实践指南](../../wiki/queries/contact-rich-manipulation-guide.md)
- [Demo Data Collection Guide](../../wiki/queries/demo-data-collection-guide.md)
- Luo et al., *DEFT: Dexterous Fine-Grained Manipulation Transformer* (2024)

### 推荐做什么
- 尝试在真机上部署 ACT 策略，使用 FT sensor 记录接触力
- 分析失败案例，找出接触力误差模式

### 学完输出什么
- 能识别 contact-rich sim2real 迁移的主要瓶颈
- 能设计 FT sensor feedback 的简单修正机制

---

## 关联页面

- [Manipulation](../../wiki/tasks/manipulation.md)
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- [Bimanual Manipulation](../../wiki/tasks/bimanual-manipulation.md)
- [Imitation Learning](../../wiki/methods/imitation-learning.md)
- [Behavior Cloning](../../wiki/methods/behavior-cloning.md)
- [Query：接触丰富操作实践指南](../../wiki/queries/contact-rich-manipulation-guide.md)
