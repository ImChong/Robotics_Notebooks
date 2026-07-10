---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.16012"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dualthor.md
summary: "开发能在真实场景做复杂交互任务的具身智能体，仍是具身 AI 的根本挑战。DualTHOR 是一个面向复杂双臂人形机器人的物理仿真平台，构建在扩展版 AI2-THOR 之上。它包含：真实世界机器人资产、双臂协作任务套件、以及面向人形形态优化的逆运动学（IK）求解器；并引入一个纳入执行失败的「意外（contingency）」机制，让仿真更贴近现实的不确定性。论文用它评测视觉语言模型（VLM）在家务任务上的表现，发现当前 VLM 在双臂协调上能力有限、面对现实意外时鲁棒性下降，凸显该平台对发展更强具身 AI 的价值。"
---

# DualTHOR

**DualTHOR: A Dual-Arm Humanoid Simulation Platform for Contingency-Aware Planning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

开发能在真实场景做复杂交互任务的具身智能体，仍是具身 AI 的根本挑战。DualTHOR 是一个面向复杂双臂人形机器人的物理仿真平台，构建在扩展版 AI2-THOR 之上。它包含：真实世界机器人资产、双臂协作任务套件、以及面向人形形态优化的逆运动学（IK）求解器；并引入一个纳入执行失败的「意外（contingency）」机制，让仿真更贴近现实的不确定性。论文用它评测视觉语言模型（VLM）在家务任务上的表现，发现当前 VLM 在双臂协调上能力有限、面对现实意外时鲁棒性下降，凸显该平台对发展更强具身 AI 的价值。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| AI2-THOR | 一个具身 AI 室内仿真环境 |
| Dual-Arm Humanoid | 双臂人形 |
| Contingency | 意外/突发，含执行失败 |
| IK | Inverse Kinematics，逆运动学 |
| VLM | Vision-Language Model |
| Task Suite | 任务套件 |

## 为什么重要

- **"意外感知"是从仿真走向现实的关键缺口**：现实充满执行失败，仿真应建模之；
- **双臂人形平台**填补单臂仿真的空白；
- **VLM 仍难做双臂协调**，提示高层规划与底层执行的鸿沟；
- 与本仓 11 其它仿真/基准平台共同丰富评测生态。

## 解决什么问题

具身 AI 缺**双臂人形 + 贴近现实意外**的仿真： - 多数平台面向单臂/简化抓取； - 缺**执行失败/意外**建模，与真实差距大； - 不清楚 VLM 在双臂协调与意外下表现如何。

DualTHOR 要：一个**双臂人形、含意外机制**的物理仿真平台 + VLM 评测。

## 核心机制

1. **双臂人形物理仿真平台**：扩展 AI2-THOR，含真实资产、双臂任务、人形 IK；
2. **意外机制**：纳入执行失败，支持意外感知规划研究；
3. **VLM 评测**：揭示当前 VLM 双臂协调与抗意外的不足；
4. **研究资源**：为更强具身 AI 提供测试床。

方法拆解（深读笔记小节）：扩展 AI2-THOR 的双臂人形物理仿真；意外（contingency）机制；VLM 评测；发现；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning.html> |
| arXiv | <https://arxiv.org/abs/2506.16012> |
| 作者 | Boyu Li、Siyuan He、Hang Xu、Haoqi Yuan、Junpeng Yue、Börje F. Karlsson、Zongqing Lu 等 |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dualthor.md](../../sources/papers/humanoid_pnb_dualthor.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning.html>
- 论文：<https://arxiv.org/abs/2506.16012>

## 推荐继续阅读

- [机器人论文阅读笔记：DualTHOR](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning/DualTHOR__A_Dual-Arm_Humanoid_Simulation_Platform_for_Contingency-Aware_Planning.html)
