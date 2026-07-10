---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.00833"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoidgen.md
summary: "现有机器人数据集与仿真基准多面向机械臂平台；对装备双臂 + 灵巧手的人形，仿真任务与高质量演示明显匮乏。双手灵巧操作更复杂——需协调臂运动与手操作，自主采集难。HumanoidGen 是一个自动化任务创建与演示采集框架，利用原子灵巧操作与 LLM 推理生成关系约束。具体：基于原子操作为资产与灵巧手提供空间标注，再用 LLM 规划器依据物体可供性（affordance）与场景生成一串可执行的臂运动空间约束；并用蒙特卡洛树搜索（MCTS）变体增强 LLM 在长时程任务与标注不足下的推理。实验里新建一个含增强场景的基准评估数据质量，结果显示 2D 与 3D 扩散策略的性能可随生成数据规模提升。"
---

# HumanoidGen

**HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

现有机器人数据集与仿真基准多面向机械臂平台；对装备双臂 + 灵巧手的人形，仿真任务与高质量演示明显匮乏。双手灵巧操作更复杂——需协调臂运动与手操作，自主采集难。HumanoidGen 是一个自动化任务创建与演示采集框架，利用原子灵巧操作与 LLM 推理生成关系约束。具体：基于原子操作为资产与灵巧手提供空间标注，再用 LLM 规划器依据物体可供性（affordance）与场景生成一串可执行的臂运动空间约束；并用蒙特卡洛树搜索（MCTS）变体增强 LLM 在长时程任务与标注不足下的推理。实验里新建一个含增强场景的基准评估数据质量，结果显示 2D 与 3D 扩散策略的性能可随生成数据规模提升。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Bimanual Dexterous | 双手灵巧（操作） |
| Atomic Operation | 原子操作，可复用的基本灵巧动作 |
| Affordance | 可供性，物体可被如何操作 |
| LLM Planner | 大模型规划器，生成空间约束链 |
| MCTS | 蒙特卡洛树搜索，增强长时程推理 |
| Diffusion Policy | 扩散策略（2D/3D） |

## 为什么重要

- **自动数据生成是双手灵巧操作的关键瓶颈解法**：自主采集难，LLM 规划 + 仿真生成是出路；
- **可供性 + 空间约束**把语义规划接到底层动作；
- **MCTS 补 LLM 长时程推理**是实用组合；
- 与 DexMimicGen、HumanoidGen 等数据生成工作共同壮大人形操作数据。

## 解决什么问题

人形**双手灵巧操作**缺数据： - 现有数据/基准多为**单臂**； - 双手灵巧需**协调臂 + 手**，**自主采集难**； - 缺高质量演示来训策略。

HumanoidGen 要：**自动**生成双手灵巧任务与高质量演示。

## 核心机制

1. **人形双手灵巧数据生成框架**：自动任务创建 + 演示采集；
2. **LLM 规划生成空间约束链**：依可供性/场景把双手协作显式化；
3. **MCTS 增强**：应对长时程与稀疏标注；
4. **数据有效性**：新基准上 2D/3D 扩散策略随数据规模提升。

方法拆解（深读笔记小节）：原子操作 + 空间标注；LLM 规划器生成空间约束链；MCTS 增强长时程推理；基准与结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html> |
| arXiv | <https://arxiv.org/abs/2507.00833> |
| 作者 | Zhi Jing、Siyuan Yang、Jicong Ao、Ting Xiao、Yu-Gang Jiang、Chenjia Bai |
| 发表 | 2025 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoidgen.md](../../sources/papers/humanoid_pnb_humanoidgen.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html>
- 论文：<https://arxiv.org/abs/2507.00833>

## 推荐继续阅读

- [机器人论文阅读笔记：HumanoidGen](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html)
