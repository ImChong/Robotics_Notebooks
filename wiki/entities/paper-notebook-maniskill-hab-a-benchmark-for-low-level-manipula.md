---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2412.13211"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_maniskill-hab.md
summary: "高质量基准是具身 AI 的基础，能推动长时程导航、操作与重排的进展。本文提出 MS-HAB（ManiSkill-HAB）：一个 GPU 加速的家庭助理基准（Home Assistant Benchmark, HAB）实现，提供真实的低层控制，相比此前\"魔法抓取（magical grasp）\"实现取得 3 倍以上提速且显存更省。作者训练了 RL 与 IL 基线，并开发一个基于规则的轨迹过滤系统，以大规模生成可控的演示数据。这把以往偏抽象/魔法抓取的家务重排基准，落到真实低层操作与高效仿真上。"
---

# ManiSkill-HAB

**ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

高质量基准是具身 AI 的基础，能推动长时程导航、操作与重排的进展。本文提出 MS-HAB（ManiSkill-HAB）：一个 GPU 加速的家庭助理基准（Home Assistant Benchmark, HAB）实现，提供真实的低层控制，相比此前"魔法抓取（magical grasp）"实现取得 3 倍以上提速且显存更省。作者训练了 RL 与 IL 基线，并开发一个基于规则的轨迹过滤系统，以大规模生成可控的演示数据。这把以往偏抽象/魔法抓取的家务重排基准，落到真实低层操作与高效仿真上。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| MS-HAB | ManiSkill-HAB，本文基准 |
| HAB | Home Assistant Benchmark，家庭助理基准 |
| Low-Level Control | 低层控制（真实物理操作，非魔法抓取） |
| Magical Grasp | 魔法抓取，抽象的瞬时抓取 |
| Trajectory Filtering | 轨迹过滤，筛选高质量演示 |
| RL / IL | 强化学习 / 模仿学习 |

## 为什么重要

- **"真实低层 vs 魔法抓取"很关键**：抽象抓取会高估能力，真实操作才有迁移价值；
- **GPU 加速仿真**是规模化训练/采集的前提；
- **规则过滤生成可控演示**是低成本扩数据的实用手段；
- 虽以移动机械臂为主，重排/低层操作经验对人形家务同样适用。

## 解决什么问题

家务**重排**基准常用"**魔法抓取**"（抽象抓取），与真实**低层操作**脱节，且**仿真慢**： - 缺**真实低层控制**的家居重排基准； - 仿真**效率低**，难规模化训练/采集。

ManiSkill-HAB 要：一个 **GPU 加速、真实低层、可大规模生成演示**的家居重排基准。

## 核心机制

1. **真实低层家居重排基准**：取代魔法抓取，落到真实操作；
2. **GPU 加速 >3x、显存更省**：高效仿真；
3. **RL/IL 基线**：提供可比较参考；
4. **规则轨迹过滤大规模生成演示**：支撑模仿学习。

方法拆解（深读笔记小节）：GPU 加速 + 真实低层控制；RL / IL 基线；规则化轨迹过滤生成演示；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks.html> |
| arXiv | <https://arxiv.org/abs/2412.13211> |
| 作者 | Arth Shukla、Stone Tao、Hao Su（UC San Diego） |
| 发表 | 2024 年 12 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_maniskill-hab.md](../../sources/papers/humanoid_pnb_maniskill-hab.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks.html>
- 论文：<https://arxiv.org/abs/2412.13211>

## 推荐继续阅读

- [机器人论文阅读笔记：ManiSkill-HAB](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks/ManiSkill-HAB__A_Benchmark_for_Low-Level_Manipulation_in_Home_Rearrangement_Tasks.html)
