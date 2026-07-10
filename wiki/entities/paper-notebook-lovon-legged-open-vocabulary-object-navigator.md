---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.06747"
related:
  - ../overview/paper-notebook-category-08-navigation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_lovon.md
summary: "开放世界里的「找东西并走过去」是一个长时程任务：既要能识别任意自然语言指定的目标（开放词汇），又要把「找 → 搜 → 接近 → 完成」拆成可执行的子动作（高层规划），还要扛住足式机器人行走时的画面抖动、盲区、目标暂时丢失等现实问题。LOVON 把 LLM 分层规划 + 开放词汇视觉检测 + 拉普拉斯方差滤波稳像 + 一套鲁棒执行逻辑组合起来，做成一个即插即用、能在 Go2 / B2 / H1-2 上直接跑的目标导航框架。"
---

# LOVON

**LOVON: Legged Open-Vocabulary Object Navigator** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：08_Navigation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

开放世界里的「找东西并走过去」是一个长时程任务：既要能识别任意自然语言指定的目标（开放词汇），又要把「找 → 搜 → 接近 → 完成」拆成可执行的子动作（高层规划），还要扛住足式机器人行走时的画面抖动、盲区、目标暂时丢失等现实问题。LOVON 把 LLM 分层规划 + 开放词汇视觉检测 + 拉普拉斯方差滤波稳像 + 一套鲁棒执行逻辑组合起来，做成一个即插即用、能在 Go2 / B2 / H1-2 上直接跑的目标导航框架。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| LLM | Large Language Model | 大语言模型，负责把长任务拆成有序基础指令 |
| Open-Vocabulary Detection | — | 开放词汇检测，无需预定义类别即可按自然语言识别目标 |
| L2MM | Language to Motion Model | 把语言指令映射到运动控制目标的模块 |
| Laplacian Variance Filtering | — | 拉普拉斯方差滤波，用于剔除运动模糊帧、视觉稳像 |
| Long-Horizon Task | — | 长时程任务，需要多步规划与持续执行 |

## 为什么重要

- **开放世界导航**：用开放词汇检测摆脱固定类别限制，让人形能「按一句话去找任意东西」
- **长时程任务**：LLM 分层规划把复杂任务拆成可执行子步骤，支撑长程自主作业
- **行走视觉退化**：拉普拉斯方差滤波给「边走边看」的足式感知提供了简单有效的稳像思路
- **通用部署**：一套框架通吃四足/人形，降低不同本体上的工程迁移成本

## 解决什么问题

传统目标导航方法难以同时满足三件事：

1. **开放世界检测**：现实目标千变万化，预定义类别根本覆盖不全； 2. **高层任务规划**：「去厨房拿那个红杯子」要拆成一连串可执行的搜索与移动动作； 3. **现实鲁棒性**：足式机器人边走边看，画面**抖动/模糊**，目标进**盲区**或**暂时消失**是常态。

## 核心机制

1. **统一框架**：首次把 LLM 分层规划、开放词汇检测与足式运动控制整合到一个长时程目标导航系统中。
2. **现实鲁棒设计**：拉普拉斯方差滤波 + 盲区/目标丢失处理，专门针对足式行走的视觉退化场景。
3. **跨本体即插即用**：同一框架可直接迁移到四足与人形多种机器人，部署门槛低。

方法拆解（深读笔记小节）：LLM 分层任务规划；开放词汇视觉检测；语言到运动（L2MM）；鲁棒执行逻辑（针对现实退化）；跨本体即插即用。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 08_Navigation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html> |
| arXiv | <https://arxiv.org/abs/2507.06747> |
| 机构 | **香港科技大学（广州）HKUST(GZ) / 北京人形机器人创新中心 / 香港科技大学 HKUST** |
| 作者 | **Daojie Peng**, Jiahang Cao, Qiang Zhang, Jun Ma |
| 发表 | 2025-07-09 (arXiv) |
| 项目主页 | [daojiepeng.github.io/LOVON](https://daojiepeng.github.io/LOVON/) |
| 源码 | [github.com/DaojiePENG/LOVON](https://github.com/DaojiePENG/LOVON) |
| 笔记阅读日期 | 2026-06-24 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-08-navigation](../overview/paper-notebook-category-08-navigation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_lovon.md](../../sources/papers/humanoid_pnb_lovon.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html>
- 论文：<https://arxiv.org/abs/2507.06747>

## 推荐继续阅读

- [机器人论文阅读笔记：LOVON](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/LOVON__Legged_Open-Vocabulary_Object_Navigator/LOVON__Legged_Open-Vocabulary_Object_Navigator.html)
