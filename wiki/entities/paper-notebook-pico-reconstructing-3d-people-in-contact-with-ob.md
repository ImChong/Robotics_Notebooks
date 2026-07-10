---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2504.17695"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_pico.md
summary: "从单张彩色图恢复 3D 人-物交互（HOI）很难：深度歧义、遮挡、物体形状外观差异巨大。以往工作需受控设置（已知物体形状/接触）且只处理有限物体类。PICO 想泛化到自然图像与新物体类，用两条思路：① 构建 PICO-db ——自然图像，唯一地配对身体与物体网格上的稠密 3D 接触：借视觉基础模型从数据库检索合适 3D 物体网格，再用一种每补丁仅 2 次点击的新方法把（DAMON 的）身体接触补丁投影到物体，以最小人工建立丰富的身-物接触对应；② 用 PICO-fit ——一种渲染-比较（render-and-compare）拟合方法，为 SMPL-X 身体推断接触、从 PICO-db 检索可能的 3D 物体网格与接触，并据接触迭代拟合身体与物体网格到图像证据。PICO 对许多现有方法无法处理的物体类别都работает（泛化好）。"
---

# PICO

**PICO: Reconstructing 3D People In Contact with Objects** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

从单张彩色图恢复 3D 人-物交互（HOI）很难：深度歧义、遮挡、物体形状外观差异巨大。以往工作需受控设置（已知物体形状/接触）且只处理有限物体类。PICO 想泛化到自然图像与新物体类，用两条思路：① 构建 PICO-db ——自然图像，唯一地配对身体与物体网格上的稠密 3D 接触：借视觉基础模型从数据库检索合适 3D 物体网格，再用一种每补丁仅 2 次点击的新方法把（DAMON 的）身体接触补丁投影到物体，以最小人工建立丰富的身-物接触对应；② 用 PICO-fit ——一种渲染-比较（render-and-compare）拟合方法，为 SMPL-X 身体推断接触、从 PICO-db 检索可能的 3D 物体网格与接触，并据接触迭代拟合身体与物体网格到图像证据。PICO 对许多现有方法无法处理的物体类别都работает（泛化好）。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| HOI | Human-Object Interaction，人-物交互 |
| PICO-db | 本文接触标注数据集 |
| PICO-fit | 渲染-比较拟合方法 |
| SMPL-X | 参数化人体模型 |
| Render-and-compare | 渲染-比较优化 |
| Contact Correspondence | 身-物接触对应 |

## 为什么重要

- **接触是人-物交互的核心线索**，呼应本仓多篇"接触为中心"的人形交互/操作工作；
- **身-物双侧接触标注**对学习抓取/操作的接触先验有价值；
- **VFM 检索 + 最小人工标注**是低成本造数据的范式；
- 人-物交互重建可为人形操作提供目标/接触监督。

## 解决什么问题

单图 3D 人-物交互重建难： - **深度歧义、遮挡、物体多样**； - 以往需**已知物体/接触**、只限少数类； - 缺**身-物双侧接触**标注。

PICO 要：泛化到**自然图像 + 新物体类**的 3D HOI 重建。

## 核心机制

1. **PICO-db**：自然图像 + 身-物双侧稠密 3D 接触（VFM 检索 + 2 点击投影）；
2. **PICO-fit**：基于接触的渲染-比较拟合；
3. **泛化新物体类**：无需预知物体几何/接触；
4. **大规模 HOI 理解**：从单图恢复交互。

方法拆解（深读笔记小节）：PICO-db：身-物双侧稠密接触数据；PICO-fit：渲染-比较拟合；泛化；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PICO__Reconstructing_3D_People_In_Contact_with_Objects/PICO__Reconstructing_3D_People_In_Contact_with_Objects.html> |
| arXiv | <https://arxiv.org/abs/2504.17695> |
| 作者 | Alpár Cseke、Shashank Tripathi、Sai Kumar Dwivedi、Michael J. Black、Dimitrios Tzionas（MPI / 图宾根等） |
| 发表 | 2025 年 4 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_pico.md](../../sources/papers/humanoid_pnb_pico.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PICO__Reconstructing_3D_People_In_Contact_with_Objects/PICO__Reconstructing_3D_People_In_Contact_with_Objects.html>
- 论文：<https://arxiv.org/abs/2504.17695>

## 推荐继续阅读

- [机器人论文阅读笔记：PICO](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/PICO__Reconstructing_3D_People_In_Contact_with_Objects/PICO__Reconstructing_3D_People_In_Contact_with_Objects.html)
