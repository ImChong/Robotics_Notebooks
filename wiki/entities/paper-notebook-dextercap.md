---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2601.05844"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_dextercap.md
summary: "精细的「在手」灵巧操作很难采集：手指挨得很近导致严重自遮挡，且动作幅度细微，传统光学动捕要么相机昂贵、要么后处理人工成本巨大。DexterCap 用密集的「字符编码」标记贴片（高对比棋盘格，每格带唯一双字符 ID）贴满手部各刚性区域，配合三级（marker → edge → tag）检测识别模型在自遮挡下稳定追踪，再用自动化重建流水线把 3D 标记拟合到 MANO 手模型与物体模型，恢复逐帧手参数与物体位姿/铰接状态——低成本、少人工地采到从简单基元到魔方等复杂铰接物的精细手-物交互，并发布 DexterHand 数据集与代码。"
---

# DexterCap

**DexterCap: An Affordable and Automated System for Capturing Dexterous Hand-Object Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

精细的「在手」灵巧操作很难采集：手指挨得很近导致严重自遮挡，且动作幅度细微，传统光学动捕要么相机昂贵、要么后处理人工成本巨大。DexterCap 用密集的「字符编码」标记贴片（高对比棋盘格，每格带唯一双字符 ID）贴满手部各刚性区域，配合三级（marker → edge → tag）检测识别模型在自遮挡下稳定追踪，再用自动化重建流水线把 3D 标记拟合到 MANO 手模型与物体模型，恢复逐帧手参数与物体位姿/铰接状态——低成本、少人工地采到从简单基元到魔方等复杂铰接物的精细手-物交互，并发布 DexterHand 数据集与代码。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| DexterCap | Dexterous Capture | 本文的低成本光学动捕系统 |
| DexterHand | — | 本文配套发布的精细手-物交互数据集 |
| MANO | hand Model with Articulated and Non-rigid defOrmations | 业界标准的可微参数化手模型 |
| Marker Patch | 标记贴片 | 贴在手部刚性区域的高对比棋盘格标记 |
| Character-Coded | 字符编码 | 每个白格内含唯一双字符 ID，用于自动标号 |
| Self-Occlusion | 自遮挡 | 手指相互遮挡，是手-物动捕的核心难点 |

## 为什么重要

- **为灵巧操作攒数据**：精确的手-物轨迹是 dexterous manipulation / 模仿学习的稀缺燃料，本系统把采集成本与人工大幅压低。
- **抗自遮挡的标记设计可迁移**：字符编码 + 三级检测的思路，可用于其它密集、易遮挡的标记动捕场景（如脚、面部、柔性物）。
- **MANO 输出对接生态**：逐帧 MANO 参数能直接喂给手部重定向 / 仿真，衔接现有手-物交互工具链。
- **限制**：仍需贴片（对真实「裸手」野外采集不适用）；极端遮挡或贴片磨损会影响识别；物体需有可拟合的模型。

## 解决什么问题

1. **手-物精细交互采集难**：手指间距小 → **严重自遮挡**；在手操作动作**细微**，普通视觉重建容易丢失指节位姿。 2. **传统光学动捕成本高**：高端商业系统相机昂贵；而且标记**自动标号（auto-labeling）失败率高**，需要大量人工逐帧修正。 3. **缺乏高质量精细数据**：下游灵巧操作学习（dexterous manipulation）渴求大规模、精确的手-物交互数据，但采集管线缺位。

**目标**：用**廉价硬件 + 自动化流水线**，稳健采集严重自遮挡下的灵巧手-物交互，并开源数据与代码。

## 核心机制

1. **字符编码密集标记**：每格唯一双字符 ID（324 标签），在严重自遮挡下也能可靠区分与自动标号，解决传统 auto-labeling 易错的痛点。
2. **三级检测识别模型**：marker → edge → tag 级联，鲁棒提取部分可见标记。
3. **自动化重建流水线**：3D 标记 → MANO 手 + 物体模型拟合，逐帧恢复手参数与物体位姿/铰接，少人工。
4. **低成本硬件**：同步工业灰度相机即可，显著降低采集门槛。
5. **DexterHand 数据集 + 开源**：覆盖从基元到魔方等复杂铰接物的精细手-物交互。

方法拆解（深读笔记小节）：字符编码标记贴片（Character-Coded Marker Patches）；采集硬件（低成本）；三级检测与识别（Marker → Edge → Tag）；自动化重建流水线（Automated Reconstruction）；DexterHand 数据集。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object.html> |
| arXiv | <https://arxiv.org/abs/2601.05844> |
| 发表 | 2026-01-09 (arXiv) |
| 项目主页 | [pku-mocca.github.io/Dextercap-Page](https://pku-mocca.github.io/Dextercap-Page/) |
| 源码 | [PKU-MoCCA/dextercap](https://github.com/PKU-MoCCA/dextercap) |
| 笔记阅读日期 | 2026-06-15 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_dextercap.md](../../sources/papers/humanoid_pnb_dextercap.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object.html>
- 论文：<https://arxiv.org/abs/2601.05844>

## 推荐继续阅读

- [机器人论文阅读笔记：DexterCap](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object/DexterCap__An_Affordable_and_Automated_System_for_Capturing_Dexterous_Hand-Object.html)
