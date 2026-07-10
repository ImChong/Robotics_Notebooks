---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.23523"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_h-rdt.md
summary: "机器人操作模仿学习面临大规模高质量机器人演示稀缺的根本难题。近期机器人基础模型常在跨本体机器人数据上预训练以扩规模，但不同本体的形态与动作空间差异大，统一训练难。H-RDT（Human to Robotics Diffusion Transformer）用人类操作数据增强机器人操作：核心洞察是带配对 3D 手姿标注的大规模第一视角人类操作视频蕴含丰富行为先验，能惠及机器人策略学习。采用两阶段：① 在大规模第一视角人类操作数据上预训练；② 用模块化动作编/解码器在机器人专属数据上做跨本体微调。模型是 2B 参数的扩散 Transformer，用流匹配建模复杂动作分布。仿真/真机较从零训练分别 +13.9% / +40.5%，超过 Pi0 与 RDT 基线。"
---

# H-RDT

**H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

机器人操作模仿学习面临大规模高质量机器人演示稀缺的根本难题。近期机器人基础模型常在跨本体机器人数据上预训练以扩规模，但不同本体的形态与动作空间差异大，统一训练难。H-RDT（Human to Robotics Diffusion Transformer）用人类操作数据增强机器人操作：核心洞察是带配对 3D 手姿标注的大规模第一视角人类操作视频蕴含丰富行为先验，能惠及机器人策略学习。采用两阶段：① 在大规模第一视角人类操作数据上预训练；② 用模块化动作编/解码器在机器人专属数据上做跨本体微调。模型是 2B 参数的扩散 Transformer，用流匹配建模复杂动作分布。仿真/真机较从零训练分别 +13.9% / +40.5%，超过 Pi0 与 RDT 基线。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| H-RDT | Human to Robotics Diffusion Transformer |
| Bimanual | 双臂 |
| 3D Hand Pose | 3D 手姿标注 |
| Cross-Embodiment | 跨本体 |
| Modular Encoder/Decoder | 模块化动作编/解码器 |
| Flow Matching | 流匹配 |

## 为什么重要

- **人类视频比跨本体机器人数据更易扩展**：用它作预训练先验是聪明的绕过数据稀缺之道；
- **模块化动作编/解码器**是跨本体微调的实用结构；
- **扩散 Transformer + 流匹配**是当前 VLA/操作模型的主流；
- 与 Being-H0、In-N-On 等"人类数据驱动操作"路线一致。

## 解决什么问题

机器人演示数据稀缺，跨本体统一训练难： - 直接跨本体机器人预训练受**形态/动作空间差异**限制； - 需要更**可扩展**的先验来源。

H-RDT 要：用**大规模人类操作视频（含 3D 手姿）**作行为先验，增强机器人双臂操作。

## 核心机制

1. **人类数据增强机器人操作**：人类视频 + 3D 手姿作行为先验；
2. **两阶段训练**：人类预训练 + 模块化跨本体微调；
3. **2B 扩散 Transformer + 流匹配**：建模复杂动作分布；
4. **显著提升**：仿真 +13.9%、真机 +40.5%，超 Pi0/RDT。

方法拆解（深读笔记小节）：洞察：人类视频 + 3D 手姿 = 行为先验；两阶段训练；架构：2B 扩散 Transformer + 流匹配；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2507.23523> |
| 作者 | Hongzhe Bi、Lingxuan Wu、Tianwei Lin、Hengkai Tan、Hang Su、Jun Zhu 等（清华 TSAIL / 地平线等） |
| 发表 | 2025 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_h-rdt.md](../../sources/papers/humanoid_pnb_h-rdt.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation.html>
- 论文：<https://arxiv.org/abs/2507.23523>

## 推荐继续阅读

- [机器人论文阅读笔记：H-RDT](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation/H-RDT__Human_Manipulation_Enhanced_Bimanual_Robotic_Manipulation.html)
