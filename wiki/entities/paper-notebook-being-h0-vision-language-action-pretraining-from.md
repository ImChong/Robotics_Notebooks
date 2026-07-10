---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.15597"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_being-h0.md
summary: "Being-H0 是一个在大规模人类视频上训练的灵巧视觉-语言-动作模型（VLA）。现有 VLA 在高灵巧操作上吃力、对新场景泛化差，主因是依赖有 sim-to-real 差距的合成数据或缺规模与多样性的遥操作演示。为破数据瓶颈，本文把人手当作基础操作器（foundation manipulator），利用网络数据中丰富的灵巧性与可扩展性。方法核心是物理指令微调（physical instruction tuning）：结合大规模人类视频 VLA 预训练、3D 推理的物理空间对齐、以及面向机器人任务的后训练适配。还提出部件级运动 token 化（part-level motion tokenization），达毫米级重建精度以建模精确手部轨迹；并构建融合动捕、VR、RGB-only 视频的百万级运动指令数据集。实验显示 Being-H0 在手部动作生成与指令跟随上优异，随模型与数据规模良好扩展，并在真机操作上随物理指令微调见效。"
---

# Being-H0

**Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

Being-H0 是一个在大规模人类视频上训练的灵巧视觉-语言-动作模型（VLA）。现有 VLA 在高灵巧操作上吃力、对新场景泛化差，主因是依赖有 sim-to-real 差距的合成数据或缺规模与多样性的遥操作演示。为破数据瓶颈，本文把人手当作基础操作器（foundation manipulator），利用网络数据中丰富的灵巧性与可扩展性。方法核心是物理指令微调（physical instruction tuning）：结合大规模人类视频 VLA 预训练、3D 推理的物理空间对齐、以及面向机器人任务的后训练适配。还提出部件级运动 token 化（part-level motion tokenization），达毫米级重建精度以建模精确手部轨迹；并构建融合动捕、VR、RGB-only 视频的百万级运动指令数据集。实验显示 Being-H0 在手部动作生成与指令跟随上优异，随模型与数据规模良好扩展，并在真机操作上随物理指令微调见效。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| VLA | Vision-Language-Action 模型 |
| Physical Instruction Tuning | 物理指令微调（本文范式） |
| Motion Tokenization | 运动 token 化（部件级，毫米级精度） |
| Foundation Manipulator | 基础操作器（人手） |
| Physical Space Alignment | 物理空间对齐（3D 推理） |
| Post-training Adaptation | 后训练适配到机器人任务 |

## 为什么重要

- **"人手 = 基础操作器"**是把网络视频转成操作先验的有力视角；
- **物理空间对齐**让 2D 视频学到 3D 可执行动作，弥合 sim-to-real；
- **运动 token 化**把连续手轨离散化，便于 VLA 建模；
- 与 H-RDT、In-N-On 等共同壮大"人类视频 → 灵巧操作"路线。

## 解决什么问题

VLA 高灵巧操作难、泛化差： - 合成数据有 **sim-to-real 差距**； - 遥操作演示**缺规模与多样性**。

Being-H0 要：把**人手**当基础操作器，从**网络规模人类视频**学灵巧 VLA，破数据瓶颈。

## 核心机制

1. **人手作基础操作器**：从网络规模人类视频学灵巧 VLA；
2. **物理指令微调**：VLA 预训练 + 物理空间对齐 + 机器人适配；
3. **部件级运动 token 化**：毫米级精度建模手轨迹；
4. **百万级多源数据 + 规模化**：动捕/VR/RGB，随规模扩展、真机见效。

方法拆解（深读笔记小节）：物理指令微调（核心范式）；部件级运动 token 化（毫米级）；多源数据管线；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos.html> |
| arXiv | <https://arxiv.org/abs/2507.15597> |
| 作者 | Hao Luo、Yicheng Feng、Wanpeng Zhang、Sipeng Zheng、Haoqi Yuan、Qin Jin、Zongqing Lu 等（北大 / BAAI 等） |
| 发表 | 2025 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_being-h0.md](../../sources/papers/humanoid_pnb_being-h0.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos.html>
- 论文：<https://arxiv.org/abs/2507.15597>

## 推荐继续阅读

- [机器人论文阅读笔记：Being-H0](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos/Being-H0__Vision-Language-Action_Pretraining_from_Large-Scale_Human_Videos.html)
