---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2507.07095"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_go-to-zero.md
summary: "本文要把文本到动作（text-to-motion）推向零样本泛化的新阶段。为此提出一套高效动作标注机制：从网络规模人类动作视频中自主采集人类动作——用运动学回归（kinematic regression）从无标注视频中提取动作，并用先进视觉语言模型生成语义丰富的描述（caption）。据此构建 MotionMillion ——迄今最大的人类动作数据集（2000+ 小时、200 万条高质量动作序列）。还提出 MotionMillion-Eval ——最全面的零样本动作生成评测基准。作者把模型规模化到 7B 参数并在 MotionMillion-Eval 上验证：结果对域外（out-of-domain）与复杂组合（compositional）动作展现强泛化，是迈向零样本人类动作生成的重要一步。代码开源。"
---

# Go to Zero

**Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文要把文本到动作（text-to-motion）推向零样本泛化的新阶段。为此提出一套高效动作标注机制：从网络规模人类动作视频中自主采集人类动作——用运动学回归（kinematic regression）从无标注视频中提取动作，并用先进视觉语言模型生成语义丰富的描述（caption）。据此构建 MotionMillion ——迄今最大的人类动作数据集（2000+ 小时、200 万条高质量动作序列）。还提出 MotionMillion-Eval ——最全面的零样本动作生成评测基准。作者把模型规模化到 7B 参数并在 MotionMillion-Eval 上验证：结果对域外（out-of-domain）与复杂组合（compositional）动作展现强泛化，是迈向零样本人类动作生成的重要一步。代码开源。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Zero-shot | 零样本（未见文本/动作） |
| MotionMillion | 本文百万级动作数据集 |
| Kinematic Regression | 运动学回归（从视频提动作） |
| VLM Caption | 视觉语言模型生成描述 |
| MotionMillion-Eval | 零样本评测基准 |
| 7B | 70 亿参数模型 |

## 为什么重要

- **零样本是动作生成的下一个高地**，数据规模 + 模型规模是关键，对人形动作生成同样适用；
- **自动从视频造"动作 + 描述"**是规模化数据的范式（与 Scaling Large Motion Models、SCHUR 同向）；
- **零样本评测基准**对衡量真正泛化很重要；
- 大规模动作模型可作人形"语言→动作"的强先验。

## 解决什么问题

文本到动作缺**零样本泛化**： - 现有数据集**小**，难覆盖多样文本/动作； - 缺**自动标注**把网络视频转成"动作 + 描述"； - 缺**零样本评测基准**。

Go to Zero 要：**自动**造百万级数据、训大模型、建零样本基准，实现零样本动作生成。

## 核心机制

1. **高效自动标注机制**：运动学回归 + VLM 描述，从网络视频造数据；
2. **MotionMillion**：迄今最大（2000h+/200 万序列）；
3. **MotionMillion-Eval**：最全面零样本评测基准；
4. **7B 模型 + 强零样本泛化**：域外与复杂组合动作。

方法拆解（深读笔记小节）：高效自动标注（视频 → 动作 + 描述）；MotionMillion 数据集；7B 大模型 + MotionMillion-Eval；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data.html> |
| arXiv | <https://arxiv.org/abs/2507.07095> |
| 发表 | 2025 年 7 月 |
| 项目主页 | [vankouf.github.io/MotionMillion](https://vankouf.github.io/MotionMillion/) · [code](https://github.com/VankouF/MotionMillion-Codes) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_go-to-zero.md](../../sources/papers/humanoid_pnb_go-to-zero.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data.html>
- 论文：<https://arxiv.org/abs/2507.07095>

## 推荐继续阅读

- [机器人论文阅读笔记：Go to Zero](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data/Go_to_Zero__Towards_Zero-shot_Motion_Generation_with_Million-scale_Data.html)
