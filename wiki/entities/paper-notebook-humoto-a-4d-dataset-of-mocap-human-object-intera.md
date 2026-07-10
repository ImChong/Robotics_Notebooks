---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2504.10414"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humoto.md
summary: "人-物交互（HOI）数据「录得真、标得准」一直很难：真实交互里手/物频繁遮挡、物体几何难精确、动捕清洗成本高，导致已有数据集要么规模小、要么脚滑穿模。HUMOTO 用三步把质量做上去——(1) 让 LLM 按场景「编剧」 出有逻辑、有目的的完整任务序列（如做饭、野餐），保证动作是「为完成任务」而非随机摆拍；(2) 动捕 + 多相机 专门设计以应对遮挡；(3) 专业人工清洗与校验，最大限度消除脚滑与物体穿透。最终得到 735 段、约 7875 秒（30fps） 的高保真序列，含 63 件精确建模物体 与 72 个铰接人体部件（Mixamo 兼容绑定），可直接喂给动作生成、姿态估计和机器人/具身 AI 研究。"
---

# HUMOTO

**HUMOTO: A 4D Dataset of Mocap Human Object Interactions** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人-物交互（HOI）数据「录得真、标得准」一直很难：真实交互里手/物频繁遮挡、物体几何难精确、动捕清洗成本高，导致已有数据集要么规模小、要么脚滑穿模。HUMOTO 用三步把质量做上去——(1) 让 LLM 按场景「编剧」 出有逻辑、有目的的完整任务序列（如做饭、野餐），保证动作是「为完成任务」而非随机摆拍；(2) 动捕 + 多相机 专门设计以应对遮挡；(3) 专业人工清洗与校验，最大限度消除脚滑与物体穿透。最终得到 735 段、约 7875 秒（30fps） 的高保真序列，含 63 件精确建模物体 与 72 个铰接人体部件（Mixamo 兼容绑定），可直接喂给动作生成、姿态估计和机器人/具身 AI 研究。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| HOI | Human-Object Interaction，人-物交互 |
| Mocap | Motion Capture，动作捕捉 |
| 4D | 3D 几何 + 时间维度（随时间演化的人体姿态与物体网格） |
| Articulated Parts | 铰接部件（72 个可动人体关节部位） |
| Rigging | 骨骼绑定（此处 Mixamo 兼容，便于复用现成动作） |

## 为什么重要

- **人-物交互数据稀缺**是人形「loco-manipulation」落地的关键瓶颈，HUMOTO 提供了带物体网格与接触细节的成对人-物运动，可作**重定向 / 模仿学习**的高质量素材；
- **物理一致性（去脚滑/穿透）**对下游「人动作→机器人动作」迁移尤为重要，减少违反接触约束的坏样本；
- **LLM 编剧的「目的性任务序列」**思路与人形「语言指令→长时程操作」高度契合，可为任务级数据合成提供范式；
- 与本仓库 [WHOLE](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/WHOLE__World-Grounded_Hand-Object_Lifted_from_Egocentric_Videos/WHOLE__World-Grounded_Hand-Object_Lifted_from_Egocentric_Videos.html)、[Efficient and Scalable Monocular HOI](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html) 的「人-物交互重建」形成数据侧互补。

## 解决什么问题

现有人-物交互数据集普遍存在三类痛点：

- **交互质量差**：真实抓取/操作中手与物体互相遮挡，重建常出现脚滑、物体穿透； - **缺乏「目的性」**：很多序列是零散摆拍，没有「为完成某个任务」的逻辑连贯动作； - **物体几何粗糙**：物体网格不精确，难以支撑接触/物理层面的下游研究。

## 核心机制

1. **高保真 4D HOI 数据集**：735 段、63 精确物体、72 铰接人体部件，兼顾几何精度与时间连贯；
2. **场景驱动 LLM 编剧管线**：自动生成有逻辑、有目的的完整任务序列；
3. **抗遮挡采集 + 专业清洗**：显著减少脚滑与物体穿透，保证物理一致性；
4. **多任务基准**：面向动作生成、姿态估计、机器人仿真、2D 图像编辑等给出评测与对比。

方法拆解（深读笔记小节）：场景驱动的 LLM 编剧管线；抗遮挡的动捕 + 相机录制；专业清洗与校验；数据规模与基准；🧭 数据构建与下游用途（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions.html> |
| arXiv | <https://arxiv.org/abs/2504.10414> |
| 机构 | 德州大学奥斯汀分校（UT Austin） · Adobe Research |
| 作者 | Jiaxin Lu, Chun-Hao Paul Huang, Uttaran Bhattacharya, Qixing Huang, Yi Zhou |
| 发表 | 2025 年 4 月（arXiv v1：2025-04-14；v2：2025-10-15）· ICCV 2025 |
| 项目主页 | [jiaxin-lu.github.io/humoto](https://jiaxin-lu.github.io/humoto/) · [数据访问 adobe-research.github.io/humoto](https://adobe-research.github.io/humoto/) |
| 源码 | 🌟 [github.com/Jiaxin-Lu/humoto](https://github.com/Jiaxin-Lu/humoto) |
| 笔记阅读日期 | 2026-07-09 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humoto.md](../../sources/papers/humanoid_pnb_humoto.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions.html>
- 论文：<https://arxiv.org/abs/2504.10414>

## 推荐继续阅读

- [机器人论文阅读笔记：HUMOTO](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions/HUMOTO__A_4D_Dataset_of_Mocap_Human_Object_Interactions.html)
