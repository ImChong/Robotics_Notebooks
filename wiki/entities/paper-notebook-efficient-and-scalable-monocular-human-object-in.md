---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2512.00960"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_efficient-and-scalable-monocular-human-object-in.md
summary: "机器人要学操作离不开大规模、多样化的「人-物交互（HOI）」数据，但高精度动捕系统又贵又受限、采不到户外运动 / 工业作业这类真实场景。本文主张直接从普通单目互联网视频里抠出 4D HOI 数据：用稀疏接触标注把昂贵的逐帧密集标注降到「平均 6.7 个点 / 约 10 分钟一条」，再用 InterPoint 多模态预测器 + 4DHOISolver 两阶段优化把人、物、接触对齐成时空连贯且物理合理的轨迹，产出 Open4DHOI 数据集，并用 RL 动作模仿证明重建质量足以驱动仿真智能体复现交互动作。"
---

# Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction

**Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

机器人要学操作离不开大规模、多样化的「人-物交互（HOI）」数据，但高精度动捕系统又贵又受限、采不到户外运动 / 工业作业这类真实场景。本文主张直接从普通单目互联网视频里抠出 4D HOI 数据：用稀疏接触标注把昂贵的逐帧密集标注降到「平均 6.7 个点 / 约 10 分钟一条」，再用 InterPoint 多模态预测器 + 4DHOISolver 两阶段优化把人、物、接触对齐成时空连贯且物理合理的轨迹，产出 Open4DHOI 数据集，并用 RL 动作模仿证明重建质量足以驱动仿真智能体复现交互动作。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| HOI | Human-Object Interaction，人-物交互 |
| 4D | 3D 几何 + 时间维，即随时间变化的三维人 / 物状态 |
| Sparse Contact Annotation | 稀疏接触标注：只在关键接触点打少量标签，而非逐帧密集标注 |
| InterPoint | 本文的多模态接触点预测器 |
| 4DHOISolver | 本文的两阶段优化求解器（几何对齐 → 梯度精修） |
| IK | Inverse Kinematics，逆运动学 |
| MPJPE | Mean Per-Joint Position Error，平均关节位置误差 |
| IoU | Intersection over Union，投影掩码重合度 |

## 为什么重要

- **低成本扩交互数据**：人形灵巧操作 / loco-manipulation 极缺「人怎么用手与物体交互」的真实数据；本文从普通视频里抠 4D HOI、把标注成本降到分钟级，是搭「交互数据飞轮」的一条可扩展路径，与本模块 WHOLE、EmbodMocap 的「从野外视频还原具身数据」思路同源；
- **接触是第一性约束**：用接触 / 碰撞 / 掩码损失把重建约束成物理合理，再用接触引导奖励驱动 RL 模仿——这条「接触约束贯穿重建与控制」的链路，对人形动作跟踪 / 操作策略的奖励设计有直接借鉴；
- **重建即可仿真训练数据**：重建结果能直接喂给 RL 智能体在仿真里复现交互，呼应 Kimodo / OMG 等「生成 / 重建动作作为下游策略数据源」的范式；
- **手物耦合的脆弱性提醒**：手部误差会放大到物体位姿，提示人形操作里手部估计精度的关键性。

## 解决什么问题

机器人（尤其是灵巧操作 / 人形）要稳健地学会与物体交互，需要**大规模、多样、贴近真实**的 HOI 数据。但现状两难：

- **高精度动捕系统**（多相机 / 多传感器、受控环境）虽准，但**贵、受限于实验室**，物体种类少，户外运动、工业任务这类真实活动根本采不到； - **互联网单目视频**内容海量、场景多样，却**没人解决「如何准确且可扩展地从中抠出 4D 交互数据」**——人和物的相对位姿、接触关系、物理合理性都很难恢复，逐帧密集标注又贵到不可扩展。

## 核心机制

- 核心机制以深读笔记为准（见 [参考来源](#参考来源)）。

方法拆解（深读笔记小节）：稀疏接触标注范式（Sparse Contact Annotation）；InterPoint：多模态接触点预测器；4DHOISolver：两阶段优化求解；Open4DHOI 数据集；RL 动作模仿验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html> |
| arXiv | <https://arxiv.org/abs/2512.00960> |
| 机构 | 上海交通大学（SJTU）· SII · 复旦大学（FDU）· 北京交通大学（BJTU）· 浙江大学（ZJU） |
| 作者 | Boran Wen、Ye Lu、Sirui Wang、Keyan Wan、Jiahong Zhou、Junxuan Liang、Xinpeng Liu、Bang Xiao、Ruiyang Liu、Yong-Lu Li |
| 发表 | 2025年11月30日 |
| 项目主页 | [wenboran2002.github.io/open4dhoi](https://wenboran2002.github.io/open4dhoi/) |
| 源码 | [wenboran2002/open4dhoi_code](https://github.com/wenboran2002/open4dhoi_code)（论文声明数据与代码将公开） |
| 笔记阅读日期 | 2026-06-20 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_efficient-and-scalable-monocular-human-object-in.md](../../sources/papers/humanoid_pnb_efficient-and-scalable-monocular-human-object-in.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html>
- 论文：<https://arxiv.org/abs/2512.00960>

## 推荐继续阅读

- [机器人论文阅读笔记：Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction/Efficient_and_Scalable_Monocular_Human-Object_Interaction_Motion_Reconstruction.html)
