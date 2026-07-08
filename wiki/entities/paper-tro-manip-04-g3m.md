---
type: entity
tags: [paper, manipulation, tro-manip-survey, video-pretraining, cross-embodiment, bit]
status: complete
updated: 2026-07-08
summary: "G3M：视频帧抽象为物体/视觉动作顶点图，图到图生成预训练条件化下游策略；20% 标注达全量性能。"
related:
  - ../overview/tro-manip-5-papers-technology-map.md
  - ../overview/tro-manip-category-03-video-pretraining.md
  - ../methods/mimic-video.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/tro_manip_survey_04_g3m.md
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# G3M

**G3M**（*Learning From Videos Through Graph-to-Graphs Generative Modeling for Robotic Manipulation*）收录于 [深蓝具身智能 · T-RO 2026 操作学习精选](https://mp.weixin.qq.com/s/nswA-jCGC3kr9iQjhRRuXQ) **第 04/5** 篇，归类为 **03 无标签视频预训练**。CVPR 2025 姊妹版题为 **GraphMimic**。

## 一句话定义

将人类操作视频帧抽象为 **物体顶点 + 视觉动作顶点** 的结构化图，通过 **图到图生成建模** 预训练未来图序列，再作为条件信号驱动下游机器人控制策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G3M | Graph-to-Graphs Generative Modeling | 本框架简称 |
| IL | Imitation Learning | 模仿学习 |
| LfD | Learning from Demonstrations | 从演示中学习 |

## 为什么重要

- 带精确关节力矩/动作标签的机器人演示 **成本极高**；互联网人类视频规模巨大但 **缺动作标签**。
- 图结构编码 **可迁移的空间关系**，而非本体特定像素或关节细节，利于 **跨本体迁移**。
- 仅用 **20% 带标签数据** 即可达到与全量标注相当的策略性能（文内/report 归纳）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 04/5 |
| 分组 | 03 无标签视频预训练 |
| 机构 | 北京理工大学（BIT） |
| 出处 | IEEE T-RO 2026 · CVPR 2025 GraphMimic |
| 论文/项目 | <https://doi.org/10.1109/TRO.2026.3658211> · [CVPR 2025 GraphMimic](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_GraphMimic_Graph-to-Graphs_Generative_Modeling_from_Videos_for_Policy_Learning_CVPR_2025_paper.html) |

## 核心机制（归纳）

### 1）图抽象

- **物体顶点**：操作对象状态。
- **视觉动作顶点**：手/效应器与物体的交互关系。
- **边**：元素间空间关系。

### 2）两阶段 pipeline

1. **预训练**：给定当前帧图，生成未来帧图 → 隐式学习操作物理/行为逻辑。
2. **下游策略**：以生成图为条件，映射为机器人动作。

### 3）实验亮点（策展）

- 仿真 **+17%+**、真机 **+23%+**（相对当时 SOTA；T-RO 期刊版部分指标略高）。
- 跨本体迁移 **+33%+**（期刊版 **+35%+**）。

## 常见误区

1. **G3M ≠ GraphMimic 重复节点**：同一研究线的期刊/会议版本；wiki 以 T-RO **G3M** 为主，GraphMimic 在此交叉引用。
2. 图预训练提供 **结构化条件**，不替代下游策略本身的 sim-to-real 与接触鲁棒性工程。

## 实验与评测

- 仿真、真机与跨本体迁移；长程任务表现见 T-RO 正文。

## 与其他页面的关系

- 技术地图：[tro-manip-5-papers-technology-map.md](../overview/tro-manip-5-papers-technology-map.md)
- [mimic-video](../methods/mimic-video.md)
- [Ego 9 篇技术地图](../overview/ego-9-papers-technology-map.md)

## 参考来源

- [tro_manip_survey_04_g3m.md](../../sources/papers/tro_manip_survey_04_g3m.md)
- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- T-RO：<https://doi.org/10.1109/TRO.2026.3658211>

## 推荐继续阅读

- [T-RO 5 篇技术地图](../overview/tro-manip-5-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)
