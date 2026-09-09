---
type: entity
tags: ['paper', 'manipulation', 'relational', 'object-centric']
status: complete
updated: 2026-09-09
arxiv: "2609.08743"
venue: "ICRA 2026 Beyond Teleoperation Workshop"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
  - ../methods/action-chunking.md
  - ../methods/dmp.md
  - ../methods/diffusion-policy.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-3dway.md
sources:
  - ../../sources/papers/foci_policy_arxiv_2609_08743.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "FOCI Policy（arXiv:2609.08743）：从演示提取紧凑交互片段，用任务相关物体间相对 SE(3) 轨迹表示关系操作；RLBench + 真机 one-shot。"
---

# FOCI Policy

**FOCI Policy**（*Focus on Object-Centric Interactions for Relational Manipulation Policies*，[arXiv:2609.08743](https://arxiv.org/abs/2609.08743)，[项目/代码](https://fitz0401.github.io/foci-page/)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

关系操作的泛化难点在物体间约束而非末端轨迹——FOCI 用变点检测切交互片段，预测实体间相对 SE(3) 而非 gripper 绝对轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOCI | Focus on Object-Centric Interactions | 本文策略框架 |
| SE(3) | Special Euclidean group | 刚体变换群 |
| RLBench | RLBench | 仿真关系操作基准 |
| BC | Behavior Cloning | 演示学习 |

## 为什么重要

- 空间抽象降轨迹方差 + 时间抽象 isolate 关键交互段
- RLBench 每任务 1 demo；真机 cross-gripper /  clutter 组合 motion planning

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08743](https://arxiv.org/abs/2609.08743) |
| **开源** | **未开源** |
| **项目/代码** | [https://fitz0401.github.io/foci-page/](https://fitz0401.github.io/foci-page/) |

## 核心原理

- 空间抽象降轨迹方差 + 时间抽象 isolate 关键交互段
- RLBench 每任务 1 demo；真机 cross-gripper /  clutter 组合 motion planning
- KU Leuven；项目页有视频，无 GitHub URL

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 与其他工作对比

> 下表只做**定位对照**，不做跨设定横比：本页 Highlights 来自公众号归纳 + 项目页摘要（见参考来源），未逐条核对原文实验表，与下列各页不共享同一评测协议。

| 对照 | 差异读法 |
|------|----------|
| **末端绝对轨迹 BC**（本文要替代的默认做法） | 差别在**预测什么**：预测 gripper 在世界系下的绝对轨迹，物体一挪、夹爪一换，轨迹分布就变；FOCI 预测任务相关物体**之间的相对 SE(3)**，把这部分方差从学习目标里去掉——这正是其 RLBench 每任务 1 demo、真机 cross-gripper 泛化的机制来源 |
| [3DWay](./paper-3dway.md) | 同批盘点里的另一条空间中间表示，**参考系相反**：3DWay 给绝对 3D 路标（依赖外参标定准），FOCI 给物体间相对变换（对全局位姿漂移更宽容，但要求物体检测/位姿估计可靠） |
| [Action Chunking](../methods/action-chunking.md) | 同为时间抽象，但切法不同：action chunking 按**固定长度**切动作段，FOCI 用变点检测按**交互事件**切，段长随任务变；后者段边界有语义，代价是依赖变点检测的准确性 |
| [DMP](../methods/dmp.md) | 经典对照：DMP 也用相对目标坐标系做泛化，靠动力学系统保形；FOCI 用学习的相对 SE(3) 预测，表达力更强但没有 DMP 的收敛性保证 |
| [Diffusion Policy](../methods/diffusion-policy.md) | 常见 BC 基线；FOCI 改的是**动作空间的参考系与分段**，扩散策略改的是动作分布的建模方式——两者正交，可叠加 |
| [接触密集操作](../concepts/contact-rich-manipulation.md) | 边界提醒：相对 SE(3) 描述的是**几何关系**，插销/拧紧这类靠力反馈判成败的环节不在其表达范围内 |

## 结论

**FOCI Policy 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 未开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)
- [3DWay](./paper-3dway.md) — 参考系相反的空间中间表示
- [Action Chunking](../methods/action-chunking.md) / [DMP](../methods/dmp.md) — 时间抽象与相对坐标系的既有做法
- [Diffusion Policy](../methods/diffusion-policy.md) — 正交的动作分布建模
- [接触密集操作](../concepts/contact-rich-manipulation.md) — 几何关系表达不到的那部分

## 参考来源

- [foci_policy_arxiv_2609_08743.md](../../sources/papers/foci_policy_arxiv_2609_08743.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08743](https://arxiv.org/abs/2609.08743)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08743)
- [项目/代码](https://fitz0401.github.io/foci-page/)
