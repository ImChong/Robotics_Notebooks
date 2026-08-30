---
type: entity
tags:
  - paper
  - mobile-manipulation
  - quadruped
  - action-chunking
  - vla
  - hunan
status: complete
updated: 2026-08-30
arxiv: "2608.22296"
related:
  - ../tasks/loco-manipulation.md
  - ../methods/action-chunking.md
  - ../methods/vla.md
  - ../tasks/teleoperation.md
  - ../overview/glancewam-vla-crew-10-papers-technology-map.md
sources:
  - ../../sources/papers/tonav_arxiv_2608_22296.md
  - ../../sources/sites/tonav-haochen611.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_glancewam_vla_crew_2026-08-30.md
summary: "TONAV（arXiv:2608.22296，湖南大学）：任务导向导航 + 位置–速度动作块，填补四足铰接物体「到达」与「稳定接触」空档；学习代码 Coming Soon。"
---

# TONAV：导航从一开始就服务于接触

**TONAV**（*Task-Oriented Navigation and Action-Velocity Chunk Learning for Articulated Object Quadrupedal Mobile Manipulation*，[arXiv:2608.22296](https://arxiv.org/abs/2608.22296)，[项目页](https://haochen611.github.io/TONAV)）由 **湖南大学（Hunan University）** 提出：用视觉语言推理把底座持续调到操作就绪位姿，再用位置–速度动作块在铰接物体上保持稳定接触。

## 一句话定义

**移动操作的关键不是「导航后再操作」，而是让导航从一开始服务于接触任务——到达可操作构型，并用速度监督稳住持续接触。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TONAV | Task-Oriented Navigation and Action-Velocity | 本文框架 |
| P–V | Position–Velocity | 耦合遥操作与动作块监督 |
| DP | Diffusion Policy | 项目页操作对照 |
| ACT | Action Chunking Transformer | 项目页操作对照 |
| LLM | Large Language Model | 导航子目标分解（Qwen-3.7-Max 等） |

## 为什么重要

- **空档真实存在：** 「靠近目标」常留下够得到却使不上力的构型。
- **示范动力学被丢掉：** 只记位置的遥操作会放大跟踪滞后与抖动。
- **真机铰接物体：** 关抽屉、马桶盖、台灯——接触必须连续。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 湖南大学（Hunan University） |
| **任务** | 四足底座 + 臂，铰接物体移动操作 |
| **开源** | **部分开源 / 待发布** — 学习代码 Coming Soon；`haochen611/TONAV` 仅为 Pages |

## 核心原理（方法）

三段：

1. **位置–速度耦合遥操作** 采集平滑、时间一致的示范。
2. **任务导向导航** 用视觉语言推理拆子目标，并持续 refinement 底座到操作就绪位姿。
3. **动作–速度块** 联合建模关节位置及其时间变化，速度监督改善持续接触。

### 流程总览

```mermaid
flowchart LR
  Tele[P-V 遥操作示范] --> Nav[任务导向导航\nVLM 子目标]
  Nav --> Ready[操作就绪位姿]
  Ready --> Chunk[动作-速度块]
  Chunk --> Contact[持续接触交互]
```

## 工程实践

| 项 | 建议 |
|----|------|
| **源码运行时序图** | **不适用**（学习代码 Coming Soon；Pages 仓无可运行训练入口） |
| 采集 | 不要只录位置轨迹；P–V 是后续 chunk 监督的前提 |
| 导航停止条件 | 停在「可操作」而不是「距离阈值」 |
| 对照 | 同一次导航到达后换 DP / ACT，才能隔离操作头 |

## 实验与评测

项目页以真机视频与对照为主，公开文字未给统一百分比总表。作者报告：任务导向导航与完整移动操作成功率高于对照，并缓解跟踪滞后、抖动与接触不稳。对照包括 TONAV / DP / ACT、有无 P–V、以及 Doubao-Seed-2.1-Pro vs Qwen-3.7-Max 的导航消融。

## 结论

**四足移动操作要先把「导航停止条件」改成操作就绪，再用速度监督补上接触段的动力学。**

1. **距离阈值是假完成** — 够得到不等于能关抽屉。
2. **P–V 示范不是锦上添花** — 位置-only 会把滞后写进数据集。
3. **LLM 只负责任务分解** — 底座 refinement 仍要闭环，不能一次规划走完。
4. **操作头对照应固定导航** — 项目页的单次到达多试操作是正确拆法。
5. **学习代码未发布** — 现阶段跟视频与遥操作说明，不要排训练复现。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [GlanceWAM](./paper-glancewam.md) | 桌面 WAM 异步想象；TONAV 是四足导航–接触衔接 |
| [DreamMimic](./paper-dreammimic.md) | 人形视觉全身蒸馏；TONAV 是四足铰接物体真机 |
| DP / ACT | 同到达构型下的操作头；TONAV 多了速度监督与导航闭环 |

## 局限与风险

- 论文 HTML 未给出可引用的统一成功率表，数字以项目页视频与后续 PDF 表为准。
- 学习代码未开，遥操作仓页上单列但独立 URL 未在 Pages 仓出现。
- LLM 导航依赖云端模型，现场延迟与稳定性未在摘要展开。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Action Chunking](../methods/action-chunking.md)
- [VLA](../methods/vla.md)
- [Teleoperation](../tasks/teleoperation.md)
- [48ms WAM / 编排 10 篇地图](../overview/glancewam-vla-crew-10-papers-technology-map.md)

## 参考来源

- [tonav_arxiv_2608_22296](../../sources/papers/tonav_arxiv_2608_22296.md)
- [TONAV 项目页](../../sources/sites/tonav-haochen611.md)
- [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_glancewam_vla_crew_2026-08-30.md)

## 推荐继续阅读

- [arXiv:2608.22296](https://arxiv.org/abs/2608.22296)
- [项目页](https://haochen611.github.io/TONAV)
