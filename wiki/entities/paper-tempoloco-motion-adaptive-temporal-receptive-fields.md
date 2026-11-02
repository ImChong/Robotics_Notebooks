---
type: entity
project_id: tempoloco-motion-adaptive-temporal-receptive-fields
tags:
  - paper
  - quadruped
  - locomotion
  - perceptive-locomotion
  - depth
status: complete
updated: 2026-10-14
arxiv: "2610.14753"
project: https://capability-aware.github.io/
related:
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../methods/pie-perceptive-locomotion.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/tempoloco_motion_adaptive_temporal_receptive_fields_arxiv_2610_14753.md
  - ../../sources/sites/tempoloco-motion-adaptive-temporal-receptive-fields.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "TempoLoco（arXiv:2610.14753）：从长深度历史学习互补时间视图，再按本体状态与地形动态融合，让视觉提前量适应运动能力。"
---

# TempoLoco（arXiv:2610.14753）

**TempoLoco**（*Beyond Fixed Visual History: Learning Motion-Adaptive Temporal Receptive Fields for Complex-Terrain Quadruped Locomotion*，[arXiv:2610.14753](https://arxiv.org/abs/2610.14753)，[项目页](https://capability-aware.github.io/)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「导航、足式与规划」组。

## 一句话定义

**从长深度历史学习互补时间视图，再按本体状态与地形动态融合，让视觉提前量适应运动能力。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TRF | Temporal Receptive Field | 时间感受野 |
| B2 | Unitree B2 | 四足平台 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

- 固定视觉历史无法随速度/地形变化调整提前量。
- 开源结论：**部分开源**（步骤 2.5，2026-10-14）。项目页链接 HaoZhangRobotics/capability_rl 与 capabilitylab（可达），与本文代码的对应关系待核实；页面另含「coming soon」。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    D["深度历史 FIFO（37 槽，每槽 18×32）"] --> S["Stride bank：M=4 个全局可学习连续步长<br/>按步长对 FIFO 线性插值取 K=8 帧"]
    S --> V["4 个时间视图 H(i)（各 8×18×32）"]
    V --> E["尺度共享视觉编码器 Eψ<br/>每个视图 → 128 维特征 z(i)"]
    P["本体历史 P_t（5 步，260 维）"] --> G
    E --> G["体态条件时间门控 Gη<br/>softmax 得到融合权重 α_t"]
    E --> F["加权融合 → z_v（128 维）"]
    G --> F
    F --> X["与 P_t 拼接 → 388 维 actor 输入"]
    P --> X
    X --> M["4 专家 locomotion MoE<br/>MLP 专家 + 动作专家门控"]
    M --> A["12 个腿关节目标 → PD 跟踪"]
    F -. critic 独立分支，融合后追加 33 维特权信息 .-> C["Critic"]
```

图依据：论文 §III-A/B/C/D 与 Fig. 2（actor 路径；actor 与 critic 各有独立 stride bank、编码器与门控，PPO 联合优化；训练期另有均匀融合 warm-up、多样性与利用率正则，未入图）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14753](https://arxiv.org/abs/2610.14753) |
| **开源** | **部分开源** |
| **要点** | 复杂地形仿真 + 带固定机械臂的 B2 真机。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

**部分开源** — 项目页链接 HaoZhangRobotics/capability_rl 与 capabilitylab（可达），与本文代码的对应关系待核实；页面另含「coming soon」。 待可运行代码公开后补图。

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**感知时间窗应随运动状态自适应，而非固定帧数。**

1. 开源边界：**部分开源** — 项目页链接 HaoZhangRobotics/capability_rl 与 capabilitylab（可达），与本文代码的对应关系待核实；页面另含「coming soon」。
2. 核心机制：复杂地形仿真 + 带固定机械臂的 B2 真机。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [stair-obstacle-perceptive-locomotion](../tasks/stair-obstacle-perceptive-locomotion.md)
- [pie-perceptive-locomotion](../methods/pie-perceptive-locomotion.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [tempoloco_motion_adaptive_temporal_receptive_fields_arxiv_2610_14753.md](../../sources/papers/tempoloco_motion_adaptive_temporal_receptive_fields_arxiv_2610_14753.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14753](https://arxiv.org/abs/2610.14753)
- [arXiv PDF](https://arxiv.org/pdf/2610.14753)：已通读方法部分（实验与附录以略读为准）：§I Introduction、§III Method（A–D）、§IV-A 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14753)
- [项目页](https://capability-aware.github.io/)
