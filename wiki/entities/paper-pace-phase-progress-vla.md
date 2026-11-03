---
type: entity
tags:
  - paper
  - vla
  - long-horizon
  - credit-assignment
  - manipulation
status: complete
updated: 2026-09-27
arxiv: "2608.15026"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
sources:
  - ../../sources/papers/pace-phase-progress-vla_arxiv_2608_15026.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md
summary: "PACE（VLA 长程信用）（arXiv:2608.15026）：融合局部视觉–动作变化识别阶段与进度，修正剩余代价并为每步生成信用；先保护预训练再高信用增强推理。"
---

# PACE（VLA 长程信用）（arXiv:2608.15026）

**PACE（VLA 长程信用）**（*PACE: Phase-Progress-Aware Credit for Long-Horizon Embodied Manipulation*，[arXiv:2608.15026](https://arxiv.org/abs/2608.15026)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) **长程记忆** 段。

## 一句话定义

**融合局部视觉–动作变化识别阶段与进度，修正剩余代价并为每步生成信用；先保护预训练再高信用增强推理。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

## 为什么重要

- 长任务仅末端成功信号无法告诉机器人哪步做对；PACE 做阶段–进度感知信用。
- 策展机构：大连理工大学；吉林大学
- 开源结论：**待核实**（步骤 2.5，2026-09-27）。

## 流程总览

```mermaid
flowchart TD
    A["后训练轨迹 + 多任务数据"] --> B["GLC-Critic 局部时序编码<br/>窗口 Transformer + 相邻帧运动差分分支"]
    B --> C["阶段头 z_t + 阶段内进度头 p_t"]
    B --> D["单帧分布式基础 critic<br/>剩余代价期望 C_raw"]
    C --> E["修正量 Δ_t（轻量 MLP）"]
    D --> F["修正后剩余代价 C_hat = C_raw + λΔ_t"]
    E --> F
    F --> G["步级信用分数 s_t"]
    G --> H["按任务分位阈值重标注信用"]
    H --> I{"PPD 路由：平均 rollout 成功率<br/>约 50% 及以下？"}
    I -->|"是"| J["PR-CFG：仅正信用条件<br/>推理时正向残差引导"]
    I -->|"否"| K["FACD：正 / 负信用条件<br/>条件以 0.3 概率丢弃"]
    J --> L["flow-matching VLA 策略更新"]
    K --> L
    L --> M["新一轮 rollout 采集"]
    M --> A
    L --> N["仿真与真机部署"]
```

图依据：论文 GLC-Critic / PPD 两节、Fig. 2、Fig. 3。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.15026](https://arxiv.org/abs/2608.15026) |
| **开源** | **待核实** |
| **文内评测** | LIBERO-Long；AgileX PiPER 双臂实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LIBERO-Long；AgileX PiPER 双臂实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md) | 同批 14 篇横向索引；本文属 **长程记忆** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**PACE（VLA 长程信用） 适合作为本期「长程记忆」路线的快速索引页。**

1. 核心贡献：融合局部视觉–动作变化识别阶段与进度，修正剩余代价并为每步生成信用；先保护预训练再高信用增强推理。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part4.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- [arXiv:2608.15026](https://arxiv.org/abs/2608.15026)
- [arXiv PDF](https://arxiv.org/pdf/2608.15026)：已通读方法部分（实验与附录以略读为准）：GLC-Critic、PPD（PR-CFG / FACD）两节及 Fig. 2/3、训练设定段

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.15026)
