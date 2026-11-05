---
type: entity
tags:
  - paper
  - vla
  - recovery
  - inference-time
  - manipulation
status: complete
updated: 2026-09-27
arxiv: "2608.14822"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md
sources:
  - ../../sources/papers/core-vla-counterfactual-realignment_arxiv_2608_14822.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md
summary: "CoRe（arXiv:2608.14822）：推理期反事实重对齐：跑偏后用合成画面在内部设想恢复路径，小步把状态接回再让原 VLA 继续，无需失败数据或重训。"
---

# CoRe（arXiv:2608.14822）

**CoRe**（*Imagining Recovery: Inference-Time Counterfactual Realignment for Vision-Language-Action Models*，[arXiv:2608.14822](https://arxiv.org/abs/2608.14822)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第四篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) **架构模块** 段。

## 一句话定义

**推理期反事实重对齐：跑偏后用合成画面在内部设想恢复路径，小步把状态接回再让原 VLA 继续，无需失败数据或重训。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| PRM | Process Reward Model | 过程/进度奖励模型 |
| OOD | Out-of-Distribution | 分布外场景或轨迹 |

## 为什么重要

- 目标/布局突变时 VLA 易偏离；CoRe 把试错留在想象空间，保留已完成进度。
- 策展机构：凯斯西储大学（美）
- 开源结论：**待核实**（步骤 2.5，2026-09-27）。

## 流程总览

```mermaid
flowchart TD
    A["冻结 VLA 正常执行<br/>记录 trace：观测、动作、关节构型 q、物体位姿 p"] --> B["在线 suffix OOD 检测<br/>动作专家 suffix 隐向量 + 马氏距离 s_t，阈值取自成功 rollout 校准"]
    B -->|"连续 n_consec 次超阈值"| C["相位感知恢复锚点<br/>在 trace 中找最近且保留已完成进度的帧；chunk 内按子步 h* 细化"]
    C --> D["基于合成观测的反事实续跑<br/>体素存储渲染 + 机械臂 FK 渲染 + 空洞修补"]
    D --> E["冻结 VLA 在目标指令 ℓ' 下闭环想象 rollout<br/>q_{t+1} = FK(q_t, a_t)"]
    E --> F["TCP 靠近物体阈值 d_prox 时终止<br/>最近的接触前帧作为 rejoin 状态 c*"]
    F --> G["基于探针的最小恢复集合<br/>强制集 F + 候选子集，虚拟 probe 对比全恢复参考，取首个 d_arm < ε 的 S*"]
    G --> H["物理重排：S* 中物体按自上而下 pick-and-place 复位<br/>验证，必要时报告人工介入"]
    H --> I["机械臂移到 q*，控制权交还冻结 VLA<br/>trace 截断至锚点"]
```

图依据：论文 Methodology 各小节（Online Suffix OOD Detection 至 Physical Rearrangement and Handoff）、Fig. 2、Fig. 3。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.14822](https://arxiv.org/abs/2608.14822) |
| **开源** | **待核实** |
| **文内评测** | LangSwitch、LIBERO-Long；UFACTORY xArm6 实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LangSwitch、LIBERO-Long；UFACTORY xArm6 实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md) | 同批 14 篇横向索引；本文属 **架构模块** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**CoRe 适合作为本期「架构模块」路线的快速索引页。**

1. 核心贡献：推理期反事实重对齐：跑偏后用合成画面在内部设想恢复路径，小步把状态接回再让原 VLA 继续，无需失败数据或重训。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第四篇技术地图](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第四篇）](../overview/vla-weekly-trends-2026-08-10-part4-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part4.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part4.md)
- [arXiv:2608.14822](https://arxiv.org/abs/2608.14822)
- [arXiv PDF](https://arxiv.org/pdf/2608.14822)：已通读方法部分（实验与附录以略读为准）：Introduction 概述、Methodology 全部小节（含 Fig. 2/3）

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.14822)
