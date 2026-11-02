---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2608.12308"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/dreamfly-aerial-vln_arxiv_2608_12308.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "DreamFly（arXiv:2608.12308）：因果记忆只回看已见画面；滚动 horizon 扩散规划多步但只执行第一步；独立机制判断到达与停车，提升熟悉/陌生环境导航与停准。…"
---

# DreamFly（arXiv:2608.12308）

**DreamFly**（[arXiv:2608.12308](https://arxiv.org/abs/2608.12308)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **长程记忆/导航** 段。

## 一句话定义

**因果记忆只回看已见画面；滚动 horizon 扩散规划多步但只执行第一步；独立机制判断到达与停车，提升熟悉/陌生环境导航与停准。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 因果记忆只回看已见画面；滚动 horizon 扩散规划多步但只执行第一步；独立机制判断到达与停车，提升熟悉/陌生环境导航与停准。
- 策展机构：西安工业大学
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    O["当前 RGB 观测 Ot"] --> Z["Projector(VisionEncoder(Ot))<br/>当前图像 token Zt"]
    I["导航指令 I"] --> R["冻结 CLIPSeg 稠密路由 + 冻结 OWLv2 区域路由<br/>按指令提取候选"]
    R --> MEM["因果对齐历史记忆 M<t<br/>16 个长期 slot (anchor + 可选 prototype)<br/>仅含 t 之前的观测"]
    Z --> CA["门控 cross-attention<br/>Zt 为 query, 记忆 slot 为 key/value"]
    MEM --> CA
    CA --> ZT["记忆条件化视觉 token"]
    ZT --> BB["Dream-VLA 双向扩散骨干 + 指令<br/>离散扩散联合去噪 K 个 action slot"]
    I --> BB
    BB --> CH["K 步动作块 (a0 ... aK-1)<br/>同时保留初始全 mask 的 logit 网格 H(0)"]
    CH --> LS["LiteStop<br/>由 H(0) 估计停止概率, 与阈值比较"]
    LS -->|"触发"| STOP["终止, 不执行动作"]
    LS -->|"未触发"| EX["仅执行首个动作 a0<br/>(a0 为 Stop 时同样终止), 其余丢弃"]
    EX --> NEXT["获得新观测 Ot+1, 重新规划"]
    NEXT -.-> O
    O -.->|"read-before-write: 决策后才写入记忆"| MEM
```

图依据：论文 §3.2–§3.5 / Fig.1、Fig.2

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.12308](https://arxiv.org/abs/2608.12308) |
| **文内评测** | OpenFly |
| **开源** | **待核实**（2026-09-26） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** OpenFly
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**DreamFly 适合作为本期「长程记忆/导航」路线的快速索引页。**

1. 核心贡献：因果记忆只回看已见画面；滚动 horizon 扩散规划多步但只执行第一步；独立机制判断到达与停车，提升熟悉/陌生环境导航与停准。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2608.12308](https://arxiv.org/abs/2608.12308)
- [arXiv PDF](https://arxiv.org/pdf/2608.12308)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3.1 问题定义、§3.2 框架总览、§3.3 因果对齐历史记忆、§3.4 Receding-Horizon 扩散规划、§3.5 LiteStop、§4.1 实验设定

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.12308)
