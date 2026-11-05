---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2608.13924"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/bicpo-vla_arxiv_2608_13924.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "BICPO-VLA（arXiv:2608.13924）：新指令下 VLA 常因行为意图不清与状态漂移导致切换延迟；先据指令与进度确定应继续的行为，再拆主要走势与细节修正分阶段合成，并按真实切换位姿调整新动作。…"
---

# BICPO-VLA（arXiv:2608.13924）

**BICPO-VLA**（[arXiv:2608.13924](https://arxiv.org/abs/2608.13924)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **架构/异步** 段。

## 一句话定义

**新指令下 VLA 常因行为意图不清与状态漂移导致切换延迟；先据指令与进度确定应继续的行为，再拆主要走势与细节修正分阶段合成，并按真实切换位姿调整新动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 新指令下 VLA 常因行为意图不清与状态漂移导致切换延迟；先据指令与进度确定应继续的行为，再拆主要走势与细节修正分阶段合成，并按真实切换位姿调整新动作。
- 策展机构：北航、北理工、中科院自动化所等
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    A["请求时刻 r：观测 o_r + 指令 ℓ + 历史 H_r"] --> B["指令引导的视觉路由<br/>语言 cross-attention 过滤视觉 token"]
    B --> C["因果 Mamba 历史编码器 → 行为状态 z_r"]
    C --> D["相位条件解码器 + 检索的长程记忆 m_r<br/>→ 先验动作 μ_r"]
    D --> E["行为条件 b_r = (ℓ, z_r, m_r, μ_r)"]
    F["推理期间仍在执行的外发命令 U"] --> G["冻结动作状态模型 rollout 到 handoff 时刻<br/>得 h_bar 与 Δh，经 P_ψ 投影为 handoff 条件 g_k"]
    E --> H["Haar 分解：先验与动作 chunk 同坐标"]
    G --> H
    H --> I["第 1 阶段：flow 生成 scaffold 系数 C（逐对运动趋势）"]
    I --> J["第 2 阶段：条件于 C 的 flow 生成 residual 系数 D"]
    J --> K["Haar 逆变换精确重建动作 chunk"]
    L["离线 BICPO：同一 fiber 内不同噪声采样两候选<br/>按 handoff 跳变 + 运动趋势失配代价排序"] --> M["参考相对 Flow-DPO 微调<br/>冻结语义通路与参考专家，仅更新 P_ψ、W_g 与部分专家参数"]
    M -.-> J
```

图依据：论文 §3.1–3.5、Fig. 2。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.13924](https://arxiv.org/abs/2608.13924) |
| **文内评测** | CALVIN ABC→D、RoboTwin 2.0（10 困难任务）、LIBERO；实机 6 双臂任务 |
| **开源** | **待核实**（2026-09-26） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** CALVIN ABC→D、RoboTwin 2.0（10 困难任务）、LIBERO；实机 6 双臂任务
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**BICPO-VLA 适合作为本期「架构/异步」路线的快速索引页。**

1. 核心贡献：新指令下 VLA 常因行为意图不清与状态漂移导致切换延迟；先据指令与进度确定应继续的行为，再拆主要走势与细节修正分阶段合成，并按真实切换位姿调整新动作。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2608.13924](https://arxiv.org/abs/2608.13924)
- [arXiv PDF](https://arxiv.org/pdf/2608.13924)：已通读方法部分（实验与附录以略读为准）：§3.1–3.5（含 Fig. 2）

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.13924)
