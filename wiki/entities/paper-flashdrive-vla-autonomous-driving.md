---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2608.12932"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/flashdrive-vla-autonomous-driving_arxiv_2608_12932.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "FlashDrive（arXiv:2608.12932）：智驾 VLA 受视觉重复编码、LLM 上下文重复、token 串行与 flow 多步去噪拖累；相邻帧共享计算、批量推理 token、按动作变化复用去噪步，加速决策且保持安全指标。…"
---

# FlashDrive（arXiv:2608.12932）

**FlashDrive**（[arXiv:2608.12932](https://arxiv.org/abs/2608.12932)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **性能提升/智驾** 段。

## 一句话定义

**智驾 VLA 受视觉重复编码、LLM 上下文重复、token 串行与 flow 多步去噪拖累；相邻帧共享计算、批量推理 token、按动作变化复用去噪步，加速决策且保持安全指标。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 智驾 VLA 受视觉重复编码、LLM 上下文重复、token 串行与 flow 多步去噪拖累；相邻帧共享计算、批量推理 token、按动作变化复用去噪步，加速决策且保持安全指标。
- 策展机构：UCSD、普林斯顿大学
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart LR
    V["多视角滑动窗口视频<br/>(4 帧 x 4 视角, 每步前进 1 帧)"] --> E["Encode<br/>仅编码最新帧"]
    E --> P["Prefill<br/>复用前序帧 KV cache<br/>(pre-RoPE 键 + streaming attention mask)"]
    P --> D["Decode<br/>CoC 推理 token<br/>DFlash 扩散草稿器并行起草 + 验证"]
    D --> A["Action<br/>flow matching 去噪<br/>8 步中 4 步复用缓存速度"]
    A --> T["轨迹输出"]
    F["streaming 微调<br/>冻结 VLM, 仅微调 action expert"] -.-> A
    Q["W4A8 量化 VLM 骨干<br/>(action expert 保持 BF16)"] -.-> P
    S["系统层: CUDA Graph + kernel fusion<br/>(每个阶段编译为 CUDA Graph)"] -.-> E
    S -.-> P
    S -.-> D
    S -.-> A
```

图依据：论文 §1、§3.1–§3.5 / Fig.1–Fig.5

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.12932](https://arxiv.org/abs/2608.12932) |
| **文内评测** | NVIDIA AV Dataset、AlpaSim |
| **开源** | **待核实**（2026-09-26） |
- **备注：** ez-lab.ai/projects/flashdrive

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** NVIDIA AV Dataset、AlpaSim
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**FlashDrive 适合作为本期「性能提升/智驾」路线的快速索引页。**

1. 核心贡献：智驾 VLA 受视觉重复编码、LLM 上下文重复、token 串行与 flow 多步去噪拖累；相邻帧共享计算、批量推理 token、按动作变化复用去噪步，加速决策且保持安全指标。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2608.12932](https://arxiv.org/abs/2608.12932)
- [arXiv PDF](https://arxiv.org/pdf/2608.12932)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3.1 Encode & Prefill、§3.2 Decode、§3.3 Action、§3.4 Quantization、§3.5 System Optimizations

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.12932)
