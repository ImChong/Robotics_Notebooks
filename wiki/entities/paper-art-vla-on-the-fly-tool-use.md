---
type: entity
tags:
  - paper
  - vla
status: complete
updated: 2026-09-26
arxiv: "2608.14047"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md
sources:
  - ../../sources/papers/art-vla-on-the-fly-tool-use_arxiv_2608_14047.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md
summary: "ART（arXiv:2608.14047）：为 VLA 配备视觉识别、操作判断等工具并学习何时/如何串联调用，减少直接在巨大连续动作空间盲搜；较少数据完成仿真与真机任务。CVPR 2026。…"
---

# ART（arXiv:2608.14047）

**ART**（[arXiv:2608.14047](https://arxiv.org/abs/2608.14047)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第三篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) **类 Agent** 段。

## 一句话定义

**为 VLA 配备视觉识别、操作判断等工具并学习何时/如何串联调用，减少直接在巨大连续动作空间盲搜；较少数据完成仿真与真机任务。CVPR 2026。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化（FIRE-VLA / Temporal GRPO 语境） |
| RL | Reinforcement Learning | 强化学习后训练 |

## 为什么重要

- 为 VLA 配备视觉识别、操作判断等工具并学习何时/如何串联调用，减少直接在巨大连续动作空间盲搜；较少数据完成仿真与真机任务。CVPR 2026。
- 策展机构：星尘智能、清华大学等
- 与 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md) 同批工作可横向对照。

## 流程总览

```mermaid
flowchart TD
    A["原始观测 õ_t<br/>图像 + 语言指令 + 本体状态<br/>（可能退化或视角/初始状态偏移）"] --> B["启用 LoRA 的 π-FAST<br/>每 H 步预测一次推理 token a_r 与工具使用 token a_t"]
    B --> C["工具使用动作：n 位二进制开关<br/>映射到词表末尾 N 个 token"]
    C --> D["激活外部工具并作用于接下来 H 步观测"]
    D --> E["视觉增强：去噪、低光增强、抖动校正、去模糊等 10 类"]
    D --> F["可供性增强：深度估计、目标检测"]
    D --> G["本体增强：相机旋转、缩放、机器人状态复位"]
    E --> H["增强后观测 o_t"]
    F --> H
    G --> H
    H --> I["不带 LoRA 的 VLA 骨干<br/>自回归生成 FAST 动作 token"]
    I --> J["动作 chunk"]
```

图依据：论文 §3.1–3.3、§4.1、Fig. 2（数据采集流程见 §4.2 / Fig. 3，此处未画）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.14047](https://arxiv.org/abs/2608.14047) |
| **文内评测** | LIBERO、LIBERO-Plus；实机 Astribot S1 |
| **开源** | **待核实**（2026-09-26） |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。

## 实验与评测

- **文内口径：** LIBERO、LIBERO-Plus；实机 Astribot S1
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [Temporal GRPO](./paper-temporal-grpo.md) | 同周「训练范式」线：阶段信用 vs 失败驱动 teacher（若适用） |
| [VLA 方法页](../methods/vla.md) | 本页为单篇索引；机制细节以原文为准 |

## 结论

**ART 适合作为本期「类 Agent」路线的快速索引页。**

1. 核心贡献：为 VLA 配备视觉识别、操作判断等工具并学习何时/如何串联调用，减少直接在巨大连续动作空间盲搜；较少数据完成仿真与真机任务。CVPR 2026。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第三篇技术地图](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第三篇）](../overview/vla-weekly-trends-2026-08-10-part3-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part3.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part3.md)
- [arXiv:2608.14047](https://arxiv.org/abs/2608.14047)
- [arXiv PDF](https://arxiv.org/pdf/2608.14047)：已通读方法部分（实验与附录以略读为准）：§3.1–3.3、§4.1–4.2、§5.1 训练设定

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.14047)
