---
type: entity
tags:
  - paper
  - manipulation
  - force-control
  - audio
  - video-generation
  - zero-shot
status: complete
updated: 2026-09-20
arxiv: "2609.19137"
related:
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-fetch-my-beer.md
  - ../overview/constraint-control-11-papers-technology-map.md
sources:
  - ../../sources/papers/dreaming-sound-of-contact_arxiv_2609_19137.md
  - ../../sources/sites/dreaming-sound-of-contact.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md
summary: "Dreaming the Sound of Contact（arXiv:2609.19137）：Seedance 2.0 联合生成视频与音频；视频得运动轨迹，音频响度构造期望力曲线，1 kHz 阻抗+力调节闭环执行；四类任务 40 次试验力感知 90% vs 运动学 20%。"
---

# Dreaming the Sound of Contact（arXiv:2609.19137）

**Dreaming the Sound of Contact**（*Dreaming the Sound of Contact: Leveraging Video and Audio Generation for Zero-Shot Force-Aware Manipulation and Data Generation*，[arXiv:2609.19137](https://arxiv.org/abs/2609.19137)，[项目页](https://dreamingcontactsound.github.io/)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)（2026-09-20）。

## 一句话定义

**Seedance 2.0 联合生成视频与音频；视频得运动轨迹，音频响度构造期望力曲线，1 kHz 阻抗+力调节闭环执行；四类任务 40 次试验力感知 90% vs 运动学 20%。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DP | Diffusion Policy | 扩散策略模仿学习 |
| EE | End-Effector | 末端执行器 |
| SAM | Segment Anything Model | 分割与音频分离模块族 |
| Zero-Shot | Zero-Shot | 无任务特定训练直接执行 |

## 为什么重要

- 擦拭、剥离、按压等任务正确轨迹不等于正确接触力；视频生成只显示「去哪」不显示「多用力」。
- 开源结论：**待发布**（步骤 2.5，2026-09-20）。
- 与 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.19137](https://arxiv.org/abs/2609.19137) |
| **开源** | **待发布** |
| **要点** | MolmoPoint+SAM2+TAPIP3D 从视频得 EE 路径；SAM-Audio 分离接触声并映射响度→力曲线；Franka 力调节闭环。 |
| **文内指标** | 四类任务共 40 次：力感知流程 36/40 vs 运动学 8/40；Diffusion Policy 数据引擎 34/40（含力输入）vs 29/40。 |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。


## 实验与评测

- 四类任务共 40 次：力感知流程 36/40 vs 运动学 8/40；Diffusion Policy 数据引擎 34/40（含力输入）vs 29/40。
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [11 篇技术地图](../overview/constraint-control-11-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**接触音频可作为零样本力监督信号；音频形力曲线优于常数力阶跃，避免 overshoot 触发安全停机。**

1. 开源边界：**待发布** — 以项目页实际链接为准（入库日 2026-09-20）。
2. 核心机制：MolmoPoint+SAM2+TAPIP3D 从视频得 EE 路径；SAM-Audio 分离接触声并映射响度→力曲线；Franka 力调节闭环。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [paper-fetch-my-beer](./paper-fetch-my-beer.md)

## 参考来源

- [dreaming-sound-of-contact_arxiv_2609_19137.md](../../sources/papers/dreaming-sound-of-contact_arxiv_2609_19137.md)
- [wechat_embodied_station_11_papers_constraint_control_2026-09-20.md](../../sources/blogs/wechat_embodied_station_11_papers_constraint_control_2026-09-20.md)
- [arXiv:2609.19137](https://arxiv.org/abs/2609.19137)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.19137)
- [项目页](https://dreamingcontactsound.github.io/)

