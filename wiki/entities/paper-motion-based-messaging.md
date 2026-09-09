---
type: entity
tags: ['paper', 'multi-robot', 'communication', 'steganography']
status: complete
updated: 2026-09-09
arxiv: "2609.08920"
venue: "arXiv 2026"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/motion_based_messaging_arxiv_2609_08920.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "Motion-based messaging（arXiv:2609.08920）：在预训练策略动作噪声中编码短消息，远程传感（视频/mocap）可解码；真机 50 Hz 100% 恢复、0.67 bit/s。"
---

# Motion-based messaging

**Motion-based messaging**（*Remotely Detectable Keyed Communication through Motion*，[arXiv:2609.08920](https://arxiv.org/abs/2609.08920)，[项目/代码](https://sites.google.com/view/motionbasedmessaging/home)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

无线不可用时，机器人运动本身能否成为可远程观测的通信信道——本文把任意短消息写进策略动作噪声而不显著伤害任务表现。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HRI | Human-Robot Interaction | 多机/人机交互场景 |
| VMAS | Vectorized Multi-Agent Simulator | 仿真基准之一 |
| MuJoCo | Multi-Joint dynamics with Contact | 物理仿真后端 |
| bps | bits per second | 通信速率 |

## 为什么重要

- Discovery / Reacher / Lunar Lander / Football 四环境验证
- 真机 RoboMaster 多机：路径 10% 处即可解码目标意图

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08920](https://arxiv.org/abs/2609.08920) |
| **开源** | **部分开源** |
| **项目/代码** | [https://sites.google.com/view/motionbasedmessaging/home](https://sites.google.com/view/motionbasedmessaging/home) |

## 核心原理

- Discovery / Reacher / Lunar Lander / Football 四环境验证
- 真机 RoboMaster 多机：路径 10% 处即可解码目标意图
- 50 Hz 控制下 100% 消息恢复，0.67 bit/s

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**Motion-based messaging 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 部分开源 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [motion_based_messaging_arxiv_2609_08920.md](../../sources/papers/motion_based_messaging_arxiv_2609_08920.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08920](https://arxiv.org/abs/2609.08920)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08920)
- [项目/代码](https://sites.google.com/view/motionbasedmessaging/home)
