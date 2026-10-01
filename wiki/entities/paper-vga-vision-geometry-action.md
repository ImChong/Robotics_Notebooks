---
type: entity
tags:
  - paper
  - vla
  - 3d
  - geometry
  - manipulation
status: complete
updated: 2026-10-01
arxiv: "2604.12908"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/vga-vision-geometry-action_arxiv_2604_12908.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "VGA（arXiv:2604.12908）：用语义/视频预训练骨干难保 3D 精度；改用学过三维结构的骨干并联合学动作与物体 3D 属性，OOD 视角抓取更稳。"
---

# VGA（arXiv:2604.12908）

**VGA**（*Robotic Manipulation is Vision-to-Geometry Mapping: Vision-Geometry Backbones over Language and Video Models*，[arXiv:2604.12908](https://arxiv.org/abs/2604.12908)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **空间感知** 段。

## 一句话定义

**用语义/视频预训练骨干难保 3D 精度；改用学过三维结构的骨干并联合学动作与物体 3D 属性，OOD 视角抓取更稳。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- 操作需要精确空间关系；VGA 把 manipulation 收成 vision-to-geometry 映射（ACM MM 2026）。
- 策展机构：中山大学；广东省大数据分析与处理重点实验室；拓元智慧；美团龙猫；广东工业大学
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2604.12908](https://arxiv.org/abs/2604.12908) |
| **开源** | **待核实** |
| **文内评测** | LIBERO、RoboTwin2.0、LIBERO-Plus；Franka Panda 实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LIBERO、RoboTwin2.0、LIBERO-Plus；Franka Panda 实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **空间感知** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**VGA 适合作为本期「空间感知」路线的快速索引页。**

1. 核心贡献：用语义/视频预训练骨干难保 3D 精度；改用学过三维结构的骨干并联合学动作与物体 3D 属性，OOD 视角抓取更稳。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2604.12908](https://arxiv.org/abs/2604.12908)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2604.12908)
