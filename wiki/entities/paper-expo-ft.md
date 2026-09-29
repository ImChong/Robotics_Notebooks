---
type: entity
tags:
  - paper
  - vla
  - reinforcement-learning
  - fine-tuning
  - stanford
status: complete
updated: 2026-09-29
arxiv: "2605.25477"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md
sources:
  - ../../sources/papers/expo_ft_arxiv_2605_25477.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
summary: "EXPO-FT（arXiv:2605.25477，Stanford）：样本高效 VLA RL 微调；chunk 采样 + edit + Q 选择 + 回灌。"
---

# EXPO-FT

**EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models**（arXiv:[2605.25477](https://arxiv.org/abs/2605.25477)）— **斯坦福大学（Stanford）**。多模空间 [2026.08.17–08.23 周报](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md) 策展条目；细节以 arXiv 为准。

## 一句话定义

样本高效 VLA RL 微调；chunk 采样 + edit + Q 选择 + 回灌。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习后训练或微调 |
| TTA | Test-Time Augmentation / Adaptation | 测试时增强或适配 |
| SR | Success Rate | 任务成功率 |
| LIBERO | LIBERO Benchmark | 常见操作仿真基准套件 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 斯坦福大学（Stanford） |
| **评测** | 8 个单臂真机任务 |
| **开源** | 待发布（项目页无 GitHub，2026-09-29） |

## 为什么重要

- 纳入 [一周 VLA 趋势（2026.08.17 第一篇）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md) 横切面索引。
- 与 [VLA](../methods/vla.md) 方法页及同周其他 **16/16 独立 canonical 节点** 交叉对照。

## 结论

**EXPO-FT 在本库中作为 arXiv:2605.25477 的 canonical 详情节点；部署与复现前请对照原文 PDF/HTML 与项目页。**

1. **canonical 唯一性** — 全库仅此一页绑定 arXiv:2605.25477。
2. **读法** — 先读公众号策展摘要，再读 arXiv 方法与实验节。
3. **开源** — 待发布（项目页无 GitHub，2026-09-29）。
4. **与 Real-Time EXPO-FT 区分** — 本文 **2605.25477** 为 CoRL 样本高效微调基线；延迟感知续作 **2609.18207** 见 [Real-Time EXPO-FT](./paper-real-time-expo-ft.md)。

## 关联页面

- [VLA](../methods/vla.md)
- [Real-Time EXPO-FT](./paper-real-time-expo-ft.md)
- [Manipulation](../tasks/manipulation.md)
- [一周 VLA 趋势地图（2026.08.17）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md)

## 参考来源

- [expo_ft_arxiv_2605_25477.md](../../sources/papers/expo_ft_arxiv_2605_25477.md)
- [多模空间周报归档](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)
- arXiv：<https://arxiv.org/abs/2605.25477>
- 项目页：<https://pd-perry.github.io/expo-ft>

## 推荐继续阅读

- [arXiv 摘要页](https://arxiv.org/abs/2605.25477)
