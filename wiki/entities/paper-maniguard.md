---
type: entity
tags:
  - paper
  - vla
  - safety
  - benchmark
  - manipulation
status: complete
updated: 2026-09-29
arxiv: "2608.17386"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md
sources:
  - ../../sources/papers/maniguard_arxiv_2608_17386.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
summary: "MANIGUARD（arXiv:2608.17386）：规格化安全约束 + 监控 + 安全标签数据；成功≠安全。"
---

# MANIGUARD

**MANIGUARD: A Benchmark and Data Suite for Specification-Grounded Safety Evaluation and Improvement of Robotic Manipulation**（arXiv:[2608.17386](https://arxiv.org/abs/2608.17386)）— **西北大学（美）、斯坦福大学、威廉玛丽学院**。多模空间 [2026.08.17–08.23 周报](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md) 策展条目；细节以 arXiv 为准。

## 一句话定义

规格化安全约束 + 监控 + 安全标签数据；成功≠安全。

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
| **机构** | 西北大学（美）、斯坦福大学、威廉玛丽学院 |
| **评测** | ManiGuard-Bench（自建） |
| **开源** | 待核实（截至 2026-09-29） |

## 为什么重要

- 纳入 [一周 VLA 趋势（2026.08.17 第一篇）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md) 横切面索引。
- 与 [VLA](../methods/vla.md) 方法页及同周其他 **16/16 独立 canonical 节点** 交叉对照。

## 评测与指标

- **基准规模：** 6 个接触丰富的家居任务族 → **200** 个锁定基础任务（技能 × 约束分类）；每任务 1 个分布内 + 4 个单轴分布外扰动，共 **1,000** 个锁定场景。
- **判定方式：** 用 **LTLf 自动机监视器**基于物理谓词做运行时检查（不用学习型分类器或 LLM 裁判），仿真与 Franka 实机都跑。
- **数据：** 发布 **8,000** 条安全标注示教（每基础任务 40 条）。
- **主要发现：**（i）**6–21%** 的成功 rollout 违反规约；（ii）在本套件上微调后，安全完成率从近零升至 **7.5–29.8%**，投入且安全比例从 16–40% 升至 **51–72%**；（iii）扩大示教仍有 **21–42%** 的投入 rollout 违规，两个任务族所有策略安全成功率均 < 2%（数值摘自 arXiv 摘要，完整表格与基线设定以原文为准）。总计 **23,000+** 次 rollout。

## 与其他工作对比

| 维度 | MANIGUARD | 对照 |
|------|-----------|------|
| 安全判据 | 形式化规约（LTLf），与任务成功独立 | [LIBERO-VIFO](./paper-libero-vifo.md)：以是否服从未授权视觉提示衡量安全 |
| 改进手段 | 安全标注数据 + 安全感知微调 | [CrossSafe](./paper-crosssafe.md)：形态感知的潜空间安全过滤，不改策略数据 |
| 通用安全基准 | 操作任务专用 | [Safety-Gymnasium](./painode-323-safetygymnasium.md)：约束 RL 通用基准 |

评测基准选型见 [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)。

## 结论

**MANIGUARD 在本库中作为 arXiv:2608.17386 的 canonical 详情节点；部署与复现前请对照原文 PDF/HTML 与项目页。**

1. **canonical 唯一性** — 全库仅此一页绑定 arXiv:2608.17386。
2. **读法** — 先读公众号策展摘要，再读 arXiv 方法与实验节。
3. **开源** — 待核实（截至 2026-09-29）。
4. **安全/评测类**（若适用）— 勿把任务成功率等同于安全或授权跟随。

## 关联页面

- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [一周 VLA 趋势地图（2026.08.17）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md)

## 参考来源

- [maniguard_arxiv_2608_17386.md](../../sources/papers/maniguard_arxiv_2608_17386.md)
- [多模空间周报归档](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)
- arXiv：<https://arxiv.org/abs/2608.17386>
- 项目页：<https://nu-ideas-lab.github.io/ManiGuard>

## 推荐继续阅读

- [arXiv 摘要页](https://arxiv.org/abs/2608.17386)
