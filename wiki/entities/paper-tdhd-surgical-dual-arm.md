---
type: entity
tags:
  - paper
  - vla
  - medical
  - dual-arm
  - safety
status: complete
updated: 2026-09-19
arxiv: "2608.09125"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/tdhd_surgical_dual_arm_arxiv_2608_09125.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "双臂手术 VLA 的轨迹分歧视界决策：生成两份轻微扰动的动作计划，分歧扩大时提前停下重规划，减少固定长度执行累积偏差，提升针与组织操作可靠性。"
---

# TDHD（arXiv:2608.09125）

**TDHD**（[arXiv:2608.09125](https://arxiv.org/abs/2608.09125)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **架构模块** 段。

## 一句话定义

**双臂手术 VLA 的轨迹分歧视界决策：生成两份轻微扰动的动作计划，分歧扩大时提前停下重规划，减少固定长度执行累积偏差，提升针与组织操作可靠性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 双臂手术 VLA 的轨迹分歧视界决策：生成两份轻微扰动的动作计划，分歧扩大时提前停下重规划，减少固定长度执行累积偏差，提升针与组织操作可靠性。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.09125](https://arxiv.org/abs/2608.09125) |
| **开源** | **待核实** |
| **文内评测** | 实机 RM65-B 双臂 + dVRK 改造末端 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** 实机 RM65-B 双臂 + dVRK 改造末端
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [Hermite Curves VLA](./paper-hermite-curves-vla-trajectory-priors.md) | 同批同为治理「固定长度执行的累积偏差」，一测一训：TDHD 在**执行期**用两份轻微扰动计划的分歧当触发器提前停下重规划，Hermite 在**训练期**把动作引向平滑曲线。TDHD 不改训练目标，Hermite 不加运行时开销 |
| [Action Chunking](../methods/action-chunking.md) | TDHD 实质是给 chunk 配一个**提前终止判据**；该页讲「部署不必等于播放整段 chunk」，本文给的是判断该在哪一步截断的一种具体信号 |
| [WA-SpecDec](./paper-wa-specdec.md) | 同批同为「多算一份再决定用不用」，目标相反：WA-SpecDec 用猜测–校验**换速度**，TDHD 用双计划分歧**换可靠性**（宁可停下重规划）。安全敏感场景按后者读 |
| [双臂操作](../tasks/bimanual-manipulation.md) | 该页给双臂协同的一般约束；手术场景把误差容限压到针与组织的量级，是本文选「提前停」而非「事后纠」的直接原因，也是它难以直接搬到桌面抓放的原因 |

## 结论

**TDHD 适合作为本期「架构模块」路线的快速索引页。**

1. 核心贡献：双臂手术 VLA 的轨迹分歧视界决策：生成两份轻微扰动的动作计划，分歧扩大时提前停下重规划，减少固定长度执行累积偏差，提升针与组织操作可靠性。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.09125](https://arxiv.org/abs/2608.09125)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.09125)
