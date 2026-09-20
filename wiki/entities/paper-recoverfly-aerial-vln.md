---
type: entity
tags:
  - paper
  - vln
  - aerial
  - rl
  - post-training
status: complete
updated: 2026-09-19
arxiv: "2608.09467"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/recoverfly_aerial_vln_arxiv_2608_09467.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "空中 VLN 失败感知 RL 后训练：token 级优化、反复学习未解失败案例，分阶段长尾训练与参考策略约束，提升闭环纠错与泛化。"
---

# RecoverFly（arXiv:2608.09467）

**RecoverFly**（[arXiv:2608.09467](https://arxiv.org/abs/2608.09467)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **训练范式** 段。

## 一句话定义

**空中 VLN 失败感知 RL 后训练：token 级优化、反复学习未解失败案例，分阶段长尾训练与参考策略约束，提升闭环纠错与泛化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 空中 VLN 失败感知 RL 后训练：token 级优化、反复学习未解失败案例，分阶段长尾训练与参考策略约束，提升闭环纠错与泛化。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.09467](https://arxiv.org/abs/2608.09467) |
| **开源** | **待核实** |
| **文内评测** | TravelUAV |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** TravelUAV
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [TEMPO](./paper-tempo.md) | 同批「训练范式」段的另一条 RL 后训练，组织方式不同：本文做 **token 级**优化并反复重练未解的失败案例（分阶段长尾课程 + 参考策略约束），TEMPO 走的是冻结 VLM、对 projection 与 action expert 分设 TD3 环双频更新。前者围绕**失败样本**做课程，后者围绕**模块分工**做解耦 |
| [ActiveFly-Bench](./paper-activefly-bench.md) | 同批同为空中具身，一训一测：本文给失败感知的训练范式，ActiveFly 给分层评测基准。两页之间缺的正是「用哪套口径判定纠错成功」——本文报的是 TravelUAV，不与 ActiveFly 自建基准同协议 |
| [WNM-3D](./paper-wnm-3d-vln.md) | 同批同为闭环 VLN，改的层不同：WNM-3D 换**条件表征**（3D 场景条件 + 联合生成未来视角与动作），本文换**训练信号**。表征与训练法正交，可叠加 |
| [VLN 任务页](../tasks/vision-language-navigation.md) | 该页给任务定义与常见评价口径；本文把重点从「一次走对」移到**走错之后能否纠回**，这一轴在多数 VLN 榜上不单列，是读本文成功率时最容易错配的地方 |

## 结论

**RecoverFly 适合作为本期「训练范式」路线的快速索引页。**

1. 核心贡献：空中 VLN 失败感知 RL 后训练：token 级优化、反复学习未解失败案例，分阶段长尾训练与参考策略约束，提升闭环纠错与泛化。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.09467](https://arxiv.org/abs/2608.09467)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.09467)
