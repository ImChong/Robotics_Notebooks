---
type: entity
tags: [paper, manipulation, vla, agent, simulation]
status: complete
updated: 2026-09-26
arxiv: "2609.30249"
related:
  - ../overview/embodied-research-12-papers-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/rapid-agentic-programming_arxiv_2609_30249.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md
summary: "RAPID（arXiv:2609.30249）：从示范推断目标/原语/可交互环境并迭代验证程序；对象关系表征利于迁移；项目页未列代码（入库日）。"
---

# RAPID

**RAPID**（*Robot Agentic Programming from Demonstrations*，[arXiv:2609.30249](https://arxiv.org/abs/2609.30249)，[项目页](https://yuyaoliu.me/projects/rapid)）收录自 [具身智能小站 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)。

## 一句话定义

**一次示范后推断可执行 agent 程序，并在仿真中迭代验证，而不是假设即时学会全部细节。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| IL | Imitation Learning | 模仿学习 |
| SR | Success Rate | 任务成功率 |
| WM | World Model | 世界模型 |

## 为什么重要

- 纳入 [12 篇具身研究清单](../../wiki/overview/embodied-research-12-papers-technology-map.md) 主线，与同期 VLA / 接触 / 规划 / 安全论文可横向对照。
- 公众号强调的可操作读法：先看 **任务信息需求**（如 PolyUMI 旋灯泡仍以视觉最优）与 **评测口径**（如 Self-Adaptive 多 trial、BeyondRetarget 仿真片段非真机 SR）。
- 开源状态（步骤 2.5）：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.30249](https://arxiv.org/abs/2609.30249) |
| **项目页** | https://yuyaoliu.me/projects/rapid |
| **代码** | 截至入库日未列 |
| **开源** | **待发布** |

## 实验与评测（公众号口径）

- 指标与消融以 [公众号盘点](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md) 与 **原文 PDF** 为准；本页不复制整表。
- 读复现前先核对：样本规模、是否仿真/真机、是否允许多次 attempt。

## 源码运行时序图

**不适用**（截至 2026-09-26 项目页未提供可运行官方代码仓库；见 [sources/papers/rapid-agentic-programming_arxiv_2609_30249.md](../../sources/papers/rapid-agentic-programming_arxiv_2609_30249.md)）。

## 结论

**总判：RAPID 适合作为「一次示范后推断可执行 agent 程序，并在仿真中迭代验证，而不是假设即时学会全…」方向的入口页；细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md) 对照选型，避免与同名不同 arXiv 的工作混淆（如 RAPID vs RAPID-VLM-RL）。
2. 开源为 **待发布** 时优先从项目页 Code 区核实，再写复现计划。
3. 长程 / 部署类条目（AdaHVLA、HarnessPAI、Self-Adaptive VLA）同时记录 **成功率定义** 与 **失败恢复预算**。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [RAPID（VLM-RL）](./paper-rapid-vlm-rl.md) | **同名不同 arXiv**：那篇是 VLM 偏好奖励 + GPU 并行 RL 的训练加速，与本文无关 |
| Code-as-Policies 类代码策略 | 通常需人工给出任务规格、动作原语与验证环境；RAPID 从 **单次视觉人类示范** 自动推断三者，再由编码 agent 生成—验证—修订程序 |
| [HarnessPAI](./paper-harnesspai.md) | 演化代码 harness 去组织已有动作模型；RAPID 的原语本身是 **轨迹优化程序**，用对象级关系约束组合，面向接触丰富的非抓取操作 |
| [Code-as-World](./paper-code-as-world.md) | 用可执行代码表示 **物理世界** 做推理；RAPID 用代码表示 **策略**，强调对象中心关系表示以泛化到物体位姿/形状/材质与环境变化 |
| [LIBERO](./libero-benchmark.md) | 抓取类任务在 LIBERO-Pro 上评测；八个非抓取任务另在 Franka 真机全部部署 |

## 关联页面

- [具身研究 12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md)
- [Manipulation](../tasks/manipulation.md)
- [VLA](../methods/vla.md)

## 参考来源

- [rapid-agentic-programming 论文归档](../../sources/papers/rapid-agentic-programming_arxiv_2609_30249.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)

## 推荐继续阅读

- [arXiv:2609.30249](https://arxiv.org/abs/2609.30249)
- [项目页](https://yuyaoliu.me/projects/rapid)
