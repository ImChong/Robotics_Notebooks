---
type: entity
tags:
  - paper
  - world-model
  - continual-learning
  - benchmark
  - manipulation
status: complete
updated: 2026-09-21
arxiv: "2609.22055"
related:
  - ../overview/contact-rich-sim-10-papers-technology-map.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/wm_compositional_cl_benchmark_arxiv_2609_22055.md
  - ../../sources/sites/wm-compositional-cl-benchmark.md
  - ../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md
summary: "Compositional CL WM Benchmark（arXiv:2609.22055）：把 world model 持续学习拆成 action/perception 组合轴，用重组已见 primitives 的任务序列单独测 reuse 与 forgetting；模块化 PWM 更平衡但仍未完全解决。"
---

# Compositional CL WM Benchmark（arXiv:2609.22055）

**Compositional CL WM Benchmark**（*Benchmarking World Models for Continual Learning on Compositional Tasks*，[arXiv:2609.22055](https://arxiv.org/abs/2609.22055)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)（策展档位：**扫读**）。

## 一句话定义

**把 world model 持续学习拆成 action/perception 组合轴，用重组已见 primitives 的任务序列单独测 reuse 与 forgetting；模块化 PWM 更平衡但仍未完全解决。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| SR | Success Rate | 任务成功率 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| WAM | World-Action Model | 联合未来观测与动作的策略 |

## 为什么重要

- 公众号将本文归入「接触丰富操作为何总在仿真里失真」专题；扫读档位。
- **牛津大学**；开源结论：**待发布**（步骤 2.5，2026-09-21）。
- 现有 CL benchmark 常把复用与快速学习新交互混在一起；该工作用组合任务末端隔离知识复用。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.22055](https://arxiv.org/abs/2609.22055) |
| **开源** | **待发布** |
| **策展摘要** | 把 world model 持续学习拆成 action/perception 组合轴，用重组已见 primitives 的任务序列单独测 reuse 与 forgetting；模块化 PWM 更平衡但仍未完全解决。 |

## 源码运行时序图

**不适用**（截至 2026-09-21 项目页/arXiv 未发布可运行官方代码，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**；数字未与下列各页核对同一评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [RAPID](./paper-rapid-vlm-rl.md) | 同专辑 **深读** 档：并行仿真 + VLM 奖励管线，关注训练吞吐与 API 成本 |
| [CRISP](./paper-crisp.md) | 同专辑 **跟进** 档：接触仿真几何与求解器，关注 peg-in-hole/装配物理准确性 |
| [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **扫读** 档位 |

## 结论

**Compositional CL WM Benchmark 代表「扫读」档位的 world-model 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [RAPID](./paper-rapid-vlm-rl.md) / [GALA](./paper-gala.md) / [CRISP](./paper-crisp.md) 形成「并行奖励 → 跨形态表征 → 接触仿真」阅读链。
3. 若做工程选型，先对齐传感器栈、仿真器与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [contact-rich-sim-10-papers-technology-map](../overview/contact-rich-sim-10-papers-technology-map.md)
- [sim2real](../concepts/sim2real.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [wm_compositional_cl_benchmark_arxiv_2609_22055.md](../../sources/papers/wm_compositional_cl_benchmark_arxiv_2609_22055.md)
- [wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)
- [arXiv:2609.22055](https://arxiv.org/abs/2609.22055)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.22055)
- [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md)
