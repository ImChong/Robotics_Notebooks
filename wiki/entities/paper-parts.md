---
type: entity
tags:
  - paper
  - rl
  - long-horizon
  - manipulation
  - residual-policy
status: complete
updated: 2026-09-21
arxiv: "2609.21788"
related:
  - ../overview/contact-rich-sim-10-papers-technology-map.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/parts_arxiv_2609_21788.md
  - ../../sources/sites/parts.md
  - ../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md
summary: "PARTS（arXiv:2609.21788）：冻结预训练策略，仅在抓取/插入等瓶颈子任务学 residual correction；YAM 32%→61%，Franka 50%→95%。"
---

# PARTS（arXiv:2609.21788）

**PARTS**（*From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention*，[arXiv:2609.21788](https://arxiv.org/abs/2609.21788)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)（策展档位：**扫读**）。

## 一句话定义

**冻结预训练策略，仅在抓取/插入等瓶颈子任务学 residual correction；YAM 32%→61%，Franka 50%→95%。**

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
- **德克萨斯大学奥斯汀分校；Autel US；加州大学伯克利分校**；开源结论：**待发布**（步骤 2.5，2026-09-21）。
- 长时程任务常卡在少数瓶颈；PARTS 把真机 RL 预算集中到这些子任务。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.21788](https://arxiv.org/abs/2609.21788) |
| **开源** | **待发布** |
| **策展摘要** | 冻结预训练策略，仅在抓取/插入等瓶颈子任务学 residual correction；YAM 32%→61%，Franka 50%→95%。 |

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

**PARTS 代表「扫读」档位的 rl 方向样本——部署前以开源状态与评测协议为准绳。**

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

- [parts_arxiv_2609_21788.md](../../sources/papers/parts_arxiv_2609_21788.md)
- [wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)
- [arXiv:2609.21788](https://arxiv.org/abs/2609.21788)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.21788)
- [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md)
