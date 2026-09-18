---
type: entity
tags:
  - paper
  - locomotion
  - quadruped
  - rl
  - sim2real
status: complete
updated: 2026-09-18
arxiv: "2609.20566"

related:
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ./paper-recmorph.md
  - ../overview/contact-wm-10-papers-technology-map.md
sources:
  - ../../sources/papers/omnimimic_arxiv_2609_20566.md
  - ../../sources/sites/omnimimic.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md
summary: "OmniMimic（arXiv:2609.20566）：时间反转 + 动力学补全 + 矢状面反射扩展四足步态监督；共享 actor + 软门控残差专家；仿真 RMSE −63.1%，真机无微调部署。"
---

# OmniMimic（arXiv:2609.20566）

**OmniMimic**（*Dynamics-completed Motion Augmentation for Multi-style Omnidirectional Quadruped Locomotion*，[arXiv:2609.20566](https://arxiv.org/abs/2609.20566)，[项目页](https://omnimimic.github.io/)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)（策展档位：**扫读**）。

## 一句话定义

**时间反转 + 动力学补全 + 矢状面反射扩展四足步态监督；共享 actor + 软门控残差专家；仿真 RMSE −63.1%，真机无微调部署。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World-Action Model | 联合预测未来观测与动作的策略 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| SR | Success Rate | 任务成功率 |
| IL | Imitation Learning | 模仿学习 |
| GS | Gaussian Splatting | 高斯溅射三维表示 |

## 为什么重要

- 公众号将本文归入「接触时视觉之外还需预测什么」专题；扫读档位。
- **（待论文正式披露）**；开源结论：**待发布**（步骤 2.5，2026-09-18）。
- 与 tactile/WAM、主动视角、多智能体场景理解、Sim2Real、人形导航等主线交叉。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.20566](https://arxiv.org/abs/2609.20566) |
| **开源** | **待发布** |
| **策展摘要** | 时间反转 + 动力学补全 + 矢状面反射扩展四足步态监督；共享 actor + 软门控残差专家；仿真 RMSE −63.1%，真机无微调部署。 |


## 源码运行时序图

**不适用**（截至 2026-09-18 项目页/arXiv 未发布可运行官方代码）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 SR/延迟/路径长度等 headline 数字。

## 结论

**OmniMimic 代表「扫读」档位的 locomotion 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [Agile-WAM](./paper-agile-wam.md) / [INSPECT](./paper-inspect-view-selection.md) 等形成「触觉 WAM → 主动视角 → 系统平台」阅读链。
3. 若做工程选型，先对齐传感器栈与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- ./paper-recmorph.md
- [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md)

## 参考来源

- [omnimimic_arxiv_2609_20566.md](../../sources/papers/omnimimic_arxiv_2609_20566.md)
- [wechat_embodied_station_10_papers_contact_wm_2026-09-18.md](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
- [arXiv:2609.20566](https://arxiv.org/abs/2609.20566)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.20566)
