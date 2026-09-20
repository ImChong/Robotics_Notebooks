---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - rl
  - sim2real
status: complete
updated: 2026-09-20
arxiv: "2609.19340"

related:
  - ../tasks/loco-manipulation.md
  - ../tasks/locomotion.md
  - ./paper-wholebodywam.md
  - ../concepts/sim2real.md
  - ../overview/contact-wm-10-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-14_18.md
  - ../../sources/papers/viloman_arxiv_2609_19340.md
  - ../../sources/sites/viloman.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md
summary: "ViLoMan（arXiv:2609.19340）：人类–物体交互重定向 + 特权 teacher + 在线 DAgger 蒸馏；Unitree G1 机载深度 + 本体 whole-body 关门。"
---

# ViLoMan（arXiv:2609.19340）

**ViLoMan**（*Learning Visual-Proprioceptive Whole-Body Loco-Manipulation Skills for Humanoid Robots*，[arXiv:2609.19340](https://arxiv.org/abs/2609.19340)，[项目页](https://viloman-anonymous.pages.dev/)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)（策展档位：**扫读**）。

## 一句话定义

**人类–物体交互重定向 + 特权 teacher + 在线 DAgger 蒸馏；Unitree G1 机载深度 + 本体 whole-body 关门。**

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
- **（匿名审稿）**；开源结论：**待发布**（步骤 2.5，2026-09-18）。
- 与 tactile/WAM、主动视角、多智能体场景理解、Sim2Real、人形导航等主线交叉。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.19340](https://arxiv.org/abs/2609.19340) |
| **开源** | **待发布** |
| **策展摘要** | 人类–物体交互重定向 + 特权 teacher + 在线 DAgger 蒸馏；Unitree G1 机载深度 + 本体 whole-body 关门。 |


## 源码运行时序图

**不适用**（截至 2026-09-18 项目页/arXiv 未发布可运行官方代码）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 SR/延迟/路径长度等 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**：G1 关门演示与下列各页不共享任务与评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [WholeBodyWAM](./paper-wholebodywam.md) | 同为人形全身操作，中间件不同：WholeBodyWAM 走世界模型预测，ViLoMan 走特权 teacher 蒸馏到机载深度 + 本体。预测未来 vs 压缩专家 |
| [RebarSim](./paper-rebarsim.md) | 同批次里同一套「特权 teacher → 视觉 student + DAgger」配方的另一端：RebarSim 用在毫米级插入，ViLoMan 用在全身关门。配方能跨这个尺度本身是看点 |
| [OmniMimic](./paper-omnimimic.md) | 同批次另一条「先造监督」：OmniMimic 在同 embodiment 内增广步态，ViLoMan 跨 embodiment 重定向人–物交互。监督从哪来，决定覆盖边界 |
| [Loco-manipulation](../tasks/loco-manipulation.md) | 该页给任务族评测口径；ViLoMan 属「机载感知 + 全身接触」一支，与分离式先导航后操作一支的取舍是**接触时机是否需要与步态协同** |
| [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **扫读** 档位 |

## 结论

**ViLoMan 代表「扫读」档位的 humanoid 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [Agile-WAM](./paper-agile-wam.md) / [INSPECT](./paper-inspect-view-selection.md) 等形成「触觉 WAM → 主动视角 → 系统平台」阅读链。
3. 若做工程选型，先对齐传感器栈与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [locomotion](../tasks/locomotion.md)
- ./paper-wholebodywam.md
- [sim2real](../concepts/sim2real.md)
- [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md)

## 参考来源

- [viloman_arxiv_2609_19340.md](../../sources/papers/viloman_arxiv_2609_19340.md)
- [wechat_embodied_station_10_papers_contact_wm_2026-09-18.md](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
- [arXiv:2609.19340](https://arxiv.org/abs/2609.19340)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.19340)
