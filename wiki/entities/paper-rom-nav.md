---
type: entity
tags:
  - paper
  - humanoid
  - navigation
  - rl
  - caltech
  - amazon
status: complete
updated: 2026-09-18
arxiv: "2609.19272"

related:
  - ../overview/navigation-slam-autonomy-stack.md
  - ../tasks/locomotion.md
  - ./paper-cap-perception-blind-humanoid.md
  - ../methods/reinforcement-learning.md
  - ../overview/contact-wm-10-papers-technology-map.md
sources:
  - ../../sources/papers/rom_nav_arxiv_2609_19272.md
  - ../../sources/sites/rom-nav.md
  - ../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md
summary: "Learning Safe Humanoid Navigation from Reduced Order Models（arXiv:2609.19272）：RoM-Nav：RoM 导航 kickstart 全人形 + Poisson safety filter；Unitree G1 无地图跨楼层 >10 m 高差 / 100 m 路径。"
---

# Learning Safe Humanoid Navigation from Reduced Order Models（arXiv:2609.19272）

**Learning Safe Humanoid Navigation from Reduced Order Models**（*Learning Safe Humanoid Navigation from Reduced Order Models*，[arXiv:2609.19272](https://arxiv.org/abs/2609.19272)，[项目页](https://wdc3iii.github.io/rom-nav/)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)（策展档位：**扫读**）。

## 一句话定义

**RoM-Nav：RoM 导航 kickstart 全人形 + Poisson safety filter；Unitree G1 无地图跨楼层 >10 m 高差 / 100 m 路径。**

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
- **加州理工学院（Caltech）；亚马逊（Amazon）**；开源结论：**待发布**（步骤 2.5，2026-09-18）。
- 与 tactile/WAM、主动视角、多智能体场景理解、Sim2Real、人形导航等主线交叉。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.19272](https://arxiv.org/abs/2609.19272) |
| **开源** | **待发布** |
| **策展摘要** | RoM-Nav：RoM 导航 kickstart 全人形 + Poisson safety filter；Unitree G1 无地图跨楼层 >10 m 高差 / 100 m 路径。 |


## 源码运行时序图

**不适用**（截至 2026-09-18 项目页/arXiv 未发布可运行官方代码）。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 SR/延迟/路径长度等 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**：>10 m 高差 / 100 m 路径是本文现场演示的规模，与下列各页不共享评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [CAP（感知盲人形）](./paper-cap-perception-blind-humanoid.md) | 同为人形在未知环境里走，感知假设相反：CAP 探的是**没有外部感知**能走多远，RoM-Nav 用降阶模型给导航 kickstart 并加 Poisson 安全滤波。一个减信息，一个加结构 |
| [PASSAGE](./paper-passage.md) | 同为 onboard 无预建图穿越，行为来源不同：PASSAGE 靠大规模场景对齐人体 motion，RoM-Nav 靠 RoM 先验 + 安全滤波。数据驱动 vs 模型驱动，对应采数据与建模两种成本 |
| **端到端 RL 直接训全人形导航**（本文要替代的默认做法） | 同为出导航动作，差别在**探索从哪起步**：端到端从零探索样本效率低，RoM-Nav 先在降阶模型上学会导航再 kickstart 全人形。代价是 RoM 与全身动力学的差距要由后续训练吸收 |
| [导航 / SLAM / 自主栈](../overview/navigation-slam-autonomy-stack.md) | 该页给导航栈分层；本文的「无地图」指不建全局地图，不等于不需要局部几何 |
| [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **扫读** 档位 |

## 结论

**Learning Safe Humanoid Navigation from Reduced Order Models 代表「扫读」档位的 humanoid 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**待发布**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [Agile-WAM](./paper-agile-wam.md) / [INSPECT](./paper-inspect-view-selection.md) 等形成「触觉 WAM → 主动视角 → 系统平台」阅读链。
3. 若做工程选型，先对齐传感器栈与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [navigation-slam-autonomy-stack](../overview/navigation-slam-autonomy-stack.md)
- [locomotion](../tasks/locomotion.md)
- ./paper-cap-perception-blind-humanoid.md
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [10 篇技术地图](../overview/contact-wm-10-papers-technology-map.md)

## 参考来源

- [rom_nav_arxiv_2609_19272.md](../../sources/papers/rom_nav_arxiv_2609_19272.md)
- [wechat_embodied_station_10_papers_contact_wm_2026-09-18.md](../../sources/blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
- [arXiv:2609.19272](https://arxiv.org/abs/2609.19272)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.19272)
