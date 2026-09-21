---
type: entity
tags:
  - paper
  - vla
  - rl
  - simulation
  - reward-learning
status: complete
updated: 2026-09-21
arxiv: "2609.21767"
code: https://github.com/rapid-vlm/rapid-vlm-rl
related:
  - ../overview/contact-rich-sim-10-papers-technology-map.md
  - ../concepts/sim2real.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/rapid_vlm_rl_arxiv_2609_21767.md
  - ../../sources/repos/rapid-vlm-rl.md
  - ../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md
summary: "RAPID（arXiv:2609.21767）：GPU 并行 rollout + 单请求偏好标注 + 代表性图像采样 + 自动奖励稳定化；5 个 IsaacLab Franka 任务平均训练 9.18h→3.13h，全组件 1.15h、896 次 API 调用。"
---

# RAPID（arXiv:2609.21767）

**RAPID**（*Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation*，[arXiv:2609.21767](https://arxiv.org/abs/2609.21767)）来自 [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)（策展档位：**深读**）。

## 一句话定义

**GPU 并行 rollout + 单请求偏好标注 + 代表性图像采样 + 自动奖励稳定化；5 个 IsaacLab Franka 任务平均训练 9.18h→3.13h，全组件 1.15h、896 次 API 调用。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习 |
| SR | Success Rate | 任务成功率 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| WAM | World-Action Model | 联合未来观测与动作的策略 |

## 为什么重要

- 公众号将本文归入「接触丰富操作为何总在仿真里失真」专题；深读档位。
- **（待论文正式披露）**；开源结论：**已开源**（步骤 2.5，2026-09-21）。
- VLM 奖励 RL 的瓶颈常在标注吞吐而非策略网络；RAPID 把 rollout、标注、采样与更新统一调度。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.21767](https://arxiv.org/abs/2609.21767) |
| **开源** | **已开源** |
| **策展摘要** | GPU 并行 rollout + 单请求偏好标注 + 代表性图像采样 + 自动奖励稳定化；5 个 IsaacLab Franka 任务平均训练 9.18h→3.13h，全组件 1.15h、896 次 API 调用。 |

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant Dev as 开发者
  participant Repo as 官方仓库
  participant Sim as 仿真/数据
  Dev->>Repo: clone + README 环境依赖
  Dev->>Sim: 准备 Isaac Sim/仿真或数据集
  Dev->>Repo: 训练/推理/示例入口
  Repo-->>Dev: 指标、日志或部署输出
```

节点对齐 [`sources/repos/rapid-vlm-rl.md`](../../sources/repos/rapid-vlm-rl.md) 与 README 入口。

## 实验与评测

- 定量指标与 baseline 协议以 arXiv PDF 与项目页为准；本页为清单级摘要。
- 读法：先确认任务设定（仿真/真机、传感器、成功定义）再对比 headline 数字。

## 与其他工作对比

> 本页为清单级摘要，下表只做**定位对照**；数字未与下列各页核对同一评测协议，不可横比。

| 对照 | 差异读法 |
|------|----------|
| [RAPID](./paper-rapid-vlm-rl.md) | 同专辑 **深读** 档：并行仿真 + VLM 奖励管线，关注训练吞吐与 API 成本 |
| [CRISP](./paper-crisp.md) | 同专辑 **跟进** 档：接触仿真几何与求解器，关注 peg-in-hole/装配物理准确性 |
| [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md) | 同批次横向对照入口：本文列 **深读** 档位 |

## 结论

**RAPID 代表「深读」档位的 vla 方向样本——部署前以开源状态与评测协议为准绳。**

1. 开源状态：**已开源**；勿凭 PDF 臆断可复现性。
2. 与同专辑 [RAPID](./paper-rapid-vlm-rl.md) / [GALA](./paper-gala.md) / [CRISP](./paper-crisp.md) 形成「并行奖励 → 跨形态表征 → 接触仿真」阅读链。
3. 若做工程选型，先对齐传感器栈、仿真器与任务是否匹配文内设定。
4. 关注项目页/arXiv 版本更新与代码发布。

## 关联页面

- [contact-rich-sim-10-papers-technology-map](../overview/contact-rich-sim-10-papers-technology-map.md)
- [sim2real](../concepts/sim2real.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [rapid_vlm_rl_arxiv_2609_21767.md](../../sources/papers/rapid_vlm_rl_arxiv_2609_21767.md)
- [wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md](../../sources/blogs/wechat_embodied_station_contact_rich_sim_10_papers_2026-09-21.md)
- [arXiv:2609.21767](https://arxiv.org/abs/2609.21767)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.21767)
- [10 篇技术地图](../overview/contact-rich-sim-10-papers-technology-map.md)
