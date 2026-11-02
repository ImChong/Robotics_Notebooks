---
type: entity
tags: [paper, robotics, robot-learning]
status: complete
updated: 2026-10-18
project_id: h2rbench
arxiv: "2609.24778"
related:
  - ../overview/humanoid-motion-intelligence-day6-engineering-deployment.md
  - ../concepts/sim2real.md
  - ../overview/embodied-station-2026-10-18-26-papers-technology-map.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day6_engineering_deployment_2026_10_07.md
  - ../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md
summary: "H2RBench把人类视频到机器人操作的迁移放进统一real-to-sim评测协议。"
---

# H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer

H2RBench把人类视频到机器人操作的迁移放进统一real-to-sim评测协议。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| RL | Reinforcement Learning | 利用交互反馈优化机器人策略。 |
| Sim2Real | Simulation-to-Real | 将仿真训练或验证迁移至实体机器人。 |
| WBC | Whole-Body Control | 协调机器人全身自由度与任务约束。 |

## 为什么重要

H2RBench把人类视频到机器人操作的迁移放进统一real-to-sim评测协议。 评读时应将问题、观测条件与执行接口一并记录，避免从单个演示或指标外推能力。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["阶段一: Real-to-Sim 重建"]
        SC["真实场景捕获"] --> SG["场景重建: Marble (3D mesh + Gaussian Splatting)"]
        OB["多视角参考图 -> TRELLIS 重建物体<br/>按真实尺寸缩放"] --> ENV
        SG --> ENV["Isaac Lab 仿真环境 (基于 PolaRiS)"]
    end
    subgraph S2["阶段二: 任务数据采集 (4 个任务)"]
        HD["真实人类示范: 第三视角 RGB 视频<br/>相机位姿与仿真场景一致"]
        RD["机器人示范: 仿真 (cuRobo 规划) / 真机 (Franka + GELLO)"]
    end
    ENV --> RD
    subgraph S3["阶段三: 仿真中训练与评测"]
        M["H2R 方法: Phantom / Point Policy / AMPLIFY / GHOST<br/>机器人示范数量固定, 变化人类示范数量"]
        EV["多阶段成功率 (最终阶段 SR + 各阶段平均)"]
    end
    HD --> M
    RD --> M
    M --> EV
    EV --> SVR["Sim-vs.-Real 验证: 真机管线同样训练评测<br/>Pearson / Spearman / MMRV 比较方法排名"]
```

图依据：论文 §3.1 / §3.3 / §3.4 / §4 / Fig.2

## 方法栈

共同比较人类示范、仿真机器人数据和真实执行，并区分交互意图与精细动作控制迁移。

## 实验与评测

文章指出策略可能理解操作顺序，却无法准确找到接触位置；任务集与对象范围决定结论边界。

## 与其他工作对比

相比只报告任务成功率，它区分意图理解和接触执行两类迁移误差。

## 工程实践

当前归档仅核实到arXiv入口，未核实可运行官方代码。

## 源码运行时序图

**不适用**：当前没有核实到该论文的可运行官方训练、推理或部署入口。

## 结论

**一句话总判：** H2RBench把人类视频到机器人操作的迁移放进统一real-to-sim评测协议。

1. 核对方法实际改变的是数据、表征、规划、控制还是评测。
2. 只把论文覆盖的任务、平台与扰动范围当作证据。
3. 复现前确认代码、权重、数据许可和硬件依赖状态。

## 局限与风险

相比只报告任务成功率，它区分意图理解和接触执行两类迁移误差。 当前导读以用户提供的文章和预印本题录为索引；详细配置与结果应核对论文全文。

## 关联页面

- [Day 6 导读](../overview/humanoid-motion-intelligence-day6-engineering-deployment.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [Day 6 原文归档](../../sources/blogs/humanoid_motion_intelligence_day6_engineering_deployment_2026_10_07.md)
- [arXiv:2609.24778](https://arxiv.org/abs/2609.24778)
- [arXiv PDF](https://arxiv.org/pdf/2609.24778)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 相关工作、§3 H2RBench（3.1–3.4）、§4 被评测方法、§5.1 实验设置；其余实验与附录略读

## 推荐继续阅读

- [Day 6 导读](../overview/humanoid-motion-intelligence-day6-engineering-deployment.md)
