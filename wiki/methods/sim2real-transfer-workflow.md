---
type: method
tags: [sim2real, simulation, robotics]
status: complete
updated: 2026-11-26
summary: "将仿真中的感知模型或控制策略接入真实系统，逐项测量、缩小并验证实际迁移误差。"
sources:
  - ../../sources/papers/tan_quadruped_sim2real_reading_arxiv_1804_10332.md
original:
  - ../../sources/papers/tan_quadruped_sim2real_reading_arxiv_1804_10332.md
related:
  - ../concepts/sim2real.md
  - sim2real-joint-sysid-experiment-design.md
  - domain-adaptation-for-sim2real.md
  - ../concepts/domain-randomization.md
---

# Sim-to-Real Transfer Workflow · Sim2Real 迁移工作流

## 一句话定义

将仿真中的感知模型或控制策略接入真实系统，逐项测量、缩小并验证实际迁移误差。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| SI | System Identification | 从实际输入输出估计仿真参数 |
| DR | Domain Randomization | 对残余不确定量做分布训练 |
| RL | Reinforcement Learning | 可用于仿真中学习反馈策略 |
| HIL | Hardware-in-the-Loop | 软硬件接口的分级验证方式 |

## 为什么重要

仿真回报高不代表能通过真机验收。执行器、控制周期或观测时延如果错配，策略甚至无法取得它训练时预期的反馈。

## 主要技术路线（核心机制）

先冻结传感／动作合约并对时，再用测量标定差距最大的环节；对可量化但不能唯一确定的残差做随机化。Tan 等人的四足案例把执行器模型与时延显式纳入 Bullet 仿真，同时用了动态参数随机化、外扰与紧凑观测；它验证的是组合系统在 Minitaur 上的步法，不是单靠随机化。

```mermaid
flowchart LR
    A["冻结观测·动作·时钟接口"] --> B["真机离线测量、辨识"]
    B --> C["仿真标定与参数残差覆盖"]
    C --> D["离线重放及受控实验"]
    D --> E["真机指标与失败记录"]
    E -->|定位偏差| B
```

## 工程实践

1. 对齐坐标系、单位、关节顺序、观测采集时刻、命令时间戳与实际周期；用日志比对设定、执行与反馈。
2. 用保留的系统辨识数据验证仿真，再在第二仿真器或离线重放中检查接口与边界。日志中区分平均时延、尾部抖动、丢包和饱和。
3. 由受控、低风险的动作开始；先定义现场保护、停机条件和失败回退，再报告速度跟踪、约束超限率、成功率和扰动恢复。

## 局限与风险

模型标定不能消除全部接触、磨损或传感偏差；随机化范围过大也可能拒学有用行为。仿真器一致不是安全证明；没有独立真机数据不要写“迁移通过”。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [关节辨识实验](sim2real-joint-sysid-experiment-design.md)
- [视觉域适应](domain-adaptation-for-sim2real.md)
- [域随机化](../concepts/domain-randomization.md)

## 参考来源

- [Tan 等人的四足 Sim2Real 一手资料归档](../../sources/papers/tan_quadruped_sim2real_reading_arxiv_1804_10332.md)

## 推荐继续阅读

- [Tan 等人的四足 Sim2Real 原文](https://arxiv.org/html/1804.10332v2)
