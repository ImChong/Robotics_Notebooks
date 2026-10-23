---
type: entity
tags: [paper, humanoid, fall-recovery, perception, motion-steering, unitree-g1]
status: draft
updated: 2026-10-23
project_id: rise-humanoid-fall-recovery
arxiv: "2610.30061"
project: https://osheraz.github.io/RISE/
related:
  - ./paper-host-humanoid-standingup.md
  - ./paper-unified-walk-run-recovery-sdamp.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/rise_fall_recovery_arxiv_2610_30061.md
  - ../../sources/sites/osheraz-rise.md
summary: "RISE uses visual prediction of future body states and test-time task costs to steer how a fallen Unitree G1 recovers, without retraining for each recovery objective."
---

# RISE：感知式人形起身的测试时运动引导

**一句话定义：** RISE 把未来身体状态作为可引导的运动意图，在测试时按目标路径、朝向或起身速度选择人形机器人的起身方式。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RISE | Test-Time Motion Steering for Perceptive Humanoid Fall Recovery | 本文的感知式起身与测试时引导方法 |
| G1 | Unitree G1 Humanoid | 论文展示真机起身的类人机器人平台 |
| VDP | Visual Diffusion Planner | 根据视觉观测预测未来身体状态的规划器 |

## 为什么重要

起身不只是从地面回到站立状态：靠墙或处于不平地形时，恢复动作的路径和支撑转换会影响是否碰撞、能否继续任务。RISE 的重点是让任务目标改变恢复动作的选择，而不必为每个目标重新训练策略。

## 核心原理

RISE 先从兼容动作片段构建可组合的恢复动作库，再训练动作词汇和反馈解码器，将运动 latent 转成马达动作。视觉扩散规划器预测未来身体状态；路径、朝向或速度相关的任务代价引导这一预测，随后条件化生成运动 latent 并闭环执行。

项目页展示的主干可概括为：

`观测 → 未来身体状态 → 任务代价引导 → motion latent → 反馈动作`

## 方法栈

- **恢复动作策展：** 匹配录制动作中的兼容帧，组合成不同长度的恢复片段并粗略对齐地形。
- **动作词汇：** 教师策略学习执行参考恢复动作；词汇与反馈解码器从教师执行中学习运动表示。
- **测试时引导：** 视觉规划器使用任务代价引导未来身体状态，覆盖目标朝向、起身速度及路径。

## 实验与评测

论文摘要报告了仿真中的多样化恢复、地形适应、无需重训的测试时引导，以及 Unitree G1 真机演示。项目页展示深度感知绕障与不同恢复路线。详细量化结果应回到论文全文表格核对。

## 与其他工作对比

| 方法 | 重点 | 恢复目标的选择 |
|------|------|----------------|
| RISE | 感知 + 未来状态意图引导 | 测试时指定路径、朝向或速度 |
| [HoST](./paper-host-humanoid-standingup.md) | 多初始姿态的多阶段强化学习起身 | 策略按学到的奖励恢复 |
| [SD-AMP](./paper-unified-walk-run-recovery-sdamp.md) | AMP 先验统一走、跑与起身 | 统一运动策略内恢复 |

## 工程实践

- 输入需要包含身体状态及视觉深度观测，才能利用地形与障碍信息。
- 如果任务要求指定恢复方向或通行路线，测试时任务代价可作为可调接口。
- **代码不适用：** 项目页标注 “Code Coming soon”，本次核查没有可运行官方代码入口。

## 结论

**RISE 的工程价值在于把“如何起身”变成测试时可调的任务条件，同时保留反馈执行。**

1. 路径、朝向与起身速度是不同的引导目标，不能把成功起身率当作唯一指标。
2. 视觉深度让恢复动作能够考虑附近障碍与支撑地形。
3. 目前公开证据包括论文、视频和 G1 演示；项目页尚未发布代码链接。

## 局限与风险

- 公开摘要未提供足以判断所有场景成功率的详细指标；具体数字应查论文实验表。
- 测试时任务代价需要与安全、可达性和稳定性约束相容。
- 代码尚未通过项目页开放，复现状态待后续更新。

## 关联页面

- [HoST：跨多样姿态的人形起身控制](./paper-host-humanoid-standingup.md)
- [SD-AMP：统一走跑起身](./paper-unified-walk-run-recovery-sdamp.md)
- [Unitree G1](./unitree-g1.md)

## 参考来源

- [RISE 论文归档](../../sources/papers/rise_fall_recovery_arxiv_2610_30061.md)
- [RISE 官方项目页归档](../../sources/sites/osheraz-rise.md)
- [arXiv:2610.30061](https://arxiv.org/abs/2610.30061)

## 推荐继续阅读

- [RISE 官方项目页](https://osheraz.github.io/RISE/) — 方法图示与真机演示
- [HoST 原论文](https://arxiv.org/abs/2502.08378) — 多初始姿态起身的对照路线
