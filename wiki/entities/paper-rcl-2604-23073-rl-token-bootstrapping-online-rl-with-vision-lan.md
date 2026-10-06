---
type: entity
tags:
- paper
- pi
- vla
- reinforcement-learning
status: complete
updated: 2026-10-06
arxiv: '2604.23073'
venue: '2026'
summary: RL Token 将预训练 VLA 特征压成紧凑状态表示，在冻结大模型后用轻量 actor/critic 在线修正动作块，提高精密操作的学习效率。
related:
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/generative-world-models.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
- ./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md
- ../methods/reinforcement-learning.md
sources:
- ../../sources/papers/rcl_awesome_wam_2604_23073_rl-token-bootstrapping-online-rl-with-vi.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
- ../../sources/sites/pi-memory-rlt-fast.md
---

# RL Token：用 VLA 表示启动在线强化学习

## 一句话定义

RL Token 将预训练 VLA 特征压成紧凑状态表示，在冻结大模型后用轻量 actor/critic 在线修正动作块，提高精密操作的学习效率。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作 |
| RL | Reinforcement Learning | 用交互反馈优化策略 |
| RLT | RL Token | VLA 内部特征的压缩表示 |
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | 双 critic、延迟 actor 更新的离策略算法 |

## 为什么重要

- 精密操作的接触反馈重要，直接在线更新大 VLA 成本高。
- 将通才先验与小网络的反馈学习结合，可以分别控制推理和训练负担。

## 核心原理

1. 用 encoder–decoder 重建 VLA 表征，训练一个压缩 token；可与示范微调共同进行。
2. 适配后冻结 VLA 与 token 编码器。
3. actor 输入 token、本体状态和 VLA 候选动作 chunk，学习局部动作修正；critic 用 chunk transition 做离策略 TD 学习。
4. 用靠近参考动作的正则约束探索，减少离开示范分布的漂移。

论文在线 critic 采用 TD3 思路。RL 不是从零产生无约束动作，也不是在部署时持续训练整个大模型。

## 源码运行时序图

**不适用**：本次没有确认官方可运行代码；论文算法流程不足以建立与源码目录对齐的运行时序图。

## 工程实践

1. 先适配通才策略和表示，再开启在线反馈；记录干预、复位与有效交互时间。
2. 固定在线数据预算，比对残差策略、噪声空间优化等基线。
3. 同时读成功率和 **每 10 分钟完成量**；部分已有高成功率的任务主要改善速度。
4. 本次官网项目页返回 403，官方完整运行实现 **未确认**；原始论文是当前归纳依据。

## 评测与指标

论文用 π₀.₆ 基座评估螺丝、扎带、以太网插头和充电接口等 **四项精密操作**。任务先收集 **1–10 小时示范**，在线训练约 **400–1000 episodes**；报告有效交互约 **15 分钟–5 小时**，不含复位等开销。控制为 **50 Hz**，14 维动作、10 步 chunk；成功率与吞吐须按同预算对照。

评测口径须区分：四项任务的主要对比从**已部分完成的状态**开始，只评精密关键阶段，每项各 50 次；完整任务评测只补测螺丝与扎带。训练仍有人提供奖励、纠正及基座 VLA / RL 策略的阶段切换，不能把这些结果理解成四项任务全程自主学习与执行。

## 结论

**RLT 的收益来自强 VLA 初始化、紧凑表示与受约束的在线动作修正。**

1. 对齐示范和在线交互预算后比较。
2. 统计包括复位与人工操作的真实总成本。
3. 保留吞吐指标，避免忽略已高成功率任务的速度收益。

## 与其他工作对比

论文与 HIL-SERL、Probe-Learn-Distill、DSRL、DAgger 对比；单步在线动作策略与 chunk 策略的时间跨度不同。读消融时重点看 token 表示、chunk、BC 正则与参考动作输入，并按相同有效交互预算比较关键阶段吞吐。

## 局限与风险

- 关键阶段的吞吐改善与完整任务成功率须分开；人工奖励、纠正与阶段切换也是实际训练成本。

- 论文有效机器人数据时间不含全部复位和操作开销；不能直接视作现场总工时。
- 结果依赖强通才初始化、任务示范与人工反馈条件。
- 官网发布日期与 arXiv 提交月份不同，公司时间线按官网文章口径。

## 关联页面

- [π₀.₆](./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md)
- [强化学习](../methods/reinforcement-learning.md)
- [VLA](../methods/vla.md)

## 参考来源

- [PI 一手资料补核](../../sources/sites/pi-memory-rlt-fast.md)

## 推荐继续阅读

- [RL Token 论文](https://arxiv.org/abs/2604.23073)
