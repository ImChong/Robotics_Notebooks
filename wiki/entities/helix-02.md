---
type: entity
status: complete
updated: '2026-10-05'
summary: Helix 02 在 Helix 的语义与动作双系统下增加 System 0：200 Hz 全身关节目标由 1 kHz 身体控制层执行，联合移动、接触与灵巧操作。
tags:
- figure-ai
- vla
- whole-body-control
- loco-manipulation
related:
- ./figure-ai.md
- ./helix-25.md
- ../concepts/embodied-three-layer-control-architecture.md
sources:
- ../../sources/sites/figure-helix-models.md
---

# Helix 02：Figure 全身自主系统

## 一句话定义

Helix 02 在 Helix 的语义与动作双系统下增加 System 0：200 Hz 全身关节目标由 1 kHz 身体控制层执行，联合移动、接触与灵巧操作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作 |
| WBC | Whole-Body Control | 全身平衡、运动与接触协调 |
| Hz | Hertz | 每秒执行或更新次数 |
| RL | Reinforcement Learning | 用交互反馈优化策略 |

## 为什么重要

- 把“看图出动作”推进到全身闭环，展示语言、视觉、触觉和本体感知的分层接口。
- 明确快慢模块频率，适合作为 VLA 与低层稳定控制分工的闭源对照。

## 核心原理

| 系统 | 输入/职责 | 输出与频率 |
| --- | --- | --- |
| System 2 | 理解图像、指令与任务步骤 | 语义 latent，官方未在本文给固定频率 |
| System 1 | 头/掌相机、指尖触觉、全身本体状态；受 S2 条件化 | 全身关节目标，200 Hz |
| System 0 | 跟踪目标、处理平衡和接触；约 10M 参数 | 关节执行命令，1 kHz |

System 0 使用超过 **1000 小时重定向人动作**，在超过 **20 万并行仿真环境**中训练并做域随机化。Helix 02 的掌部视觉与触觉依赖 Figure 03 硬件；“所有执行器输出”与层级系统并存，不能理解成单一频率的无层级网络。

## 工程实践

1. 按 **S2 latent → S1 joint targets → S0 actuator commands** 对齐自己系统的模块边界。
2. 评估掌相机和触觉解决的自遮挡/接触反馈问题，而非直接把演示归因于更大的 VLM。
3. **未确认可复现模型发布（2026-10-05）**：官方页未提供训练代码、权重和数据；源码运行时序图不适用。

## 评测与指标

官方 2026-01-27 展示约 **4 分钟 / 61 个动作**的洗碗机装卸连续任务，以及拧瓶盖、取药片、推注射器和分拣金属件。本文把这些作为定性闭环与感知证据，未据此建立跨模型排名。

## 局限与风险

- 四分钟任务是连续演示证据，不能替代随机环境、多次运行的成功率统计。
- 公布训练规模不能证明硬件外迁移或任意家庭零样本能力。
- 模块时序来自官方架构说明，完整实现未开放。

## 关联页面

- [Figure 公司](./figure-ai.md)
- [Helix 2.5](./helix-25.md)
- [具身三层控制架构](../concepts/embodied-three-layer-control-architecture.md)

## 参考来源

- [两代 Helix 官方技术发布](../../sources/sites/figure-helix-models.md)

## 推荐继续阅读

- [Helix 02 官方文章](https://www.figure.ai/news/helix-02)
