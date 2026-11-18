# XMimic and XSoma: Learning Humanoid Motion Tracking and Locomotion in Under a Minute / 一分钟内训练人形动作跟踪与地形行走

- **类型：** paper
- **论文：** https://arxiv.org/abs/2611.20452
- **全文：** https://arxiv.org/html/2611.20452v1
- **原文版本：** arXiv v1；核对日期 2026-11-18
- **核对范围：** arXiv v1 联合全文、XMimic README/训练入口和 XSoma README/训练入口/环境与指标协议，以及站内旧 XMimic 归档以防重复。
- **通读：** 已完成
- **项目：** 无单独项目主页；官方代码见下
- **代码：** https://github.com/nico-bohlinger/xsoma
- **限制：** 来源和结果以论文中各自实验协议为准；未实际运行训练或实机测试。

## 核心摘录

1. XMimic 对 LAFAN1 的 40 段参考动作进行 Unitree G1 全身跟踪；与 BlinkMimic 共享世界模型想象的策略梯度思路，使用 JAX/XLA 完整编译并将 actor、世界模型、策略学习和失败探测分流到最多四卡。 对 wiki 的映射：[核心原理](../../wiki/entities/paper-xmimic-xsoma-2611-20452.md)。
2. XSoma 复用 Holosoma 的 G1 地形速度跟踪任务、奖励、随机化与终止条件；先在真实模拟交互采集转移、拟合世界模型，再穿过约一秒预测轨迹直接反传奖励来更新策略，源码明确不使用 critic。 对 wiki 的映射：[核心原理](../../wiki/entities/paper-xmimic-xsoma-2611-20452.md)。
3. “一分钟内”仅指连续检查点达标的学习墙钟时间，不包括启动、首次编译、真实机器部署；四卡 XMimic 和 XSoma 处理的是不同任务与不同目标阈值。 对 wiki 的映射：[核心原理](../../wiki/entities/paper-xmimic-xsoma-2611-20452.md)。

## 实验与评测口径

- XMimic：4×H100 NVL 上 40 段 LAFAN1 G1 参考跟踪，严格目标是归一化回报至少 0.82、40 段零失败、连续 3 个评测点保持，平均 0.86 分钟；BlinkMimic 单卡 3.19 分钟。
- XSoma：4×H100 上在 Holosoma 粗糙地形速度/角速度指标（score ≥0.95、≥0.8）连续 3 个检查点达标，平均 0.77 分钟；单卡 Holosoma FastSAC 平均 9.44 分钟。
- 训练耗时均排除启动和编译；4 GPU 更短的墙钟时间不等于使用的 GPU·分钟更省；两种任务的达标阈值不同。

## 局限、可复现入口与证据

- 全部关键成绩来自训练基准中的仿真任务，未证明真机部署的鲁棒性。
- 多卡同步、旧策略/旧世界模型流水陈旧性以及机器资源成本需要单列。
- 本论文以 XSoma 代码 URL 作为主项目入口；XMimic 已在站内有更早独立工作记录，本页不复制其独立论文主身份。
