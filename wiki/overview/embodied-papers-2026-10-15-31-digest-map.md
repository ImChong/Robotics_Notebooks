---
type: overview
tags: [overview, vla, humanoid, manipulation, tactile, navigation, digest]
status: complete
updated: 2026-10-15
related:
  - ../methods/vla.md
  - ../entities/paper-morphact.md
  - ../entities/paper-acro.md
  - ../entities/paper-tacclaw-infinity.md
sources:
  - ../../sources/blogs/wechat_embodied_station_robot_31_papers_2026-10-15.md
  - ../../sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md
summary: "2026-10-15 具身智能小站汇总的 31 篇新论文索引：按主题把每篇映射到独立 paper-* 详情节点，并标注文章的阅读优先级；开源状态以各详情页为准。"
---

# 2026-10-15 具身新论文 31 篇：技术地图索引

> 本页是一份**索引**，不替代各论文详情页。每篇论文的方法、实验与开源核查见对应 `paper-*` 节点。

## 一句话定义

把公众号单日汇总的 31 篇具身智能论文，按「VLA 适配与恢复 / 人形与移动操作 / 灵巧与接触 / 数据与仿真 / 导航与空间 / 基础表示与综述」归类，每篇一个独立详情节点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉语言动作模型，直接输出机器人动作 |
| WAM | World Action Model | 联合预测视频与动作的世界-动作模型 |
| GMT | General Motion Tracking | 通用全身运动跟踪 |
| RL | Reinforcement Learning | 强化学习 |
| BC | Behavior Cloning | 行为克隆 |

## 主题分组（文章阅读优先级：深读 1 / 跟进 2 / 扫读 28）

### VLA 适配、执行恢复与推理加速

- [MorphAct](../entities/paper-morphact.md)（2610.20019，文章标注「深读」）— 情境相关的低秩参数更新
- [ACRO](../entities/paper-acro.md)（2610.19677，文章标注「跟进」）— 失败后回退重试的 Critic 编排
- [Atomic-WAM](../entities/paper-atomic-wam.md)（2610.20157）— 执行反馈校准原子动作推理
- [CapPro](../entities/paper-cappro.md)（2610.19746）— 单段人类视频引导的 VLA 适配
- [ObjDream](../entities/paper-objdream.md)（2610.20223）— 物体级身份与未来状态
- [PICES](../entities/paper-pices.md)（2610.19656）— 扰动感知的动作块与执行长度选择
- [RAIN](../entities/paper-rain-adaptive-action-refinement.md)（2610.19765）— 按需精修动作块
- [GR-BC](../entities/paper-gr-bc.md)（2610.20504）— 图重加权的行为克隆
- [离散动作与连续控制](../entities/paper-discrete-actions-continuous-control.md)（2610.20429）— 离散动作头的瓶颈分析
- [Robota-Embed](../entities/paper-robota-embed.md)（2610.19495）— 统一 VLA 嵌入用于上下文学习

### 人形、移动操作与乒乓

- [PingpongDuo](../entities/paper-pingpongduo.md)（2610.20579）— 双人形乒乓双打
- [KinoGen](../entities/paper-kinogen.md)（2610.20552）— 运动学约束视频生成的人形参考
- [HiRe-MoMa](../entities/paper-hire-moma.md)（2610.20607）— 移动操作分层残差
- [Cracking Mobile Manipulation](../entities/paper-cracking-mobile-manipulation.md)（2610.19600）— 失效诊断与 agentic 系统
- [人形策略权重扩散](../entities/paper-humanoid-policy-weight-diffusion.md)（2610.20606）— 直接生成策略权重

### 灵巧操作、接触与力觉

- [HiFun](../entities/paper-hifun.md)（2610.19518）— 功能性灵巧操作分层框架
- [ForceBench](../entities/paper-forcebench.md)（2610.19604）— 腕力融合基准
- [FOMI](../entities/paper-fomi.md)（2610.19839）— 力反馈手持采集接口
- [TacClaw-Infinity](../entities/paper-tacclaw-infinity.md)（2610.20320）— 视触觉语言动作数据集
- [人类示范：重定向物体](../entities/paper-human-demos-object-retargeting.md)（2610.20351）— 跨本体示范的物体级对齐
- [触觉引导机器人学习综述](../entities/paper-tactile-guided-robot-learning-survey.md)（2610.20134）— 综述

### 数据、仿真与世界生成

- [MetaGenSim](../entities/paper-metagensim.md)（2610.19479）— 学习该仿真什么数据
- [PhySGA](../entities/paper-physga.md)（2610.20560）— 物理空间 3D 组合
- [地牢与可见性世界生成器](../entities/paper-dungeon-visibility-world-generator.md)（2610.19523）— 探索实验环境

### 导航、空间与认知

- [C-Slerp](../entities/paper-c-slerp-hyperspherical.md)（2610.20409）— 超球面表示上的规划与探索
- [HM3D-M3](../entities/paper-hm3d-m3.md)（2610.20121）— 多智能体多目标导航基准
- [目标泛化失效的策略对齐](../entities/paper-goal-misgeneralization-policy-alignment.md)（2610.19731）— 三类失效
- [Eye, Robot](../entities/paper-eye-robot-gaze.md)（2610.19635）— VLM 引导的注视动力学

### 硬件、几何与综述

- [统一连续体机器人驱动平台](../entities/paper-unified-continuum-actuation.md)（2610.19798）— 驱动平台
- [管路中心线精修](../entities/paper-pipe-centerline-refinement.md)（2610.19403）— 拓扑保持的几何优化
- [Towards Agentic Robots](../entities/paper-towards-agentic-robots-survey.md)（2610.20442）— 机器人智能体综述

## 读法提醒

- 文章的「深读 / 跟进 / 扫读」分档是公众号编辑的判断，不是本库的评级。
- 多数论文在入库日为「待发布」或「未开源」，复现前以各详情页的开源核查为准。
- 文章中部分机构与许可证信息为转述，详情页已标注未核实项。

## 关联页面

- [VLA](../methods/vla.md) — VLA 方法概览
- [MorphAct](../entities/paper-morphact.md) / [ACRO](../entities/paper-acro.md) — 本期深读与跟进的两篇

## 参考来源

- [公众号汇总归档](../../sources/blogs/wechat_embodied_station_robot_31_papers_2026-10-15.md)
- [原始抓取](../../sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)

## 推荐继续阅读

- [原文（具身智能小站）](https://mp.weixin.qq.com/s/ZfE6_UgqmHFePROtf2eE2w)
