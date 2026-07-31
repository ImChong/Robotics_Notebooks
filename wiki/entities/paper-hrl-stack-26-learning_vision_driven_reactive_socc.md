---
type: entity
tags:
  - paper
  - humanoid
  - soccer
  - rl
  - amp
  - perception
  - motion-control
  - body-system-stack
  - tsinghua
  - bytedance
  - cau
  - booster
status: complete
updated: 2026-07-28
arxiv: "2511.03996"
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/humanoid-amp-motion-prior-survey.md
  - ../tasks/humanoid-soccer.md
  - ../methods/amp-reward.md
  - ../methods/reinforcement-learning.md
  - ./paper-notebook-learning-soccer-skills-for-humanoid-robots.md
  - ./paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md
  - ./paper-robonaldo-humanoid-soccer-shooting.md
  - ../../roadmap/depth-humanoid-soccer.md
sources:
  - ../../sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md
  - ../../sources/sites/humanoid-kick-vision-driven-soccer.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md
summary: "Vision-Driven Reactive Soccer（arXiv:2511.03996）：AMP 扩展到真实动态感知；encoder-decoder + 虚拟感知恢复特权态；主动看球；Booster 平台 RoboCup 类场景；项目页暂无 Code 链。"
---

# Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots

**Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots**（[arXiv:2511.03996](https://arxiv.org/abs/2511.03996)，[项目页](https://humanoid-kick.github.io)）由 **清华大学 · 字节跳动 Seed · 中国农业大学** 提出：用统一 RL 控制器把 **视觉感知与运动控制** 直接耦合，将 [AMP](../methods/amp-reward.md) 扩展到真实动态环境，并通过 **虚拟感知系统 + encoder-decoder** 从不完美观测恢复球位等特权态，使「看球」成为策略一部分。收录于具身智能研究室 **42 篇 humanoid RL 运动控制** 长文 **第 26/42**（03 感知式高动态运动），并作为本库 [人形足球纵深 Stage 2](../../roadmap/depth-humanoid-soccer.md) 推荐读物。

## 一句话定义

**别把检测和踢球拆成慢半拍的模块流水线——用虚拟感知对齐真机视觉误差，让策略在 AMP 运动先验下直接从部分观测做出反应式踢球，并主动把球留在视野里。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | PPO 训练视觉–运动统一策略 |
| PPO | Proximal Policy Optimization | 策略优化；多 critic 估值 |
| AMP | Adversarial Motion Prior | 判别器约束状态转移接近专家运动 |
| Sim2Real | Simulation to Real | 虚拟感知建模真实视觉特性后迁移 |
| YOLO | You Only Look Once | 部署侧常用球检测族（项目页强调 onboard camera） |

## 为什么重要

- **针对解耦栈的延迟与行为不连贯：** 足球是紧耦合感知–动作环；模块化易「看见了但踢晚了」。
- **AMP 进真实动态感知：** 把运动模仿先验接到含视觉误差的闭环，而不是只在特权本体上玩风格。
- **主动感知：** 躯干/头部/身体调整成为动作的一部分，服务「把球留在更好视野」。
- **纵深 Stage 2 入口：** 帮助理解机载视觉如何进入反应式技能，再衔接到 Stage 3 的 PAiD / RoboNaldo / Agile Striker。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 清华大学；字节跳动 Seed；中国农业大学 |
| **作者** | Yushi Wang、Changsheng Luo、Penghui Chen 等 |
| **平台** | 致谢 Booster Robotics 提供机器人与场地 |
| **编号** | 42 篇栈 **26/42** · 03 感知式高动态运动 |
| **开源** | **项目页暂无 Code 链**（截至 **2026-07-28**）：[humanoid-kick.github.io](https://humanoid-kick.github.io) 仅 Paper/arXiv；**勿默认可复现训练代码** |

## 流程总览

```mermaid
flowchart TB
  subgraph train [训练]
    virt["虚拟感知系统<br/>建模真实视觉误差"]
    enc["Encoder-Decoder<br/>历史 → 恢复特权态"]
    amp["PPO + AMP 判别器<br/>+ 多 critic"]
    virt --> enc --> amp
  end
  subgraph deploy [部署]
    cam["机载相机球检测"]
    odom["里程计估门位"]
    pi["策略 → 反应式踢球"]
    cam --> pi
    odom --> pi
  end
  amp --> deploy
```

## 核心机制（方法栈）

### 1）虚拟感知 + 状态恢复

- 训练时用虚拟感知模拟真实视觉特性；actor 拿部分观测，经 encoder-decoder 从历史重建更完整状态（含球位等）。
- 目的：缩小「干净仿真观测」与「抖动检测」之间的错位。

### 2）AMP 风格的感知–运动联合

- 回报来自环境任务项 + 运动先验判别器；多 critic 提供价值估计。
- 相对纯手工奖励踢球，先验帮助保持连贯全身运动。

### 3）主动感知

- 策略显式协调躯干/头部等，使球处于更有利视野——「看」与「踢」同一优化。

### 4）部署接口（项目页）

- 机载相机检测球位直接进策略；门位由里程计模块从长期信息估计。

## 源码运行时序图

**不适用（截至 2026-07-28）。** 项目页未提供 GitHub/训练入口；仅有 arXiv PDF。若后续发布代码，应补 `sources/repos/` 与本图。

## 与其他工作对比

| 维度 | 本文 | PAiD | Agile Striker |
|------|------|------|---------------|
| **感知切口** | 虚拟感知 + 主动看球 | 骨盆系球/门 + LSTM | 显式噪声/延迟/丢帧模型 |
| **运动先验** | **AMP** | MoCap tracking | 行走改造奖励 |
| **开源** | 暂无 Code | TeleHuman 已开源 | Daffan 已开源 |
| **纵深位置** | Stage 2 感知进技能 | Stage 3 主线 | Stage 3 主线 |

## 实验与评测

- 项目页展示室外、RoboCup 比赛、连续表现与多方向敏捷行为；**具体 SR/球速表以 PDF 为准**。
- 本页在 42 篇栈策展 + 项目页方法摘要上归纳；量化对比请回 arXiv:2511.03996。

## 结论

**反应式人形足球的关键是把视觉误差与主动看球写进同一 RL 环，而不是在检测模块后再挂一个开环踢球器。**

1. **虚拟感知是 Sim2Real 对齐器** — 先模拟真实检测特性，再谈策略鲁棒。
2. **AMP 管风格与连贯** — 在动态球场景保持全身运动不「散架」。
3. **主动感知是一等公民** — 头/躯干动作服务视野，而不只是副作用。
4. **选型** — 读感知进环与 AMP 结合 → 本文；要可复现 G1/T1 踢球课程 → PAiD / Agile Striker / RoboNaldo。
5. **工程预期** — 当前以论文与演示为主，缺官方训练仓时不要排复现里程碑。

## 局限与风险

- **无公开训练代码**（项目页核查）；复现成本高。
- 策展来源含公众号摘要，细节以 PDF / 项目页为准，避免二次转述误差。
- 与联赛战术栈正交：本文聚焦单机反应式技能。

## 与其他页面的关系

- 总框架：[人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md)
- AMP：[amp-reward](../methods/amp-reward.md) · [AMP survey](../overview/humanoid-amp-motion-prior-survey.md)
- 任务 / 纵深：[Humanoid Soccer](../tasks/humanoid-soccer.md) · [depth-humanoid-soccer](../../roadmap/depth-humanoid-soccer.md)
- Stage 3 对照：[PAiD](./paper-notebook-learning-soccer-skills-for-humanoid-robots.md) · [Agile Striker](./paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md)

## 参考来源

- [humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md](../../sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md)
- [humanoid-kick-vision-driven-soccer.md](../../sources/sites/humanoid-kick-vision-driven-soccer.md)
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md)
- [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
- 论文：<https://arxiv.org/abs/2511.03996>
- 项目页：<https://humanoid-kick.github.io>

## 推荐继续阅读

- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
- [人形足球纵深路线](../../roadmap/depth-humanoid-soccer.md)
