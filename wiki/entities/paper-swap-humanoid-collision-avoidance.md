---
type: entity
project_id: swap-humanoid-collision-avoidance
tags: [paper, humanoid, whole-body-control, collision-avoidance, diffusion-policy, reinforcement-learning]
status: complete
updated: 2026-10-25
arxiv: "2610.25711"
project: "https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/"
related:
  - ../tasks/humanoid-locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../concepts/whole-body-control.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/swap_humanoid_oa_arxiv_2610_25711.md
  - ../../sources/sites/swap-steerable-humanoid-oa-website.md
summary: "SWAP（arXiv:2610.25711）用感知条件化扩散策略提出多模态全身避障动作，低层 RL 控制器负责稳定跟踪；仿真与 Unitree G1 实验覆盖静态和动态障碍，测试时引导使目标奖励平均提升 34%，代码待开放。"
---

# SWAP：测试时可控的人形全身避障

**SWAP**（*Steerable Whole-body Avoidance Policy*， [arXiv:2610.25711](https://arxiv.org/abs/2610.25711)，[项目页](https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/)；Zeji Yi、Guanqi He、Xiaofeng Guo、Guannan Qu、Pieter Abbeel、Yan Duan、Karen Liu、Guanya Shi，Carnegie Mellon University 与 Amazon FAR）把「选一条安全的全身通过动作」与「稳定执行该动作」分层：高层感知条件化扩散策略生成多种避障提案，低层全身 RL 控制器跟踪提案并维持动态平衡。

## 一句话定义

先生成多种碰撞安全的全身动作候选，再按速度、朝向或姿态等目标在测试时引导采样，交由低层控制器稳定跟踪。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SWAP | Steerable Whole-body Avoidance Policy | 论文提出的可引导全身避障策略 |
| G1 | Unitree G1 Humanoid | 论文中的真实人形实验平台 |
| RL | Reinforcement Learning | 低层控制器通过交互学习动作跟踪与平衡 |
| MC | Monte Carlo | 测试时从扩散策略采样候选动作的搜索方法 |
| WBC | Whole-Body Control | 协调全身动作与动态平衡的控制范式 |

## 为什么重要

- **避障动作具有多解：** 同一个障碍可以侧身、转向或调整姿态通过；单一确定性策略容易遗漏有效方案。
- **把动作提案与稳定性拆开：** 高层策略处理感知和避障多模态选择，低层 RL 控制器负责动态平衡和动作跟踪。
- **支持测试时目标调整：** 无需繁复重调训练奖励，即可用简单代价引导偏好的速度、朝向或姿态。

## 核心原理

### 1. 高层生成可避障动作提案

高层策略根据机器人感知条件，用扩散模型产生多个碰撞感知的全身动作候选。保留多模态候选让策略能够针对同一环境找到不同的可行穿越方式。

### 2. 低层 RL 跟踪并稳住机器人

低层 whole-body RL 控制器执行高层动作提案，同时维持动态平衡。该分层将避障决策与稳定控制的职责分开。

### 3. 通过 hindsight labeling 构造训练样本

训练时让低层控制器在随机指令下滚动执行，记录身体实际扫过的空间；只在未被动作扫过的区域放置障碍，并重新渲染机器人当时会看到的感知输入。这样构造的高层监督提案按生成方式保证无碰撞。

### 4. 测试时用简单代价引导扩散采样

部署时用 Monte Carlo 采样产生候选，再根据任务相关的简单代价选择偏好的动作方向。论文报告静态和动态障碍实验中避障率保持较高，测试时 steering 使目标奖励平均提高 **34%**；这里的 34% 是目标奖励提升，不是避障成功率提升。

## 流程总览

```mermaid
flowchart LR
  obs["感知输入"] --> high["高层扩散策略"]
  high --> proposals["多模态全身动作提案"]
  proposals --> steer["测试时 Monte Carlo 代价引导"]
  steer --> low["低层 RL 全身跟踪与平衡"]
  low --> robot["Unitree G1"]
  robot --> obs
```

## 评测与结果

- **平台：** 仿真与真实 Unitree G1。
- **环境：** 静态及动态障碍。
- **指标口径：** 测试时 steering 对速度、朝向、姿态等目标奖励平均提升 **34%**，同时论文报告保持较高碰撞规避率；摘要未给出可据此引用的统一成功率数值。

## 工程实践与开源状态

| 项 | 当前信息 |
|----|----------|
| 控制分层 | 感知条件化扩散策略提案 → 低层全身 RL 稳定跟踪 |
| 训练标签 | 随机指令 rollouts + 动作扫掠空间之外放置障碍 + 重新渲染感知 |
| 测试时引导 | Monte Carlo 候选采样 + 速度 / 朝向 / 姿态等简单代价 |
| 硬件 | Unitree G1 |
| 代码 | 截至 2026-10-21 待开放，尚无可用仓库链接 |

### 源码运行时序图

**不适用：** 截至入库日没有可用官方代码仓库或训练 / 部署入口，暂时无法基于源码描述运行时序。

## 与其他工作对比

| 路线 | 决策输出 | 平衡职责 | 测试时调整 |
|------|----------|----------|------------|
| 单一确定性策略 | 每个状态给出单一动作 | 取决于策略本身 | 通常需改变策略或奖励设置 |
| 仅低层全身控制器 | 跟踪参考或指令并稳定执行 | 由低层控制器承担 | 不负责从感知中产生多模态避障路线 |
| **SWAP** | 感知条件化扩散生成多个全身避障提案 | 低层 RL 跟踪并保持动态平衡 | Monte Carlo 采样结合简单目标代价引导 |

SWAP 将候选生成、动作稳定和目标选择分别放在不同环节；是否优于其他方案仍需按相同障碍、机器人和指标设置比较。

## 局限与风险

- 论文摘要称保持「较高避障率」，但摘要没有给出统一的碰撞率数字；引用时应以论文表格和实验设置为准。
- 测试时 Monte Carlo 搜索需要额外采样预算；实际控制周期、采样数和机载延迟应结合完整论文及实现复核。
- 低层稳定性依赖于其动作跟踪能力；将高层提案迁移到不同机器人时，仍需验证动力学、关节范围和接触条件。
- 项目页代码待开放，当前不能据论文摘要复现训练或部署管线。

## 结论

**SWAP 将人形全身避障写成「扩散策略提出多种动作、低层 RL 保持平衡、测试时按目标引导选择」的分层问题。**

- 评估时把 **34%** 读作目标奖励平均提升，不能替代碰撞率或任务成功率。
- 若要复现，优先确认高层感知输入、动作提案的时域和低层跟踪频率；摘要未提供这些工程参数。
- G1 真机与仿真覆盖静态、动态障碍，但不能直接推断到其他形态或高速人群混行。
- 代码待开放；项目页发布可运行实现后再补充运行时序图与依赖信息。

## 关联页面

- [Humanoid Locomotion](../tasks/humanoid-locomotion.md) — 人形平衡与全身移动
- [楼梯与障碍感知移动](../tasks/stair-obstacle-perceptive-locomotion.md) — 感知输入下的障碍通过任务
- [Whole-Body Control](../concepts/whole-body-control.md) — 分层全身动作执行与稳定性
- [Reinforcement Learning](../methods/reinforcement-learning.md) — 低层跟踪策略的学习范式

## 参考来源

- [SWAP 论文摘录（arXiv:2610.25711）](../../sources/papers/swap_humanoid_oa_arxiv_2610_25711.md)
- [SWAP 项目页归档](../../sources/sites/swap-steerable-humanoid-oa-website.md)

## 推荐继续阅读

- Yi et al., [SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance](https://arxiv.org/abs/2610.25711)
- [项目页与 G1 演示](https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/)
