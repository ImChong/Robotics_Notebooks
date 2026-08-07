---
type: method
tags: [robotics, motion-retargeting, humanoid, trajectory-optimization, contact-implicit, multiple-shooting, contact-rich, sim2real, caltech, depaul]
status: complete
updated: 2026-08-07
related:
  - ../entities/paper-shooting-for-contact.md
  - ../concepts/motion-retargeting.md
  - ../concepts/motion-retargeting-pipeline.md
  - ./dynaretarget-sbto-motion-retargeting.md
  - ./spider-physics-informed-dexterous-retargeting.md
  - ./motion-retargeting-gmr.md
  - ../entities/paper-hrl-stack-03-omniretarget.md
  - ../overview/hub-motion-retargeting.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/shooting_for_contact_arxiv_2608_03116.md
  - ../../sources/sites/shooting-for-contact-github-io.md
  - ../../sources/repos/shooting-for-contact.md
summary: "DSMS：用可微仿真器离散转移作多重打靶 NLP 动力学，接触隐式解析，把运动学参考转为全身动力学可行轨迹并支持任意路径约束；开源于 sesteban951/shooting-for-contact。"
---

# DSMS（接触隐式直接仿真多重打靶）

**DSMS**（Direct Simulation-based Multiple Shooting）是 [Shooting for Contact](../entities/paper-shooting-for-contact.md)（Esteban 等，arXiv:[2608.03116](https://arxiv.org/abs/2608.03116)）提出的 **动力学感知运动重定向** 核心算法：把轨迹优化转录为 NLP，动力学用可微仿真器的离散转移 \(\mathbf{F}\) 表示，从而在 **不显式建模接触约束** 的情况下获得全身可行轨迹。参考实现：[sesteban951/shooting-for-contact](https://github.com/sesteban951/shooting-for-contact)。

## 一句话定义

把 MuJoCo（或任意可微仿真器）嵌进多重打靶 NLP：接触在仿真里解决，NLP 只负责跟踪、作动与任务约束。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DSMS | Direct Simulation-based Multiple Shooting | 仿真器在环的直接多重打靶 |
| NLP | Nonlinear Program | IPOPT 求解的有限维优化问题 |
| MPC | Model Predictive Control | 高动态动作上的 receding-horizon 包装 |
| FD | Finite Difference | MuJoCo 提供一阶灵敏度的方式之一 |
| ZOH | Zero-Order Hold | 控制 spline 基的一种 |

## 为什么重要

- **接触丰富重定向的工程瓶颈** 常是「时刻表 / 互补约束难写」；DSMS 把接触丢给仿真器。
- **相对运动学 SOCP**（[OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md)）：补上 whole-body 动力学与作动极限。
- **相对采样式 SBTO**（[DynaRetarget](./dynaretarget-sbto-motion-retargeting.md)）：保留 **任意等式/不等式约束** 与稀疏 KKT 结构（IPOPT + ma57）。
- **形态无关：** 换模型即可人形/四足；开源仓已含 G1 与 Go2。

## 主要技术路线

| 阶段 | 输入 / 输出 | 作用 |
|------|-------------|------|
| **参考接入** | 运动学 / SRB / MoCap clips | 提供待精炼轨迹与可选 twist 网格 |
| **DSMS NLP** | shooting 状态 \(\mathbf{X}\) + 控制 spline \(\mathbf{U}\) | MuJoCo 细步 \(\mathbf{F}\) 作 defect；接触隐式 |
| **约束层** | 跟踪代价 + \(g,h\) | 作动限、limit-cycle 闭合、任务等式/不等式 |
| **求解模式** | one-shot 或 receding-horizon MPC | 周期步态库 vs 高动态拼接 |
| **下游 RL** | 动力学可行参考 | mjlab PPO imitation / 命令条件化 asymmetric AC |

## 核心原理

### 输入 / 输出

| | 内容 |
|--|------|
| **输入** | 运动学或降阶参考轨迹；机器人 MuJoCo 模型；可选任务/边界约束 |
| **输出** | 满足仿真动力学的全身状态–控制轨迹；或命令索引的周期步态库 |
| **求解** | `cyipopt`/IPOPT；一阶梯度 + L-BFGS |

### 机制要点

1. **多重打靶：** 区间独立积分 + defect 缝合，降低长时域开环敏感度。
2. **细步 / 粗节点：** 区间内 \(N_s\) 仿真子步解析刚性接触；决策只在粗 shooting 节点，控制 NLP 规模。
3. **接触隐式：** 无 contact force 决策变量、无 complementarity 松弛、无预设 schedule。
4. **作动双接口：** \(\mathbf{u}\) 可为力矩或 PD 目标角；正则作用在 **实现力矩** 上。

### 流程总览

```mermaid
flowchart LR
  ref[运动学/降阶参考] --> shoot[决策 X,U]
  shoot --> F["仿真器 F：接触/摩擦/冲击"]
  F --> defect[Defect 连续性]
  defect --> cost[跟踪 + 力矩正则]
  cost --> ipopt[IPOPT]
  ipopt --> feas[动力学可行轨迹]
  feas --> rl[Motion-imitation / 命令策略]
```

## 工程实践

| 步骤 | 说明 |
|------|------|
| 1. 装环境 | `make install` → `conda activate dsms`（见 [repo 归档](../../sources/repos/shooting-for-contact.md)） |
| 2. 接参考 | `trajectories/` 下 CSV；example `config.py` 注册 |
| 3. 选题型 | 周期步态用 one-shot `g1_gait`；高动态用 `*_mpc` receding-horizon |
| 4. 调权 | 状态跟踪 vs body tracking 互补；软 no-slip 勿写成硬约束（爬行需滑） |
| 5. 下游 | 论文用 mjlab PPO；仓内用 `replay.py` 先验视觉查接触 |

**调试指标：** NLP 迭代是否过 defect；回放 contact 箭头是否合理；下游 PPO exploration noise \(\sigma\) 下降速度与落地成功率。

## 局限与风险

- 接触模式切换导致曲率突变时，L-BFGS 可能需良好初值 / warm-start。
- 任意约束能力强，但 **调权与约束可行性** 仍依赖工程经验。
- 开源仓 **不含** 论文 RL 训练；对比 Table II 需自备 tracking 环境。
- 与 GMR/PHC 等 **运动学前端** 是串联关系，不是替代关系。

## 关联页面

- 论文实体：[Shooting for Contact](../entities/paper-shooting-for-contact.md)
- 采样对照：[DynaRetarget / SBTO](./dynaretarget-sbto-motion-retargeting.md)、[SPIDER](./spider-physics-informed-dexterous-retargeting.md)
- 运动学前端：[GMR](./motion-retargeting-gmr.md)、[OmniRetarget](../entities/paper-hrl-stack-03-omniretarget.md)
- 概念枢纽：[Motion Retargeting](../concepts/motion-retargeting.md)、[Hub](../overview/hub-motion-retargeting.md)

## 参考来源

- [shooting_for_contact_arxiv_2608_03116.md](../../sources/papers/shooting_for_contact_arxiv_2608_03116.md)
- [shooting-for-contact-github-io.md](../../sources/sites/shooting-for-contact-github-io.md)
- [shooting-for-contact.md](../../sources/repos/shooting-for-contact.md)

## 推荐继续阅读

- 论文：<https://arxiv.org/abs/2608.03116>
- 代码：<https://github.com/sesteban951/shooting-for-contact>
- 项目页：<https://shooting-for-contact.github.io/>
