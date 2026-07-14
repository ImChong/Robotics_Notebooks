---
type: method
tags: [bfm, teacher-student, multi-skill, humanoid, reinforcement-learning]
status: complete
updated: 2026-07-14
summary: "BFM 路线之一：用 Teacher-Student 在仿真特权信息下学习多动作/多技能跟踪，再蒸馏为可部署策略；与 BFM-Zero（FB 无监督）和 SONIC（规模化跟踪）并列。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../entities/paper-bfm-zero.md
  - ./sonic-motion-tracking.md
  - ./teacher-student-dagger-training.md
  - ../concepts/privileged-training.md
  - ../../roadmap/depth-bfm.md
sources:
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# Teacher-Student 多动作 BFM 学习

飞书 Know-How 在 **BFM 行为基础模型** 下分三线：**BFM-Zero（FB 无监督）**、**SONIC（DeepMimic 规模化跟踪）**、**基于 Teacher-Student 的多动作学习**。本页独立对应第三条：用特权教师在仿真中学会**多参考动作/多技能**跟踪，再蒸馏为单一可部署学生，作为「运控大模型」的一种工程路径。

## 一句话定义

一个身体策略覆盖多种动作库，训练靠特权教师解决多任务探索，部署靠学生只带真机传感器。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 多行为统一身体控制基座 |
| TS | Teacher-Student | 特权→可部署蒸馏 |
| RL | Reinforcement Learning | 多任务跟踪常用 PPO 族 |
| WBT | Whole-Body Tracking | 多动作统一目标形式 |
| Sim2Real | Simulation to Real | 蒸馏与域随机关键环节 |
| IL | Imitation Learning | 可与 RL 奖励混合 |

## 为什么重要

- **动作库扩展**：比单动作 DeepMimic 更接近「基础模型」叙事。
- **与 BFM-Zero 对照**：无监督 FB 表示 vs 显式多任务 TS 蒸馏，飞书并列呈现路线分歧。
- **数据筛选争议**：文档提到特权学习筛数据 vs 人工筛的工程冲突，多动作 TS 常涉此问题。

## 核心原理

1. **多参考库**：动捕/重定向轨迹集合 $\{\xi_k\}$。
2. **教师**：观测 $s_{\mathrm{priv}}$（含参考相位、接触真值、地形）→ 动作 $a$。
3. **学生**：$s_{\mathrm{deploy}}$ → $a$，蒸馏损失 + 可选 RL 微调。
4. **技能切换**：参考 ID、相位编码或 latent 条件输入。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| BFM 无监督 | [BFM-Zero](../entities/paper-bfm-zero.md) | FB 表示 |
| 规模化跟踪 | [SONIC](./sonic-motion-tracking.md) | 多动作跟踪 |
| 特权多技能 | [Privileged Training](../concepts/privileged-training.md) | 本页 TS 路线 |

## 工程实践

- 统一 **动作接口**（参考关节/根轨迹）与 **失败采样** 策略，避免多动作互相干扰。
- 评估 **未见动作泛化** 与 **单动作最优性** 的权衡。
- 交叉阅读 [SONIC](./sonic-motion-tracking.md) 规模化路线与 [BFM-Zero](../entities/paper-bfm-zero.md)。

## 局限与风险

- **蒸馏瓶颈**：学生容量不足时多技能互相干扰。
- **BFM-Zero / FB 无监督（全文）：** 动力学连续、动作「友好」，但 FB 对观测动力学未来占据取平均 → **难泛化到动力学突变**（如 in-the-air 动作）；网络容量限制覆盖度。
- **特权依赖**：真机无对应观测时需额外估计器。
- **评测不统一**：飞书指出 benchmark 缺失。

## 关联页面

- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [Teacher-Student + DAgger](./teacher-student-dagger-training.md)
- [SONIC](./sonic-motion-tracking.md)、[BFM-Zero](../entities/paper-bfm-zero.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [depth-bfm 纵深路线](../../roadmap/depth-bfm.md)
