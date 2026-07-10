---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2509.20717"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robotdancing.md
summary: "长时程、高动态的人形动作追踪之所以脆，是因为「绝对关节指令」无法补偿仿真-实机的动力学差异，误差会随时间累积。 RobotDancing 让策略不再输出绝对关节角，而是在参考轨迹之上预测残差修正量 q^tar = q^ref + a；再配合单阶段（single-stage）非对称 actor-critic PPO、统一的观测/奖励/超参，以及\"只对髋/膝 pitch 关节加残差\"的选择性残差化，就能把 LAFAN1 里三分钟一段的舞蹈（含跳跃、360° 旋转、侧手翻、冲刺）零样本部署到 Unitree G1 上。"
---

# RobotDancing

**RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：13_Physics-Based_Animation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

长时程、高动态的人形动作追踪之所以脆，是因为「绝对关节指令」无法补偿仿真-实机的动力学差异，误差会随时间累积。 RobotDancing 让策略不再输出绝对关节角，而是在参考轨迹之上预测残差修正量 q^tar = q^ref + a；再配合单阶段（single-stage）非对称 actor-critic PPO、统一的观测/奖励/超参，以及"只对髋/膝 pitch 关节加残差"的选择性残差化，就能把 LAFAN1 里三分钟一段的舞蹈（含跳跃、360° 旋转、侧手翻、冲刺）零样本部署到 Unitree G1 上。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- **动作表示**："参考 + 残差"是比"绝对目标"更适合 sim-to-real 的动作参数化——把 domain gap 变成一个可学的小修正量，值得作为追踪类任务的默认选择
- **选择性纠偏**：不必对所有关节一视同仁；识别"物理最吃紧"的少数 DoF 集中施加控制，能降方差、稳训练
- **工程简洁性**：单阶段、统一超参、无部署期后处理，是很强的可复现性/工程价值信号，与"15 分钟 sim-to-real"等极简管线思路一脉相承
- **采样课程**：长尾动作分布下，"分布均衡 + 失败优先"的采样比纯优先采样更稳，可迁移到其他长序列/多技能训练

## 核心机制

1. **残差动作表示**：把动作从绝对目标改成"参考轨迹 + 残差"，给策略一个显式补偿模型-实机差异的自由度，直击长时程误差累积；
2. **选择性残差化**：只对承重最吃紧的髋/膝 pitch 加残差，其余透传参考，兼顾纠偏能力与训练稳定；
3. **单阶段统一配置**：一套观测/奖励/超参覆盖所有舞蹈序列，无需分阶段或逐动作调参，端到端到零样本部署；
4. **多分钟高能真机验证**：LAFAN1 全 8 段在 G1 上跑通，含跳跃/旋转/侧手翻/冲刺，并迁移到 H1/H1-2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking.html> |
| arXiv | <https://arxiv.org/abs/2509.20717> |
| 机构 | 慕尼黑工业大学（TUM）等 |
| 作者 | Zhenguo Sun, Yibo Peng, Yuan Meng, Xukun Li, Bo-Sheng Huang, Zhenshan Bing, Xinlong Wang, Alois Knoll |
| 发表 | 2025-09-25（arXiv） |
| 源码 | 论文未公开代码/项目页（截至写入时未见官方仓库） |
| 笔记阅读日期 | 2026-07-08 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robotdancing.md](../../sources/papers/humanoid_pnb_robotdancing.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking.html>
- 论文：<https://arxiv.org/abs/2509.20717>

## 推荐继续阅读

- [机器人论文阅读笔记：RobotDancing](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking/RobotDancing__Residual-Action_RL_Enables_Robust_Long-Horizon_Motion_Tracking.html)
