---
type: method
tags: [locomotion, perception, footstep-planning, attention, reinforcement-learning]
status: complete
updated: 2026-07-14
summary: "Attention 落足点优化：用注意力机制在机器人中心高程图或候选落脚点集合上选择可行走区域，是飞书 Know-How 中感知 loco 与落足规划交叉模块；站内代表实现含 AME 等。"
related:
  - ../entities/paper-ame-attention-based-map-encoding.md
  - ../concepts/footstep-planning.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ./pie-perceptive-locomotion.md
  - ../../roadmap/depth-perceptive-locomotion.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# Attention 落足点优化

飞书 Know-How 中的 **Attention 落足点优化算法** 指：在**感知地形表示**（高程图、垫脚石候选、射线高度采样等）上用 **注意力机制** 选择下一步落脚点或 foothold 权重，使策略在稀疏/不规则支撑面上稳定行走。本仓库将这一**方法主题**独立成节点；具体论文实现见 [AME](../entities/paper-ame-attention-based-map-encoding.md) 等。

## 一句话定义

让网络「看向」地形图上真正可踩的区域，而不是均匀处理整张高程图或手工固定采样点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AME | Attention-based Map Encoding | 2.5D 高程图 + CNN + 多头注意力的代表工作 |
| RL | Reinforcement Learning | 常与 PPO 两阶段或单阶段感知 loco 结合 |
| MPC | Model Predictive Control | 模型派落足规划对照 |
| ZMP | Zero Moment Point | 模型派平衡约束，与 RL 落足可串联 |
| WBC | Whole-Body Control | 跟踪落足参考的底层执行器 |
| Locomotion | Robot Locomotion | 本方法主要服务足式/人形行走 |

## 为什么重要

- **稀疏地形**：梁、踏脚石、沟槽等需要**选择性关注**可支撑区域。
- **可解释性**：注意力权重可视化 foothold，比黑箱端到端更易调试。
- **飞书课程位置**：与 DreamWaQ、PIE 并列，构成感知 loco 三条线之一。

## 核心原理

典型管线：

1. **构建机器人中心地图**（2.5D elevation / 局部点云投影）。
2. **CNN + Multi-Head Attention** 编码为策略可用特征（AME 路线）。
3. **策略输出** 足端目标或残差，底层 PD/WBC/全身跟踪执行。
4. **训练** 常配合特权 critic、课程地形与落足奖励（接触、防滑、姿态）。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| 感知编码 | [AME 论文](../entities/paper-ame-attention-based-map-encoding.md) | CNN + 多头注意力高程图 |
| 落足规划 | [Footstep Planning](../concepts/footstep-planning.md) | 模型派落脚点优化对照 |
| 纵深路线 | [感知越障纵深](../../roadmap/depth-perceptive-locomotion.md) | Stage 1–3 感知 loco 学习顺序 |

## 工程实践

- 对比 **均匀池化高程图** vs **注意力编码** 在相同地形集上的样本效率。
- 记录注意力热图与失败案例（踩空、边缘踩偏）。
- 人形注意双足交替与支撑多边形约束，勿照搬四足单图编码尺寸。

## 局限与风险

- **地图滞后与漂移**：定位/状态估计误差直接污染注意力输入。
- **与 PIE 边界**：PIE 偏端到端跑酷估计器；Attention 落足偏**可解释 foothold 选择**，可组合而非互斥。
- **计算预算**：高分辨率地图 + 注意力增加推理延迟。

## 关联页面

- [AME 论文实体](../entities/paper-ame-attention-based-map-encoding.md)
- [感知越障纵深路线](../../roadmap/depth-perceptive-locomotion.md)
- [PIE](./pie-perceptive-locomotion.md)、[DreamWaQ](./dreamwaq.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [AME arXiv 论文页](../entities/paper-ame-attention-based-map-encoding.md) 内链项目与代码
