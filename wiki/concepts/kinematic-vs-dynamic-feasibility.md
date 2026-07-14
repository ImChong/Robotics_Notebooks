---
type: concept
tags: [humanoid, retargeting, feasibility, dynamics, kinematics]
status: complete
updated: 2026-07-14
summary: "运动学可行（姿态/轨迹几何可达）不等于动力学可行（力矩、摩擦、接触与稳定可执行）；人形重定向与跟踪失败常源于混淆二者。"
related:
  - ./humanoid-vs-other-robots.md
  - ../concepts/motion-retargeting.md
  - ../methods/dynaretarget-sbto-motion-retargeting.md
  - ../queries/humanoid-motion-control-know-how.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
sources:
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
---

# 运动学可行与动力学可行

飞书 Know-How 将 **运动学可行和动力学可行** 列为控制问题框架的核心分叉：前者回答「关节角度/末端轨迹是否存在」，后者回答「在给定力矩、摩擦与接触下是否稳定可执行」。

## 一句话定义

能摆出这个姿势（运动学）不代表在真实物理里站得住、跟得上（动力学）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IK | Inverse Kinematics | 运动学可达性常用工具 |
| ID | Inverse Dynamics | 动力学可行性所需力矩分析 |
| WBC | Whole-Body Control | 在动力学约束下协调任务 |
| ZMP | Zero Moment Point | 动力学平衡判据之一 |
| RL | Reinforcement Learning | 可在仿真中隐式学动力学可行策略 |
| GMR | General Motion Retargeting | 常见只保证运动学，需后处理动力学 |

## 为什么重要

- **重定向管线**：人体 MoCap → 机器人参考常先运动学匹配，需 SBTO/跟踪器补动力学。
- **模仿学习筛选**：飞书提到人工筛 vs 特权筛数据；动力学不可行片段会污染训练。
- **演示误导**：仿真里 kinematic replay 看起来「像」，真机穿透或跌倒。

## 核心原理

| | 运动学可行 | 动力学可行 |
|---|-----------|-----------|
| 关心量 | 关节限位、自碰撞、末端轨迹 | 力矩限、摩擦锥、接触力、稳定性 |
| 典型工具 | IK、重定向优化 | 逆动力学、QP/WBC、仿真 rollout |
| 失败现象 | 无解/奇异 | 打滑、饱和、跌倒、穿透 |

## 工程实践

- 重定向后加 **动力学精炼**（见 [DynaRetarget/SBTO](../methods/dynaretarget-sbto-motion-retargeting.md)）。
- 跟踪策略训练前统计 **力矩使用率、ZMP/支撑多边形、足端滑移**。
- 区分「retarget 成功」与「policy 可跟踪」验收标准。

## 局限与风险

- **作者经验链：** 重定向改进（PHC → GMR 硬 IK → OmniRetarget SQP）越好，后续 RL **越省奖励调参**；最终效果 **人形本体 > 重定向轨迹 > RL 算法**。
- **完全动力学验证昂贵**：高 DoF 人形全轨迹 ID+仿真仍耗时。

## 关联页面

- [Motion Retargeting](./motion-retargeting.md)
- [人形 vs 其他机器人](./humanoid-vs-other-robots.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)

## 推荐继续阅读

- [DynaRetarget 论文实体](../entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)
