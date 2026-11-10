---
type: concept
tags: [humanoid, retargeting, feasibility, dynamics, kinematics]
status: complete
updated: 2026-11-07
summary: "运动学可行（姿态/轨迹几何可达）不等于动力学可行（力矩、摩擦、接触与稳定可执行）；人形重定向与跟踪失败常源于混淆二者。"
related:
  - ./humanoid-vs-other-robots.md
  - ../concepts/motion-retargeting.md
  - ../methods/dynaretarget-sbto-motion-retargeting.md
  - ../overview/humanoid-motion-control-know-how-technology-map.md
sources:
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/dynaretarget_arxiv_2602_06827.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
  - ../../sources/raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md
  - ../../sources/papers/humanoid_motion_control_know_how.md
original:
  - ../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md
  - ../../sources/papers/dynaretarget_arxiv_2602_06827.md
  - ../../sources/papers/omniretarget_arxiv_2509_26633.md
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

- **重定向管线**：人体 MoCap → 机器人参考常先运动学匹配，需 SBTO/跟踪器补动力学。DynaRetarget 论文指出，主流重定向（PHC / GMR 等）解的是运动学问题，在 loco-manipulation 中易出现物理与几何不一致；它用采样式轨迹优化（SBTO）把运动学参考精炼为动力学可行轨迹，在 285 段 OmniRetarget 数据上成功率约为同类采样式 MPC（SPIDER）的两倍。
- **伪影会传导到策略**：GMR 的受控实验（BeyondMimic，几乎不调奖励，LAFAN1 21 段，Unitree G1）显示，地面穿透、自穿透和关节值突变这三类运动学伪影显著降低策略鲁棒性；OmniRetarget 则在伪影少的参考上只用 5 项奖励 + 4 项域随机化就零样本上真机。
- **模仿学习筛选**：飞书提到人工筛 vs 特权筛数据；动力学不可行片段会污染训练（飞书作者经验，未被上述论文逐条验证）。
- **演示误导**：仿真里 kinematic replay 看起来「像」，真机穿透或跌倒。

## 核心原理

| | 运动学可行 | 动力学可行 |
|---|-----------|-----------|
| 关心量 | 关节限位、自碰撞、末端轨迹 | 力矩限、摩擦锥、接触力、稳定性 |
| 典型工具 | IK、重定向优化 | 逆动力学、QP/WBC、仿真 rollout |
| 失败现象 | 无解/奇异 | 打滑、饱和、跌倒、穿透 |

> **补充（原文核实）**：运动学重定向的"可行"也只是**几何意义**上的——GMR 的目标是关键刚体位置 / 朝向误差 + 关节限位，没有力矩、摩擦或接触力项；OmniRetarget 在运动学层额外加入了 SDF 非穿透、关节 / 速度界、stance 脚不滑动等硬约束，但仍属运动学，作者自述约束线性化会带来偶发小穿透，由下游 RL 修复；DynaRetarget 用仿真器 rollout 的采样优化把"缺失接触、穿透、不连续"变成动力学一致的轨迹，并发现参考中手-物接触突变会让它失败。

## 工程实践

- 重定向后加 **动力学精炼**（见 [DynaRetarget/SBTO](../methods/dynaretarget-sbto-motion-retargeting.md)）：SBTO 逐步增加优化结点数（0.25 s 一个 PD 目标结点，仿真步长 0.01 s），约 20 秒算力 / 每秒运动（112 核 CPU）。
- 先用**运动学指标**把关：穿透（时间占比 + 最大深度）、脚滑（stance 脚速度）、接触保持（OmniRetarget 的口径）；关节单步跳变、自碰帧占比、关节极限帧占比（NMR 的口径）。
- 跟踪策略训练前统计 **力矩使用率、ZMP/支撑多边形、足端滑移**（经验清单，推测；以上三篇论文未采用这套指标）。
- 区分「retarget 成功」与「policy 可跟踪」验收标准：GMR 的成功判据要求策略完整走完整段参考；DynaRetarget 的重定向成功判据是物体平均位置误差 <10 cm 且旋转误差 <25°。
- **起始帧很关键**：GMR 论文发现同一策略仅改变参考动作起始帧，sim2sim 成功率就能大幅变化；建议起始位姿可安全到达、结束于稳定姿态。

## 局限与风险

- **作者经验链（飞书，二手）：** 重定向改进（PHC → GMR 硬 IK → OmniRetarget SQP）越好，后续 RL **越省奖励调参**；最终效果 **人形本体 > 重定向轨迹 > RL 算法**。前半句有论文支持（GMR 受控实验、OmniRetarget 的最小奖励配方）；"人形本体 > 重定向轨迹 > RL 算法"的排序属作者经验，上述论文未做对照实验。
- **完全动力学验证昂贵**：高 DoF 人形全轨迹 ID+仿真仍耗时；DynaRetarget 的精炼本身就要约 20 秒算力 / 每秒运动，并且对低质量参考（手-物接触突变、物体朝向翻转）会失败。
- **证据范围有限**：GMR 的实验只用 LAFAN1 与 Unitree G1；DynaRetarget 只在 G1 与箱子交互的数据上评测。

## 关联页面

- [AgenticGenPlan：代码代理泛化 TAMP](../entities/paper-agenticgenplan-generalized-tamp-coding-agents.md) — 在同时含运动学和动力学约束的仿真任务中，代码代理合成可跨实例复用的程序；并非任意跨引擎迁移

- [Motion Retargeting](./motion-retargeting.md)
- [人形 vs 其他机器人](./humanoid-vs-other-robots.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)
- [物理保真度与 Sim2Real 差距](./physics-fidelity-sim2real-gap.md) — 动力学可行性在仿真里能否复现，取决于物理保真度
- [仿真物理保真度选型](../queries/simulation-physics-fidelity.md) — 保真度层级与选型闭环

## 参考来源

- [GMR（Retargeting Matters）通读归档](../../sources/papers/gmr_retargeting_matters_arxiv_2510_02252.md) — arXiv:2510.02252
- [DynaRetarget 通读归档](../../sources/papers/dynaretarget_arxiv_2602_06827.md) — arXiv:2602.06827
- [OmniRetarget 通读归档](../../sources/papers/omniretarget_arxiv_2509_26633.md) — arXiv:2509.26633
- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md) — 飞书 Know-How 整理（二手，未列入 original）

## 推荐继续阅读

- [DynaRetarget 论文实体](../entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)
- Dhédin et al., *DynaRetarget*：<https://arxiv.org/abs/2602.06827>
- Araujo et al., *Retargeting Matters*：<https://arxiv.org/abs/2510.02252>
