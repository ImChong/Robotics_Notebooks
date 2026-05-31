# OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction（arXiv:2509.26633）

> 来源归档（ingest · PHP 上游依赖）

- **标题：** OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction
- **类型：** paper / humanoid motion retargeting + data generation
- **arXiv：** <https://arxiv.org/abs/2509.26633>（HTML：<https://arxiv.org/html/2509.26633v1>）
- **项目页：** <https://omniretarget.github.io/>
- **作者：** Lujie Yang*, Xiaoyu Huang*, Zhen Wu*, Angjoo Kanazawa†, Pieter Abbeel†, Carmelo Sferrazza†, C. Karen Liu†, Rocky Duan†, Guanya Shi†（* equal；† Amazon FAR co-lead）
- **机构：** Amazon FAR、MIT、UC Berkeley、Stanford、CMU
- **入库日期：** 2026-05-31
- **一句话说明：** 基于 **interaction mesh** 的交互保留重定向引擎：最小化人–机 mesh 的 **Laplacian 形变能**，在**硬约束**（无穿透、关节/速度限、脚粘附）下生成人形 loco-manipulation / 场景交互运动学轨迹，并支持单演示到多 embodiment / 地形 / 物体的**数据增广**；下游 RL 仅用 **5 项 reward** + 轻量 DR 即可在 Unitree G1 上零样本实机执行长达 **30 s** 的 parkour / 操作序列。

## 摘要级要点（与 abs 一致）

- **痛点：** 常见重定向（PHC、GMR、VideoMimic 等）易产生脚滑、穿透，且**不显式保留**人–物–地形交互关系，导致参考质量差、下游 RL 需大量 ad-hoc reward。
- **方法：** Delaunay 四面体 **interaction mesh** 连接关键关节 + 物体/环境采样点；每帧解 SOCP 序列，目标为 Laplacian 坐标差最小 + 时间平滑，约束含 SDF 非穿透、关节/速度界、stance 脚位置固定。
- **增广：** 固定源 demonstration mesh，变化目标物体位姿/形状或地形高度，重新优化得新轨迹（物体局部系建 mesh 以免整体刚体漂移）。
- **下游 RL：** 本体感知跟踪 + 极简 5 项 reward（DeepMimic 式 body/object tracking、action rate、软关节限、自碰惩罚）+ 共享 DR；**无 curriculum** 多任务。
- **规模：** OMOMO、LAFAN1、自采 MoCap → **8+ 小时**重定向数据；kinematic 指标优于 PHC/GMR/VideoMimic 等基线。

## 与 PHP 的关系（为何随 PHP 一并入库）

- **PHP [43] 明确使用 OmniRetarget** 将人类跑酷 MoCap 转为 G1 可执行**原子技能库**，再交给 motion matching 做长程合成；无高质量交互保留重定向，PHP 的稀疏技能库难以扩展为多样参考。
- 同一作者团队（Amazon FAR + 伯克利/CMU/Stanford）在 **2025 重定向数据层** 与 **2026 感知跑酷策略层** 形成衔接，适合在知识库中**互链**而非混为同一 arXiv 条目。

## 对 wiki 的映射

- 深化实体页：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)
- 交叉：[`wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md`](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)、[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)、[`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)

## 关联原始资料

- 42 篇栈策展（保留）：[`humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md`](humanoid_rl_stack_03_omniretarget_interaction_preserving_data_generat.md)
