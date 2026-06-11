# Perceptive BFM（CoRL 2026 submission）

> 来源归档（ingest）

- **标题：** Perceptive Behavior Foundation Model: Adapting Human Motion Priors to Robot-Centric Terrain
- **缩写：** **Perceptive BFM** / **PMT**（Perceptive Motion Tracking）
- **类型：** paper / humanoid / behavior-foundation-model / perceptive-locomotion / motion-tracking
- **项目页：** <https://acodedog.github.io/perceptive-bfm/>
- **arXiv：** TBA（截至 2026-06-11）
- **PDF：** 项目页提供 submission PDF 链接（CoRL 2026 under review）
- **代码：** TBA
- **作者：** Wang, Zifan; Li, Yizhao; Ma, Teli; Zhang, Qiang; Fan, Yudong; Xu, Hao; Yang, Shuo; Liang, Junwei
- **机构：** 妙动科技；香港科技大学（广州）；香港科技大学；中国科学技术大学人工智能研究院
- **会议：** CoRL 2026（submission，under review）
- **入库日期：** 2026-06-11
- **一句话说明：** 在 **保留原始人体运动参考为部署命令** 的前提下，用 **机器人中心地形感知** 把人类运动先验落地到楼梯、坡、稀疏支撑与户外真机；**TCRS** 离线合成地形一致监督，**PMT** 经盲 teacher、视觉 student、目标帧对齐蒸馏与 identity-gated 残差微调，在 **Unitree G1** 上单策略覆盖跟踪、风格动作、杂技与 mocap 遥操作。

## 核心论文摘录（MVP）

### 1) 问题：Operator–Environment Mismatch（Abstract / 项目页）

- **核心贡献：** 人形 BFM 从广泛人体运动先验学可复用全身策略，但既有 motion-centric foundation policy 假设参考已与机器人周围 **物理兼容**；当演示者/操作员与机器人 **环境分离** 时，参考只给 **行为意图**，不给 **落脚、间隙、身体高度、接触时序**。
- **对 wiki 的映射：**
  - [Perceptive BFM 实体](../../wiki/entities/paper-perceptive-bfm.md)
  - [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md) — goal-conditioned 线补 **地形感知** 维度
  - [楼梯与障碍 Locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)

### 2) TCRS：Terrain-Conformal Reference Synthesis（§Method Stage 1）

- **核心贡献：** 离线将 **原始人体 locomotion 片段 + 采样高程场** 转为地形一致监督：接触感知落脚构造、足几何摆动优化（mid-foot 帧 **MPPI**）、支撑感知根重建、碰撞修复、多点 IK。相对 Z-offset 基线，足-地形穿透深度 **5.48 → 2.38 cm**（−56.6%），间隙违规 −48.3%。**TCRS 仅训练期使用，部署永不查询。**
- **对 wiki 的映射：**
  - [Privileged Training](../../wiki/concepts/privileged-training.md) — adapted 参考作 teacher 监督
  - [Footstep Planning](../../wiki/concepts/footstep-planning.md) — 落脚与摆动几何修正

### 3) PMT 训练栈：盲 teacher → 视觉 student → 对齐蒸馏（§Method Stage 2–4）

- **核心贡献：** (1) 盲 Transformer teacher 用 PPO 跟踪 TCRS 参考（特权状态、无感知）；(2) 部署 student 接收 **raw 参考 + 机器人中心高程图**，Transformer 骨干 + MapTransformer 地形支路；(3) **Target-frame action alignment** 蒸馏：`a* = (q_reftcrs + μ_tea) − q_refraw`，使 teacher 在 adapted 帧、student 在 raw 帧下仍可对齐；(4) **Identity-gated** 残差通路 `tanh(α)` 初值≈0，策略初始等同纯 raw-reference tracker，仅在地形需要时学修正。全 PMT vs 无视觉：**54.6 vs 3.6** reward；去蒸馏 **54.6 → 50.1**。
- **对 wiki 的映射：**
  - [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) — WBT + 感知修正扩展
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) — PPO 跟踪训练

### 4) 实验与真机（项目页 / 摘要）

- **仿真：** 受控环境训练消融；Transformer 骨干相对 MLP/GRU/CNN **+5–8** reward（同算力预算 **48×A800**）。
- **真机（Unitree G1）：** 单 raw-reference 策略在楼梯、坡、稀疏支撑、凹障、草地、室内外不规则面执行 locomotion、风格动作、杂技（台阶后空翻等）、mocap 遥操作；**无 per-skill 调参、无测试期 TCRS**。
- **局限：** TCRS 为运动学合成、静态刚体高程假设；足部中心适应保留上身命令 → 上肢碰撞风险。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BFM](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) — 无感知跟踪 foundation 对照

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-perceptive-bfm.md`](../../wiki/entities/paper-perceptive-bfm.md)
- 任务索引：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)
- 站点归档：[`sources/sites/perceptive-bfm-github-io.md`](../sites/perceptive-bfm-github-io.md)
