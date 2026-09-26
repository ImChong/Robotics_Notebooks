# Morphometric Imitation（arXiv:2609.28660）

> 来源归档（ingest）

- **标题：** Morphometric Imitation: From Morphology and Contact Aware Hand Retargeting to Sim-to-Real Visuomotor Policy
- **缩写：** **Morphometric Imitation** / **MMO**（第一阶段 morphometric optimization）
- **类型：** paper / dexterous-manipulation / hand-retargeting / residual-rl / visuomotor / sim2real
- **arXiv：** <https://arxiv.org/abs/2609.28660>
- **PDF：** <https://arxiv.org/pdf/2609.28660>
- **项目页：** <https://morphometricimitation.github.io/> — 归档见 [`sources/sites/morphometricimitation-github-io.md`](../sites/morphometricimitation-github-io.md)
- **代码：** <https://github.com/tsadja/morphometric> — 归档见 [`sources/repos/morphometric.md`](../repos/morphometric.md)（截至入库日 README 写 **Code will be released soon**）
- **作者：** Tara Sadjadpour、Siming He、C.K. Wolfe、Haozhi Qi、Lea Wilken、S. Shankar Sastry、Claire Tomlin*、Jitendra Malik*（* 共同 advising）
- **机构：** 加州大学伯克利分校 EECS
- **入库日期：** 2026-09-26
- **开源状态（步骤 2.5，2026-09-26）：** GitHub 占位仓已建，**训练/推理代码待发布**；项目页链 Code → 同上仓库。

## 核心论文摘录（MVP）

### 1) 问题：自然 HOI → 多指灵巧手的形态差、动力学可行性与 sim-to-real

- **链接：** <https://arxiv.org/abs/2609.28660>
- **核心贡献：** 从 **重建的人手–物交互（HOI）** 学灵巧策略时，运动学重定向常 **丢失示范接触**、动态重定向难处理 **自然桌面动作与桌碰撞**，视觉策略还需 **零样本 sim-to-real**。Morphometric Imitation 用 **三阶段**：MMO（形态+接触感知运动学重定向）→ **残差 RL 动态重定向**（物体位姿+接触）→ **仿真 IL 蒸馏 visuomotor**，强调 **一条人类示范、任意三/四/五指手**。
- **对 wiki 的映射：**
  - [Morphometric Imitation 论文实体](../../wiki/entities/paper-morphometric-imitation.md)
  - [Manipulation 任务页](../../wiki/tasks/manipulation.md)

### 2) MMO：先 morph 人手再恢复接触，相对向量类基线保 contact F1

- **核心贡献：** **Morphometric optimization** 优化 MANO 形态以对齐目标机器人手，再 **恢复** 形态变换后的 **手–物接触**；相对 DexPilot / AnyTeleop / Position / Contact-Aware PyRoki / OmniRetarget 等 **五个运动学基线**，在 **三种机器人手 × 十条 GRAB HOI** 上 contact **F1 至少 +8 pt**（相对最强基线），并降低 patch distance（论文 Table II）。
- **对 wiki 的映射：**
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)（灵巧 HOI 子链路）
  - [DemoMimic](../../wiki/entities/paper-demomimic.md)（同属单次示范 + 接触中心下游 IL 对照）

### 3) 残差 RL + visuomotor：动态参考 → 89.3% 真机零样本

- **核心贡献：** 残差策略在观测/奖励/终止中同时使用 **物体位姿与接触信息**，把 MMO 运动学参考变为 **动力学可行、避桌碰撞** 的机器人示范；再蒸馏 **visuomotor** 策略。动态重定向相对最强基线 **任务成功率最高 +35 pt**（Table III）；**300 次真机试验、30 物体、10 类、随机初始位姿** 报告 **89.3%** zero-shot 成功率（项目页 TLDR）。
- **对 wiki 的映射：**
  - [Morphometric Imitation 论文实体](../../wiki/entities/paper-morphometric-imitation.md)
  - [DemoMimic](../../wiki/entities/paper-demomimic.md)（Table I 对照：单次示范 + zero-shot sim-to-real visuomotor）
