# RoboNaldo（arXiv:2606.11092）

> 来源归档（ingest）

- **标题：** RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting via Motion-Guided Curriculum Reinforcement Learning
- **缩写：** **RoboNaldo**
- **类型：** paper / humanoid soccer / high-impulse interaction / curriculum RL
- **arXiv：** <https://arxiv.org/abs/2606.11092>
- **PDF：** <https://arxiv.org/pdf/2606.11092>
- **项目页：** <https://opendrivelab.com/RoboNaldo/>
- **作者：** Yichao Zhong†, Yidan Lu†, Yuhang Lu, Tianyang Tang, Haoguang Mai, Yixuan Pan, Tianyu Li‡, Li Chen‡, Jingbo Wang, Zhongyu Li, Peng Lu‡, Hongyang Li‡（† 共同一作；‡ 共同指导）
- **机构：** 香港大学；香港中文大学；Archon Robotics
- **入库日期：** 2026-06-10
- **一句话说明：** 以 **单条人类踢球参考** 为 scaffold 的 **三阶段 motion-guided curriculum RL**：先学稳定全身踢球先验，再适应任意球位定点射门，最后经 **locomotion 命令 + kick-trigger 接口** 泛化到 **来球射门**；仿真误差较基线 **−48.6%**、球速 **2.96×**；**Unitree G1** 机载 LiDAR/IR 感知在真草场外实现 **3 m 平均误差 0.73 m（任意球）/ 0.86 m（来球）**、触球后球速最高 **13.10 m/s**。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2606.11092>
- **摘录要点：** 精英人形射门同时要求 **全身稳定**、**毫秒级高冲量足–球接触** 与 **点级瞄准**；纯 motion tracking 给协调但 **固定参考** 难适应球位与触球时机，纯 task RL 又难从零探索有效踢球。RoboNaldo 用 **三阶段课程** 把「稳定先验 → 瞄准适应 → 移动球时机」拆开；训练期由 **启发式高层规划器** 驱动 locomotion/kick 接口，推理期可换其他高层控制器。论文主张在 **点级精度、球速、来球射门、机载感知、室外部署** 五维上相对既有 humanoid soccer 系统 **首次联合达标**（Table 1）。
- **对 wiki 的映射：**
  - [RoboNaldo 论文实体](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md)
  - [Humanoid Soccer](../../wiki/tasks/humanoid-soccer.md)

### 2) 三阶段课程与观测接口（§3.1–3.2）

- **链接：** <https://arxiv.org/abs/2606.11092> §3
- **摘录要点：**
  - **Stage 1 · Shooting Motion Tracking：** 侧脚踢球参考经 **GVHMR + GMR** 重定向；仿 **BeyondMimic** 纯跟踪，无球/任务奖励，建立平衡与摆腿结构。
  - **Stage 2 · Shooting Adaptation：** 引入球、目标与射门奖励；随机化球 spawn，策略须 **偏离参考** 调整触球点、方向与冲量——输出 **任意球策略** 并初始化 Stage 3。
  - **Stage 3 · Task Generalization：** 来球射门需 **接近 + 触球时机**；用 **locomotion command** 与 **kick-trigger** 将 episode 分为接近 / 踢球 / 稳定三模式；训练期 **启发式规划器** 预测球位、对齐朝向并在最近接近距离阈值触发踢球；**proximity-based tracking relaxation** 在触球附近放松 motion tracking 权重（脚速项 $\mu{=}0.05$）。
- **对 wiki 的映射：**
  - [RoboNaldo 论文实体](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — Mermaid 三阶段流程；BeyondMimic / GMR 见实体页交叉引用

### 3) 奖励与仿真训练（§3.3 / Appendix）

- **摘录要点：**
  - **Instant Interaction Reward** 与 **Densified Shooting Reward**：针对 **<10 ms 冲量** 与 **延迟球–目标反馈**，外推触球后球状态以密集监督。
  - **PPO（RSL-RL）**：4096 并行环境、Isaac Lab / PhysX GPU；50 Hz 控制；每阶段最多 $10^5$ iter；actor/critic 为 $512{\to}256{\to}128$ MLP。
  - **域随机化：** 摩擦、关节偏置、CoM、执行延迟、随机推力；球材质 restitution 0.95。
- **对 wiki 的映射：**
  - [RoboNaldo 论文实体](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 奖励设计归纳见实体页

### 4) 真机部署与感知（Appendix D）

- **摘录要点：**
  - **平台：** Unitree G1，29-DoF；策略 **50 Hz ONNX** 推理；**关节索引 BFS↔DFS 置换** 必须与 SDK/MuJoCo 对齐。
  - **近距球：** 头部 **Livox MID-360** LiDAR，利用 retro-reflective 球 **反射率 + 球体拟合 + Kalman**；**远距：** D435 **IR 灰度** 亮斑分割（优于 RGB YOLO/HSV 在快球模糊下的表现）。
  - **目标：** RealSense D435 **AprilTag** 或固定坐标。
  - **室外真草：** 任意球与来球射门；最佳单次 **17 cm @ 3 m**、**13.10 m/s** 触球后球速（约职业男足开球射门速度的 **59%**、女足 **71%** 叙述）。
- **对 wiki 的映射：**
  - [RoboNaldo 论文实体](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 真机感知与 Sim2Real 见实体页

### 5) 实验与相对基线（§1 / Table 1 / Appendix E）

- **仿真（Stage 2 任意球，5 m）：** 平均误差 **0.899 m**；**65.5%** 射门误差 <1 m；球速 **14.79 m/s**；相对 prior work **误差约 0.5×、速度 2.96×**。
- **仿真（Stage 3 来球）：** **63.3%** 射门 <1 m 误差。
- **真机（3 m）：** 任意球平均 **0.73 m**；来球 **0.86 m**；来球试验 **74%** 有效触球。
- **谱系对照：** 相对 **PAiD†、HumanX†、Reactive、Striker** 等，RoboNaldo 在公开 Table 1 中唯一同时勾选 **点级精度 / 球速 / 来球 / 自中心感知 / 室外**。
- **对 wiki 的映射：**
  - [RoboNaldo 论文实体](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 与 PAiD 对照见实体页与 [PAiD Framework](../../wiki/methods/paid-framework.md)

## 对 wiki 的映射（汇总）

- [paper-robonaldo-humanoid-soccer-shooting.md](../../wiki/entities/paper-robonaldo-humanoid-soccer-shooting.md) — 主沉淀页
- 交叉更新：[humanoid-soccer.md](../../wiki/tasks/humanoid-soccer.md)

## 引用（项目页 BibTeX）

```bibtex
@article{zhong2026robonaldo,
  title   = {RoboNaldo: Accurate, Stable and Powerful Humanoid Soccer Shooting
             via Motion-Guided Curriculum Reinforcement Learning},
  author  = {Zhong, Yichao and Lu, Yidan and Lu, Yuhang and Tang, Tianyang
             and Mai, Haoguang and Pan, Yixuan and Li, Tianyu and Chen, Li
             and Wang, Jingbo and Li, Zhongyu and Lu, Peng and Li, Hongyang},
  note    = {Under review},
  year    = {2026}
}
```
