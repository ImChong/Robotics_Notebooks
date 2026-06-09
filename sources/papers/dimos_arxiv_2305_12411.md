# DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes（arXiv:2305.12411）

> 来源归档（ingest）

- **标题：** DIMOS: Synthesizing Diverse Human Motions in 3D Indoor Scenes
- **类型：** paper / human-scene interaction / character animation / reinforcement learning
- **arXiv abs：** <https://arxiv.org/abs/2305.12411>
- **会议：** ICCV 2023
- **作者：** Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, Siyu Tang（ETH Zürich / Google）
- **项目页：** <https://zkf1997.github.io/DIMOS/>
- **代码：** <https://github.com/zkf1997/DIMOS>
- **入库日期：** 2026-06-09
- **一句话说明：** 用 **RL + 生成式运动基元潜空间** 让人体在复杂室内 3D 场景中自主导航、坐下/躺下并与家具交互，无需成对的「人体运动–场景」监督数据。

## 摘要级要点

- **问题：** 现有 human-scene interaction 合成多依赖**同步采集的人体运动与 3D 场景**配对数据，成本高、覆盖窄，难以泛化到未见交互与复杂室内布局。
- **核心思路：** 将「人在场景中生活」建模为 **MDP**：感知为状态、**CVAE 运动基元的潜变量**为动作、目标为奖励；在 **AMASS + SAMP** 上学 latent motion primitive，再用 **PPO actor-critic** 学场景感知策略。
- **两大策略模块：**
  1. **场景感知 locomotion：** 局部 **16×16 可走性二值图**（1.6 m×1.6 m）+ 骨盆目标方向特征；**穿透惩罚** 抑制与家具碰撞。
  2. **细粒度物体交互：** **体表 marker 目标**（由 [COINS](https://arxiv.org/abs/2207.08761) 从语义生成静态交互姿态）+ **物体 SDF 距离与梯度** 编码人–物邻近；同时学 **坐下/躺下** 与 **站起** 以串联多段活动。
- **系统编排：** **NavMesh + A\*** 生成中间路点 → locomotion 策略到达 → interaction 策略执行坐/躺 → 可循环组合（演示：走向凳子坐下 → 换椅再坐 → 走向沙发躺下）。
- **训练环境：** locomotion 在 **ShapeNet 随机 clutter 场景**；interaction 在 **PROX 静态交互 retarget 到 ShapeNet 家具** 上随机初始/目标姿态（含正反采样学站起）。
- **与基线对比：** 相对 **GAMMA**（仅 waypoint 行走、常穿模）与 **SAMP**（需配对数据、脚滑明显），DIMOS 在 **多样性、物理合理性、感知分数** 上更优；可 **OOD** 泛化到 novel 椅形、重建房间、点云街景、text-to-3D 奇怪形状椅子。
- **局限（项目页自述）：** **运动学** 方法，人–场景穿透仍可见；**躺姿** 训练数据不足导致部分不自然姿态。

## 核心摘录（面向 wiki 编译）

### 方法栈（Fig. 2 归纳）

| 模块 | 输入 | 输出 / 作用 |
|------|------|-------------|
| Motion primitive (CVAE) | 过去 1–2 帧 marker 历史 | 0.25 s 未来 marker 片段 → MLP 回归 SMPL-X |
| Locomotion policy | marker 种子 + 可走性图 + 骨盆目标方向 | 采样潜动作，避障走向路点 |
| Interaction policy | marker 种子 + marker 目标 + 物体 SDF 特征 | 坐/躺/站起，贴近 COINS 目标姿态 |
| COINS | 物体点云 + action-object 语义 | 静态交互 marker 目标 |
| Path finding | NavMesh | 长程路点序列 |
| Tree sampling (test) | 同训练奖励 | 每步保留 top-K 潜动作样本，提升推理质量 |

### 奖励结构（统一骨架）

\(r = r_{goal} + r_{contact} + r_{pene}\)

- \(r_{contact}\)：鼓励脚–地接触、抑制脚滑（容差 0.05 m / 0.075 m/s，沿用 GAMMA）。
- Locomotion：\(r_{pene}\) 惩罚 bbox 与不可走格重叠；\(r_{goal}\) 含骨盆距离与朝向对齐。
- Interaction：\(r_{goal}\) 为 marker 距离；\(r_{pene}\) 惩罚 SMPL-X 顶点在物体 SDF 内的负值。

### 与 GAMMA / SAMP 的分工（Related Work 表）

| 方法 | 数据需求 | 能力 | 主要短板 |
|------|----------|------|----------|
| GAMMA | 无 goal-motion 配对 | 多样体形、高保真 locomotion | 不懂坐/躺，易穿模 |
| SAMP | 物体–运动配对 | 室内坐/躺 | 多样性/物理性弱，脚滑 |
| **DIMOS** | AMASS+SAMP 基元 + 合成场景 RL | 导航 + 坐/躺 + 序列组合 + marker 细控 | 运动学穿透、躺姿质量 |

### 代码与复现要点（GitHub README）

- 环境：Ubuntu 22.04, CUDA 11.8, `conda env create -f env.yml`
- 依赖数据：SMPL-X, VPoser, AMASS, SAMP, BABEL, ShapeNet, Replica, PROX-S 等（demo 可部分省略）
- 许可：主体 **Apache 2.0**；SMPL-X / AMASS 等遵循各自许可
- 后续更新：数据预处理与训练文档、清洗版 locomotion 权重

## 对 wiki 的映射

- 沉淀实体页：[DIMOS（ICCV 2023）](../../wiki/entities/paper-dimos-human-scene-motion-synthesis.md)
- 项目页归档：[sources/sites/dimos-zkf1997-github-io.md](../sites/dimos-zkf1997-github-io.md)
- 代码归档：[sources/repos/dimos.md](../repos/dimos.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2305.12411>
- 项目页：<https://zkf1997.github.io/DIMOS/>
- 代码：<https://github.com/zkf1997/DIMOS>
