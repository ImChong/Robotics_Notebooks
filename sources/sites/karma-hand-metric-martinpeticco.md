# martinpeticco.com/karma（KaRMA 项目页）

- **标题：** KaRMA — A Kinematic Metric for Fine Manipulation Ability in Robotic Hands
- **类型：** site / project-page
- **URL：** <https://martinpeticco.com/karma/>
- **配套论文：** [arXiv:2605.15548](https://arxiv.org/abs/2605.15548) — [`sources/papers/karma_hand_metric_arxiv_2605_15548.md`](../papers/karma_hand_metric_arxiv_2605_15548.md)
- **配套仓：** [mfpeticco/karma-hand-metric](https://github.com/mfpeticco/karma-hand-metric)
- **机构：** 麻省理工学院（MIT）Improbable AI Lab（Martin Peticco · Pulkit Agrawal）
- **入库日期：** 2026-09-17

## 一句话摘要

仅 URDF 运动学评估 **拇指–食指 rolling pinch** 内物体 **平移 + 重定向** 能力的三分数指标（KaRMA-T/R/S），带交互 voxel 可视化与 16 手对比榜；**CPU、秒–分钟级、无训练**。

## 开源核查（2026-09-17）

| 入口 | 结果 |
|------|------|
| 本页 | 交互 viewer、16 手表、Allegro vs D'Claw 等案例叙事 |
| GitHub | **已开源**；`run_metric.py` / `run_viser_app.py` / 16 手 `results/` |
| 论文 PDF | IROS 2026；arXiv 2605.15548 |

**判定：已开源（指标 + 可视化 + 论文结果可复现）。**

## 公开要点

### 三分数

| 分数 | 含义 |
|------|------|
| **KaRMA-T** | 抓取内可达球心体积 / 手尺寸归一化（Translation reach） |
| **KaRMA-R** | 228 HEALPix 姿态 bin 中可达比例（Rotation coverage；五最佳位姿均值） |
| **KaRMA-S** | 最优 seed vs median seed 比值（Seed robustness） |

### 管线（与论文一致）

1. Seed grasps → 2. Translation BFS + rolling QP → 3. Rotation HEALPix → 4. Scoring

### 设计案例（页内 Selected comparisons）

- **LEAP vs Allegro** — 最高 T vs 最高 R，解耦传统 DoF/workspace 叙事
- **Allegro vs D'Claw** — 相近 voxel 数但颜色（R 覆盖）差大
- **Shadow vs D'Claw** — 9 DoF < 6 DoF 的 T（joint range > count）
- **Dex3** — 2D 可达流形（5 DoF pinch 缺一平移维）

### 范围声明

- **Lower bound** on thumb–index rolling pinch dexterity；无 regrasp/gait；**kinematics only**
-  meant to **complement** task benchmarks（DexMachina、ISyHand 等同向一致）

## 关联资料

- 论文归档：[`karma_hand_metric_arxiv_2605_15548.md`](../papers/karma_hand_metric_arxiv_2605_15548.md)
- 代码：[`karma-hand-metric.md`](../repos/karma-hand-metric.md)
- wiki：[KaRMA](../../wiki/entities/paper-karma-hand-metric.md)
