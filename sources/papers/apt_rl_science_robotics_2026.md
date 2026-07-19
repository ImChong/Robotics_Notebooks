# Agile Perceptive Multi-Skill Locomotion for Quadrupedal Robots in the Wild（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Agile perceptive multi-skill locomotion for quadrupedal robots in the wild
- **类型：** paper / quadruped locomotion / multi-gait / perceptive RL / sim2real / transformer prior
- **期刊：** Science Robotics, July 2026 — **封面文章**
- **DOI：** <https://doi.org/10.1126/scirobotics.adz7397>
- **arXiv：** <https://arxiv.org/abs/2607.13579>
- **项目页：** <https://skillquadsr.github.io/>（详见 [`sources/sites/skillquadsr-github-io.md`](../sites/skillquadsr-github-io.md)）
- **作者：** Jun-Gill Kang†、Jaehyun Park†、Tae-Gyu Song、Joon-Ha Kim、Seungwoo Hong‡、Hae-Won Park‡（† 同等一作；‡ 通讯作者）
- **机构：** 韩国国防科学研究所（ADD）、KAIST 机械工程、DIDEN Robotics、高丽大学机械工程 / 智能移动学院
- **平台：** KAIST HOUND（自研四足）；Intel RealSense D435 + 2D LiDAR；Isaac Gym 训练
- **代码与数据：** 项目页 **未列 GitHub**；论文 Data availability 写明数据与作图代码见 [Zenodo:20645964](https://zenodo.org/records/20645964)（**非** 完整训练/部署仓库）
- **入库日期：** 2026-07-19
- **一句话说明：** 提出 **APT-RL**：用 **2D 轨迹优化** 大规模生成 **trot/bound 力矩先验**，经 **TVAE 表征学习 + PPO 潜/辅助动作联合 RL + 深度/LiDAR 感知蒸馏**，在 **单策略** 内实现 **感知条件下步态与机动技能切换**；HOUND 在校园/森林 **零样本 sim2real** 完成 **1.1 km / 0.34 km** 野外路线，瞬时峰值 **6 m/s**，并演示 **60 cm 高台、三级楼梯、垫脚石、沟、倒木** 等障碍。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [skillquadsr.github.io](https://skillquadsr.github.io/) | 演示视频、三阶段框图、野外路线与 gait 选择对比 |
| 数据 / 作图 | [Zenodo 20645964](https://zenodo.org/records/20645964) | 论文声明的 data & figure generation code |
| 多步态盲走对照 | [Learning to Adapt（Nature MI 2025）](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md) | **8 步态 + 仅本体**；本文 **2 gait + 深度/LiDAR 感知 + 更高速度** |
| MoB 参数化对照 | [Walk These Ways](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md) | **行为参数 b** 预埋多样步态；本文 **潜动作选 trot/bound + 辅助力矩扩展** |
| 四足感知跑酷 | [SWAP](../../wiki/entities/paper-swap-parkour.md) / [Extreme Parkour](../../wiki/entities/extreme-parkour.md) | 同为感知敏捷 locomotion；本文强调 **TO 力矩先验 + 野外长程 + 步态切换** |
| 楼梯/障碍中心节点 | [楼梯与障碍 Locomotion](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md) | 本论文实机含多级楼梯、高台、垫脚石、沟等离散地形 |

## 摘要级要点

- **问题：** 野外四足需要 **多种机动技能**、**稳定步态过渡** 与 **宽速度域感知控制**；分 gait 专训或手工切换规则难以在 **高速 bound** 与 **精细 trot 攀台** 间无缝切换。
- **APT-RL 三阶段：**
  1. **表征学习：** 2D 单刚体轨迹优化在 **8 分钟** 生成 **180,000** 条 trotting/bounding 轨迹（含力矩）；**TVAE** 学统一潜空间 + **步态专属力矩解码器**。
  2. **强化学习：** PPO 输出 **潜动作**（经预训练解码器）+ **辅助动作**（PD 力矩）；在 rough/discrete/stair/high-step/stepping-stone 等地形课程训练；**Auto** 策略相对固定 trot/bound 在 **成功率、速度跟踪、1/COT** 上整体更优。
  3. **感知蒸馏：** 特权 **2.5D 高程** teacher → **深度 CNN + 45-bin LiDAR 高程 + 本体 GRU** 学生；DAgger 式蒸馏；双模态优于单模态（LiDAR 擅长远距/楼梯/沟/高台，深度擅长栏/垫脚石）。
- **力矩组成分析：** 平地/熟悉动作以 **decoder 力矩** 为主；**越障起跳、腿损、偏航** 等 OOD 场景 **辅助力矩** 占比上升，可生成预训练集未覆盖的 3D 行为（区别于固定基策略的 residual RL）。
- **Gait 先验选择：** 在 trot/bound/pace/gallop/pronk 五解码器对比中，**trot + bound** 在多样地形与 **-1~7 m/s** 命令下综合最优，故作为主干先验。
- **实机亮点：**
  - **校园 1.1 km：** 多级楼梯、草、坡；下楼 **1 m/s trot**，高速段 **bound**，三级楼梯瞬时 **6 m/s**。
  - **森林 0.34 km：** 倒木、树根、湿滑落叶；不规则区 **trot**，跨木 **bound**，提速后回到 bound。
  - **60 cm 高台：** 瞬时 **4.25 m/s**；**几何相似下降障碍** 按高度选 trot vs bound（同命令速度）。
  - **工程：** **>4 m/s** 为 LiDAR 加装 **3D 打印减振**；全栈机载，无动捕。
- **局限（论文 Discussion）：** 当前侧重 **矢状面前后运动**；仅 **trot/bound** 两 gait；四足专用，未扩展到人形/轮足。

## 核心摘录（面向 wiki 编译）

### 1) 训练管线与 APT 模块

| 阶段 | 输入 / 输出 | 要点 |
|------|-------------|------|
| TO 数据集 | 2D flat trot/bound | 180k 轨迹，含状态与 **τ**；8 min 生成 |
| TVAE | 3 帧本体历史 **s** | 潜 **z∈R^16**；重建 + 步态解码器 **L_recon** |
| RL policy | 观测 → **z_action + a_aux** | **τ = τ_decoder(z) + τ_PD(a_aux)** |
| 感知 student | depth + LiDAR + proprio | CNN(**z_depth∈R^32**) + 45-bin LiDAR + GRU |

### 2) 与 AMP / vanilla RL 对比（论文 Fig. 7）

- 与 **同数据集 AMP**、**无先验 vanilla PPO** 比：APT-RL 在多样障碍上 **成功率更高、1/COT 更优**；辅助动作 + 结构化先验优于纯对抗模仿或从零训练。
- **对 wiki 的映射：** [`wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md`](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md)

### 3) 野外与步态选择（实机）

- **地形条件：** 同命令下较低障碍 **trot**、较高平台 **bound**。
- **速度条件：** 同楼梯 **1.8 m/s trot** vs **4.3 m/s bound**。
- **对 wiki 的映射：** [`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)

### 4) 开源状态

- 项目页 **无 GitHub**；Zenodo 提供 **数据 + 作图代码** → wiki「局限与风险」写明 **训练栈未公开**。
- **对 wiki 的映射：** [`sources/sites/skillquadsr-github-io.md`](../sites/skillquadsr-github-io.md)

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md`](../../wiki/entities/paper-apt-rl-agile-perceptive-quadruped-locomotion.md)**
- 交叉：**[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)**、**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)**
- 谱系：**[`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)**、**[`wiki/entities/paper-walk-these-ways-quadruped-mob.md`](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md)**、**[`wiki/entities/paper-swap-parkour.md`](../../wiki/entities/paper-swap-parkour.md)**
