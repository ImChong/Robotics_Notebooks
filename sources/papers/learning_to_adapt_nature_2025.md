# Learning to Adapt through Bio-Inspired Gait Strategies for Versatile Quadruped Locomotion（Nature Machine Intelligence, 2025）

> 来源归档（ingest）

- **标题：** Learning to adapt through bio-inspired gait strategies for versatile quadruped locomotion
- **类型：** paper / quadruped locomotion / multi-gait / bio-inspired / DRL / blind sim2real
- **期刊：** Nature Machine Intelligence, vol. 7, pp. 1141–1153 (2025)
- **DOI：** <https://doi.org/10.1038/s42256-025-01065-z>
- **论文页：** <https://www.nature.com/articles/s42256-025-01065-z>
- **作者：** Joseph Humphreys, Chengxu Zhou
- **代码与数据：** <https://github.com/ihcr/learning_to_adapt>（详见 [`sources/repos/ihcr_learning_to_adapt.md`](../repos/ihcr_learning_to_adapt.md)）
- **仿真栈：** RaiSim ≥1.1.6；策略 PyTorch；Colcon 工作空间；依赖子模块 `ihcr/bio_gait`
- **入库日期：** 2026-06-06
- **一句话说明：** 在 **仅本体感知（interoceptive）** 条件下，用 **分层 DRL** 同时嵌入动物运动三要素——**步态切换策略（πG）**、**步态程序性记忆（BGS）** 与 **实时运动微调（πL）**——实现 **8 种步态** 的流畅切换与 **盲零样本** 复杂地形部署；步态选择由 **CoT / τ% / Wext / 接触跟踪误差** 等生物力学指标统一驱动。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 官方代码 | [ihcr/learning_to_adapt](https://github.com/ihcr/learning_to_adapt) | 论文 demo 脚本、图表复现数据、`bio_gait` 子模块安装说明 |
| 同作者 UCL 路线 | [E-SDS（arXiv:2512.16446）](../../wiki/entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md) | Chengxu Zhou 等人 **人形 + 感知 VLM 奖励** 路线；本文为 **四足 + 盲本体 + 生物力学步态选择** |
| 多行为参数化对照 | [Walk These Ways（MoB）](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md) | 用 **行为参数 b** 预埋多样步态；本文用 **πG 在线选 gait ID + BGS 参考** |
| 步态生成概念 | [Gait Generation](../../wiki/concepts/gait-generation.md) | CPG / 参数化 / RL 涌现步态；本文属 **显式多步态 + 生物力学切换** |

## 摘要级要点

- **问题：** 多数四足 DRL 部署 **单一固定步态**，难以像动物那样在 Froude 表征 locomotion（约占日常运动 70–90%）中 **按速度与地形切换步态**、**快速调用多种 gait**、并在 **非名义接触态** 下微调摆腿轨迹。
- **动物三要素 → 机器人模块映射：**
  1. **Gait transition strategies** → 步态选择策略 **πG**（最小化生物力学启发的切换指标）
  2. **Gait procedural memory** → **Bio-inspired Gait Scheduler（BGS）** 输出 **βL / βG**（名义与过渡参考）
  3. **Adaptive motion adjustments** → 运动策略 **πL** 结合 **状态估计器 SE** 与 **βL**，按当前状态 **偏离名义摆腿**
- **仅本体感知：** 部署 **无额外外感知**（无深度/LiDAR）；传感器经 SE 形成状态 **s** 与观测 **oL / oG**。
- **8 步态：** stand, trot, run, bound, pronk, limp, amble, hop；**Γ\* ∈ [0,7]** 由 πG 输出，与速度命令 **U^cmd = [v_x, v_y, ω_z, Γ\*]** 一并送入 BGS。
- **名义 vs 辅助步态：** trot / run 为 **名义步态**（低/高速）；bound, pronk, limp, amble, hop 为 **辅助步态**（非名义扰动、稳定性恢复）。
- **πL 训练：** **仅在平地** 训练，但凭 **βL + s** 在未见 rough terrain 上仍成功（对比无 βL 或无 SE 基线频繁失败）。
- **πG 统一指标 πG^uni：** 联合 **CoT（能耗）**、**τ%（力矩饱和）**、**Wext（外功）**、**c_avg^err（足端接触跟踪 / 稳定性代理）**；单一指标策略无法复现动物 **过渡相** 行为。
- **与动物数据对照：** 加速时 **trot↔run 过渡相**、stride frequency 随速度上升；rough terrain 上启用辅助步态；**πG^uni** 在 CoT / fgrf / Wext / stride CV 上与犬马数据趋势一致。
- **实机零样本：** 木屑、大台阶、裂缝混凝土、深石、草坡、树根、落叶、松散木材、低摩擦坡、扰动、窄木梁等；草地 CoT 相对固定 trot / run **降 18% / 30%**。

## 核心摘录（面向 wiki 编译）

### 1) 控制栈与数据流

| 模块 | 角色 | 关键 I/O |
|------|------|----------|
| **πG** | 最优步态选择 | 输入 **oG**（含 **βG**）；输出 **Γ\*** |
| **BGS** | 伪步态程序性记忆 | 由 **s** 与 **Γ\*** 生成 **βL**（给 πL）与 **βG**（给 πG）；含 **任意步态对过渡参考** |
| **SE** | 状态估计（类比小脑–中脑协调） | **σ**（IMU + 本体）→ **s** |
| **πL** | 运动执行 + 自适应微调 | **oL** 含 **βL** 与 **s**；输出关节目标 **q\*** → **PD → τ\*** |
| **训练/部署一致性** | sim2real | 控制框架 **训练与部署结构相同** |

### 2) 消融要点（locomotion 对比）

- **πL^bio**（完整） vs **πL^noβL**（无 BGS 参考） vs **πL^noSE**（观测直接取自仿真器）：
  - 平地：**v_err / c_err / ω_err** 分别约低 **15% / 21% / 10%**；
  - rough terrain：后两者 **高速/高 roughness 大量失败**，**πL^bio 全 trial 成功**（训练仍仅平地）。
- **固定 trot 遇 5% 台阶：** 无自适应摆腿导致 **基座姿态/高度持续振荡**；**πL^bio** 可恢复。

### 3) 步态选择（πG^uni）

-  demanding 速度轨迹：低速 **trot**，高速 **run**，加速段 **trot↔run 振荡** 提高 stride frequency；**v=0** 时自动 **stand** 最小化 τ% / Wext / c_err。
- rough terrain + 高加速：**bound / hop / limp** 等辅助步态介入；radar 图显示辅助步态在 **τ% / Wext / c_err** 上相对 trot/run 提升。
- **单指标 πG**（仅 CoT / 仅 τ% / 仅 Wext / 仅 c_err）均 **无法同时** 复现动物 **过渡相** 与多数据集一致性；**统一指标必要**。

### 4) 与现有 DRL 四足路线的差异

| 维度 | 本文 | 常见单步态 DRL | 参考 motion / 多策略蒸馏 |
|------|------|----------------|--------------------------|
| 步态数 | **8 + 在线切换** | 1 | 多但常需参考轨迹或分 gait 专训 |
| 感知 | **仅 interoceptive** | 各异 | 各异 |
| 切换逻辑 | **生物力学指标 πG** | 无 / 手工 | 速度查表或 human 调参 |
| 摆腿自适应 | **βL + SE + πL** | 隐式或缺失 | 依赖 retarget 质量 |
| 仿真 | **RaiSim** | IsaacGym 系为主 | — |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)**
- 交叉：**[`wiki/concepts/gait-generation.md`](../../wiki/concepts/gait-generation.md)**（RL + 生物力学切换）、**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)**
- 谱系：**[`wiki/entities/paper-walk-these-ways-quadruped-mob.md`](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md)**（MoB 参数化多样行为 vs 本文 gait ID + BGS）

## 当前提炼状态

- [x] 摘要与三模块架构摘录
- [x] 8 步态与生物力学指标
- [x] 消融与实机要点
- [x] wiki 映射与代码入口
