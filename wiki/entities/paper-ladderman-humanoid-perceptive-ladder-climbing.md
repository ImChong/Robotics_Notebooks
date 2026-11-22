---
type: entity
tags:
  - paper
  - humanoid
  - locomotion
  - perceptive-locomotion
  - ladder-climbing
  - loco-manipulation
  - depth
  - teacher-student
  - dagger
  - ppo
  - sim2real
  - unitree-g1
  - isaac-sim
  - amazon-far
  - open-source
status: complete
updated: 2026-11-21
arxiv: "2606.05873"
venue: "CoRL 2026 Spotlight"
code: https://github.com/amazon-far/LadderMan
original:
  - ../../sources/papers/ladderman_arxiv_2606_05873.md
  - ../../sources/repos/ladderman.md
related:
  - ../overview/humanoid-motion-cerebellum-technology-map.md
  - ../overview/motion-cerebellum-category-08-real-tasks.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../tasks/humanoid-locomotion.md
  - ../tasks/loco-manipulation.md
  - ../concepts/terrain-adaptation.md
  - ../concepts/privileged-training.md
  - ../concepts/sim2real.md
  - ../methods/dagger.md
  - ./unitree-g1.md
  - ./paper-rpl-robust-humanoid-perceptive-locomotion.md
  - ./paper-hrl-stack-22-perceptive-humanoid-parkour.md
  - ./paper-pilot-perceptive-loco-manipulation.md
  - ./paper-faststair-humanoid-stair-ascent.md
sources:
  - ../../sources/papers/ladderman_arxiv_2606_05873.md
  - ../../sources/sites/ladderman-robot-github-io.md
  - ../../sources/repos/ladderman.md
  - ../../sources/papers/motion_cerebellum_64_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md
summary: "LadderMan（CoRL 2026 Spotlight，arXiv:2606.05873，Amazon FAR）两阶段：hybrid motion tracking 从单参考学多几何攀爬专家，hybrid DAgger+RL 蒸馏为深度统一策略；VFM+RFM 零样本 sim-to-real；梯上双智能体遥操作；G1 真机约 3.4 s/踏棍；amazon-far/LadderMan 已开源。"
---

# LadderMan：人形感知梯子攀爬与梯上操作

**LadderMan**（*Learning Humanoid Perceptive Ladder Climbing*，Amazon FAR 等，arXiv:[2606.05873](https://arxiv.org/abs/2606.05873)，[项目页](https://ladderman-robot.github.io/)）提出 **统一 visuomotor 系统**：在 **稀疏窄踏棍** 上完成 **全身手–足多接触攀爬**，并进一步支持 **梯顶稳定遥操作**。攀爬策略经 **两阶段学习**——先用 **hybrid motion tracking** 从 **单条参考动作** 得到多梯子几何 **专家**，再以 **hybrid DAgger + RL** 蒸馏为 **深度统一策略**；真机部署用 **视觉基础模型（VFM）深度** 与 **rung-focused masking（RFM）** 实现 **零样本 sim-to-real**；梯上操作采用 **双智能体** 解耦下身稳定与上身跟踪。

## 一句话定义

**用一条梯子参考动作先「长出」多几何攀爬专家，再焊成只看深度、能双向爬多样梯子并在梯顶稳着干活的统一策略——稀疏踏棍上的感知全身协调，不靠改硬件。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LadderMan | Learning Humanoid Perceptive Ladder Climbing | 本文梯子攀爬 + 梯上操作系统 |
| VFM | Vision Foundation Model | 真机深度用 Fast-FoundationStereo 等预训练立体模型 |
| RFM | Rung-Focused Masking | 训练时遮非踏棍区，聚焦攀爬相关几何 |
| DAgger | Dataset Aggregation | 专家在策略诱导状态下纠偏的模仿蒸馏 |
| PPO | Proximal Policy Optimization | 专家与统一策略的 RL 优化算法 |
| RL | Reinforcement Learning | Stage 2 与专家训练中的策略优化 |
| Sim2Real | Simulation to Real | 深度 VFM + 轻量噪声实现零样本迁移 |
| WBC | Whole-Body Control | 手–足多接触全身协调控制 |
| G1 | Unitree G1 Humanoid | 宇树 1.3 m、29-DoF 实验平台 |

## 为什么重要

- 在 [运动小脑 64 篇技术地图](../overview/humanoid-motion-cerebellum-technology-map.md) 中归类为 **H 真实任务**（59/64）：任务：爬梯和梯上操作暴露手脚协同难题。
- **填补「梯子」空白：** 仓库内楼梯/跑酷文献多，但 **梯子稀疏踏棍 + 扶手** 对感知精度与全身协调要求更高；LadderMan 是 **端到端深度攀爬** 在 **人形真机** 上的系统案例。
- **单参考 → 多几何泛化：** **Hybrid motion tracking**（上身松弛、下身严格 + 梯子接触/攀爬奖励）避免为每种 $(\phi,z)$ 收集大动捕库——与 [RPL](./paper-rpl-robust-humanoid-perceptive-locomotion.md) 的 **分地形高程专家** 形成互补范式。
- **深度 sim-to-real 新配方：** 相对重度 depth randomization，**VFM 真机深度 + RFM + 极简噪声** 在 **细薄踏棍** 上更关键（真机消融：w/o VFM **3/10**，w/o RFM **0/10**）。
- **梯上 loco-manipulation：** **双智能体** 下身稳、上身跟 VR，相对 **TWIST2** 等现成全身遥操作在梯顶不易失稳——把「能爬上去」延伸到 **维护/搬运类任务**。

## 核心原理

| 模块 | 作用 |
|------|------|
| **Stage 1 专家** | 对每个目标梯子 $(\phi,z)$ 训 **状态专家** $\pi^{\text{expert}}_{\phi,z}$；**hybrid tracking** + **接触/攀爬奖励** + 梯子相对位姿观测 |
| **Stage 2 统一策略** | **DAgger + PPO + KL 到专家**；输入 **本体 + 深度 + 上下攀爬指令**；NVIDIA **Warp** 深度渲染 |
| **VFM 深度** | 真机 **Fast-FoundationStereo**；减轻原始 D435i 缺失像素与噪声 |
| **RFM** | 10% 概率遮非踏棍区，泛化不同扶手/支撑结构 |
| **梯上操作** | **双智能体** $\pi^l$ 稳定接触与骨盆，$\pi^u$ 跟踪 PICO 4 Ultra 上肢目标；训练上肢目标来自 **AMASS** |
| **部署** | G1 + **RealSense D435i** + **Jetson Orin**；**无硬件改装** |

### 流程总览

```mermaid
flowchart TB
  subgraph s1 [Stage 1 · 多几何专家]
    ref["单条 OptiTrack 参考\n梯子 A: 65.5° / 24.8 cm"]
    hmt["Hybrid motion tracking\n上身松弛 + 下身严格"]
    contact["梯子中心接触奖励\n+ 攀爬任务奖励"]
    experts["π^expert_{φ,z} 专家库"]
    ref --> hmt --> contact --> experts
  end
  subgraph s2 [Stage 2 · 统一 visuomotor]
    warp["NVIDIA Warp 深度渲染"]
    rfm["RFM 踏棍聚焦遮罩"]
    dagger["DAgger + PPO + KL"]
    vis["π^visual 深度攀爬策略"]
    warp --> rfm --> dagger
    experts --> dagger --> vis
  end
  subgraph deploy [真机部署]
    vfm["VFM 立体深度\nFast-FoundationStereo"]
    cmd["上下攀爬二值指令"]
    g1["Unitree G1\n零样本 sim-to-real"]
    vis --> vfm --> g1
    cmd --> vis
  end
  subgraph manip [梯上操作 · 双智能体]
    climb["攀爬策略到梯顶"]
    lower["π^l 下身稳定"]
    upper["π^u 上身遥操作\nPICO 4 Ultra"]
    tasks["调画 / 换灯泡 / 递箱"]
    g1 --> climb --> lower
    upper --> tasks
    lower --> tasks
  end
  s1 --> s2
  s2 --> deploy
  deploy --> manip
```

## 实验要点（归纳）

| 设置 | 要点 |
|------|------|
| 平台 | Unitree G1（1.3 m、29-DoF）；**IsaacSim** 训练 |
| 参考动作 | **仅 1 条** OptiTrack（梯子 A） |
| 仿真评估 | $\phi \in \{55°,60°,65°,70°\}$，$z \in \{20,\ldots,30\}$ cm；踏棍 24–28 cm、55–65° **>95%** 成功率 |
| 盲基线 | 无感知 BeyondMimic 式 tracking：近参考 **49%**，OOD **近零** |
| 真机梯子 | A/B/C 不同材料与几何；**双向攀爬 ~20 s** |
| 速度 | **~3.4 s/踏棍**（人类 ~3.2 s） |
| 真机消融（梯子 A, 10 次） | Ours **9/10**；w/o RL **2/10**；w/o VFM **3/10**；w/o RFM **0/10** |
| 代码 | **已开源** [amazon-far/LadderMan](https://github.com/amazon-far/LadderMan)（Apache-2.0，2026-11 发布）；真机部署对接 [EGalahad/sim2real](https://github.com/EGalahad/sim2real) |

## 源码运行时序图

图依据：基于 `amazon-far/LadderMan@9a8d5ed` 的 `README.md`、`scripts/train_teachers.sh`、`scripts/train.py`、`active_adaptation/learning/ppo/ppo_me_dagger.py` 与 `scripts/play.py` 源码核实。

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Setup as setup.sh / IsaacLab v2.2.0
    participant Teacher as scripts/train.py (algo=teacher)
    participant Vec as data/vecnorm/ladder_vecnorm.pt
    participant Student as ppo_me_dagger.py (algo=student)
    participant Play as scripts/play.py (export_policy=true)
    participant Real as EGalahad/sim2real (G1)
    Dev->>Setup: 安装 Isaac Sim 4.5.0 + Isaac Lab v2.2.0
    Dev->>Teacher: train_teachers.sh 按倾角 x 间距网格训专家 (7500 iter)
    Teacher->>Vec: 共享冻结观测归一化统计
    Dev->>Student: 传入 teacher_checkpoint_path_list 训深度学生
    Note over Student: 7500 iter 纯 ME-DAgger -> 2500 线性衰减 -> 3000 纯 PPO
    Student-->>Dev: checkpoints/checkpoint_final.pt
    Dev->>Play: 仿真回放并导出 .onnx / .json / .yaml
    Play->>Real: 部署 ONNX 至 Unitree G1 实机
```

## 工程实践

| 环节 | 官方入口与工程要点 |
|------|--------------------|
| **环境与仿真后端** | `bash setup.sh` 安装 Python 3.10、`isaacsim==4.5.0` 与 `IsaacLab v2.2.0`（注意与论文正文笼统提及的 IsaacSim 区别）；运行需 `export OMNI_KIT_ACCEPT_EULA=YES` |
| **共享 VecNorm 契约** | 所有教师与学生必须共享同一冻结 `data/vecnorm/ladder_vecnorm.pt`；修改观测或参考动作后需先以 `vecnorm=train vecnorm_path=null` 重算 |
| **显存与多专家网格** | 默认 `4096 envs` 在单张 NVIDIA L40S（48 GB）运行；蒸馏阶段在 BC 权重归零前所有教师驻留显存，显存不足时下调 `task.num_envs` |
| **依赖故障排查** | Conda 内启动 Isaac Sim 若报 `GLIBCXX` 缺失，按 README 设置 `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` |
| **导出与真机部署** | `scripts/play.py export_policy=true` 将归一化层直接打包进 ONNX，输出 `.onnx` / `.json` / `.yaml` 后对接 [EGalahad/sim2real](https://github.com/EGalahad/sim2real) |

## 结论

**细稀悬空踏棍上，端到端深度 + 踏棍聚焦 + RL 恢复，比「盲跟踪一条参考」关键得多；梯子不是楼梯的特例。**

1. **单参考也能覆盖几何带** — 仅 1 条 OptiTrack（梯子 A）；仿真 \(\phi\in\{55°,60°,65°,70°\}\)、\(z\in\{20,\ldots,30\}\) cm，踏棍 24–28 cm、55–65° 成功率 **>95%**。
2. **盲基线在 OOD 近乎失效** — 无感知 BeyondMimic 式 tracking：近参考 **49%**，OOD **近零**。
3. **真机消融：RFM/VFM/RL 都硬** — 梯子 A 10 次：Ours **9/10**；w/o RL **2/10**；w/o VFM **3/10**；w/o RFM **0/10**。
4. **速度接近人类** — 双向攀爬约 **20 s**；约 **3.4 s/踏棍**（人类约 **3.2 s**）；A/B/C 不同材料与几何。
5. **两阶段蒸馏可操作** — Stage 1 按 \((\phi,z)\) 训 hybrid-tracking 专家；Stage 2 DAgger+PPO+KL 蒸馏到深度策略；真机用 Fast-FoundationStereo，无硬件改装。
6. **操作与攀爬分策略** — \(\pi^l/\pi^u\) 双智能体做梯上遥操作；2026-11 已开源 `amazon-far/LadderMan` 训练与导出代码（基于 HDMI `active_adaptation`）。

## 常见误区或局限

- **误区：「梯子 = 另一种楼梯」。** 踏棍 **细、稀、悬空**，高程图常难以精确表达；LadderMan 强调 **端到端深度** 与 **踏棍聚焦**，与 [FastStair](./paper-faststair-humanoid-stair-ascent.md) 的 **重复踢面 + DCM 监督上楼** 问题形态不同。
- **误区：「有 DAgger 就够」。** 动态接触-rich 任务中纯模仿误差会累积；论文真机 **w/o RL** 仅 **2/10**——**RL 恢复行为** 不可或缺。
- **误区：「深度 randomization 越多越好」。** 梯子场景用 **VFM + 轻量噪声** 替代重度 hand-tuned DR；RFM 针对 **结构化几何差距**（扶手/背景）。
- **局限（论文 §5）：** (1) 目前主要在倾角 **$\le 75°$** 斜梯上验证，尚不能覆盖对接触力与稳定裕度要求更苛刻的 **垂直梯（vertical ladders）**；(2) 缺乏 **灵巧手（dexterous end-effectors）** 限制了梯上精细操作品类；(3) 攀爬与梯上操作 **分策略** 独立运行，极端上肢遮挡仍可能干扰深度观测。

## 与其他工作对比

| 维度 | LadderMan | RPL | PHP | FastStair | PILOT |
|------|-----------|-----|-----|-----------|-------|
| 机构 | Amazon FAR | Amazon FAR | Amazon FAR | LimX 等 | 上海交大 |
| 任务 | **梯子攀爬 + 梯上操作** | 双向多地形 + 载荷 | 跑酷技能链 | 高速上楼梯 | loco-manipulation LLC |
| 感知 | **深度 + VFM** | 多视角深度 | 单深度 | 机载高程 | LiDAR 高程 |
| 专家来源 | **单参考 hybrid tracking** | 分地形高程专家 | MM 长程参考 | DCM 监督 | — |
| 蒸馏 | **DAgger + PPO + KL** | DAgger | DAgger + PPO | 分速专家 | 单阶段 MoE |
| 操作 | **梯上双智能体 VR** | 2 kg 载荷行走 | — | — | 全身 LLC |
| 平台 | **G1** | G1 | G1 | LimX Oli | G1 |

## 工程实践

- **教师–学生深度表征蒸馏：** Stage 1 特权教师依赖梯档真值位姿先学会多点抓握与脚踏切换；Stage 2 通过 DAgger 蒸馏至机载双摄深度图输入，并对强反光金属梯柱加入图像噪声与掩码随机化。
- **攀爬与梯上操作解耦：** 到达目标高度后锁死脚/单手抓牢梯柱作为锚点，再交由上肢遥操作完成涂画或灯泡更换。

## 关联页面

- [楼梯与障碍 Locomotion 中心节点](../tasks/stair-obstacle-perceptive-locomotion.md) — 离散接触地形索引（含梯子）
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 梯上操作与双智能体解耦
- [Humanoid Locomotion](../tasks/humanoid-locomotion.md) — 人形行走任务总览
- [Terrain Adaptation](../concepts/terrain-adaptation.md) — 深度感知闭环
- [Sim2Real](../concepts/sim2real.md) — VFM 深度迁移语境
- [Privileged Training](../concepts/privileged-training.md) — 专家 → 学生蒸馏范式
- [DAgger](../methods/dagger.md) — Stage 2 蒸馏算法
- [RPL](./paper-rpl-robust-humanoid-perceptive-locomotion.md) — 同系 Amazon FAR 多向深度行走
- [Agile Perceptive Traversal](./paper-agile-perceptive-traversal-sparse-3d.md) — ETH 猴架荡杆：原始 LiDAR + 钩具，与梯子攀爬形成稀疏结构对照（arXiv:2608.29769）
- [Unitree G1](./unitree-g1.md) — 实验平台

## 参考来源

- [LadderMan 论文摘录（arXiv:2606.05873）](../../sources/papers/ladderman_arxiv_2606_05873.md)
- [LadderMan 官方仓库归档（amazon-far/LadderMan）](../../sources/repos/ladderman.md)
- [LadderMan 项目页归档](../../sources/sites/ladderman-robot-github-io.md)

## 推荐继续阅读

- 官方仓库：<https://github.com/amazon-far/LadderMan>
- 论文 HTML：<https://arxiv.org/html/2606.05873>
- 论文 PDF：<https://arxiv.org/pdf/2606.05873>
- 项目页（演示视频）：<https://ladderman-robot.github.io/>
- [RPL](./paper-rpl-robust-humanoid-perceptive-locomotion.md) — 同平台双向深度行走与载荷对照
- [PHP](./paper-hrl-stack-22-perceptive-humanoid-parkour.md) — 同系深度感知跑酷技能链
- [PILOT](./paper-pilot-perceptive-loco-manipulation.md) — 单阶段 LiDAR 感知全身 LLC 对照
