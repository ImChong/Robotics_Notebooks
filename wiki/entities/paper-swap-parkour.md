---
type: entity
tags: [paper, quadruped, reinforcement-learning, locomotion, perception, parkour, world-model, symmetry, equivariant, rssm, amp, sim2real, zju]
status: complete
updated: 2026-07-19
arxiv: "2606.19928"
related:
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
  - ../concepts/terrain-adaptation.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../methods/amp-reward.md
  - ./extreme-parkour.md
  - ./dreamwaq-plus.md
  - ./isaac-gym-isaac-lab.md
sources:
  - ../../sources/papers/swap_parkour_arxiv_2606_19928.md
  - ../../sources/sites/swap-parkour-github-io.md
summary: "SWAP（arXiv:2606.19928）在 RSSM 潜变量世界模型与 Actor-Critic 上硬嵌入左右镜像等变（SE-CNN/MLP/GRU），端到端训练四足极限跑酷；Apollo 实机 2.13 m 远跳 / 1.63 m 攀台，并对镜像地形与户外场景零样本泛化。"
---

# SWAP：对称等变世界模型四足跑酷

**SWAP**（*Symmetric Equivariant World-Model for Agile Robot Parkour*，Lan et al., [arXiv:2606.19928](https://arxiv.org/abs/2606.19928)，[项目页](https://swap-parkour.github.io/)）提出 **端到端对称等变潜变量世界模型 + 对称合规 Actor-Critic**，在 **Apollo 四足**（70×55 cm、72 kg）上实现 **213 cm 远跳** 与 **163 cm 攀台**——论文称在对比四足跑酷系统中为 **最远 / 最高** 绝对纪录；并对 **仅单侧训练、镜像 OOD 评测** 与多样 **户外非结构化场景** 展现 **零样本** 鲁棒性。

## 一句话定义

**把四足形态与环境的左右镜像对称写进 RSSM 潜动态与策略网络拓扑（等变 Actor + 不变 Critic），用单一端到端 RL 阶段学几何感知的世界模型跑酷策略——避免 WMP 类无约束 latent WM 重复编码对称模式。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SWAP | Symmetric Equivariant World-Model for Agile Robot Parkour | 本文框架：对称等变世界模型 + 跑酷策略 |
| RSSM | Recurrent State-Space Model | Dreamer/PlaNet 系确定性记忆 + 随机潜变量动态模块 |
| SMDP | Symmetric Markov Decision Process | 对状态/动作定义群变换且奖励与转移不变的 MDP 扩展 |
| SE-CNN | Symmetric Equivariant CNN | 在 $\mathbb{Z}_2$ 反射群下等变的卷积网络，处理深度图 |
| SE-MLP | Symmetric Equivariant MLP | 满足镜像同态约束的多层感知机 |
| SE-GRU | Symmetric Equivariant GRU | 潜动态中的对称等变门控循环单元 |
| MFRL | Model-Free Reinforcement Learning | 不显式学习潜动态的 RL（如 Extreme Parkour 端到端路线） |
| AMP | Adversarial Motion Priors | 对抗运动先验风格正则；本文判别器亦约束为等变 |
| PPO | Proximal Policy Optimization | 本文 Actor-Critic 优化算法（隐含于 RL 管线） |
| WMP | World Model-based Perception | ICRA 2025 视觉足式 locomotion RSSM 基线；本文「w/o Eq」消融近似 |
| RL | Reinforcement Learning | 强化学习范式 |
| Sim2Real | Simulation to Real | 仿真训练、真机零样本部署 |

## 为什么重要

- **世界模型跑酷的新结构先验：** 在 WMP（Lai et al., ICRA 2025，RSSM 视觉跑酷）之后，证明 **硬等变约束** 比纯数据驱动或 SymLoss 软惩罚更能 **压缩潜空间、抑制探索坍缩**，尤其在 **双侧协调多接触攀台** 任务上。
- **纪录级实机边界：** 相对 [Extreme Parkour](./extreme-parkour.md)（Go1、~80 cm gap / 50 cm box 量级）与 PIE 等，SWAP 在 **更大 Apollo 机体** 上把绝对 gap/box 推到 **2.13 m / 1.63 m**，且 **单一统一策略** 户外零样本。
- **镜像 OOD 可验证：** 训练地形恒为「左高右低」，测试直接水平镜像——无等变基线 **深度重建误差陡升** 且成功率崩溃；全框架 **轻微误差上升**，说明潜空间真编码对称结构而非记忆外观。
- **训练成本可控：** Isaac Gym **6000** 环境、RTX 4090 **~10 h** 端到端，与当代四足跑酷 RL 量级可比。

## 核心结构

| 模块 | 作用 |
|------|------|
| **Symmetric Equivariant Encoder** | 深度图经 **SE-CNN**；本体 proprio 经 **SE-MLP**；满足 $e_\phi(\mathcal{F}_o(o))=\mathcal{F}_l(e_\phi(o))$ |
| **SE-RSSM（低频）** | 每 $k$ 步更新 $(h_t, z_t)$；SE-GRU 递推 + 等变 prior/posterior；KL + 重建联合优化 |
| **SE Decoder** | 重建深度、本体与 **body/foot heightmap** 辅助目标，压缩地形几何 |
| **Equivariant Actor（高频）** | 以 `sg(h_t)` + proprio 历史 + 指令采样动作；$\mu_\theta$ 严格等变 |
| **Invariant Critic（高频）** | 特权观测（含 terrain heightmap 等）估计价值；输出为 **平凡表示**（镜像不变） |
| **Equivariant AMP** | 风格判别器输出镜像不变，防止对称破坏下的风格漂移 |

### 流程总览

```mermaid
flowchart TB
  subgraph wm["低频 Symmetric Equivariant World Model"]
    dep["前向深度图"]
    prop["本体 proprio"]
    enc["SE-CNN + SE-MLP Encoder"]
    rssm["SE-GRU RSSM\n(h_t, z_t) 每 k 步"]
    dec["SE Decoder\n深度 + 本体 + heightmap"]
    dep --> enc
    prop --> enc
    enc --> rssm
    rssm --> dec
  end
  subgraph pol["高频 Equivariant Actor-Critic"]
    hist["proprio 短历史 + 速度指令"]
    actor["Equivariant Actor μ_θ"]
    critic["Invariant Critic V_ψ\n(特权 terrain 等)"]
    act["12-DoF 关节动作"]
    hist --> actor
    rssm -.->|sg(h_t)| actor
    rssm -.->|sg(h_t)| critic
    actor --> act
  end
  act --> env["Isaac Gym 跑酷课\n→ Apollo 零样本"]
  env --> dep
  env --> prop
```

## 方法栈与对比（提炼）

- **与 MFRL 跑酷对比：** [Extreme Parkour](./extreme-parkour.md) 直接 obs→action + 蒸馏，**无多步 latent 想象**；SWAP 强调 **长程 proactive 预测** 与 **对称结构化潜空间** 对极限探索效率的贡献。
- **与 WMP 对比：** 消融 **SWAP (w/o Eq)** 即标准网络 RSSM 跑酷（论文明示等同 WMP 基线）；全 SWAP 在 **box 攀台**（大状态-动作、双侧力平衡敏感）上优势最大。
- **与软对称对比：** SymLoss 仅边际改善；**网络拓扑硬约束** 才能避免 **单腿依赖、撞墙逃避** 等不对称次优陷阱。
- **与高台攀爬对照（轮足）：** [MUJICA](./paper-mujica-wheel-legged-multi-skill.md) 在 **Go2-W 轮足** 上以 **纯本体单策略** 做 **1 m 高台**（轮–腿钩挂）；SWAP 为 **纯四足 + 深度 WM**，Apollo **1.63 m 攀台**——机体与感知栈不同，可对照「高台」任务边界。
- **对称群：** 反射群 $\mathbb{Z}_2$；关节/速度/地形 heightmap 的 **左右置换规则** 见论文 Table I（如 abad 关节符号翻转、terrain 左中右 块置换）。

## 实验要点

- **仿真消融（各 1500 次独立试验）：** gap 跳跃 SWAP ≈ SWAP w/o Eq-Policy 均高；box 攀台 **SWAP > w/o Eq-Policy > SymLoss > w/o Eq**。
- **镜像迁移：** 单侧倾斜 box/楼梯/gap 训练 → 水平镜像测试；SWAP 全难度维持高成功率。
- **实机室内：** 复现仿真策略——gap 前肢撑台缘、后肢蹬地；攀台 **双侧对称接触**。
- **实机户外：** 湿花岗岩 1 m 台（前腿打滑仍成功）、户外沟壑落点坡、浅水反光、暗楼梯、高草/纸板、碎石与地毯等 **统一策略** 穿越。

## 常见误区或局限

- **误区：** 把 SWAP 当作「世界模型 + 数据增强 镜像」——其核心是 **计算图级等变架构**，论文明确不纳入纯增广基线（样本效率与泛化劣于硬编码）。
- **误区：** 认为仅 WM 等变足够——**box 攀台** 上 **SWAP w/o Eq-Policy** 仍次优，需 **WM 与 Actor 同时约束** 才达最优双侧协调。
- **局限：** 主要在 **结构化 box/gap 跑酷课** 验证；论文承认 **离散踏脚石（stepping stones）** 等更复杂非结构化场景尚未充分评测；代码开源状态以项目页为准。

## 参考来源

- [SWAP 论文摘录（arXiv:2606.19928）](../../sources/papers/swap_parkour_arxiv_2606_19928.md)
- [SWAP 项目页归档](../../sources/sites/swap-parkour-github-io.md)

## 关联页面

- [Extreme Parkour（端到端四足感知跑酷）](./extreme-parkour.md) — MFRL + 双重蒸馏跑酷标杆（Go1）
- [楼梯与障碍 Locomotion（感知中心节点）](../tasks/stair-obstacle-perceptive-locomotion.md) — 四足跑酷谱系挂接
- [Locomotion](../tasks/locomotion.md) — 四足 RL 与跑酷任务地图
- [Terrain Adaptation](../concepts/terrain-adaptation.md) — 地形几何进入策略/潜空间
- [AMP（RSL-RL）](./amp-rsl-rl.md) — 对抗运动先验工程实现
- [DreamWaQ++](./dreamwaq-plus.md) — 四足感知 loco 姊妹路线（隐式地形想象）

## 推荐继续阅读

- Lan et al., [SWAP: Symmetric Equivariant World-Model for Agile Robot Parkour](https://arxiv.org/abs/2606.19928)（2026）
- Lai et al., *World Model-based Perception for Visual Legged Locomotion*（ICRA 2025）— RSSM 视觉跑酷直接前作 / 无等变基线
- Cheng et al., [Extreme Parkour with Legged Robots](https://arxiv.org/abs/2309.14341)（ICRA 2024）— MFRL 跑酷对照
- Nie et al., *Coordinated Humanoid Robot Locomotion with Symmetry Equivariant RL*（AAAI 2026）— 人形对称等变 RL 姊妹思路
