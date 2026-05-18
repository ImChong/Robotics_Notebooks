# BFM：Behavior Foundation Model for Humanoid Robots（arXiv:2509.13780）

> 来源归档（ingest）

- **标题：** Behavior Foundation Model for Humanoid Robots
- **缩写：** **BFM**（Behavior Foundation Model）
- **类型：** paper / humanoid whole-body control + 生成式策略预训练
- **arXiv：** <https://arxiv.org/abs/2509.13780>（HTML 版便于检索结构与公式：<https://arxiv.org/html/2509.13780v1>）
- **PDF：** <https://arxiv.org/pdf/2509.13780>
- **项目页：** <https://bfm4humanoid.github.io/>（演示视频，代码标注 *In Coming*）
- **作者：** Weishuai Zeng, Shunlin Lu, Kangning Yin, Xiaojie Niu, Minyue Dai, Jingbo Wang, Jiangmiao Pang
- **机构：** 北京大学、香港中文大学（深圳）、上海交通大学、复旦大学、**上海人工智能实验室**
- **入库日期：** 2026-05-18
- **一句话说明：** 以 **掩码在线蒸馏 + CVAE** 把人形 **全身控制（WBC）** 的多种控制接口（根、关节、关键点等）抽象成 **位级掩码可选** 的 **条件生成模型**，在 Unitree G1 上做 motion tracking / VR 遥操作 / locomotion 等多任务统一策略，并通过 **潜空间插值 + classifier-free 调制 + 残差解码器** 实现行为组合与新技能（如 Side Salto）少样本获取。

## 摘要级要点（与 abs / HTML 一致）

- **问题：** 现有 WBC 路线 **任务专一**、奖励工程沉重，跨控制模式（locomotion / 遥操作 / motion tracking）难复用；论文主张这些任务的本质都是「**生成把机器人引向目标状态的合适行为**」，因此把 **WBC 当作条件生成问题**。
- **核心方法：** 三件套
  1. **统一控制接口（masked control）**：低层 mode 含 **(i) 根平移/朝向/速度、(ii) 关节角、(iii) 关键点在局部坐标系下的位置**；每个元素用 **位级二值掩码** 激活子集，理论上可任意组合，**告别为每种 mode 单独写 reward**。
  2. **掩码在线蒸馏（masked online distillation）**：BFM 在仿真中 rollout，**特权 proxy agent** 提供参考动作；loss 为 `L_DAgger + λ_KL · L_KL`（动作 MSE + 编码器对先验的 KL）。掩码按 **B(0.5)** 直接采样；冷启动用 **mask curriculum**（采样概率从 1.0 退火到 0.5）。
  3. **CVAE 生成器**：包含 **prior ρ**、**encoder ε**、**decoder D**。encoder 显式输入当前 **mask m_t**，decoder **不直接见目标状态**，迫使潜变量 z 承载行为知识；推理时可只用 prior + decoder，遥操作/locomotion 等只给可观测的目标子集即可。
- **行为组合 / 调制：** **潜空间线性插值**（如 root-only mode + keypoint-only mode → Roundhouse Kick）；**classifier-free 风格调制** `z = (1+λ)·μ^ρ(s^p, s^g) − λ·μ^ρ(s^p, ∅)` 在恢复平衡等设置中以 λ≈0.5 提升 Butterfly Kick 跟踪稳定性。
- **新技能少样本获取：** **冻结预训练 BFM** 并训练 **残差解码器** `π(Δa_t | s^p, z)`，最终动作 `a_t' = a_t + Δa_t`；对未见过的 **Side Salto** 收敛速度与跟踪精度均优于 RL from scratch。
- **训练数据与平台：** **AMASS** 两阶段重定向到 **Unitree G1**（先 shape，再 joint）；**IsaacGym 8192 并行环境**；proxy agent 用 **PPO + 域随机化 + 硬负例挖掘 + motion filtering**。**Unitree G1**（原 29 DoF，腕部冻结为 **23 DoF**）。
- **量化结论（论文 Table III/IV）：**
  - **Motion Tracking** E_mpjpe **0.2226 rad**（vs Specialist 0.2247、HOVER 0.2416）；E_mpkpe **61.12 mm**（vs Specialist 73.63 mm）。
  - **VR Teleoperation** E_mpjpe **0.2235 rad**（vs Specialist 0.2555、HOVER 0.3055）；E_mpkpe **63.14 mm**。
  - **Locomotion** E_lin,xy **0.2116 m/s**、E_ang,z **0.6744 rad/s**，与 specialist 同档。
  - BFM **持续显著优于 RL from Scratch**，validating 预训练价值；同 HOVER（多模态 WBC 基线）相比通常更稳，对 **fixed control mode** 的依赖更弱。
- **真机：** 论文同时给出 **Mujoco sim-to-sim** 与 **Unitree G1 sim-to-real** 演示（具体真机指标以论文为准）。

## 关键设计与同期工作对照

- 与 [HOVER](https://hover-versatile-humanoid.github.io/)（Zhengyi Luo 等）类似都做 **多模式 WBC**，但 HOVER 偏 **multi-mode multitask RL**；BFM 显式建生成式潜空间，**支持插值与 CFG 调制**，并允许 **任意位级掩码组合**。
- 与 [GMT / OmniH2O](../../wiki/tasks/teleoperation.md) 等专家策略相比，BFM 一个策略覆盖跟踪/遥操作/locomotion，性能在多数指标上 **持平或更好**。
- 与 [SONIC](../../wiki/methods/sonic-motion-tracking.md) 等「大规模 motion tracking 当预训练」叙事互补：SONIC 强调 **数据 / 网络 / 算力同步 scaling**；BFM 强调 **条件生成结构 + 蒸馏** 让 **一个 checkpoint 覆盖多种控制接口**。
- 与 **BFM-Zero**（[lecar-lab.github.io/BFM-Zero](https://lecar-lab.github.io/BFM-Zero/)，arXiv:2511.04131）同名但取向不同：BFM-Zero 用 **无监督 RL + Forward-Backward 表示** 学 dynamics-aware 潜空间；本论文走 **有监督特权蒸馏 + CVAE** 路线。两者可并列阅读以理解「Behavior Foundation Model」当前的方法谱系。
- 综述：**A Survey of Behavior Foundation Model**（Yuan 等，arXiv:2506.20487，TPAMI 2025）系统梳理 BFM 作为 humanoid WBC 下一代范式的 pre-training pipeline 分类。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-behavior-foundation-model-humanoid.md`](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) — 方法、架构、量化结论与同期工作互链
- 关联升级 / 互链：
  - [Foundation Policy（基础策略模型）](../../wiki/concepts/foundation-policy.md) — 添加 **humanoid WBC 系 BFM** 子项
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md) — 在「Learning-based & Generative WBC」一段补 BFM
  - [Sim2Real](../../wiki/concepts/sim2real.md)（间接，proxy agent 域随机化）
  - [Privileged Training](../../wiki/concepts/privileged-training.md)（teacher = proxy agent）
  - [DAgger](../../wiki/methods/dagger.md)（在线蒸馏算法）
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)（mask curriculum）
  - [AMASS](../../wiki/entities/amass.md)（重定向输入）
  - [Unitree G1](../../wiki/entities/unitree-g1.md)（部署平台）
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)（并行训练）

## 其他公开资料（非 PDF 正文）

- **项目页（演示视频）：** <https://bfm4humanoid.github.io/> — 归档见 [`sources/sites/bfm4humanoid-github-io.md`](../sites/bfm4humanoid-github-io.md)
- **BFM 综述（同名族）：** Yuan et al., *A Survey of Behavior Foundation Model: Next-Generation Whole-Body Control System of Humanoid Robots*, arXiv:2506.20487, IEEE TPAMI 2025. <https://arxiv.org/abs/2506.20487>

## 当前提炼状态

- [x] 论文摘要与核心方法摘录（CVAE + 掩码蒸馏 + 行为调制 + 残差新技能）
- [x] wiki 实体页 + 关联页面参考来源回链
- [x] 项目页归档
