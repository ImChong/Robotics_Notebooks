---
type: entity
tags: [paper, humanoid, whole-body-control, behavior-foundation-model, motion-tracking, scaling-law, transformer, ppo, unitree-g1, isaac-lab, cuhk, sjtu, zju, pku, tsinghua, galbot, shanghai-ai-lab, sim2real]
status: complete
updated: 2026-11-21
arxiv: "2607.15163"
venue: "arXiv 2026"
code: https://github.com/zengweishuai/ScaleBFM
related:
  - ./egalahad-sim2real.md
  - ../concepts/behavior-foundation-model.md
  - ../concepts/whole-body-control.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../concepts/curriculum-learning.md
  - ../concepts/domain-randomization.md
  - ../concepts/sim2real.md
  - ./nvidia-sonic.md
  - ./paper-behavior-foundation-model-humanoid.md
  - ./paper-reactivebfm.md
  - ./unitree-g1.md
  - ./amass.md
  - ./isaac-gym-isaac-lab.md
sources:
  - ../../sources/papers/scaling_bfm_arxiv_2607_15163.md
  - ../../sources/sites/scalebfm-github-io.md
  - ../../sources/repos/scalebfm.md
  - ../../sources/repos/scalebfm-plus.md
original:
  - ../../sources/papers/scaling_bfm_arxiv_2607_15163.md
  - ../../sources/repos/scalebfm.md
summary: "ScaleBFM（arXiv:2607.15163）系统给出人形 BFM 规模化配方：全局坐标系 integrated whole-body motion tracking + on-policy rollout 数量与参考运动多样性协同 + Humanoid Transformer（3M 参数）；Unitree G1 真机 50 Hz，whole-body 模式相对 SONIC 等基线 G-MPKPE 最高降约 82%（global）、L-MPKPE 降 >10%（local）。"
---

# ScaleBFM（Scaling Behavior Foundation Model for Humanoid Robots）

**ScaleBFM** 是上海人工智能实验室牵头、多校与 Galbot 合作的人形 **BFM scaling 技术报告**（arXiv:2607.15163，[项目页](https://scalebfm.github.io/)）：在 [BFM](./paper-behavior-foundation-model-humanoid.md) 同团队的「行为与任务解耦」叙事上，把 **motion tracking** 确立为可扩展预训练范式，并实证 **学习范式 × 训练数据 × Humanoid Transformer 架构** 三轴如何协同放大控制保真度与跨域泛化。

## 一句话定义

**把 PPO 全局坐标系整体动作跟踪作为可扩展预训练代理任务，用并行 GPU × 滚动步数（数量）协同多源异质动作库（多样性）与 3M 档 Humanoid Transformer（RMSNorm 超球面潜空间），让单一 G1 策略同时兼容从仅根节到全身 8 种控制模式并零样本真机部署。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的人形全身基础控制器 |
| WBC | Whole-Body Control | 多关节协调的低层执行问题 |
| PPO | Proximal Policy Optimization | 本文 BFM 预训练所用 on-policy RL 算法 |
| MPKPE | Mean Per-Keypoint Position Error | 关键点位置跟踪误差（分 global/local） |
| HT | Humanoid Transformer | 本文提出的可扩展 Transformer 策略骨干 |

## 为什么重要

- **首次系统拆解 BFM scaling 三轴**：相较 [SONIC](./nvidia-sonic.md) 对 MoCap/算力/容量的初步 scaling，本文把 **PPO on-policy 数据量（GPU 并行 × rollout horizon）** 与 **参考运动行为覆盖（多源 102M 帧）** 分开度量，并给出 **同质 vs 异质** 数据扩展的两类 regime——对后续人形 foundation 数据策展有直接指导意义。
- **全局整体跟踪范式有明确 ablation**：相对 BeyondMimic 式 reward（**BFM-Bym** 消融）与去掉根平移/根–姿态解耦的惯例，**integrated global whole-body tracking** 在 BONES / Ours 双测试集上 consistently 降低 **G-MPKPE**，说明「行为语义」与「全身协调」需要同一奖励里一起学。
- **Humanoid Transformer 证明架构 matters**：**3M 参数 M 档 Transformer** 已优于显著更大的 MLP；潜空间在 **RMSNorm 超球面** 上无辅助 loss 即呈现 locality、跨行为分区与噪声鲁棒性，且大模型下 **八控制模式潜表示收敛**——解释 mode 间 scaling 饱和现象。
- **真机栈可复用**：[Unitree G1](./unitree-g1.md) 上 BFM 50 Hz（TensorRT）+ PD 200 Hz（LCM/SDK2），global 模式用 **VIVE Ultimate Tracker** 根定位，local 模式相对根重锚定；与 [ReactiveBFM](./paper-reactivebfm.md) 共享团队与部署文化。

## 流程总览

```mermaid
flowchart TB
  subgraph data [参考运动与 on-policy 数据]
    direction TB
    src["多源开源 MoCap\nLAFAN·AMASS·OMOMO·GRAB·\nSnapMoGen·FineDance·BONES·Embody3D\n→ 102M frames @ 50Hz"]
    ret["两阶段重定向\nshape 对齐 → 逐帧 IK"]
    roll["PPO on-policy rollouts\nwidth: GPU 数\ndepth: rollout horizon"]
    src --> ret --> roll
  end
  subgraph paradigm [学习范式：全局整体 motion tracking]
    direction TB
    mask["8 种 link mask 控制模式\nRoot → … → Whole-Body"]
    rew["全局 integrated 全身跟踪 reward\n早停 0.5m + RSI + 自适应采样"]
    mask --> rew
  end
  subgraph model [Humanoid Transformer]
    direction TB
    tok["分模态 tokenizer\n本体/动作交错 + query token"]
    cross["目标 cross-attention\nRMSNorm 超球面潜空间"]
    tok --> cross
  end
  subgraph deploy [真机 G1]
    direction TB
    inf["50 Hz BFM 推理\nTensorRT"]
    pd["200 Hz 关节 PD\nSDK2 + LCM"]
    inf --> pd
  end
  roll --> paradigm
  paradigm --> model
  model --> deploy
```

## 核心机制（归纳）

### 1）问题形式化与 motion tracking 代理任务

- 将人形控制表述为 **GCRL**：行为 $\boldsymbol{B}$ 定义为 $(s^p, a)$ 轨迹，**goal $s^g$ 是外部规格而非行为本体**。
- **BFM vs 普通 tracker**：motion tracking 是 **学行为的代理任务**，而非唯一部署接口；同一 checkpoint 可通过 **不同 mask 的 goal** 接稀疏到稠密多种控制模式。
- **全局整体跟踪**：奖励要求再现 **全局坐标系下的 integrated whole-body 参考**，避免根平移缺失或根–姿态解耦带来的 **行为歧义与协调弱化**。

### 2）八模式掩码控制接口

- 目标为根相对笛卡尔 **link pose + 与当前偏差**；actor 接收 **masked 目标 + mask 向量**，critic 看 **完整未 mask 目标 + 速度项**（非对称 actor–critic）。
- **八种 curated modes**（附录详列）：从仅 pelvis 的 Root Mode 到 14 link 的 Whole-Body Mode；训练时 $\boldsymbol{m}_i \sim \mathcal{M}$ 均匀采样，未激活 link 由策略 **inpainting**。
- **Global vs Local 部署**：global 用外部位姿跟踪器对齐根 xy 与航向；local 将控制信号 **相对当前根重锚定**（航向仍可用 IMU + 首帧标定），适配无全局定位场景。

### 3）数据 scaling：数量 × 多样性

| 维度 | 操作 | 主要结论 |
|------|------|----------|
| **On-policy 数量** | 扩 GPU（width）与 rollout horizon（depth） | **联合扩维** 最优（如 64×64）；单扩一维不稳定 |
| **参考运动同质扩展** | XXS→S（同 BONES-SEED 域，occupancy ~0.94） | 域内 benchmark 仅边际收益 |
| **参考运动异质扩展** | S→L（+LAFAN/AMASS/OMOMO 等，occupancy→0.9995） | **Ours Test**（Xsens+100Style）大幅提升，BONES 域内增益有限 |

- **102M 帧语料**：全开源数据集聚合 + 两阶段 retarget；**早停 + RSI**（终止点邻域重初始化）与 **自适应失败加权采样** 塑造 on-policy 分布质量。

### 4）Humanoid Transformer

- **结构**：有限窗本体/动作/目标 → 共享 tokenizer；本体与动作 token **交错** 为 context；**query token** 聚合历史（context 不可 attend 到 query）；目标序列 **cross-attention** 条件化。
- **潜空间**：goal embedding 经 RMSNorm 落在 **连续超球面**；满足 locality、全局行为分区、中等噪声下跟踪指标稳定（BONES 噪声 level 20 仍 Succ 0.9646）。
- **规模与部署**：M 档 **3.00M 参数** 为精度–效率折中（真机 >50 Hz）；actor 未来窗含 **随机远帧 offset** 补偿通信延迟。

## 主要量化结果

### Whole-body 模式基准（3M Humanoid Transformer）

| Test Set | 方法 | Succ ↑ | G-MPKPE ↓ | L-MPKPE ↓ |
|----------|------|--------|-----------|-----------|
| BONES | SONIC† | 0.9239 | 0.1740 | 0.0436 |
| BONES | BFM-Bym（BeyondMimic reward） | 0.9644 | 0.1005 | 0.0406 |
| BONES | **BFM-Global（本文）** | **0.9677** | **0.0798** | 0.0400 |
| BONES | BFM-Local | 0.7286 | 0.2281 | **0.0398** |
| Ours | SONIC | 0.5937 | 0.5035 | 0.0430 |
| Ours | BFM-Bym | 0.9709 | 0.1224 | 0.0403 |
| Ours | **BFM-Global（本文）** | **0.9776** | **0.0915** | 0.0396 |

- 相对 SONIC，**G-MPKPE** 在 BONES / Ours 上约 **54% / 82%** 降幅（摘要口径）；**L-MPKPE** local 模式相对既有控制器 **>10%** 改善。
- †BONES 随机子集可能与 SONIC 训练集重叠；Ours 为跨源未见动作，更能体现 **异质数据 scaling** 价值。
- 基线还包括 GMT、TWIST；全面落后于本文 BFM-Global。

## 结论

**ScaleBFM 的主张不是「人形控制器越大越好」，而是 motion tracking 可以当作可扩展的预训练代理任务——前提是奖励在全局坐标系下整体跟踪，且 on-policy 数据量与参考运动多样性沿两条独立的轴一起扩。**

- 真正起作用的机制是 **integrated global whole-body tracking**：相对 BeyondMimic 式 reward（BFM-Bym 消融）与根–姿态解耦惯例，它在 BONES / Ours 双测试集上一致降低 G-MPKPE（相对 SONIC 约 54% / 82%），说明「行为语义」与「全身协调」要在同一个奖励里学。
- 数据扩展分两种 regime：**同质扩展**（域内 occupancy 已 ~0.94）只有边际收益，**异质扩展**（S→L，occupancy→0.9995）才在跨源未见动作的 Ours Test 上大幅提升；on-policy 侧则需 GPU width 与 rollout horizon **联合扩维**，单扩一维不稳定。
- 架构确实 matters，但不是单调「越大越好」：3M 参数的 Humanoid Transformer 已优于显著更大的 MLP，继续放大则出现 **跨控制模式的 trade-off**，与共享 RMSNorm 超球面潜空间的模式收敛直接相关。
- 适用边界在部署侧：**Global 模式依赖外部根定位**（VIVE Ultimate Tracker），换到无全局定位的 Local 模式后 whole-body Succ 掉到 0.73；读表必须把指标绑定「模式 + G-/L-MPKPE 口径」，否则容易得出相反结论。
- 工程与开源：训练在 IsaacLab、评测走 MuJoCo sim-to-sim，真机为 [Unitree G1](./unitree-g1.md) 50 Hz BFM + 200 Hz PD；官方上游仓库 [`zengweishuai/ScaleBFM`](https://github.com/zengweishuai/ScaleBFM) 现已开源 ScaleRetarget / ScaleTrack / ScaleBridge 全链路代码与预训练权重（HuggingFace `WeishuaiZeng/ScaleBFM`）。
- 定位上它与 [BFM（CVAE）](./paper-behavior-foundation-model-humanoid.md) 是互补路线而非 v2，相对 [SONIC](./nvidia-sonic.md) 的增量在于把 scaling 拆成可分别度量的三轴，而 [ReactiveBFM](./paper-reactivebfm.md) 在这层基座上叠高层闭环规划。

## 与其他工作的关系

- **[BFM（CVAE，arXiv:2509.13780）](./paper-behavior-foundation-model-humanoid.md)**：同作者团队、同「掩码多接口」哲学，但 ScaleBFM 走 **PPO + Transformer + 全局 tracking scaling 实证**，BFM 走 **CVAE 在线蒸馏**；二者是 **互补技术路线** 而非同一模型的 v2。
- **[SONIC](./nvidia-sonic.md)**：同为 motion-tracking BFM 预训练；SONIC 强调 **离散行为表征 + 多命令对齐 + NVIDIA 规模**；ScaleBFM 强调 **三轴 scaling 定律式分解 + Humanoid Transformer + 八模式稀疏接口**。
- **[ReactiveBFM](./paper-reactivebfm.md)**：共享 Zeng / Niu / Pang / Wang 等作者；ScaleBFM 提供 **低层可扩展 tracker 基座**，ReactiveBFM 在其上叠 **闭环 AR-MDM 规划**。
- **[Perceptive BFM](./paper-perceptive-bfm.md)**：补 **地形感知** 与 operator–environment mismatch；ScaleBFM 假设参考在全局/局部重锚定后可跟踪，未显式建模楼梯/块级几何。

## 源码运行时序图

节点对齐 [`zengweishuai/ScaleBFM`](https://github.com/zengweishuai/ScaleBFM)（[sources/repos/scalebfm.md](../../sources/repos/scalebfm.md) 与 `ScaleTrack/README.md`）实际入口：

```mermaid
sequenceDiagram
    autonumber
    actor U as 维护者
    participant RET as ScaleRetarget/<br/>retarget.py
    participant PKG as package_motions.py +<br/>create_yaml.py
    participant TR as rsl_rl/train.py<br/>(OnPolicyRunner)
    participant EXP as play_export_check_<br/>humanoid_transformer*.py
    participant DEP as ScaleBridge/<br/>scalebridge/run.py
    U->>RET: 多源 SMPL/BVH 重定向至 Unitree G1 (.pkl)
    U->>PKG: 重采样 50 FPS 并生成 motions.yaml
    U->>TR: 单卡/多机 torchrun (--task G1-BFM-Transformer-Tracking)
    loop 共享内存运动库 + PPO
        TR->>TR: LOCAL_RANK=0 加载 npz 至共享内存 → OnPolicyRunner.learn()
    end
    U->>EXP: 导出 mode_table.pt 与 *_tensorrt.pt / _onboard.py aarch64 编译
    EXP-->>DEP: 下发 checkpoint + metadata JSON
    DEP->>DEP: bfm_agent.before_step / get_action → MuJoCo 或实机 G1 LCM
```

## 工程实践

- **训练：** IsaacLab（Unitree G1, 29 DoF）；多机 `torchrun` 时每节点仅 `LOCAL_RANK=0` 读取 `.npz` 并写入共享 CPU 内存以节省显存与 I/O。
- **回放与模式切换：** `rsl_rl/play.py` 支持 `--mode_index 0..7` 切换 8 种掩码控制模式及 `--local_tracking`（要求当前模式含 Pelvis）。
- **真机：** G1 机载 aarch64 不可直接复用 x86_64 TensorRT 引擎，须在机器人端执行 `play_export_check_humanoid_transformer_onboard.py` 重编引擎；`ScaleBridge/scalebridge/run.py` 经 LCM + SDK2 驱动 200 Hz PD，并接入 VIVE Tracker（Global 模式）或 Wi-Fi TCP 流式指令。
- **开源与社区扩展（2026-11-21 复核）：** 上游仓库 [zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM)（1,263 文件）已公开 `ScaleRetarget`、`ScaleTrack` 与 `ScaleBridge`，并于 HF `WeishuaiZeng/ScaleBFM` 提供 M（3M）/ XL checkpoint（`model_22200.pt`）。fork [`Hxxxz0/ScaleBFM_Plus`](https://github.com/Hxxxz0/ScaleBFM_Plus) 另增 SuperMimic-style 世界模型 32 步 imagined BPTT 与 ONNX 导出（仓库自报单步 26× 加速与持平 ScaleBFM-M，不作原文结论）。

## 常见误区或局限

- **不是 CVAE-BFM 的简单放大版**：架构为 **Transformer + PPO**，与 arXiv:2509.13780 的 CVAE+DAgger 管线不同。
- **Local 模式 Succ 低于 Global**：稀疏接口 + 无全局根定位时 whole-body 任务更难（BONES BFM-Local Succ 0.73）；读表时需区分 **G- vs L-MPKPE** 与部署模式。
- **Transformer 继续放大有 mode trade-off**：某些控制模式提升伴随其他模式略降，与 **共享潜空间跨模式收敛** 相关，不是单纯「越大越好」。
- **八模式是否终极接口未定论**：论文讨论节明确指出与未来高层策略集成方式仍开放。

## 关联页面

- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Humanoid Policy Network Architecture](../concepts/humanoid-policy-network-architecture.md)
- [SONIC（规模化运动跟踪）](./nvidia-sonic.md)
- [BFM（CVAE 多接口）](./paper-behavior-foundation-model-humanoid.md)
- [ReactiveBFM（闭环规划–控制）](./paper-reactivebfm.md)
- [Unitree G1](./unitree-g1.md)
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)
- [AMASS](./amass.md)

## 参考来源

- [sources/papers/scaling_bfm_arxiv_2607_15163.md](../../sources/papers/scaling_bfm_arxiv_2607_15163.md)
- [sources/sites/scalebfm-github-io.md](../../sources/sites/scalebfm-github-io.md)
- [sources/repos/scalebfm.md](../../sources/repos/scalebfm.md)
- [sources/repos/scalebfm-plus.md](../../sources/repos/scalebfm-plus.md)
- Zeng, Yin, Niu et al. *Scaling Behavior Foundation Model for Humanoid Robots*. arXiv:2607.15163, 2026. <https://arxiv.org/abs/2607.15163>

## 推荐继续阅读

- [ScaleBFM 项目页](https://scalebfm.github.io/) — scaling 曲线、潜空间可视化、真机八模式与 loco-manip 视频
- [SONIC 项目页](https://nvlabs.github.io/GEAR-SONIC/) — motion-tracking BFM scaling 对照
- [BFM 项目页](https://bfm4humanoid.github.io/) — 同团队 CVAE 路线演示
- [ReactiveBFM 项目页](https://xiao-chen.tech/reactivebfm) — 闭环上层与 ScaleBFM 低层栈衔接
