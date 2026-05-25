# Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking（arXiv:2605.23733）

> 来源归档（ingest）

- **标题：** Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
- **类型：** paper / humanoid / whole-body tracking / cross-embodiment / PEFT
- **arXiv abs：** <https://arxiv.org/abs/2605.23733>
- **arXiv HTML：** <https://arxiv.org/html/2605.23733v1>
- **PDF：** <https://arxiv.org/pdf/2605.23733>
- **机构：** LimX Dynamics（Ming Yang, Tao Yu†, Feng Li, Hua Chen* 等）
- **硬件 / 平台：** LimX Oli（31-DoF）、LimX Luna（27-DoF）、Unitree G1、Unitree H1；源骨干含自训 **Oli-WBT**（Transformer / MLP）与开源 **Gear-SONIC**（G1 预训练 WBT）
- **入库日期：** 2026-05-25
- **一句话说明：** 将成熟 **WBT 专家策略** 后训练迁移到新的人形：先做 **运动学对齐**（观测布局 + 关节语义空间映射），再对 **动力学敏感模块** 插入 **LoRA** 做动力学适配；约 **1%** 全量训练算力与数据即可把 G1 上 Sonic 等骨干迁到 Oli/Luna，多对真机下游任务验证。

## 摘要级要点

- **问题：** 大规模 WBT 预训练（如 SONIC 量级 MoCap + GPU 小时）与目标机形态/动力学绑定；新机重训成本极高。
- **分解：** 跨具身差距 = **运动学**（DoF、观测/动作布局、闭链） + **动力学**（惯量、执行器、接触）。
- **Any2Any 两阶段：**
  1. **Kinematic Alignment：** Level-1 观测项重排；Level-2 关节级 $\Phi_r$（稀疏散射 $S_r$ + 髋解耦 $D_r$ + 并联闭链 Jacobian $J_r$），使冻结源策略在统一语义关节空间读写。
  2. **Dynamic Adaptation：** 冻结骨干 $\theta_{\mathcal{S}}$，在选定线性层加 LoRA $W'=W+BA$，仅训 $\{A,B\}$；假设 $\Delta\eta=\eta_{\mathcal{T}}-\eta_{\mathcal{S}}$ 低维，适配容量由 rank $k$ 控制。
- **网络结构假设：** 现代 WBT 常为 **Reference Motion Encoder + Action Decoder**；编码器偏可迁移运动特征，解码器更贴具身动力学——**局部 PEFT** 优于全参微调以防灾难性遗忘。
- **实验：** 5 组 source→target 迁移对；Sonic(G1)→Oli/Luna 等；训练仍用 **Isaac Lab + PPO**，除对齐与 PEFT 外与源预训练协议一致。
- **对比：** 与「多机联合预训练通用控制器」「URDF 级统一动作空间」路线互补——本文是 **已有单源 WBT 专家的后训练迁移**。

## 核心摘录（面向 wiki 编译）

### 与 SONIC / 跨具身文献的关系

| 维度 | Any2Any | SONIC 类大规模 WBT 预训练 | 多具身 generalist 预训练 |
|------|---------|---------------------------|---------------------------|
| 数据需求 | 目标机少量数据 | 亿级帧 + 万 GPU 时 | 多机大规模数据集 |
| 结构改动 | 对齐层 + 局部 LoRA | 端到端 scaling | 统一表征 / 路由 |
| 源策略 | 冻结单源专家 | 从头训 | 从头训 |
| 典型收益 | ~1% 算力达竞争跟踪性能 | 单平台最优 | 跨平台泛化 |

### 相关资料

- **源 WBT 公开叙事：** [GEAR-SONIC](https://nvlabs.github.io/GEAR-SONIC/) / [SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)
- **LimX 平台上下文：** [FastStair（arXiv:2601.10365）](faststair_arxiv_2601_10365.md) 同机构 Oli 工程线；[Unitree G1](../../wiki/entities/unitree-g1.md)
- **PEFT：** LoRA (2021)、Adapter、Prefix-Tuning（论文 Related Work）
- **组织：** [LimX Dynamics GitHub](https://github.com/limxdynamics)（tron1-rl-isaaclab 等，截至 ingest 未见 Any2Any 独立代码仓）

## 对 wiki 的映射

- 沉淀实体页：[Any2Any（arXiv:2605.23733）](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)
- 交叉补强：[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[BFM 人形基础模型](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
