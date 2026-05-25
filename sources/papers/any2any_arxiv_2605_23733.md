# Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking（arXiv:2605.23733）

> 论文来源归档（ingest）

- **标题：** Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
- **类型：** paper / humanoid / whole-body tracking / cross-embodiment / PEFT / post-training
- **arXiv abs：** <https://arxiv.org/abs/2605.23733>
- **arXiv HTML：** <https://arxiv.org/html/2605.23733v1>
- **PDF：** <https://arxiv.org/pdf/2605.23733>
- **机构：** LimX Dynamics（Ming Yang, Tao Yu†, Feng Li, Hua Chen*；†项目负责人，*通讯作者）
- **训练：** Isaac Lab + PPO；目标侧适应约 **4× NVIDIA A100**
- **源策略骨干：**
  - **Oli-WBT**（LimX Oli 上自预训练，500+ 小时动作数据，MLP / Transformer 两种 backbone）
  - **SONIC / Gear-Sonic**（Unitree G1 上开源/大规模 WBT，见 [SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)）
- **目标平台：** LimX Oli（31-DoF）、LimX Luna（27-DoF）、Unitree G1、Unitree H1
- **关联生态：** LimX GitHub 组织 <https://github.com/limxdynamics>（tron1-rl-isaaclab 等训练/部署仓，**非**本文专用 Any2Any 代码仓）；勿与遥感论文 [MiliLab/Any2Any](https://github.com/MiliLab/Any2Any)（arXiv:2603.04114）混淆
- **入库日期：** 2026-05-25
- **一句话说明：** 将成熟 **单机体 WBT 专家** 迁到新的人形：先做 **观测/动作运动学对齐**（散射矩阵 + 髋解耦 + 闭链 Jacobian），再对 **动力学敏感模块** 插入 **LoRA** 等 PEFT，仅用全量训练约 **1%** 算力与数据即可把 G1 上 Sonic 迁到 LimX Oli/Luna 等。

## 摘要级要点

- **问题：** 大规模 WBT（whole-body tracking）预训练成本高（SONIC 叙事：亿级帧、万级 GPU 小时）；新机型若从零训练重复投入；直接全参微调易 **灾难性遗忘** 源行为先验。
- **核心分解（与经典「运动学规划 + 动力学跟踪」同构）：**
  1. **Kinematic alignment（固定、无梯度）：** 把目标机 obs/动作映射到源机 **统一关节语义空间**（源机 DoF 数 $T$ 为基准）。
  2. **Dynamic adaptation（可学习）：** 冻结骨干 $\theta_{\mathcal{S}}$，仅学小参数 $\Delta\theta_{\mathcal{T}}$，$|\Delta\theta_{\mathcal{T}}|\ll|\theta_{\mathcal{S}}|$。
- **对齐三级结构（关节级 $\Phi_r$）：**
  - **(i) 稀疏散射矩阵 $S_r$：** 目标关节注入源布局，无对应位填零。
  - **(ii) 髋解耦 $D_r$：** 源机斜置髋 pitch 轴导致左右髋坐标耦合，用 $2\times2$ 块补偿。
  - **(iii) 并联闭链 Jacobian $J_r$：** 踝/腰等并联机构，$\Phi_r=J_r D_r^{-1} S_r$。
- **动力学残差动机：** 同参考 $(q,\dot q,\ddot q)$ 下 $M,C,G$ 与摩擦/接触差异产生 $\Delta\tau$；对齐后剩余缺口用 **低秩 LoRA** $W'=W+BA$ 吸收（默认机制；亦对比 Adapter / Prefix）。
- **网络结构假设：** 现代 WBT 常 **Reference Motion Encoder（偏可迁移运动语义）+ Action Decoder（偏机体动力学）**；PEFT 应插在 **源-目标差异最大** 的模块（实验验证）。
- **实验：** 5 组 source→target（含 Sonic→Oli/Luna、Oli-WBT→G1/H1/Luna）；相对 scratch 显著加速收敛；真机多下游任务部署叙事。
- **与跨机体 generalist 的区别：** HOVER / 多机体联合预训练需大数据；Any2Any 研究 **「已有单机体专家 → 新机体」** 的后训练问题。

## 核心摘录（面向 wiki 编译）

### 与 FastStair / CapVector 中「LoRA」的差异

| 场景 | LoRA 目的 | 代表 |
|------|-----------|------|
| 同机不同速度专家融合 | 分支差异、速度边界平滑 | [FastStair](faststair_arxiv_2601_10365.md) |
| VLA 能力向量正交 | 防灾难遗忘的 $\theta$ 空间 | [CapVector](capvector_arxiv_2605_10903.md) |
| 跨 embodiment WBT | 动力学残差 + 保留跟踪先验 | **本文** |
| 同机 sim2real 安全微调 | 真机样本效率 + recovery | [SLowRL](slowrl_arxiv_2603_17092.md) |

## 对 wiki 的映射

- 沉淀实体页：[Any2Any 跨机体人形 WBT 迁移（arXiv:2605.23733）](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)
- 交叉补强：[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[人形并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)、[Neural Motion Retargeting](../../wiki/methods/neural-motion-retargeting-nmr.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)
