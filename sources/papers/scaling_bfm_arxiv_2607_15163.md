# Scaling Behavior Foundation Model for Humanoid Robots（arXiv:2607.15163）

> 来源归档（ingest）

- **标题：** Scaling Behavior Foundation Model for Humanoid Robots
- **缩写：** **ScaleBFM** / **Humanoid Transformer BFM**
- **类型：** paper / humanoid / behavior-foundation-model / motion-tracking / scaling-law
- **arXiv：** <https://arxiv.org/abs/2607.15163>
- **PDF：** <https://arxiv.org/pdf/2607.15163>
- **项目页：** <https://scalebfm.github.io/>
- **代码：** <https://github.com/zengweishuai/ScaleBFM>（截至 2026-07-18 仓库仅 README，作者公告 **2026-07-26 前** 逐步释出重定向/训练/部署代码）
- **发表日期：** 2026-07-16
- **作者：** Weishuai Zeng, Kangning Yin, Xiaojie Niu, Shunlin Lu, Weixiang Zhong, Jiahe Chen, Feiyu Jia, Xiao Chen, Zirui Wang, Furui Xu, Ming Zhou, Kailin Li, Weinan Zhang, He Wang, Li Yi, Dahua Lin, Jiangmiao Pang, Jingbo Wang（* 前三位共一）
- **机构：** 香港中文大学、上海交通大学、浙江大学、北京大学、清华大学、Galbot、上海人工智能实验室
- **入库日期：** 2026-07-18
- **一句话说明：** 系统研究人形 **BFM 规模化配方**：以 **全局坐标系下整体全身轨迹跟踪** 为统一学习范式，协调 **on-policy rollout 数量（GPU 并行 × rollout horizon）** 与 **参考运动多样性（102M 帧、多开源数据集）**，并用 **Humanoid Transformer**（3M 参数 M 档）替代 MLP 骨干；Unitree G1 真机 50 Hz 推理，whole-body 模式上相对 SONIC / GMT / TWIST 等基线 **G-MPKPE 降幅最高约 82%**（global）、**L-MPKPE 降幅 >10%**（local）。

## 核心论文摘录（MVP）

### 1) 问题：BFM scaling 三要素如何协同（Abstract / §1）

- **链接：** <https://arxiv.org/abs/2607.15163>
- **核心贡献：** BFM 已证明表达力与泛化潜力，但 **学习范式、行为数据、模型架构** 如何配合才能有效 scaling 仍不清楚。本文把 diverse WBC 统一为 **全局坐标系下 integrated whole-body behavior 的再现**，并系统拆解 scaling 的三条轴。
- **对 wiki 的映射：**
  - [ScaleBFM 论文实体](../../wiki/entities/paper-scaling-bfm-humanoid.md)
  - [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md)
  - [BFM（CVAE 路线）](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)

### 2) 学习范式：全局整体跟踪 + 八模式掩码控制接口（§3.2）

- **链接：** <https://arxiv.org/pdf/2607.15163>
- **核心贡献：**
  - **与 BeyondMimic / SONIC 差异：** 要求机器人在 **全局坐标系** 再现参考的 **integrated whole-body trajectory**；去掉根平移跟踪会使「向前走」与「原地踏步」难以区分，根–姿态解耦会削弱全身协调引导。
  - **控制接口：** 根相对笛卡尔空间 **masked whole-body target poses**；从 8 种预定义 link mask（Root / Bimanual / Root-and-Hand / End-Effector / Root-and-End-Effector / Upper-Body / Whole-Body 等）随机采样，同一底层行为可用 **稀疏到稠密** 多种规格呈现；未指定 link 由策略 **inpainting**。
  - **部署：** 支持 **global**（HTC VIVE Ultimate Tracker 根定位 + 首帧标定）与 **local**（控制信号相对当前根重锚定）两模式；真机 G1、高层 BFM 50 Hz（TensorRT）、低层 PD 200 Hz（LCM + SDK2）。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md)（对照：SONIC 亦用 motion tracking 作 BFM 预训练，但 scaling 结论较初步）
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 3) 数据 scaling：on-policy 数量 × 参考多样性（§3.3）

- **核心贡献：**
  - **数量（PPO on-policy rollouts）：** 有效训练数据由 **并行 GPU 数（width）** 与 **rollout horizon（depth）** 共同决定；单独扩一维不稳定，**64 GPU × horizon 64** 在多数控制模式下最优。
  - **多样性（reference motions）：** 聚合 LAFAN、AMASS、OMOMO、GRAB、SnapMoGen、FineDance、BONES-SEED、Embody3D → **102M frames @ 50 FPS**；两阶段重定向（shape 对齐 + 逐帧 IK）。
  - **同质 vs 异质 scaling：** 仅增加同域 clip（XXS→S，occupancy rate ~0.94）收益边际；引入多源异质数据（S→L，occupancy →0.9995）在 **Ours Test Set**（Xsens + 100Style）上大幅提升，BONES 域内增益有限。
  - **RSI + 自适应采样：** 全局偏差 >0.5 m 早停；失败轨迹采样权重上调（ProtoMotions 风格）。
- **对 wiki 的映射：**
  - [AMASS](../../wiki/entities/amass.md)
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)

### 4) Humanoid Transformer 架构与潜空间（§3.4 / §4.3）

- **核心贡献：**
  - **Humanoid Transformer：** 本体/动作/目标分模态 tokenizer；本体+动作 token 交错作 context，**query token** 读全历史；目标序列经 **cross-attention** 注入；**RMSNorm 超球面潜空间** 无辅助 loss 即呈现 locality / global organization / 噪声鲁棒性。
  - **规模：** S 0.41M / M 3.00M / L 4.44M / XL 9.91M；**M 档 Transformer 已优于更大 MLP**；继续放大 Transformer 出现 **mode-dependent 饱和**（不同控制模式潜表示收敛耦合）。
  - **实时：** actor 未来窗含随机远帧 offset（缓解通信延迟）；critic 用指数间隔长窗 $\{0,1,2,4,8,16,32\}$。
- **对 wiki 的映射：**
  - [Humanoid Policy Network Architecture](../../wiki/concepts/humanoid-policy-network-architecture.md)

### 5) 基准与真机（§4.4 / Appendix）

- **仿真：** IsaacLab 训练，MuJoCo 评测；BONES Test（10k held-out）+ Ours Test（Xsens 839 + 100Style 810 seq）。
- **Whole-body 基准（3M 参数）：** BFM-Global 在 BONES 上 Succ **0.9677**、G-MPKPE **0.0798**；Ours 上 Succ **0.9776**、G-MPKPE **0.0915**；显著优于 SONIC（0.9239 / 0.5937）、GMT、TWIST；优于同管线 **BFM-Bym**（BeyondMimic reward 消融）。
- **真机：** 操作/ loco-manip / 高动态技能 / 八模式稀疏约束 / 扰动恢复；论文承诺 **开源全部资源**（仓库 README 写明分周释出）。
- **局限：** 八模式接口是否最优、与高层策略如何集成仍开放；分布式人形 BFM 训练基础设施仍不成熟；机载算力限制参数量上界。
- **对 wiki 的映射：**
  - [ReactiveBFM](../../wiki/entities/paper-reactivebfm.md)（同团队闭环上层）
  - [Sim2Real](../../wiki/concepts/sim2real.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-scaling-bfm-humanoid.md`](../../wiki/entities/paper-scaling-bfm-humanoid.md)
- 关联升级：
  - [BFM 论文实体](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) — 同作者团队 CVAE 路线；ScaleBFM 为 **PPO + Transformer scaling 技术报告**
  - [Behavior Foundation Model](../../wiki/concepts/behavior-foundation-model.md) — goal-conditioned 线补 **scaling recipe**
  - [SONIC](../../wiki/methods/sonic-motion-tracking.md) — 主要 motion-tracking scaling 对照基线
  - [ReactiveBFM](../../wiki/entities/paper-reactivebfm.md) — 共享作者与 Shanghai AI Lab 栈

## 其他公开资料

- **项目页：** <https://scalebfm.github.io/> — scaling 曲线、潜空间可视化、真机八模式与 loco-manip 演示
- **代码仓：** <https://github.com/zengweishuai/ScaleBFM>
- **Citation：** Zeng et al., arXiv:2607.15163, 2026

## 当前提炼状态

- [x] 摘要与三轴 scaling 摘录（范式 / 数据 / 架构）
- [x] 项目页与代码开放状态核查（2026-07-18）
- [x] wiki 实体页规划
