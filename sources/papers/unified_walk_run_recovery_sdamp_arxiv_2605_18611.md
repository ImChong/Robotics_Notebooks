# Unified Walking, Running, and Recovery for Humanoids via State-Dependent Adversarial Motion Priors（arXiv:2605.18611）

> 来源归档（ingest）

- **标题：** Unified Walking, Running, and Recovery for Humanoids via State-Dependent Adversarial Motion Priors
- **类型：** paper / humanoid locomotion / AMP / fall recovery / sim2real
- **arXiv HTML：** <https://arxiv.org/html/2605.18611v1>
- **arXiv abs：** <https://arxiv.org/abs/2605.18611>
- **PDF：** <https://arxiv.org/pdf/2605.18611>
- **机构：** The University of Hong Kong（Yidan Lu, Yichao Zhong, Liu Zhao, Wanyue Li；通讯作者 Peng Lu）
- **硬件：** Unitree G1（真机验证，无部署期显式模式切换）
- **入库日期：** 2026-05-25
- **开源状态（2026-07-20 再核）：** 论文 HTML / abs **无** 官方 GitHub；工程侧统一 walk/run/recovery 对照见 [AMP_mjlab](../repos/amp_mjlab.md)（非本文双判别器实现）。
- **一句话说明：** 用**状态相关 AMP（SD-AMP）**在训练期按投影重力门控切换 recovery / 速度条件 locomotion 两个判别器，**三条 LAFAN1 参考片段**即可让**单一策略**在 G1 上统一走、跑与俯卧/仰卧起身，部署为 50 Hz 冻结 ONNX、无运行时 FSM。

## 摘要级要点

- **问题：** 传统模块化 FSM（走 / 跑 / 起身分控）需手调切换、边界脆弱；全局单一 AMP 先验难以同时覆盖稳态 locomotion 与高动态 recovery 的不同状态转移分布。
- **方法：** 共享 actor $\pi_\theta$；训练时每步用门控 $g(s_t)$ 选活跃判别器：
  - **Recovery 判别器** $D_\phi^{\mathrm{rec}}$：仅 fall-recovery 参考转移；
  - **Locomotion 判别器** $D_\phi^{\mathrm{loco}}$：以归一化速度命令 $\hat{v}_t\in[0,1]$ 为条件，在 walk / run 参考间插值采样。
- **门控（固定阈值，非学习分类器）：** $|g_z+1|>0.6$（约 37° 倾角）→ recovery；否则 → locomotion。$g_z$ 为投影重力 $z$ 分量（直立时 $\approx -1$）。
- **AMP 奖励：** 按分支取 $-\log(1-D(\cdot))$；总奖励 $R_t^{\mathrm{total}}=R_t+\lambda_{\mathrm{amp}} R_{\mathrm{AMP}}$，$\lambda_{\mathrm{amp}}=0.5$。
- **参考数据：** LAFAN1 三条 retarget 到 G1：`walk1_subject1`、`run1_subject2`、`fallAndGetUp2_subject2`；locomotion 分支以概率 $(1-\hat{v}_t)$ / $\hat{v}_t$ 混合 walk/run 转移。
- **观测 / 动作：** 单帧 96 维（角速度、投影重力、速度命令、相对关节位、关节速、上步动作）；**4 帧堆叠 → 384 维**；动作 29 维关节目标位置 + 底层 PD。
- **训练 / 部署：** Isaac Lab + PPO；收敛后导出 ONNX，真机 50 Hz C++ FSM 读关节状态发 PD 目标；**无 sim2real 额外微调**叙述（仅训练期标准域随机化）。
- **实验：** 正常模式速度跟踪约 $[-0.5,1.0]$ m/s；快速模式（操作员显式启用安全约束）约 $[-1.5,3.0]$ m/s；俯卧 / 仰卧起身后连贯 **recovery → walk → run** 硬件 rollout。

## 核心摘录（面向 wiki 编译）

### 与 Selective AMP / 统一策略工程线的关系

| 维度 | 本文 SD-AMP | Selective AMP（多步态） | AMP_mjlab（工程实现） |
|------|-------------|-------------------------|------------------------|
| 先验切换依据 | **机体倾角（投影重力）** | **步态类型（周期 vs 高动态）** | 训练配置 / 参考库分区（walk-run vs recovery） |
| 判别器数量 | 2（recovery + 速度条件 loco） | 按步态是否启用 AMP | 通常 1 个 AMP 判别器 + 多类参考 |
| 部署模式逻辑 | **无** | 无（统一策略） | 无（统一策略） |
| 参考规模 | **3 条 LAFAN1 clip** | 多步态 MoCap | WalkRun + Recovery npz |

### 与 Heracles / 纯 tracking 的对比（索引级）

- **SD-AMP**：在 **RL + 对抗 motion prior** 框架内用**训练期门控**分离 recovery 与 locomotion 风格正则；部署仍是单策略。
- **Heracles（arXiv:2603.27756）**：在 **参考跟踪器之上**加 **状态条件扩散中间件**，扰动大时生成恢复关键帧再交给 tracker——属于 **生成式中间层 + 跟踪执行** 分层，而非 AMP 判别器路由。

## 对 wiki 的映射

- 沉淀实体页：[SD-AMP 统一走跑起身（arXiv:2605.18611）](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)
- 交叉补强：[AMP & HumanX](../../wiki/methods/amp-reward.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Balance Recovery](../../wiki/tasks/balance-recovery.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[AMP_mjlab](../../wiki/entities/amp-mjlab.md)、[LAFAN1](../../wiki/entities/lafan1-dataset.md)
