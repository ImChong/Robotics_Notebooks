# Athena-WBC: Capability-Aligned Policy Experts for Long-Tail Humanoid Whole-Body Control（arXiv:2607.04837）

> 来源归档（ingest）

- **标题：** Athena-WBC: Capability-Aligned Policy Experts for Long-Tail Humanoid Whole-Body Control
- **类型：** paper / humanoid WBC / motion tracking / long-tail / teacher-student / expert distillation
- **arXiv abs：** <https://arxiv.org/abs/2607.04837>
- **PDF：** <https://arxiv.org/pdf/2607.04837>
- **项目页：** 暂无公开项目页或代码链接
- **发表日期：** 2026-07-06
- **机构：** XPENG Robotics（小鹏机器人）
- **通讯作者：** Jie Chen（chenj81@xiaopeng.com）
- **入库日期：** 2026-07-08
- **一句话说明：** **Athena-WBC** 针对大规模人形 motion tracking 中 **训练集长尾残余失败**（高动态过渡、平衡关键姿态），用 **能力对齐** 的 dynamic / balance **privileged teacher**（改奖励与重力课程，而非仅重采样）→ **按动作路由** → **DAgger 蒸馏** → **RL 微调**，在 **80 kg 小鹏人形** 上相对 **SONIC-Base 配方** 改善长尾恢复与 held-out 跟踪；提出 **STC / TIS / MPJPE-W** 评测。

## 摘要级要点

- **问题定义：** 高覆盖率 WBC 下，训练集仍有 **residual long-tail**：高动态（急转、激进接触切换）与 **balance-critical**（低支撑、慢恢复）片段在 SONIC 发布权重上仍失败；部分片段 **仅对失败子集重训仍学不会** → **capability bottleneck**（默认训练配方诱导的控制能力不匹配），而非单纯曝光不足。
- **与常见对策的区别：** 难样本过采样、动作聚类分专家（BumbleBee）、MoE（GMT/EGM）等主要改 **数据分配**；Athena-WBC 改 **acquisition recipe**（奖励分解、课程、辅助正则），再压缩为单策略。
- **奖励分解：** $r_t = r^{\mathrm{track}}_t + r^{\mathrm{phys}}_t + r^{\mathrm{effort}}_t + r^{\mathrm{temp}}_t$；$r^{\mathrm{phys}}$ 为关节/力矩/脚滑等 **物理可行性**；$r^{\mathrm{effort}}/r^{\mathrm{temp}}$ 为 **保守控制偏好**，可抑制高动态可行动作。
- **Round 1 — 通用 privileged teacher** $\pi_T^{\mathrm{gen}}$：全训练集；评估每 clip $K_{\mathrm{eval}}=10$ 次 rollout，$\widehat{\mathrm{SR}}<0.8$ 划入残余集 $\mathcal{R}_{\mathrm{gen}}$。
- **Round 2 — 能力专家（同残余集、不同配方）：**
  - **Dynamic expert** $\pi_T^{\mathrm{dyn}}$：保留 tracking + physical constraint，**去掉** effort/temporal reward；用 **Grad-CAPS** 辅助损失约束动作均值二阶差分，允许大但结构化动作变化。
  - **Balance expert** $\pi_T^{\mathrm{bal}}$：**重力课程** $g_e=\alpha_e g_0$，$\alpha_e$ 从低到 1，延长早期 rollout、改善冷启动存活。
- **Motion routing：** 冻结 $\{\pi_T^{\mathrm{gen}}, \pi_T^{\mathrm{dyn}}, \pi_T^{\mathrm{bal}}\}$，每 clip 选 $\widehat{\mathrm{SR}}$ 最高 teacher 作蒸馏监督源（非手工 dynamic/balance 标注）。
- **DAgger 蒸馏 → 单 student**（可部署观测）；**RL fine-tuning**（Parkour in the Wild 式 critic warm-up + 渐解冻 actor PPO）。
- **平台与数据：** **80 kg** 行星滚柱丝杠 + 闭链 **小鹏人形**（非 G1）；训练集 **55,482 clips / 175.88 h**（AMASS 子集、Bones-Seed、BEAT、策展 mocap）；基线为 **同平台重实现的 SONIC-Base**（非 SONIC 发布权重）。
- **训练算力：** 8 GPU × 2048 env = 16,384 并行；主 checkpoint **40k iter**；全程 **adaptive motion sampling**（实现细节与 SONIC/EGM 有差异）。
- **评测：** 除 SR / MPJPE 外，**STC**（阈值缩放 SR 曲线）、**TIS**（阈值积分成功率）、**MPJPE-W**（按动作显著性加权关节误差）；held-out：**AMASS-eval**（10 h IID）、**Omni-eval**（227 clips 难动作 stress test）。
- **主要结果（RL 微调最终策略 vs SONIC-Base）：** AMASS-eval SR **98.18→99.26%**，TIS **0.7631→0.8034**，MPJPE **68.26→63.63 mm**；Omni-eval SR **91.81→94.89%**，TIS **0.7123→0.7449**。No-smoothness 基线默认 SR 更高但 **Action Rate 1.46**（不可部署）；RL 微调在 TIS/MPJPE 上更优且 Action Rate **1.03**。
- **长尾恢复（Q2）：** multi-teacher student 在 balance 子集 SR **94.73%**、MPJPE-W **88.61 mm** 优于 SONIC-Base（90.60% / 101.99 mm）；**RL 微调主要贡献 held-out 泛化而非训练集长尾恢复**。
- **局限：** 训练集覆盖率仍非 100%；RL 微调有时 **牺牲训练集覆盖换 held-out**；管线阶段多、工程重；**尚无系统真机定量结果**（内部平台开发中）。

## 对 wiki 的映射

- 沉淀实体页：[paper-athena-wbc-humanoid-longtail.md](../../wiki/entities/paper-athena-wbc-humanoid-longtail.md)
- 交叉：[SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)、[DeepInsight](../../wiki/entities/deepinsight.md)、[humanoid motion tracking 选型 query](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 参考来源（原始）

- arXiv:2607.04837（2026-07-06）
- 对比基线配方：SONIC（arXiv:2511.07820）、OmniH2O teacher 观测（CoRL 2024）、Parkour in the Wild 蒸馏+RLFT（IJRR 2026）、Grad-CAPS（IROS 2024）
