# Athena-WBC: Capability-Aligned Policy Experts for Long-Tail Humanoid Whole-Body Control（arXiv:2607.04837）

> 来源归档（ingest · 含 v2 全文核对与 v3 撤稿状态记录）

- **标题：** Athena-WBC: Capability-Aligned Policy Experts for Long-Tail Humanoid Whole-Body Control
- **作者：** Yuan Jiang\*, Ningyuan Zhang\*, Xicun Yang, Yuzhi Jiang, Jie Chen†（† 通讯：`chenj81@xiaopeng.com`）
- **类型：** paper / humanoid WBC / motion tracking / long-tail / teacher-student / expert distillation
- **arXiv abs：** <https://arxiv.org/abs/2607.04837>（v1 提交 2026-07-06；v2 2026-07-07；**v3 2026-08-24 已由作者撤稿**）
- **v2 PDF（已读）：** <https://arxiv.org/pdf/2607.04837v2>
- **项目页 / 代码：** 截至核查日无公开项目页或代码仓库链接
- **机构：** XPENG Robotics（小鹏机器人）
- **入库日期：** 2026-07-08
- **核对范围：** 核查 `https://arxiv.org/abs/2607.04837`（确认 v3 于 2026-08-24 被作者撤稿："Withdrawn by the authors due to unresolved authorization and data-governance concerns affecting the dataset used in the experiments and, consequently, the reported results"），并下载通读公开留存的 `arXiv:2607.04837v2` PDF 全文（§1–7、Table 1–6、Fig. 1–11 题注与方法公式）。未包含公开代码或真机实测。
- **通读：** 已完成
- **一句话说明：** **Athena-WBC** 针对大规模人形 motion tracking 中 **训练集长尾残余失败**（高动态过渡、平衡关键姿态），用 **能力对齐** 的 dynamic / balance **privileged teacher**（改奖励与重力课程，而非仅重采样）→ **按动作路由** → **DAgger 蒸馏** → **RL 微调**，在 **80 kg 小鹏人形** 上相对 **SONIC-Base 配方** 改善长尾恢复与 held-out 跟踪；提出 **STC / TIS / MPJPE-W** 评测。**注意：v3 已因实验数据集授权与数据治理争议被作者撤稿，引用实验数字时须附撤稿风险提示。**

## 撤稿状态记录（2026-08-24 · arXiv v3）

- **撤稿声明原文（arXiv:2607.04837v3）：** *"Withdrawn by the authors due to unresolved authorization and data-governance concerns affecting the dataset used in the experiments and, consequently, the reported results. Readers should not rely on this work while these issues are under review."*
- **当前影响：** 默认 `arxiv.org/pdf/2607.04837` 返回 404，历史版本 `v2`（`arxiv.org/pdf/2607.04837v2`）仍可访问用于核对方法机制；但实验结论受限于数据集治理审查，工程选型不得把文中定量结果视为已定论基准。

## 全文核心摘录（arXiv:2607.04837v2）

- **问题定义（§1, §3）：** 高覆盖率 WBC 下，训练集仍有 **residual long-tail**：高动态（急转、激进接触切换）与 **balance-critical**（低支撑、慢恢复）片段在 SONIC 发布权重上仍失败；部分片段 **仅对失败子集重训仍学不会** → **capability bottleneck**（默认训练配方诱导的控制能力不匹配），而非单纯曝光不足。
- **与常见对策的区别（§2）：** 难样本过采样、动作聚类分专家（BumbleBee）、MoE（GMT/EGM）等主要改 **数据分配**；Athena-WBC 改 **acquisition recipe**（奖励分解、课程、辅助正则），再压缩为单策略。与 OmniTrack「参考物理修正即可单策略覆盖」的结果在 80 kg 闭链人形上形成互补且部分对立的观察。
- **奖励分解（§3.1）：** $r_t = r^{\mathrm{track}}_t + r^{\mathrm{phys}}_t + r^{\mathrm{effort}}_t + r^{\mathrm{temp}}_t$；$r^{\mathrm{phys}}$ 为关节/力矩/脚滑等 **物理可行性**；$r^{\mathrm{effort}}/r^{\mathrm{temp}}$ 为 **保守控制偏好**，可抑制高动态可行动作。
- **Round 1 — 通用 privileged teacher（§4.1）：** $\pi_T^{\mathrm{gen}}$ 在全训练集训练；评估每 clip $K_{\mathrm{eval}}=10$ 次 rollout，$\widehat{\mathrm{SR}} < \rho_{\mathrm{fail}}=0.8$ 划入残余集 $\mathcal{R}_{\mathrm{gen}}$。
- **Round 2 — 能力专家（同残余集、不同配方，§4.1）：**
  - **Dynamic expert** $\pi_T^{\mathrm{dyn}}$：保留 tracking + physical constraint，$r^{\mathrm{effort}}=r^{\mathrm{temp}}=0$；用 **Grad-CAPS** 辅助损失 $\mathcal{L}_{\mathrm{GradCAPS}}$ 约束策略均值 $\mu_\theta(o_t)$ 二阶差分（$\mathcal{L}=\mathcal{L}_{\mathrm{PPO}}+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{GradCAPS}}$），分离「激进跟踪」与「高频抖动」。
  - **Balance expert** $\pi_T^{\mathrm{bal}}$：**重力课程** $g_e=\alpha_e g_0$，$\alpha_e \in [\alpha_{\min}, 1]$ 渐增，延长早期 rollout、改善冷启动存活。
- **观测维度与路由蒸馏（Table 1, §4.2–4.4）：** Teacher 观测 1837 维（含 OmniH2O 式特权自/任务状态），Student/SONIC-Base 可部署观测 2081 维（含 10 帧历史本体与参考目标）。冻结 $\{\pi_T^{\mathrm{gen}}, \pi_T^{\mathrm{dyn}}, \pi_T^{\mathrm{bal}}\}$，每 clip 选 $\widehat{\mathrm{SR}}$ 最高 teacher 作 DAgger 监督源；之后按 Parkour in the Wild 流程做 critic warm-up（冻结 actor + 加噪 rollout）再渐解冻 actor 做 PPO 微调（关闭特权量与课程）。全程保持 adaptive motion sampling。
- **平台与数据（§5.1）：** **80 kg** 行星滚柱丝杠 + 闭链 **小鹏人形**（非 G1）；训练集 **55,482 clips / 175.88 h**（Bones-Seed 46,341 clips / 95.66h、AMASS 7,333 clips / 25.11h、BEAT 1,574 clips / 43.47h、自采 mocap 234 clips / 11.64h）；基线为 **同平台重实现的 SONIC-Base**（非 SONIC 发布权重）。8 GPU × 2048 env = 16,384 并行；主 checkpoint **40k iter**。
- **评测指标与主要结果（§5.2–5.5, Table 3–6，注意 v3 撤稿背景）：**
  - 阈值规则：$\delta_z^0=0.20\text{ m},\,\delta_{\mathrm{yaw}}^0=0.50\text{ rad},\,\delta_{\mathrm{kb}}^0=0.50\text{ m}$；提出 **STC**（缩放因子 $s$）、**TIS**（$\int_0^\infty \mathrm{SR}(s)/(1+s)^2 ds$）、**MPJPE-W**（按显著性加权误差）。
  - Held-out（Table 3，RLFT vs SONIC-Base）：AMASS-eval（10h IID）SR 98.18→99.26%、TIS 0.7631→0.8034、MPJPE 68.26→63.63 mm；Omni-eval（227 clips / ~3h 难动作）SR 91.81→94.89%、TIS 0.7123→0.7449、MPJPE 70.39→65.11 mm。No-smoothness 基线 Action Rate 1.4634（SONIC-Base 0.5447，RLFT 1.03）。
  - 长尾恢复（Table 4）：multi-teacher student 在 $D_{\mathrm{balance}}$ SR 94.73%、MPJPE-W 88.61 mm（SONIC-Base 90.60% / 101.99 mm）；RLFT 后 $D_{\mathrm{balance}}$ SR 略降至 92.55%（RLFT 主要贡献 held-out 与部署平滑度）。
  - 频域分析（§5.5）：NoSmooth 高频（8–25 Hz）动作能量占比 0.063 vs SONIC-Base 0.009 vs Grad-CAPS 0.007（4319 条中有 3441 条低于 SONIC-Base 对角线）。
- **局限（§6）：** 训练集覆盖率仍未达 100%；Grad-CAPS 仍存在低频晃动（low-frequency sway）；多阶段管线工程重、超参多；RLFT 存在训练集覆盖与 held-out 泛化权衡；算力受限；**无系统真机定量验证**（且实验数据集后续在 v3 引发撤稿争议）。

## 对 wiki 的映射

- 沉淀实体页：[paper-athena-wbc-humanoid-longtail.md](../../wiki/entities/paper-athena-wbc-humanoid-longtail.md)
- 交叉：[SONIC 方法页](../../wiki/entities/nvidia-sonic.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)、[DeepInsight](../../wiki/entities/deepinsight.md)、[humanoid motion tracking 选型 query](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 参考来源（原始）

- arXiv:2607.04837（v2 全文 + v3 撤稿记录）：<https://arxiv.org/abs/2607.04837>
- 对比基线配方：SONIC（arXiv:2511.07820）、OmniH2O teacher 观测（CoRL 2024）、Parkour in the Wild 蒸馏+RLFT（IJRR 2026）、Grad-CAPS（IROS 2024）
