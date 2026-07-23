# Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control（arXiv:2607.20110）

> 来源归档（ingest）

- **标题：** Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist Humanoid Control
- **缩写：** **Extreme-RGMT**（前作 **RGMT** = Robust and Generalized Humanoid Motion Tracking）
- **类型：** paper / humanoid / motion-tracking / continual-learning / whole-body-control
- **arXiv：** <https://arxiv.org/abs/2607.20110>（Submitted 2026-07-22；PDF：<https://arxiv.org/pdf/2607.20110>）
- **项目页：** <https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/> — 归档见 [`sources/sites/extreme-rgmt-github-io.md`](../sites/extreme-rgmt-github-io.md)
- **代码：** **未开源**（截至 2026-07-23 项目页无 GitHub / HF 链接）
- **作者：** Yubiao Ma\*、Han Yu\*、Kai Guo、Changtai Lv、Zhengquan Mao、Boyang Xing、Xuemei Ren、Dongdong Zheng（\*共一；通讯：Dongdong Zheng、Kai Guo）
- **机构：** 北京理工大学自动化学院；人形机器人（上海）有限公司（青龙 / OpenLoong，邮箱域 `openloong.net`）；山东大学机械工程学院
- **入库日期：** 2026-07-23
- **一句话说明：** 在 RGMT 动力学条件参考编码之上，用 **两阶段** 训练：Stage I 多样本源 generalist 基座 + 按完成率分层；Stage II 以 **PACE** 非对称 acquisition/consolidation 与 **STAR** 优势优先轨迹重采样，把高动态技能叠进同一策略，并在 Unitree G1 上做固定回放与 Xsens 在线遥操作。

## 开源状态（步骤 2.5）

- **项目页核查（2026-07-23）：** 仅 arXiv PDF + BibTeX + 演示视频；**无代码 / 权重 / 数据集下载链**。
- **结论：** **确认未开源**。wiki「工程实践 / 源码运行时序图」写明不适用。

## 摘录 1：问题与主张（§I）

- **痛点：** generalist tracker 难可靠执行稀有高动态段（空中调姿、触地恢复）；specialist 优化又易冲掉已掌握日常能力；在线惯性 MoCap 参考还带时序误差 / 根漂 / 姿态不一致。
- **主张：** Extreme-RGMT = Stage I generalist 基座 → 按 tracking 统计分层 \(\mathcal{D}_m\) / \(\mathcal{D}_c\) → Stage II **PACE** + **STAR**。
- **硬件口号：** 据称是首个能对 **惯性 MoCap 在线输入** 做高动态全身跟踪的 generalist 控制器（论文 §I / §VI-C）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-extreme-rgmt.md`](../../wiki/entities/paper-extreme-rgmt.md)；与 [RGMT](../../wiki/entities/paper-hrl-stack-14-robust_and_generalized_humanoid_moti.md)、[Any2Track & RGMT](../../wiki/methods/any2track.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md) 互链。

## 摘录 2：架构与两阶段训练（§III–§V）

- **接口：** POMDP；actor 用 10 帧本体+动作历史 + 21-token 参考窗；动作 = 29 维残差关节位置 \(q^{\mathrm{tar}}=q^{\mathrm{ref}}+a\)；非对称 actor–critic（critic 特权信息）。
- **相对 RGMT 的增强：** 本体/动作历史 **分支出编码 + LayerNorm**；动力学引导 cross-attn 聚合参考；**FSQ** 瓶颈正则聚合命令特征。
- **数据（Stage I）：** LAFAN1 ≈2.44 h、AMASS ≈0.51 h、自采 Xsens ≈0.14 h，合计 ≈3.1 h；重定向到 Unitree G1、50 Hz。
- **分层：** clip 完成率 ≥80% → \(\mathcal{D}_m\)（≈2.82 h，consolidation）；其余 → \(\mathcal{D}_c\)（≈0.28 h，acquisition）。
- **PACE：** 并行环境 \(\xi=0.8\) 做 acquisition（自适应采样 \(\mathcal{D}_c\)）+\(1-\xi\) consolidation（均匀 \(\mathcal{D}_m\)）；consolidation 损失 \(\|a_\theta-a_{\mathrm{ref}}\|_2^2\)；\(\lambda_{\mathrm{con}}\) 随有效 acquisition 样本比进度自适应。
- **STAR：** 把自适应采样 bin 难度传到 transition 权重 → 难度条件 advantage 归一化 → 按 raw advantage 选高优势片段 → \(\rho_{\mathrm{star}}=0.25\) 混入 PPO mini-batch。

**对 wiki 的映射：** 实体页画两阶段 flowchart；强调「同一策略扩展高动态，而非另训 specialist」。

## 摘录 3：实验要点（§VI）

| 设定 | 关键数字（论文 Table VI / VIII） |
|------|----------------------------------|
| Generalist In-source Succ. | Full **99.76%**（Stage I 99.54%；RGMT 同数据 99.12%） |
| Generalist Unseen Succ. | Full **96.68%**（Stage I 95.13%） |
| Specialist XtremeMotion Succ. | Full **100%**（Stage I 仅 21.42%；OmniXtreme 亦 100% 但更偏其库） |
| Specialist AMASS Challenging | Full **90.91%**（OmniXtreme 36.16%；说明对低质量直接重定向更鲁棒） |
| 真机 AMASS Replay | **90%**；高动态 Xsens Teleop **85%**；日常 Xsens Teleop **100%**（各 4 动作×5 试验） |
| STAR 增益（Table VII） | Xsens 高动态 Succ. **45.5% → 86.3%**（+40.8） |

- **对照：** ExBody2、BeyondMimic、SONIC、RGMT（generalist）；OmniXtreme、直接 Fine-Tuning（specialist）。
- **局限（§VI-D）：** 根相对参考、长时全局漂移；对训练分布外协调模式泛化有限。

**对 wiki 的映射：** 用 generalist–specialist 权衡表写清相对 OmniXtreme / Fine-Tuning 的定位。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-extreme-rgmt.md`**（含流程总览；源码时序图标不适用）。
- 新建 **`sources/sites/extreme-rgmt-github-io.md`**。
- 更新 **`wiki/entities/paper-hrl-stack-14-...`**、**`wiki/methods/any2track.md`**、**`wiki/methods/sonic-motion-tracking.md`**、**`wiki/methods/beyondmimic.md`**、**`wiki/overview/humanoid-rl-motion-control-body-system-stack.md`** 交叉引用。
