# Extreme Parkour with Legged Robots（arXiv:2309.14341）

> 来源归档（ingest）

- **标题：** Extreme Parkour with Legged Robots
- **类型：** paper / 四足 locomotion / 感知跑酷 / Teacher–Student 蒸馏
- **arXiv：** <https://arxiv.org/abs/2309.14341>（PDF：<https://arxiv.org/pdf/2309.14341>）
- **会议：** ICRA 2024；CoRL 2023 Generalist / Roboletics / Deployable Workshop（Oral）
- **项目页：** <https://extreme-parkour.github.io/>
- **代码：** <https://github.com/chengxuxin/extreme-parkour>
- **作者：** Xuxin Cheng*, Kexin Shi*, Ananye Agarwal, Deepak Pathak（CMU / Pathak Lab）
- **入库日期：** 2026-06-04（2026-11-07 按 arXiv v1 全文通读复核）
- **版本：** arXiv:2309.14341v1（2023-09-25），pdftotext 提取全文
- **核对范围：** 通读摘要、§1 引言、§2 相关工作（含同期工作）、§3 方法（统一奖励式 1–4、Phase 1 scandots RL、Phase 2 双重蒸馏与 MTS）、§4 结果（实机设置、高跳 / 远跳 / 手倒立、表 2 / 表 3 / 图 7 基线对比）、§5 讨论；参考文献仅浏览。该 v1 无独立附录。未读：项目页视频内容与代码（仓库另见 `sources/repos/extreme-parkour.md`），未运行代码。
- **通读：** 已完成
- **一句话说明：** 在 **Isaac Gym + legged_gym** 上用 **两阶段 RL + 双重蒸馏**（特权 scandots→深度、oracle 航向→自预测 yaw），训练 **Unitree A1**（论文 §4.1 实机平台；非 Go1）低成本四足 **端到端深度跑酷** 策略；约 **20 小时** 仿真训练即可实机高跳 / 远跳 / 手倒立 / 斜坡变向。

## 摘要级要点

- **问题：** 传统跑酷依赖 **分模块高精度工程**（感知 / 执行 / 控制分别调参），难以泛化到新障碍组合；人类则通过练习在 **不改造硬件** 的前提下习得技能。
- **平台约束：** 小型低成本四足、**执行器不精确**、仅 **单目前向深度相机**（低频、抖动、伪影）——刻意贴近真实 onboard 限制。
- **主张：** **单神经网络** 从深度图直接输出关节控制；仿真 **大规模 RL** 克服感知与执行噪声，实机展现 **高精度** 动态动作。
- **能力（项目页 / 论文）：** **2× 身高高跳**、**2× 体长远跳**、手倒立、倾斜 ramp 上跑并 **自主变向**；对新障碍组合与不同物理属性 **零样本泛化**；CoRL 2023 **2 分钟不间断** 攀爬 / 远跳 / 跳下演示。
- **Phase 1 — 特权 scandots RL：** PPO 训练 base policy，输入 **本体 $x$ + scandots $m$ + oracle 航向 $\hat{d}$（terrain waypoint 计算）+ 行走标志 $W$ + 速度指令**；用 **ROA（Regularized Online Adaptation）** 从历史观测估计环境参数（RMA 思路）。
- **Phase 2 — 双重蒸馏：** Teacher 依赖的两类特权在部署时不可用——(1) scandots → **ConvNet–GRU 深度管线**（RMA 式）；(2) waypoint 航向 → **从深度自预测 yaw**。外感知与航向均用 **DAgger** 向 Phase 1 行为对齐。
- **统一奖励（inner-product）：** 航向不再随机采样，而由 **terrain 上 red-dot waypoint** 给出 $\hat{d}_w = (p-x)/\|p-x\|$，速度跟踪奖励与之对齐；避免「绕障碍转圈」 exploit。
- **Clearance 惩罚：** $r_{\mathrm{clearance}} = -\sum_{i=0}^{4} c_i \cdot M[p_i]$，足端接触点距地形边缘 **< 5 cm** 时惩罚——否则大 gap 上策略会 **贴边省能** 导致实机失败（项目页 ablation）。
- **Direction distillation ablation：** 无航向蒸馏时 ramp 地形 **摇杆遥控几乎不可控**（需自主选跳上角与即时变向，人类无法实时给 waypoint）。
- **训练成本（README）：** Base **10–15k iter（3090 约 8–10 h，建议 ≥15k）**；Distillation **5–10k iter（5–10 h，建议 ≥5k）**；合计 **< 20 h** 量级。

## 对 wiki 的映射

- 沉淀实体页：`wiki/entities/extreme-parkour.md`
- 交叉更新：`wiki/concepts/privileged-training.md`、`wiki/entities/legged-gym.md`、`wiki/tasks/locomotion.md`、`wiki/methods/dagger.md`
- 姊妹工作：Robot Parkour Learning（ZiwenZhuang / Unitree）、DreamWaQ++、PHP（人形跑酷）

## 2026-11-07 通读复核补充（出自论文原文）

- **平台：** Unitree **A1**（12 关节；站立时大腿关节高 26 cm、体长 40 cm）；头部 Intel RealSense D435，深度 10 ± 2 Hz；深度骨干（10 Hz）与基策略（50 Hz）都跑在 Jetson NX，经 UDP 通信；图像裁掉左侧坏像素并降采样到 58 × 87；深度延迟固定为 0.08 s、本体延迟固定 0.016 s；可部署策略在单张 3090 上不到 20 小时训练完成。
- **Table 1（与同期工作对比）：** Climb / Gap 为相对机身高度 / 体长的倍数——Rudin et al. 1.1 / 0.75；Hoeller et al.（ANYmal Parkour）2 / 1.5；Zhuang et al.（Robot Parkour Learning, A1）1.6 / 1.5；本文 2 / 2，ramp 37°，且有手倒立。
- **奖励：** 式 (2) $r_{tracking}=\min(\langle v,\hat d_w\rangle, v_{cmd})$，速度在**世界系**而非机体系（防止绕障碍转圈 exploit）；式 (3) clearance 对「距地形边缘 5 cm 内」的触地足惩罚；式 (4) stylized 奖励 $W\cdot(0.5\langle\hat v_{fwd},\hat c\rangle+0.5)^2$，手倒立取 $\hat c=[0,0,-1]^T$，$W$ 训练时随机取 {0,1}、部署时遥控器切换；另加 Cheng 2023（Legs as Manipulator）的正则项。
- **课程：** 机器人在易级别出生，走过超过一半长度则升级，走不到 $v_{cmd}T$ 的一半则降级。
- **Phase 2：** DAgger 蒸馏，用学生动作步进环境；actor 由 Phase 1 复制初始化；航向预测的深度编码器无预训练，用 MTS（mixture of teacher and student）：$|\theta_{pred}-\hat d_w|<0.6$ 时观测预测 yaw，否则用 oracle。
- **表 2（仿真 256 机器人 / 30 s，总计）：** MXD Ours 0.98 / NoInner 0.75 / NoClear 0.99 / Noisy 0.82；MEV Ours 0.03 / NoInner 0.04 / NoClear 0.08 / Noisy 0.20。NoInner 在 Step 上 MXD 仅 0.14（只会撞台阶重试）。
- **表 3（蒸馏）：** Ours MXD 0.92 / MEV 0.09，接近 Oracle 0.94 / 0.10；Both 0.12 / 0.26，Mask 0.05 / 0.00（噪声 yaw 导致分布漂移）。
- **实机（图 7）：** 每地形每难度 5 次试验，最难档上成功率比 NoClear / NoDir 高 20–80%；NoDir 由熟练操作员摇杆给方向，在跳跃与斜坡的突然变向处失败。
- **手倒立策略不带外感知**，仅凭本体感知即可在草地与下楼梯上行走。
- **论文自述的未来方向：** 把同一方法扩展到移动操作；未设专门「局限」小节。
