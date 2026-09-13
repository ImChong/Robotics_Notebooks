# EAGLE-WBC: Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control（arXiv:2602.02960）

> 来源归档（ingest）

- **标题：** Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control
- **简称：** EAGLE / EAGLE-WBC
- **类型：** paper / humanoid WBC / cross-embodiment / generalist-specialist distillation
- **arXiv abs：** <https://arxiv.org/abs/2602.02960>
- **PDF：** <https://arxiv.org/pdf/2602.02960>
- **项目页：** <https://eagle-wbc.github.io/>
- **会议：** ICRA 2026
- **发表日期：** 2026-02-03（arXiv v1）；2026-02-27（v2）
- **机构：** 上海交通大学（SJTU）；上海人工智能实验室（Shanghai AI Lab）
- **作者：** Quanquan Peng*、Yunfeng Lin*、Yufei Xue、Jiangmiao Pang、Weinan Zhang（* equal contribution）
- **入库日期：** 2026-09-13
- **一句话说明：** **EAGLE** 用迭代的 **generalist → per-embodiment specialist 微调 → DAgger 回蒸** 循环，在 **无需 per-robot 奖励重调** 的前提下，让 **单一策略** 同时控制 **H1 / G1 / T1 / N1 / Adam** 等异构人形，并统一支持 **速度 + 高度 + 躯干 pitch** 等高维指令。

## 摘要级要点

- **问题：** RL 人形 WBC 近年性能很强，但多数工作 **一台机器人一份策略 + 一套奖励**；DoF、动力学与运动学拓扑差异使 **单策略跨本体** 困难；且许多控制器只跟踪 base 速度，难以在同一接口下同时支持 **蹲、倾、转身** 等丰富行为。
- **方法 — 统一指令接口：** 命令向量 $c_t$ 合并 **任务命令** $v_t$（$v_x, v_y, \omega$）与 **行为命令** $b_t$（基座高度 $h$、躯干 pitch $p$）；与短窗本体感知 $s_t$ 组成观测 $o_t$。
- **方法 — 迭代蒸馏循环：** 每轮从当前 generalist $\pi_g^{(k)}$ **fork** 出 $N$ 个 embodiment specialist $\{\pi_{s_i}^{(k)}\}$，在各自机器人上 RL 精修；再在 pooled embodiment 集上运行 $\pi_g$，用对应 specialist **重标注动作**，以 **模仿损失（DAgger 风格）** 蒸馏出 $\pi_g^{(k+1)}$；重复至收敛。
- **本体感知：** 输入除 proprio 外含 **embodiment ID / 拓扑描述**（DoF、padding mask、URDF 语义等），使单网络在不同机器人上条件化分化。
- **训练栈（论文/笔记归纳）：** 大规模 GPU 并行仿真（Isaac Sim 类）；specialist 用 **PPO**；蒸馏用 **DAgger**。
- **实验规模：** **5** 种机器人仿真（H1、G1、T1、N1、Adam）；**4** 种真机（H1、G1、N1、T1）。
- **任务：** 变速行走、转向、蹲行、躯干前倾行走、抗扰动等。
- **对比（定性，以论文/项目页为准）：** 相对 **per-robot PPO** 相当或更好；显著优于 **不蒸馏的多本体共训**；去掉 embodiment 编码在 DoF 差异大的平台上明显掉点。
- **开源状态（2026-09-13 项目页核查）：** 项目页 **无 GitHub / Hugging Face / 数据链接**；论文与深读笔记亦标注暂未公开 → **确认未开源**。

## 对 wiki 的映射

- 沉淀实体页：[paper-notebook-embodiment-aware-generalist-specialist-distillat.md](../../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md)
- 交叉：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[跨具身迁移选型](../../wiki/queries/cross-embodiment-transfer-strategy.md)、[XHugWBC](../../wiki/entities/paper-xhugwbc-cross-humanoid.md)、[DAgger 方法页](../../wiki/methods/dagger.md)

## 参考来源（原始）

- arXiv:2602.02960（2026-02-03 / v2 2026-02-27）
- 项目页：<https://eagle-wbc.github.io/>
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html>
