# PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations（arXiv:2604.27472）

> 来源归档（paper）

- **标题：** PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations
- **类型：** paper
- **作者：** Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Fangzheng Yan, Tian Li, Haitong Tang, Sen Fu, Xuan'er Wu, Qizhen Weng, Weinan Zhang, Xiu Li, Chi Zhang, Chenjia Bai（通讯）, Xuelong Li（通讯）；中国电信 TeleAI、清华、上交、复旦
- **arXiv：** <https://arxiv.org/abs/2604.27472>（v1，2026-04-30，cs.AI）
- **PDF：** <https://arxiv.org/pdf/2604.27472v1>
- **项目页：** <https://rhodes-team-prts.github.io/>
- **代码：** https://github.com/TeleHuman/PRTS
- **权重：** <https://huggingface.co/TeleEmbodied/PRTS-4B>
- **入库日期：** 2026-11-12
- **一句话说明：** 把 VLA 预训练从纯行为克隆改成「行为克隆 + 语言目标条件对比式 RL」：语言指令当目标，状态-动作与目标嵌入的内积近似对数折扣目标占据度，单次前向（角色感知因果掩码）完成，Qwen3-VL-4B 底座。
- **核对范围：** 通读 arXiv v1 PDF 正文 §1–§7（引言、预备知识、方法、数据集、相关工作、实验 §6.1–§6.7、结论）与附录 A（贡献）、B（定理 1 证明）、C（真机任务描述与 Table 8/9）、D（Table 10）；图 1/2/5/9/10 看了图注与文字，图 4 柱状图数值以附录 Table 8 与正文为准；未核对：参考文献逐条、未运行/未读代码仓库、未读项目页视频、未独立复现任何数字。
- **通读：** 已完成

## 开源状态

- 论文首页列出代码仓库与 HuggingFace 权重（PRTS-4B）；仓库内容本页未核查（仓库侧阅读见 [telehuman_prts_smart_vla.md](../repos/telehuman_prts_smart_vla.md)，该文明确 SMART 论文与本文不是同一篇）。

## 核心摘录（均出自论文，括号为章节）

1. **目标（§1, §3.1）：** 现有 VLA 预训练基本是行为克隆；作者要让 VLM 主干获得「目标可达性」（goal reachability）感知。对比 π0.6*（需成败标签）与 VLAC（需进度标注）：PRTS 不需奖励标注、不单独训价值网络。
2. **语言条件对比 RL（§3.2）：** 目标是整条轨迹共享的语言指令 l；同任务样本互为正例，异任务语言目标作负例。用时间权重 q_ij ∝ γ^(T_j−t_j) 模拟标准 CRL 的几何采样；双向 InfoNCE：L_{sa→l}（任务判别，按 γ^(T−t) 缩放）与 L_{l→sa}（软目标 q_ij，注入时间进度）。
3. **定理 1（§3.2，附录 B）：** 在「确定性专家 π*」假设下，L_{l→sa} 的最优表示满足 ψ(l)ᵀφ(s,a) = log Q(s,a) + C(l)。证明依赖 Q = (1−γ)γ^(T−t)（确定性演示、到 T 步到达目标）与 batch 常数期望收敛。
4. **架构（§3.3, 图 1）：** Qwen3-VL-4B-Instruct；序列末尾追加 <CRL_action> 与 <CRL_goal> 两块；五角色（vision/state、instruction、AR action、CRL_action、CRL_goal）注意力掩码：AR 动作 token 保持标准因果，CRL_action 只看视觉+本体，CRL_goal 只看自身块；FAST 离散动作 token 做 BC；后训练阶段才接 675M 的 DiT 流匹配动作专家（5 步去噪）。
5. **预训练目标（式 15）：** L = L_BC + L_RobotQA + L_Subtask + λ_crl·L_{sa→l} + L_{l→sa}，λ_crl=1.0，γ=0.995；后训练只用 L_FM（式 16）。
6. **数据（§4）：** 404M 样本、167.8B token；动作数据来自 AgiBotWorld、RoboMind、Open X-Embodiment 与自采（RealMan、Agilex 双臂）；视觉推理数据含 RefCOCO、Pixmo-Point、RoboPoint、RoboRefIt、RefSpatial、RoboAfford、Cosmos-Reason1、EgoPlanIT、RoboVQA、LLaVA-Instruct 子集。
7. **训练规模（§6.1）：** 64×H100 约一周，全局 batch 256 个打包序列（长 4096），约 220K 步，约 2K 个批内负例；LIBERO 后训练仅 bs=32、30K 步（文中称为 π0.5 的 1/8 计算）。
8. **仿真结果：** LIBERO 平均 98.4%（Table 2）；LIBERO-Plus 平均 81.4%（Table 3，零样本）；LIBERO-Pro 平均 58.8%，Task 轴 31.5%（Table 4）；SimplerEnv WidowX 77.1%（Table 5）。
9. **消融（§6.3, Table 6/7）：** 同架构同数据，仅 λ_crl 0 vs 1：LIBERO 97.8→98.4，LIBERO-Plus 76.5→81.4，LIBERO-Pro 53.8→58.8；LIBERO-Pro Object 轴 84.5→82.3（−2.2）。
10. **真机（§6.2.5, Table 8/9, 图 4）：** RealMan 双臂 11 任务单一共享检查点，每格 20 次试验：PRTS 95.9%，π0.5 85.5%，π0 67.3%；Flexiv 单臂 3 任务 PRTS 90.0%、π0.5 比其低 15.0 个点（分任务 100/100/70 对 100/90/35）；任务指令泛化轴（Table 9）PRTS 平均 73.8%，π0.5 35.0%，π0 13.8%。
11. **人为干预（§6.5, 图 8）：** 4 个任务各约 5 次，作者明说是定性压力测试，不报成功率表。
12. **价值可视化（§6.6, 图 9）：** 一条预训练后未微调的 Pick Shoes 留出轨迹上，正确指令下 φᵀψ 从 0.28 升到 0.57，错误指令下 0.04~0.20；单条轨迹。
13. **效率（§6.7, 图 10）：** CuTe 融合核每层注意力前向为 FA3 基线的 1.18×（0.531 ms 对 0.45 ms）；吞吐扩展效率 32 卡 95.2%、64 卡 85.1%。

## 作者自述的局限（论文原文）

- 实机任务级指令重组泛化「仍是真实世界泛化中最难的形式之一」（§6.4）。
- 干预实验是定性压力测试，非完整仪表化基准（§6.5）。
- LIBERO-Pro 未含 Environment 轴，因官方代码尚未发布对应资产（§6.2.3 脚注）。
- 论文无独立「局限」章节。

## 对 wiki 的映射

- [paper-prts-primitive-reasoning-tasking-system](../../wiki/entities/paper-prts-primitive-reasoning-tasking-system.md)
