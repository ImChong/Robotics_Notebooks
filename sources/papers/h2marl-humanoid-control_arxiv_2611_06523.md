# H²MARL：面向人形机器人控制的分层异构多智能体强化学习（arXiv:2611.06523）

> 来源归档（paper）

- **标题：** H²MARL: Hierarchical Heterogeneous Multi-agent Reinforcement Learning for Humanoid Robot Control
- **类型：** paper
- **作者：** Xuemin Hu, Kaixin Li, Qingqing Xia, Yuxin Wang, Wei Xie, Bo Tang, Long Chen
- **机构（论文首页脚注）：** 湖北大学人工智能学院（Hubei University，武汉）；伍斯特理工学院（Worcester Polytechnic Institute，美国麻省）；中国科学院自动化研究所（北京）
- **arXiv：** <https://arxiv.org/abs/2611.06523>（v1，2026-11-04，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.06523>
- **代码：** 论文正文未给出代码或项目页链接（截至入库日 2026-11-09 通读全文未见）；未核查 arXiv 页之外的项目页
- **入库日期：** 2026-11-09
- **一句话说明：** 把人形机器人每个力矩驱动关节自由度（JDoF）当作带物理描述符的异构智能体（HAPPO 训练），上层输出位姿增量、下层用 HST 式 Transformer 执行器，并用同一条交互轨迹闭环联合优化两层。
- **核对范围：** 通读 arXiv PDF v1 正文全文（摘要、I 引言、II 相关工作、III JDoF 级策略分解、IV 分层策略-执行器耦合、V 实验含表 I–V、VI 结论）及参考文献列表，对表内数字与正文百分比逐项复算；论文无附录和补充材料（PDF 内未见）；图 1、图 2 仅读到文字层与图注，未做图像级核对；未运行任何代码，未读引用文献原文。
- **通读：** 已完成

## 开源状态

- 论文未给出 GitHub / 项目页链接，也未写 "code will be released"；按步骤 2.5「未列链接」处理，入库日 2026-11-09，后续 lint 可跟进。

## 核心摘录

1. **问题与动机（I 引言）：** 模仿学习主流方法需大量动捕或遥操作数据；单智能体 RL 难以协调高自由度关节；已有 MARL 把肢体或关节当作同构智能体；已有分层方法把高层规划与低层控制解耦训练，且高层常用大模型、推理慢。
2. **JDoF 级异构分解（III，式 2–3）：** 每个力矩驱动的自由度作为一个智能体 i，描述符 ξ_i = [c_i, u_i, τ_i^max, q_i^min, q_i^max, d_i, m_i]，依次为关节类型、驱动轴、力矩上限、运动范围、运动学位置、惯量相关属性；论文称这些值直接从仿真器读取、无需手工设置。策略 π_Θ(a|o) = ∏ π_θi(a_i | o_i, ξ_i)；沿用 HAPPO：每个智能体独立 actor、不共享参数，中心化 critic，CTDE。
3. **高层位姿规划器（IV-A，式 4–8）：** 观测 o^H = [本地本体感知, 全局身体状态, ξ_i, 智能体编号 e_i]；输出位姿增量 Δq^H = α_i·tanh(μ + η⊙ε)，α_i = α·τ_i^max·(q_i^max − q_i^min)·d_i/m_i，α 固定 0.001；目标位姿 q_t^tar = clip(q_{t-1}^tar + Δq^H, q^min, q^max)。
4. **低层关节执行器（IV-B，式 9–10）：** 基于 HST（论文引 [3]，即 HumanPlus）的时序 Transformer f_ψ(q^tar, H_t) 输出关节位置，PD 控制器（Kp=100，Kd=5）转力矩；执行器损失 L_hst = E[w_q‖q^posi−q^tar‖² + w_s‖q^posi−q^posi_{t-1}‖² + w_τ‖τ‖²]，权重 1.0 / 0.1 / 0.001。
5. **闭环联合优化 CPECO（IV-C，算法 1，式 11）：** 高层与执行器共用同一条交互轨迹 D_{Θ,ψ}；每轮先收集轨迹，再更新高层策略（HAPPO），固定高层后以 Eq.10 更新执行器，最后更新中心化 critic；论文称这避免了「冻结低层」造成的分布失配。
6. **设置（V-A）：** MuJoCo 与 Isaac Gym 的 Humanoid-v2 行走任务；机器人 10 个关节、17 个驱动力矩、每个都作智能体；观测 s_t ∈ R^376；10M 环境步；RTX 4090 + Xeon Gold 6530；基线 HAPPO、MAPPO、HATD3，均用 HAPPO 的配置。额外奖励：防滑 λ_slip=0.01、左右交替 λ_alt=0.05（式 12）。
7. **与 SOTA 对比（V-B，表 I，MuJoCo）：** FS 0.5367 / TS 0.2516 / GS 0.001039 / MS 0.3514；MAPPO 为 0.5790 / 0.3873 / 0.000905 / 0.2329；HAPPO 0.9827 / 0.5091 / 0.000150 / 7.3819；HATD3 0.8220 / 0.4739 / 0.000050 / 2.3818。相对 MAPPO，FS/TS/GS 改善 7.3% / 35.0% / 14.8%（复算一致）；MS 略逊于 MAPPO（作者自述）。
8. **消融（V-C，表 II–IV，Isaac Gym）：** 完整 H²MARL 的 VE/CoT/MJ/CAM/TS = 0.4807 / 1.5213 / 0.7448 / 1.0324 / 0.0140；w/o HA、w/o CPECO、w/o Rew、w/o All(HAPPO) 在 VE、CoT、MJ、CAM 上均更差；w/o CPECO 的 TS(0.0106) 反而最低（作者解释为牺牲整体质量换来躯干少动）。智能体粒度（表 III）：JDoF 级相对肢体级 VE/CoT/MJ/CAM/TS 降低 71.4% / 79.3% / 88.3% / 26.7% / 71.7%（复算一致），BP 级 CAM 最低（0.5330）。
9. **实时性（V-D，表 V）：** 高层均值 8.332 ms、低层 0.927 ms，合计 9.259 ms（>100 Hz）；高层比 HAPPO（7.677 ms）多 0.655 ms。

## 作者自述与核对者判断的区分

- 作者自述：性能提升来自 JDoF 级分解与策略-执行器耦合，而非 HAPPO/HST/PD 的简单拼接（I 引言第 3 点，靠表 II、III 的受控比较支持）。
- 核对者判断：全部实验为仿真（MuJoCo / Isaac Gym 的 Humanoid-v2 行走），无真机、无 sim2real；表 I 基线在 MuJoCo、表 II–IV 在 Isaac Gym，两个表之间的 H²MARL / HAPPO 数值不可直接互比；论文未报告随机种子数与方差（推测：为单次或未说明）。

## 对 wiki 的映射

- [paper-h2marl-hierarchical-heterogeneous-marl-humanoid](../../wiki/entities/paper-h2marl-hierarchical-heterogeneous-marl-humanoid.md)
