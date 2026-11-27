# EvoNav: Evolutionary Reward Function Design for Robot Navigation with Large Language Models

- **作者：** Zhikai Zhao, Chuanbo Hua, Federico Berto, Zihan Ma, Kanghoon Lee, Jiachen Li, Jinkyoo Park
- **机构：** KAIST, Radical Numerics, UC Riverside, Omelet (AI4CO)
- **版本 / 载体：** arXiv:2605.11859v1 [cs.RO]（2026-05-12，arXiv HTML 全文）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2605.11859) · [arXiv 摘要页](https://arxiv.org/abs/2605.11859)
- **项目页 / 代码：** 未确认开源代码（论文原文及补查材料未提供官方 GitHub 仓库或独立项目页链接）
- **核对范围：** 全文通读 arXiv HTML（Abstract、§1–§6、Table 1–2、Figure 1–5 图注，以及 Appendix A Algorithm 1、Appendix B MDP 定义、Appendix C Table 3–9 超参数/LLM 消融/算力成本对比、Appendix D Prompts 与 Seed Function）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 在动态人群导航（crowd navigation）的强化学习中，手工设计奖励函数依赖专家经验、易引入隐含偏置；而现有的 LLM 自动奖励设计方法（如 Eureka、CARD）通常需对每个候选奖励执行完整 RL 策略训练来评估效果，在仿真与策略学习耗时较高的机器人导航任务中算力瓶颈显著。
- **核心方法（渐进保真度三阶段「Warm-Up-Boost」演化搜索）：**
  - 奖励以可执行 Python 函数表示（文中主文描述标准接口模板 `cal_reward(s_t)`，Appendix C.6 与 D.1/D.5 给出实际交互脚本签名 `compute_reward(inst, traj)` / `seed_reward_func(inst, traj)`，每步返回标量浮点奖励）。
  - **Stage I（解析规则驱动演化 / Warm-up）：** 从 CrowdNav++ 简化种子函数出发，LLM 生成 $N=8$ 个初始候选，在 $M=100$ 个预采集场景（每场景 $N_{\text{traj}}=10$ 条轨迹、约 $F_j \approx 200$ 帧）上零策略训练评估。通过 Spearman 秩相关 $\text{Score}_1(r_\theta)$ 衡量候选累计奖励排序与规则排序（成功 > 进度 > 碰撞）的一致性，经突变、交叉、随机初始化与反思笔记迭代 $G_1=10$ 代。
  - **Stage II（轻量代理模型推演 / Boost）：** 对 Stage I 演化出的 $N=8$ 个候选，分别从零训练轻量代理策略 $\pi_\phi^{\text{proxy}}$（A2C，$K_2=8{,}000$ 步，$E_2=50$ 回合、短视野 $T_{\text{short}}=100$ 步），收集多目标指标 $M(r) = (\text{SR}, \text{NT}, \text{PL}, \text{ITR}, \text{SD})$ 反馈给 LLM 做针对性代码微调，共进行 $G_2=16$ 轮。
  - **Stage III（完整策略精调）：** 使用完整 CrowdNav++ 策略架构与 PPO 训练 $K_3=10^7$ 步（评估 $E_3=500$ 回合，$H \in \{5,10,15,20\}$），共 $G_3=3$ 轮，输出最终排序 $\mathcal{R}_3$。
- **关键实验数据（§5 & Appendix C）：**
  - 2D 连续人群仿真环境 $[0,12]\times[0,12]\ \text{m}^2$，最多 $H=20$ 名 ORCA 驱动行人；默认 LLM 为本地 vLLM 部署的 `gpt-oss-120B`。
  - **w/ random（行人可中途切换目标且参数随机化）：** EvoNav 达到 SR $93.22\pm0.61\%$、CR $4.47\pm0.42\%$、TR $2.31\pm0.10\%$，优于 CrowdNav++（SR $89.35\%$、CR $7.88\%$）、Eureka（SR $86.71\%$、CR $10.03\%$）和 CARD（SR $91.35\%$、CR $6.12\%$）。
  - **w/o random（固定行人属性与目标）：** EvoNav 达到 SR $95.61\pm0.55\%$、CR $4.01\pm0.36\%$、TR $0.38\pm0.06\%$，同样全面领先对比基线。
  - **三阶段消融（Table 2，w/ random）：** Stage I（SR $90.11\%$）→ Stage I+II（SR $92.93\%$，TR $1.35\%$）→ Full（SR $93.22\%$，CR $4.47\%$）。仅 Stage I+II 的 SR 已超过 CARD（$91.35\%$）。
  - **LLM 消融（Table 8）：** 在同一评估设置下，GPT-4o-mini（$86.2\%$）、Llama-3.1-70B（$88.5\%$）、Llama-3.1-405B（$90.1\%$）、GPT-4o（$90.8\%$）、GPT-OSS-120B（$91.2\%$）均高于手工基线 CrowdNav++（$84.3\%$）。
  - **计算成本（Table 9）：** 相比 CARD（GPU 36 小时 / \$18.00 + 72K tokens / \$0.02 = \$18.02），EvoNav 仅需 GPU 9.2 小时（\$4.60）+ 720K tokens（\$0.21）= \$4.81，GPU 成本节省 74%，总成本节省 73%。
- **对 wiki 的映射：** [EvoNav 论文实体页](../../wiki/entities/paper-agentic-2605-11859.md)
