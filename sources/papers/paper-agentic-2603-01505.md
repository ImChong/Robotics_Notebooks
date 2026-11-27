# FATE: Closed-Loop Feasibility-Aware Task Generation with Active Repair for Physically Grounded Robotic Curricula

- **作者：** Bingchuan Wei, Bingqi Huang, Jingheng Ma, Zeyu Zhang, Sen Cui（通讯作者：`cuis@mail.tsinghua.edu.cn`）
- **机构：** 清华大学航天航空学院（School of Aerospace Engineering, Tsinghua University）、清华大学集成电路学院（School of Integrated Circuits, Tsinghua University）、清华大学自动化系（Department of Automation, Tsinghua University）、北京通用人工智能研究院通用人工智能全国重点实验室（BIGAI）
- **版本 / 载体：** `arXiv:2603.01505v1 [cs.RO]`（2026-03-02 提交，16 页，4 图，CC BY 4.0）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2603.01505) | [arXiv HTML 全文](https://arxiv.org/html/2603.01505)
- **项目页 / 代码：** 未确认开源代码（Step 2.5 核查：arXiv 摘要页、HTML 全文及元信息均未提供独立项目主页或 GitHub 源码仓库链接）
- **核对范围：** 已通读 arXiv HTML 全文（摘要、第 1–6 节正文、Impact Statement、References，以及附录 A–D：线性收敛证明、分层对齐单调性证明、静态/动态修复 API 与算法流程、Embodied Brain 架构与训练数据配比；完整核对 Table 1–6 与 Algorithm 1–2）
- **通读：** 已完成

## 原文核对笔记

- **核心问题与形式化：** 针对开放式 LLM 仿真任务生成中“语义通顺但物理不可行”的 Feasibility Gap，将可行性感知任务定义为三元组 $\tau = (\mathcal{I}, \mathcal{S}, \Omega_\pi)$，其中 $\mathcal{I} = (\mathcal{L}, \Psi_{\mathcal{I}})$ 为自然语言指令与轨迹谓词，$\mathcal{S} = \{(M_j, q_j, \Lambda_j)\}_{j=1}^N$ 为物理流形、运动学状态与语义供能标注，$\Omega_\pi \subset \Theta$ 为带约束的策略流形。可行策略集测度为 $\mu(\tau) = \int_{\Omega_\pi} \mathbb{I}(\theta \in \Phi_\tau)\,d\rho_{init}(\theta)$，优化目标为在语义偏移约束 $\mathbb{E}[D(\tau, \mathcal{A}(\tau))] \le \epsilon_{sem}$ 下最小化不可行概率质量 $L_{\mathcal{G}}(\mathcal{A})$。
- **双阶段闭环架构（$\mathcal{A} = \mathcal{A}_{exec} \circ \mathcal{A}_{perc}$）：**
  1. **Phase I 静态对齐（Ante-Auditor + StaticRepair，Algorithm 2）：** 围绕可达性（Geometric Reachability）、供能匹配（Affordance Matching）、物理合理性（Physical Plausibility，穿模/放置稳定）和形态兼容性（Morphological Compatibility，尺度匹配）评估静态代理分 $\hat{\mu}_{static}$，通过 6 个静态场景 API（`SWAP_ASSET`、`SPAWN_ASSET`、`REMOVE_ASSET`、`SET_JOINT`、`TRANSFORM_POSE`、`RESCALE`）构造残差 $\Delta\tau_{static}$。
  2. **Phase II 动态对齐（In-Step Auditor + DynamicRepair）：** 将任务拆解为导航与操作原语序列 $\Pi = \{\pi_1, \dots, \pi_K\}$（无接触导航采用 MPC / BIT*；富接触操作采用 SAC 强化学习配合 MLLM 合成的势能奖励塑形）。结合状态检查点（checkpointing）与子步回滚（Sub-step Rollback），通过 7 个动态干预 API（`INJECT_REWARD`、`SET_RL_HORIZON`、`REPLAN_PATH`、`ADJUST_GRASP`、`TUNE_IMPEDANCE`、`SKIP_SUBSTEP`、`RESET_ROBOT_JOINT`）执行所谓语义策略梯度更新。
- **Embodied Brain 骨干与数据（附录 D）：** 审计器基于 **RoboBrain 2.0**（7B 参数，源于 Qwen2.5-VL，包含 Qwen-VL ViT 编码器、MLP 投影器与 Qwen2.5-7B 语言模型，支持最高 16k 视频上下文 token）。使用约 145,000 条混合样本微调（45% 具身动作：RoboBench、EgoPlan-Bench2、RoboSpatial-Home；45% 空间感知：Blink、OmniSpatial、CVBench；10% 因果推理：ERQA）。在 EmbodiedVerse 上相对 Qwen2.5-VL 基线在 Relative Shape 绝对提升 34.15%（相对提升 80.01%），Goal Decomposition 绝对提升 5.00%，Counting 提升 2.17%。
- **主要实验结果（NVIDIA Isaac Sim，RidgebackFranka 移动机械臂 + Objaverse/PartNet-Mobility）：**
  - **生成多样性（Table 1，352 个任务）：** Self-BLEU-4 0.276、S-BERT Sim 0.158、ViT Sim 0.189、CLIP Sim 0.754，整体优于对比基准 RLBench、ManiSkill2、Meta-World、Behavior-100 与 GenSim-V2。
  - **任务可行率（Table 2）：** Vanilla（无审计）SVR 41.5% / EVR 30.4% / FTR 12.6%；GenSim-V2 为 58.2% / 51.2% / 29.8%；仅静态审计 FATE-Static 为 93.8% / 69.7% / 65.4%；完整 FATE 达到 SVR 97.5% / EVR 94.5% / FTR 92.1%。
  - **审计诊断性能（Table 3，人工标注测试集）：** 语义类别 F1 0.95（P=0.96, R=0.94）、几何类别 F1 0.86（P=0.88, R=0.85）、动态类别 F1 0.83（P=0.85, R=0.81）、总体 F1 0.88（P=0.90, R=0.87）。
  - **修复成功率（Table 4，RSR）：** Ante-Repair 201/215（93.5%）、Primitive-Repair 39/42（92.9%）、RL-Repair 58/76（76.3%）、汇总 298/333（89.5%）。
  - **消融发现（Sec. 5.6）：** 几何敏感任务（如 Ingredient into Pot、Heat Food in Microwave）仅静态对齐即可相对 Vanilla 提升超过 60%；富接触任务（如 Open Sliding Window）仅静态对齐为 20.6%、仅动态对齐为 41.8%，完整流程达到约 88%。
- **局限性核对：** 依赖 VLM 采样仍存在随机漏检；聚焦几何与运动学约束，对摩擦、阻尼等精细接触动力学建模不足；RL 阶段修复率最低（76.3%），学习策略脆弱性是主要瓶颈；实验完全局限于仿真环境，未在真机部署验证。
- 对 wiki 的映射：[FATE：面向物理接地机器人课程的闭环可行性感知任务生成与主动修复框架](../../wiki/entities/paper-agentic-2603-01505.md)
