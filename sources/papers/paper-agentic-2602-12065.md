# Scene2Demo: Self-Evolving Embodied Data Generation via Object-Action Graph

- **作者：** Xiang Liu*, Sen Cui, Guocai Yao†, Zhong Cao, Jingheng Ma, Min Zhang, Changshui Zhang†
- **机构：** 清华大学（`cuis@mail.tsinghua.edu.cn`、`zcs@mail.tsinghua.edu.cn`）、北京大学（`yaoguocai@pku.edu.cn`）；项目页仍保留匿名投稿占位（Anonymous Institution）
- **版本 / 载体：** arXiv:2602.12065v3 [cs.RO]（v1 提交于 2026-02-12，v3 修订于 2026-10-17；v3 HTML 正文题名为 *Scene2Demo: Automated Embodied Data Generation for Robot Policy Learning*，摘要与项目页题名为 *Scene2Demo: Self-Evolving Embodied Data Generation via Object-Action Graph*）
- **原文：** [arXiv:2602.12065 (HTML v3)](https://arxiv.org/html/2602.12065) / [arXiv 摘要页](https://arxiv.org/abs/2602.12065) / [项目页](https://scene2demo-anon.github.io/)
- **项目页 / 代码：** 未确认开源代码（项目页 `https://scene2demo-anon.github.io/` 列有 `Paper`、`Code`、`Policy Weights` 入口，但当前无具体 GitHub 或 Hugging Face 仓库地址，且论文全文与 `discovered_urls` 中均未提供可下载运行的官方源码仓库）
- **核对范围：** 通读 `arXiv:2602.12065v3` HTML 全文，包括 Abstract、§1 Introduction、§2 Related Works、§3 Method（§3.1–§3.6，公式 1–8）、§4 Experiments（§4.1–§4.5，Table 1–3，Figure 1–5）、§5 Conclusion、References（[1]–[42]）、Appendix A（A.1–A.3、公式 9–10、Algorithm 1）与 Appendix B（B.1–B.8、Table 4–10、Figure 6–11、公式 11–14），并核对项目页文本快照。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[Scene2Demo: Self-Evolving Embodied Data Generation via Object-Action Graph](../../wiki/entities/paper-agentic-2602-12065.md)

### 1. 问题定位与输入输出

- **任务目标：** 面向离线具身智能数据生成（offline embodied data generation），输入单张 RGB 场景图 $X_0$ 与用户任务查询 $I$，自动生成交互式仿真场景 $S_0$、BDDL 风格任务配置、基于原子原语的动作流（action flow）、三视角执行视频（Global / Head / Wrist）及下游策略学习数据集。
- **场景重建基底：** 沿用 Digital Cousins / ACDC（Automated Creation of Digital Cousins）流程：使用 LLM + GroundedSAM-v2 提取物体标签与掩码、DepthAnything-v2 估计深度并提取点云，在 BEHAVIOR-1K 资产库中基于 CLIP 与 DINOv2 分层检索匹配交互式资产，最后在 OmniGibson 仿真器中布置场景并做物理稳定性修正。
- **机器人与尺度适配：** 实验采用带移动底盘的 FrankaMounted 机械臂（IK 控制器 + 机载 RGB-D）。针对单目三维重建的绝对尺度歧义，统一将机器人缩放系数设为 $s = 0.7$，抓取收敛偏置（CONVERGE offset）设为 $\delta_{\text{offset}} = 0.14 \times s$。

### 2. 五模块结构化任务生成与对象-动作图

- **Module 1（Semantic Expansion）：** 结合 `scene_json`（物体 ID、3D 坐标、包围盒）和可选的场景图像，将模糊指令接地为带唯一场景 ID 的 `task_name` 和逐步任务描述 `task_message_detail`。
- **Module 2（Subtask Decomposition）：** 将复杂长时程任务 $\mathcal{T} = \{T_1, \dots, T_K\}$ 分解为有序的简单子任务序列；每个简单任务定义为五元组 $T = \langle o_t, o_{s_1}, o_{s_2}, s_{\mathrm{init}}, s_{\mathrm{goal}} \rangle$，其中 $o_t$ 为直接操作对象，$o_{s_1}, o_{s_2}$ 分别为初始与目标支撑/关联对象。
- **Module 3（BDDL Generation）：** 为每个子任务生成 `.bddl` 格式的初始条件 $s_{\mathrm{init}}$ 与成功检测条件 $s_{\mathrm{goal}}$（如 `ontop`、`inside`、`open`）。
- **Module 4（Object Size Adjustment）：** 区分固定/铰接家具（Type A，保持 $S=1.0$）与可抓取物品（Type B）；若平行夹爪最大宽度约 $0.06\text{ m}$ 且水平最小尺寸 $d_{\min} = \min(\text{bbox}_x, \text{bbox}_y) > 0.06\text{ m}$，则缩放使 $S \times d_{\min} \approx 0.05\text{ m}$。
- **Module 5（Action Flow & Transfer Planning）：** 在 `BasicSkill` 类定义的 21 个原子动作原语库 $\mathcal{A}$ 上生成任务内动作流 $\pi(T) = \{a_1(\theta_1|T), \dots, a_n(\theta_n|T)\}$ 与相邻子任务间的动作转移 $e_k = \{a_{\mathrm{end}}(\theta_{\mathrm{end}}^{(k)}|\mathcal{T}), a_{\mathrm{start}}(\theta_{\mathrm{start}}^{(k+1)}|\mathcal{T})\}$。
  - **21 个原语分为 4 类：**（1）7 个无参上下文感知原语（`APPROACH`, `CONVERGE`, `GRASP`, `UNGRASP`, `RETREAT`, `NAVIGATE_TO_TARGET`, `NAVIGATE_TO_SUPPORT`）；（2）6 个末端执行器平移（`LIFT_EEF_UP/DOWN`, `MOVE_EEF_FORWARD/BACKWARD/LEFT/RIGHT`）；（3）6 个移动底盘平移与旋转（`MOVE_BASE_FORWARD/BACKWARD/LEFT/RIGHT`, `TURN_BASE_LEFT/RIGHT`）；（4）2 个铰接体交互（`ARTICULATE_OPEN`, `ARTICULATE_CLOSE`）。
- **Object-Action Task Graph：** 定义通用模板 $\mathcal{G}=(\mathcal{V},\mathcal{E}),\ \mathcal{V}=\mathcal{O}\times\mathcal{A}\times\mathbb{N}^+$，在重建场景 $S_0$ 下实例化为条件子图 $\mathcal{G}_{S_0}$，将长时程任务规划转化为满足初始/目标谓词及子任务转移约束的路径搜索问题。

### 3. 基于视觉反馈的自进化机制（Self-Evolution）

- **状态重置：** 每次进化尝试 $\tau$ 前将仿真器重置为该子任务保存的初始状态 $\bar{S}_{\mathrm{init}}^{(k)}$（满足 $s_{\mathrm{init}}(T_k)$），避免失败 rollout 污染后续迭代。
- **多视角与滑动窗口：** 从相机集合 $\mathcal{C} \subset \{\mathrm{Global}, \mathrm{Head}, \mathrm{Wrist}\}$ 采样（每动作最多下采样至 $L_j = \min(\lceil t_j/p_2 \rceil, 6)$ 帧），默认回看窗口 $p_1 = 1$，形成局部上下文 $X_{k,j}^{\mathcal{C}}$。
- **Robotic Safety Inspector（步级诊断）：** 针对动作 $a_j$ 输出诊断 $m_{\tau, j}$，核对运动验证（Motion Verification）、状态对比（State Contrast）与异常检测（Anomaly Detection）。
- **Robotic Task Supervisor（任务级修订）：** 汇总当前动作流 $\hat{\pi}_\tau(T_k)$、步级诊断序列 $m_\tau$、完整多视角视频 $X_k^\mathcal{C}$、任务上下文 $\mathcal{T}$ 及历史演化记录 $\mathcal{H}_{<\tau} = \{(\hat{\pi}_i(T_k), m_i, \hat{r}_{i+1})\}_{i=0}^{\tau-1}$，进行根因定位、动作序列/参数修订并输出原因 $\hat{r}_{\tau+1}$，严格限制在 21 原语库内，子任务迭代上限 $\tau_{\max}=5$。

### 4. 关键实验数据（均经全文核对）

- **102 个原子场景-任务对（无自进化，Table 1）：** 由 Qwen3-Max 与 Seedream-4.5 生成 34 张 RGB 场景图（16 厨房、10 卧室、8 书房/工作间等），每图匹配 BEHAVIOR-1K top-3 资产变体共 $34 \times 3 = 102$ 场景；总体成功率 **71.6%（73/102）**，其中 $\mathcal{T}_5$（Open/Close）为 **66.7%（24/36）**，$\mathcal{T}_6$（Pick/Place）为 **74.2%（49/66）**。
- **4 个复杂长时程任务主对比（默认使用 `qwen3-vl-235b-22a-thinking`，Table 2 & Table 4）：**
  - $\mathcal{T}_1$（Put the glass into the refrigerator）：Primitive-only SR/Subtask-Score 为 `0 / 50`，Prim+Evo 为 `40 / 84`；GenSim2 (prim) `0 / 73`，RoboGen (prim) `0 / 50`，RoboGen (prim+RL) `fail`。
  - $\mathcal{T}_2$（Put the apple into the refrigerator）：Primitive-only `0 / 50`，Prim+Evo `30 / 79`；GenSim2 `5 / 78`，RoboGen (prim) `0 / 46.5`，RoboGen (prim+RL) `fail`。
  - $\mathcal{T}_3$（Put the apple into the bowl）：Primitive-only `40 / 70`，Prim+Evo `70 / 93`；GenSim2 `55 / 91`，RoboGen (prim) `0 / 50`，RoboGen (prim+RL) `fail`。
  - $\mathcal{T}_4$（Put the cup on the table）：Primitive-only `0 / 50`，Prim+Evo `70 / 85`；GenSim2 `0 / 68`，RoboGen (prim) `0 / 50`，RoboGen (prim+RL) `fail`。
  - $\mathcal{T}_5$（Open/Close）：Ours (prim) `66.7 / 66.7`，GenSim2 `100 / 100`，RoboGen (prim) `0 / 50.2`，RoboGen (prim+RL) `success`。
  - $\mathcal{T}_6$（Pick/Place）：Ours (prim) `74.2 / 74.2`，GenSim2 `38.8 / 82.1`，RoboGen (prim) `0 / 42.8`，RoboGen (prim+RL) `success`。
- **VLM 对比（Table 3 & Appendix B.7.1）：** 在 $\mathcal{T}_1$（10 或 20 demos）和 $\mathcal{T}_3$（10 demos）上比较 Gemini-3-Flash、GPT-5.2、Qwen3-VL；$\mathcal{T}_1$ 完整任务 SR 分别为 `30%` / `20%` / `40%`（平均迭代 2.3 / 3.5 / 3.1），$\mathcal{T}_3$ 分别为 `70%` / `50%` / `50%`（平均迭代 1.6 / 0.2 / 0.4）。
- **时间开销与消融（§4.4）：** 简单任务无自进化约 2 分钟；长时程任务初始生成 35s–1m20s，每子任务仿真执行 22s–1m30s，单次 Qwen3-VL 进化迭代 1m19s–2m57s，完整长时程演示约 15–30 分钟。多视角（含 Global + Head + Wrist）优于单视角，且回看窗口 $p_1=1$ 优于 $p_1=2$。
- **下游模仿学习（§4.5 & Appendix B.8）：** 利用自主生成的 100 条演示（含物体尺度 $\pm 0.10$ 与铰接体绕 $z$ 轴旋转 $\pm\pi/12$ 随机化）训练 PointNet + 4 层 Causal Transformer（$T=10$，隐藏维 512，8 头）+ Action MLP 预测 10-DoF 动作，单张 A800 训练 200 epochs（batch size 1024，AdamW，lr $4\times 10^{-4}$）。评估各 25 次：Open Refrigerator 成功率 **96.0%（24/25）**，Pick Up Glass 成功率 **92.0%（23/25）**。

### 5. 局限与边界

- 动作空间受限于 21 个预定义原语与 IK 控制器，不支持灵巧操作（dexterous manipulation），也不含真实物理机器人迁移（仅仿真评测）。
- 依赖 Digital Cousins 单目重建与 BEHAVIOR-1K 资产匹配，存在绝对尺度误差（$s=0.7$ 仍偶发失配）与资产缺失问题。
- 离线开环自进化缺少实时连续视觉伺服，精细对齐（如苹果入碗）与运动学奇异/臂身碰撞（如机械臂下连杆碰冰箱左侧框）仍可能振荡或失败。
