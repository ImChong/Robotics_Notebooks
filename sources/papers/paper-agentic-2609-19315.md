# GAVEL: Graph World Models for Verified and Efficient Long-Horizon LLM Task Planning

- **作者：** Ruiyang Wang, Hao-Lun Hsu, Swarajh Mehta, Jiwoo Kim, Zhihao Dou, Miroslav Pajic
- **机构：** 未在 arXiv HTML 原文明确标注（按原文署名作者列表记录）
- **版本 / 载体：** arXiv:2609.19315v1 [cs.RO]（2026-09-16 提交；正文 §I–§VI，含 Algorithm 1–2、Figure 1–3、Table I–III 及 References [1]–[29]）
- **原文：** [arXiv HTML](https://arxiv.org/html/2609.19315) / [arXiv Abs](https://arxiv.org/abs/2609.19315)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2609.19315 原文及候选仓库核查均未提供独立项目页或官方代码仓库链接）
- **核对范围：** arXiv:2609.19315v1 全文（Abstract、§I–§VI、Algorithm 1–2、Figure 1–3 文字说明、Table I–III 与 References [1]–[29]）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **问题形式化（§III-A, Eq. 1–5）：** 环境建模为类型化场景图 $G=(V,E,\Phi)$，顶点 $V=\mathcal{R}\cup\mathcal{O}\cup\{\rho\}$（房间实例、任务相关物体与机器人），关系集 $E$ 包含 `near`、`under`、`room_connect`、`room_inside`、`object_inside`、`on_top`、`next_to`、`holding`，一元状态 $\Phi$ 包含 `open`、`toggled`、`cooked`、`washed`、`dried`。区分真实隐物理状态 $G_t^{\star}$ 与机器人信念图 $G_t$；对未定位物体维护房间概率分布 $b_o^t(r)=P(\mathrm{room}(o)=r\mid\mathcal{H}_t)$。优化目标是在满足所有动作前提 $\mathrm{pre}_{a_t}(G_t^{\star})$、全目标 $\mathsf{unmet}(g,G_H^{\star})=\emptyset$ 及轨迹级安全约束 $\mathsf{safe}(a_{0:H-1})$（关回打开的容器、关闭安全关键电器）的条件下，最小化期望物理导航距离 $\mathbb{E}[C_{\mathrm{exec}}(\mu)]$。
- **动作空间与前提（§IV-E）：** 定义 9 个 grounded 原语 $\mathcal{A}=\{\textsc{NavigateTo}, \textsc{Grasp}, \textsc{Release}, \textsc{PlaceOnTop}, \textsc{PlaceInside}, \textsc{Open}, \textsc{Close}, \textsc{ToggleOn}, \textsc{ToggleOff}\}$。前提约束分为 4 类：邻近性（`proximity`，不能直接对关闭容器内的隐蔽物体执行 `NavigateTo`）、可供性（`affordance`）、夹爪状态（`gripper state`）和可达性（`accessibility`，要求外层容器处于开启状态）。
- **任务理解与信念初始化（§IV-B, §IV-C）：**
  - **指令分解与解析：** 两个基于 `Qwen3-1.7B` 的轻量 LoRA 适配器分别负责提取任务相关物体/显式关系 $(\mathcal{O}_i,\mathcal{E}_i)$ 与部分图目标 $g_i$；均在 8,000 条合成指令-目标对上训练，并在 500 条独立验证集上验证。若指令已明确给出关系（如 `inside(potato, fridge)`），则直接写入信念图，无需 RSN 预测。
  - **RSN 信念初始化：** 遵循 SEEK [11] 思路，使用冻结文本编码器 `BAAI/bge-small-en-v1.5`（384 维）加三层 MLP（隐藏层 256–128–64，dropout 0.2），以加权二元交叉熵训练并用 Platt scaling 校准，预测房间类型得分 $\tilde{f}_o(\tau)$，再按可达同类型房间实例数均分并归一化得到 $b_o^0(r)$。在 51 个场景、197 类物体、37 种房间类型的 11,218 个放置样本上训练（按场景留出验证），AUC 达 **0.904**，Brier score 为 **0.057**；在 832 个留出查询中，真实房间排第一的比例仅为 **47%**。
- **规划验证、图修复与在线重排（§III-B, §IV-A, §IV-F, §IV-G, Alg. 1–2）：**
  - 单次 LLM 调用一次性为全部 $N$ 个分解子任务生成动作序列 $\{\pi_i\}_{i=1}^N$。图世界模型前滚校验每个动作的前提、目标满足和轨迹安全约束。
  - **模型推导自动修复（Alg. 2 `Repair`）：** 对动作语义可直接确定的违例（如因缺少邻近性导致抓取失败时补插 `NavigateTo`、目标在关闭容器内时补插导航并 `Open` 容器、遗漏关门/关电器时补插恢复动作）递归编辑，用已访问集合 $\mathcal{S}$ 防环，并按词典序评分 $\mathrm{rk}(\omega,\pi)=(\mathbf{1}[\omega.\mathrm{app}],\mathbf{1}[\omega.\mathrm{unmet}=\emptyset],\mathbf{1}[\omega.\mathrm{safe}],-\omega.d)$ 保留历史最优候选。仅当无模型可推导修复时，才将结构化失败反馈 $\omega_i$ 退回 LLM 重规划（上限 $T=5$ 次）。
  - **多任务信念感知重排（§III-B, §IV-G）：** 按候选房间概率降序计算未定位物体的期望搜索成本 $C_{\mathrm{srch}}$（含房间间最短路 $D$ 与遍历代价 $s(r)=A_{\mathrm{trav}}(r)/w_{\mathrm{cov}}$，实验取 $w_{\mathrm{cov}}=1.2\text{ m}$），对已定位物体用几何导航距离 $C_{\mathrm{loc}}$；用 $\mathcal{O}(N^2)$ 次信念前滚构建首任务代价 $h[j]$ 与转移代价 $A[i,j]$（四任务 $R^2=0.93$，五任务 $R^2=0.89$），在 $N\le 5$ 时枚举排列并校验拼接后无跨任务冲突的首个最优序列。仅执行排首位的子任务，随后按观测更新信念并重排剩余子任务（重排优化仅耗时 17 ms/指令）。
- **BEHAVIOR-1K 实验设计与核心数值（§V-A–V-G, Table I–III）：**
  - **仿真校验：** 为大规模运行（共 5,500 次评测），将 9 个原语映射到保留真实平面图、导航几何（机器人半径膨胀可行域 + A* 寻路 + 扇形视场前沿探索）与谓词的轻量 2D 执行器，单规划执行时间从 OmniGibson 的约 17 分钟降至约 1 秒；作者在全量 100 个单任务指令上与 OmniGibson 对拍，成功结果 **100% 一致**。结果均取 5 个随机种子均值±标准差。
  - **单任务（Table I, 100 个长时程任务 / 10 场景，平均 12.2 原语，240 目标谓词）：** `Qwen3-4B` 上 `llm-only` 成功率 **21.8±1.5%**、`sayplan` **55.4±1.9%**（3.4±1.7 次调用，27.6±12.0 s）、纯图修复不重调 `gavel-basic` **67.7±1.6%**（1.00 次调用，9.2±1.0 s）、完整 `gavel` **88.8±1.6%**（1.7±1.3 次调用，15.0±11.1 s）。`Qwen3-8B` 上 `llm-only` **41.2±1.2%**、`sayplan` **76.4±1.5%**（2.7±1.7 次调用，24.6±15.8 s）、`gavel-basic` **76.2±1.2%**、完整 `gavel` **91.8±1.3%**（1.5±1.1 次调用，15.5±9.4 s）。
  - **多任务（Table II, 500 条指令，$N\in\{2,3,4,5\}$，平均 18.8 原语、跨 4.66 房间，其中 199 条需电器语义状态，使用 Qwen3-8B）：** `llm-only` 成功率 **19.9±0.7%**（距离 86.69±0.37 m）、`sayplan` **75.6±2.1%**（83.01±0.42 m，51.4±27.6 s）、`epog` **60.2±0.0%**（84.51±0.00 m，13.9±3.9 s；全部失败均源于无法由图差分推出电器语义操作如洗盘子）、`gavel-map` **92.6±1.0%**（82.45±0.23 m）、`gavel-static` **92.6±1.0%**（79.69±0.45 m）、`gavel` **92.6±1.0%**（78.01±0.71 m，39.3±22.6 s）、`oracle` 100.0%（56.17 m）。分布信念与在线重排较 `gavel-map` 共节省 **4.45 m（约 5.4%）** 行程；GAVEL 剩余 38 例失败中，21 例来自上游物体抽取，仅 5 例为不可执行规划、9 例为未达目标。
  - **模型缩放对照（Table III, 多任务基准每 5 条取 1 条共 100 条子集）：** 单独使用 LLM 时，`Qwen3-4B` **2.4±1.5%**、`Qwen3-8B` **23.6±1.5%**、`GPT-5.6 Sol` **24.6±3.9%**、`Claude Sonnet 5` **38.6±2.2%**；接入 GAVEL 后分别升至 **75.2±2.1%**、**89.2±1.3%**、**99.2±0.8%**（耗时仍为 16.5 s）和 **99.4±0.5%**（耗时 18.0→18.2 s）。
- **局限（§VI）：** 评测停留在符号动作层，未建模运动学可达性与几何碰撞；RSN 语义先验主要依赖房间类型，未从跨回合观测中在线更新环境特定规律；残余失败主要来自上游自然语言指令到物体/目标的接地抽取阶段。
- **对 wiki 的映射：** [GAVEL: Graph World Models for Verified and Efficient Long-Horizon LLM Task Planning](../../wiki/entities/paper-agentic-2609-19315.md)
