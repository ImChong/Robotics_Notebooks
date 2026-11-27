# Mimir: A Neuro-Symbolic Memory System with Dynamic Grounding for Embodied Agents in Interactive Environments

- **作者：** Haoming Xu, Zhenlin He, Hengyi Wang, Jiafeng Xu, Hao Dong
- **机构：** PrimeBot Research Institute; Peking University; University of Chinese Academy of Sciences
- **版本 / 载体：** arXiv:2608.04933v1 [cs.RO]（2026-08-05 提交，9 页，4 张图，CC BY-NC-SA 4.0）
- **原文：** https://arxiv.org/abs/2608.04933（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2608.04933）
- **项目页 / 代码：** 宣称将开源或未列链接（原文摘要写明“Our code will be released soon.”，截至 2026-11-27 arXiv 原文及核查材料均未公布 GitHub 仓库地址或独立项目主页）
- **核对范围：** 通读 arXiv:2608.04933v1 全文（Abstract、Introduction、Related Work、Method〔Task Memory、World Memory、Grounding and Execution、Algorithm 1〕、Experiments〔Metrics、Main Results、Comparison with Agent and Memory Systems、Comparison with Foundation Models、Ablation Studies、Failure Analysis〕、Conclusion、References、Tables 1–4 与 Figures 1–3 题注）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Mimir 实体页](../../wiki/entities/paper-agentic-2608-04933.md)

1. **核心动机与双记忆形式化（Introduction & Method）**
   - 长程具身操作运行于部分可观测环境中：智能体交互并改变场景后，物体可能离开当前视野、发生状态迁移，或需借助早期探索才能识别；扁平文本历史或隐式策略状态虽保留过往观测，却未暴露“哪些世界事实支撑当前激活子目标”的显式决策接口。
   - 论文提出神经符号记忆系统 **Mimir**，将智能体记忆拆分为独立演化、动态对齐的两部分：$M_t = (T_t, W_t)$（Eq. 1），其中 $T_t$ 为任务记忆（记录执行进度），$W_t$ 为世界记忆（记录场景信念）。
   - **任务记忆 $T_t$（Eq. 2–6）**：定义为六元组 $T_t = (\mathcal{G}, \preceq, s_t, h_t, F_t, C_t)$，包含符号子目标集 $\mathcal{G}$、前置依赖序 $\preceq$、三值状态映射 $s_t(g) \in \{\mathrm{pending}, \mathrm{completed}, \mathrm{blocked}\}$、当前手持状态 $h_t$、被反馈证伪的“物体–来源”假设集 $F_t$，以及已完成子目标的支撑证据 $C_t$。每步待执行集合 $\mathcal{P}_t = \{g \in \mathcal{G} \mid s_t(g)=\mathrm{pending}\}$，激活目标 $g_t = \min_{\preceq}\mathcal{P}_t$（空时为 $\bot_{\mathrm{goal}}$）。环境反馈 $y_t = (\sigma_t, \Delta_t)$ 包含动作级结果 $\sigma_t$ 与受支撑状态/关系变化 $\Delta_t$；仅当后置条件满足（$\operatorname{Post}(g_t, \Delta_t)=1$）时 $g_t$ 才标记为 `completed`，从而把“动作执行成功”（如导航、开门）与“操作子目标完成”严格区分开。
   - **世界记忆 $W_t$（Eq. 7–9）**：以回合级结构化 **TreeMemory** 形式组织场景实体与关系 $W_t = (V_t, E_t, r, \ell_t, \phi_t)$，其中 $r$ 为场景根节点、$E_t$ 为父子关系（赋予每个非根实体唯一激活父节点 $p_t(v)$，表示记忆中的房间、支撑表面、容器或手持状态，不确定备选保留为独立假设）、$\ell_t$ 为语义标签、$\phi_t$ 为属性与感知/交互证据存储。写入分为观测更新 $W_t^+ = U_W^{\mathrm{obs}}(W_t, o_t)$ 与动作更新 $W_{t+1} = U_W^{\mathrm{act}}(W_t^+, a_t, y_t)$：失败动作不修改物理关系，仅将证伪绑定写入 $F_{t+1}$。

2. **目标条件召回、动态接地与闭环执行（Grounding and Execution, Algorithm 1）**
   - **召回（Eq. 9）**：针对激活目标 $g_t$ 构造查询 $q_t = Q(g_t)$，检索 $W_t^+$ 并返回有界排序列表 $\mathcal{R}_t = [(v_k, p_t^+(v_k), \eta_{t,k})]_{k=1}^{K_t}$（包含候选实体、记忆父节点与词法/视觉/语义/关系证据），同时过滤已在 $F_t$ 中被否定的绑定。
   - **动态接地（Eq. 10）**：算子 $G(g_t, \mathcal{R}_t, F_t, A_e, x, o_t)$ 剔除不满足合法交互或约束的候选，若仍剩多候选则由 VLM 对照当前第一人称观测 $o_t$ 与候选记忆描述，在受限候选集 $\{v_k\}_{k=1}^{K_t}$ 内做视觉消歧（防止幻觉编造记忆中不存在的物体）。接地后的目标 $\hat{g}_t = (g_t, v_t^\star, u_t^\star, d_t^\star, \eta_t^\star)$ 在指令省略来源时自动以 $p_t^+(v_t^\star)$ 补全 $u_t^\star$；若无合法候选则返回 $\bot_{\mathrm{grd}}$ 并触发探索策略 $\operatorname{Explore}_e$（失效闭锁，fail-closed）。
   - **抽象规划与具身适配（Eq. 11 & Algorithm 1）**：规划器在本体无关抽象动作空间 $\bar{\mathcal{A}}$ 生成 $\bar{a}_t = \Pi(\hat{g}_t, T_t, W_t^+)$，经具身适配器 $\rho_e: \bar{\mathcal{A}} \to A_e \cup \{\bot_{\mathrm{act}}\}$ 映射为合法动作 $a_t$；映射失败时触发一次 `Replan`，仍失败则调用 $\operatorname{Block}(T_t, g_t)$ 避免向环境发出非法动作。与依赖预设 3D 场景图的 SayPlan 不同，Mimir 直接将第一人称观测、有序任务状态、召回候选、失败记录与世界记忆快照序列化为 VLM 决策输入。

3. **实验设定与核心量化结果（Experiments & Tables 1–4）**
   - **基准与指标**：在 EmbodiedBench（Yang et al., 2025）的 **EB-ALFRED** 与 **EB-Habitat** 上评估（temperature=0 消除采样随机性），汇报 **Success Rate (SR)** 与 **Goal Condition Success Rate (GC)**（百分数）。四任务均值覆盖 `Base`、`Common-sense`、`Complex-instruction` 与 `Long-horizon`。
   - **多底座泛化（Table 1 & Table 4）**：在 12 个模型（含 Table 4 中 Qwen3-VL-32B Thinking 变体共 13 个评测底座配置）上均一致提升 SR 与 GC。例如 `InternVL3-8B` 在 EB-ALFRED 上 SR 从 17.5% 升至 60.0%（**+42.5**），GC 从 28.6% 升至 68.5%（**+39.9**）；`Gemma-3-12B` 在 EB-Habitat 上 SR 从 26.5% 升至 64.0%（**+37.5**），GC 从 33.0% 升至 75.7%（**+42.6**）；`Qwen3-VL-32B`（Instruct）在 EB-ALFRED / EB-Habitat 上 SR 分别由 50.5% 升至 **68.0%**（+17.5）、由 50.5% 升至 **71.5%**（+21.0）；`Qwen3.5-27B` 在 EB-ALFRED 上达 **72.5%** SR / **79.5%** GC。摘要汇报跨任务最大 SR 增益 **42.5%**、平均增益 **23.0%**。
   - **与智能体及记忆系统对比（Table 2，统一 Qwen2.5-VL-72B 底座，Base 与 Long-horizon 非加权均值）**：对比 Voyager、Reflexion、Cradle、RoboOS 与 RoboMemory，Mimir 在 EB-ALFRED 上取得 **68.0%** SR / **76.3%** GC（RoboMemory 为 67.0% / 78.4%），在 EB-Habitat 上取得 **90.0%** SR / **94.6%** GC（领先 RoboMemory 的 74.0% / 81.0% 达 +16.0 SR / +13.6 GC）；在同底座最佳先前系统对比下，总体平均 SR 提升 **8.5%**。
   - **与闭源基础模型对比（Table 3，EB-Habitat Long-horizon，最大生成预算 16,384 tokens）**：`Mimir (Qwen3-VL-32B)` 取得 **86.0%** SR / **93.0%** GC，高于同期评测的 Claude-Sonnet-4-6（82.0% / 86.8%）、Gemini-3.5-Flash（80.0% / 84.8%）、GPT-5.5（78.0% / 79.5%）、Gemini-3.1-Pro（70.0% / 75.0%）与 Kimi-2.5（30.0% / 34.6%）。
   - **消融实验（Table 4，四任务均值）**：
     - 去掉世界记忆后，EB-Habitat 上 SR 在 `Qwen3-VL-8B` 由 65.0% 骤降至 **12.5%**，在 `Qwen3-VL-32B` 由 71.5% 骤降至 **12.5%**；
     - 去掉任务记忆后，EB-ALFRED 上 SR 在 `Qwen3-VL-8B` 由 57.0% 降至 **34.0%**（-23.0），在 `Qwen3-VL-32B` 由 68.0% 降至 **30.5%**（-37.5）；
     - 推理模式对比（Qwen3-VL-32B）：无 Mimir 时 Thinking 在 EB-Habitat 略占优（52.0% vs 50.5%），接入 Mimir 后 Instruct 模式在两个基准的 SR/GC 四项指标均高于 Thinking（EB-ALFRED 68.0/73.0 vs 61.5/69.1；EB-Habitat 71.5/78.6 vs 67.5/75.6），表明当状态外化接地后，Instruct 模型能更直接利用结构化决策状态。

4. **失败分析与局限（Failure Analysis, Fig. 3）**
   - **语言指代歧义**：指令仅以粗粒度类别或颜色描述目标而场景中存在多个同类实例时，Mimir 虽能保留并比对候选，但输入本身不足以唯一确定评测标注的目标实例。
   - **不可见与感知证据匮乏**：严重遮挡、非现实环境配置或渲染细节不足导致观察未能写入有效世界记忆条目或条目过于模糊；记忆只能保存与组织已观测证据，无法重构未进入感知流的事实。
   - **工程与开放性边界**：实验限于仿真基准 EB-ALFRED 与 EB-Habitat，尚未验证真实物理机器人部署；官方源码尚未公开，TreeMemory 匹配细节、Prompt 模板与具身适配器实现尚待开源后核验。
