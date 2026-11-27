# Pre-Execution Safety Gate & Task Safety Contracts for LLM-Controlled Robot Systems

- **作者：** Ike Obi, Vishnunandan L.N. Venkatesh, Weizheng Wang, Ruiqi Wang, Dayoon Suh, Temitope I. Amosa, Wonse Jo, Byung-Cheol Min
- **机构：** SMART Laboratory, Department of Computer and Information Technology, Purdue University; Department of Information and Telecommunication Engineering, Incheon National University
- **版本 / 载体：** arXiv:2604.05427v1 [cs.RO]（2026-04-07 提交，CC BY 4.0）
- **原文：** https://arxiv.org/abs/2604.05427（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2604.05427）
- **项目页 / 代码：** 未确认开源代码（arXiv:2604.05427v1 原文及公开检索未列明独立项目主页或 GitHub 源码仓库链接）
- **核对范围：** 通读 arXiv:2604.05427v1 全文（Abstract、Sections I–VII、Definitions 1–5、Equations 1–10、Tables I–II、References [1]–[22]）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[SafeGate 实体页](../../wiki/entities/paper-agentic-2604-05427.md)

1. **核心问题与安全形式化（Sections I–III）**
   - 现有 LLM 机器人任务规划器普遍直接将自然语言指令转换为可执行代码或动作序列，缺少在规划与执行前拦截不安全或存在缺陷指令的门控机制；即便表面看似安全的指令，在缺乏持续约束监控时也可能在执行过程中诱发危险状态转移。
   - 论文借鉴 **ISO 12100**（风险传导链 $\text{Hazard} \to \text{Hazardous Situation} \to \text{Harm}$）与 **ISO 13482**（个人护理机器人安全规范 §5.2–5.5），将任务指令上下文形式化为 $(c, \mathcal{E}, u)$：自然语言指令 $c \in \Sigma^*$、场景上下文 $\mathcal{E}=(\mathcal{O},\mathcal{R},\mathcal{P},\mathcal{L})$（物体集合 $\mathcal{O}$、空间/支撑关系 $\mathcal{R}$、在场人员 $\mathcal{P}$、环境布局 $\mathcal{L}$）以及包含显式未知属性集 $u_? \subseteq \mathit{Attributes}$ 的用户画像 $u$。
   - 定义 1–4 形式化了危害 $\eta$（类型 $\in \{\text{physical}, \text{psychological}, \text{operational}, \text{consequential}\}$，来源 $\eta.\mathit{source}$，严重度 $\in \{\text{critical}, \text{high}, \text{moderate}, \text{low}, \text{negligible}\}$）、危险状态 $\mathit{hs}(\eta,p,s) \triangleq \mathit{Exposed}(p,\eta,s)$、运行安全性 $\mathit{Safe}(c,\mathcal{E},u)$ 与危险状态可达性 $\mathit{Reachable}_{HS}(c,\mathcal{E},u,s_0)$。

2. **SafeGate 六阶段神经符号架构（Section IV）**
   - **Stage 1 — Hazard Analysis Matrix（HAM）**：LLM 接收 $(c, \mathcal{E}, \mathcal{U}, \mathcal{R})$（指令、场景、用户画像、机器人能力声明），在 3 个信息层（Task Level $\sigma_T$、Deployment Level $\sigma_D$、User Analysis Level $\sigma_U$）$\times$ 4 类危害（H1 Physical、H2 Psychological、H3 Operational、H4 Consequential）共 12 个分析单元中系统识别危害提案集 $\mathcal{H}$ 及未知信息集 $\mathcal{X}$（含是否关键 $x.\gamma$ 标记；明确指示模型不得将未知用户属性默认假定为安全）。
   - **Stage 2 — Hazard Binding Layer**：将 $\mathcal{H}$ 绑定到人工整理的 **Hazard Template Library（HTL）**（源于 ISO 13482 家庭服务场景，含不变量、守卫与中止规则），并将模板变量实例化为指令中的具体对象，产出已匹配危害 $\mathcal{M}$ 与未映射危害 $\mathcal{U}_\bot$；若超出 HTL 覆盖，触发延迟并引导用户扩展新模板。
   - **Stage 3 — Deterministic Decision Gate**：基于严重度阈值 $\theta$（默认 $\text{high}$）按优先级规则表（Eq. 1）输出三向决策 $D \in \{\textsc{authorize}, \textsc{defer}, \textsc{reject}\}$：
     - R1：存在不可预防的高危危害（$p=\textit{unprev.} \wedge s \ge \theta$）$\to \textsc{reject}$；
     - R1b：高危危害可预防性未知（$p={?} \wedge s \ge \theta$）$\to \textsc{defer}_1$；
     - R2：存在未绑定模板的危害（$\mathcal{U}_\bot \neq \varnothing$）$\to \textsc{defer}_2$（终结型延迟，需扩展 HTL）；
     - R3：存在关键未知信息（$\mathcal{X}_c \neq \varnothing$）$\to \textsc{defer}_1$（生成澄清问题并在补全画像后重跑 Stage 1）；
     - R3b：存在不确定且致命的已匹配危害（$s=\textit{crit.} \wedge u=\textit{unc.}$）$\to \textsc{defer}_1$；
     - R4：其余情况 $\to \textsc{authorize}$。
   - **Stage 4 — Contract Compilation & Plan Generation**：仅当 $D=\textsc{authorize}$ 时，将匹配模板中的预防条件通过变量替换 $\beta^{-1}$ 编译为 **Task Safety Contract** 三元组 $\mathcal{C} = (\mathcal{I}_C, \mathcal{G}_C, \mathcal{A}_C)$（Eq. 2：不变量 $\mathcal{I}_C$、前置守卫 $\mathcal{G}_C$、中止条件 $\mathcal{A}_C$），经历提取、正则替换、默认值补全、按 $(\mathit{id},\beta)$ 去重、词汇表 $\mathcal{V}$ 校验与组装 6 步；随后将 $\mathcal{C}$ 转为自然语言规划约束，交由基于 SMART-LLM 的任务规划器生成受约束动作计划 $\pi$。
   - **Stage 5 — Static Plan Verification**：在执行前对 $\pi = \langle a_1, \dots, a_L \rangle$ 做符号执行并结合 **Z3 SMT 求解器**校验状态轨迹 $\langle s_0, \dots, s_L \rangle$（跟踪机器人位置与手持物、物体位置与热/锐/密封等属性、人员距离、门/容器状态、电源状态），核对 Eq. 3–5（每步不变量成立、动作前守卫成立、任意状态中止条件不触发）。若经 $k$ 次修复仍未通过，则将决策降级为 `reject`。
   - **Stage 6 — Runtime Contract Monitoring**：在线逐动作核查——动作前检查守卫与中止条件，动作后检查不变量与中止条件；一旦违背立即记录 `MonitorViolation` 并触发**吸收态停机（absorbing halt）**，拒绝后续所有动作以防规划器覆盖，并在结束时输出 `MonitorSummary`。

3. **实验评测与核心数据（Sections V–VI）**
   - **230 项合成基准测试（Stages 1–3，Tables I–II）**：涵盖 assistive / navigation / manipulation 三领域与 simple（<5 步）/ medium（<10 步）/ complex（>11 步）三复杂度；所有框架（SafeGate、SELP、RoboGuard）均基于 GPT-4o，另对比单提示词三分类基线 GPT-4o 与 Gemini 2.5-flash。
     - **Table I**：SafeGate 达到 $AR\text{-}S = 92.8\%$、$AR\text{-}U = 0.0\%$、$DR = 9.2\%$（$DR\text{-}S = 4.8\%$、$DR\text{-}U = 4.0\%$、$DR\text{-}A = 28.3\%$）；GPT-4o 为 $AR\text{-}S = 51.8\%$、$AR\text{-}U = 0.0\%$、$DR = 40.6\%$（$DR\text{-}S = 48.2\%$）；Gemini 2.5-flash 为 $AR\text{-}S = 12.0\%$、$AR\text{-}U = 0.0\%$、$DR = 51.5\%$（$DR\text{-}S = 79.5\%$）；SELP 为 $AR\text{-}S = 85.5\%$、$AR\text{-}U = 26.0\%$；RoboGuard 为 $AR\text{-}S = 74.7\%$、$AR\text{-}U = 39.0\%$。在 SafeGate 选择延迟的 21 项模糊任务中，SELP 直接授权了 16 项（76.2%），RoboGuard 直接授权了 15 项（71.4%）。McNemar 配对检验显示 SafeGate 在 $AR\text{-}S\%$ 上显著优于 GPT-4o（$\chi^2=32.1, p<10^{-8}$）、Gemini（$\chi^2=67.0, p<10^{-15}$）和 RoboGuard（$\chi^2=9.0, p=0.003$），相对 SELP 高 7.3%（$\chi^2=2.6, p=0.109$）。
     - **Table II（安全分类混淆矩阵，unsafe 为正类，defer 计入阻断）**：SafeGate 取得 TP=100, FN=0, FP=6, TN=77，Precision **94.3%**、Recall **100.0%**、F1 **97.1%**（每抓出 1 个真不安全任务仅误拦 0.06 个安全任务，远低于 GPT-4o 的 0.40 与 Gemini 的 0.73）；GPT-4o F1=83.3%、Gemini 2.5-flash F1=73.3%、SELP F1=79.6%（FN=26, Rec=74.0%）、RoboGuard F1=67.0%（FN=39, Rec=61.0%）。
   - **30 项 AI2-THOR 仿真场景（Sec. V-B & VI-B）**：分为组合危害推理（Compositional Hazard）、不完整信息推理（Incomplete Information）、后果状态推理（Consequential State）三类各 10 场景。SafeGate 在全部三类上实现 $CR\% = 0\%$（拦截全部 30 个危险任务）；RoboGuard 授权了 93.3%（28/30）的危险任务（在后两类上 $CR\% = 100\%$）；SELP 在后果状态上 $CR\% = 10\%$、组合危害上 $CR\% = 40\%$、不完整信息上 $CR\% = 100\%$（全部放行）。
   - **真机定性验证（Sec. V-C & VI-C）**：在物理机器人上验证 Stages 4–6（HTL 危害检测、未覆盖任务的 HTL 模板扩展提示、Task Safety Contract 编译、规划验证与运行时不变量监控），表明运行时不变量能有效作为 LLM 控制系统的安全护盾；作者指出未来工作将探索自动修复缺陷/不安全指令而非仅做拒绝。
