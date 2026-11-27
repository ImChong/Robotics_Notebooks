# VASO: Formally Verifiable Self-Evolving Skills for Physical AI Agents

- **作者：** Yunhao Yang, Neel P. Bhatt, Kevin Wang, Samuel Tetteh, Zhangyang Wang, Ufuk Topcu（* Equal Contribution）
- **机构：** The University of Texas at Austin、Iowa State University
- **版本 / 载体：** arXiv:2606.05395v1 [cs.RO]（2026-06-03），HTML 全文（CC BY 4.0）
- **原文：** [arXiv HTML v1 全文](https://arxiv.org/html/2606.05395) / [arXiv 摘要页](https://arxiv.org/abs/2606.05395) / [项目主页](https://languagegroundedriskdetection.github.io/ProjectPage/vaso-webpage/)
- **项目页 / 代码：** 宣称将开源或未列链接 — 截至 2026-11-27，论文未提供公开 GitHub 仓库链接，匿名项目主页 `https://languagegroundedriskdetection.github.io/ProjectPage/vaso-webpage/` 在 Resources -> Code 一栏明确标注「We will make the code public after acceptance due to the anonymity policy」，当前无公开可运行官方源码仓库（不建 `sources/repos/` 档）
- **核对范围：** 通读全文 §1–§7、Table 1、Figure 1–6、References [1]–[32] 及全部附录 Appendix A–D（含 Algorithm 1、Table 2–3、Listing 1–12 的技能生成提示词、Jackal 与 PX4 控制 API 及 11 条 LTL 规范、NuSMV 自动机编译示例、基线对比提示与 PDDL 域、Held-Out 复用任务提示词、地面/空中平台案例与命题对齐失败案例、四项消融分析），并核对项目主页全文
- **通读：** 已完成

## 原文核对笔记

- **核心问题与动机（§1–§2）：** 现有具身智能体（如 Voyager、AutoManual、Eureka、AutoSkill 等）多依赖执行结果、单元测试、环境奖励或 LLM 自反思等“轨迹级证据（trace-level evidence）”进化技能，只能说明技能在已采样执行上有效，无法保证技能诱导出的规划在未测试条件下满足线性时序逻辑（LTL）安全契约。另一方面，传统形式化验证（如 VerifyLLM 等）多只做单次规划的通过/拒绝过滤，或像 LAD-VF / RLVF 那样把反例吸收到扁平提示词或模型权重中，缺乏既面向形式化模型检测又面向 LLM 规划器的可复用、可解释技能载体。
- **形式化技能表示（§3）：** 将技能定义为三元组 $sk = (\mathcal{G}, \psi_{sk}, \mathcal{C}_{sk})$：
  1. **全局规范 $\mathcal{G} = \{\varphi_1, \dots, \varphi_m\}$：** 必须始终满足的全局 LTL 安全约束；
  2. **局部规则 $\psi_{sk}$：** 描述该技能预期行为与活性目标的局部 LTL 规范；
  3. **语义契约 $\mathcal{C}_{sk} = (L_{sk}, \mathcal{C}_{sk}^p)$：** 由两套耦合接口组成——面向形式化验证的**命题对齐标注函数（proposition-aligned labeling function）** $L_{sk}: \sigma_t \to 2^{\mathcal{AP}}$（将低层状态、观测与控制指令映射为原子命题真值），以及面向规划器的**文本规划模板（text-based plan template）** $\mathcal{C}_{sk}^p$（提供可执行 Python 行为先验）。
- **两级形式化验证与无梯度技能自进化闭环（§4, App. A）：**
  1. **技能级可行性检验（§4.1）：** 构造全域通用转移系统 $\mathcal{M} = (S, S_0, T, L)$（$S = 2^{\mathcal{AP}}, T = S \times S$），利用符号模型检测器（NuSMV）检验 $\mathcal{M} \models \neg\Phi_{sk}$（其中 $\Phi_{sk} = (\bigwedge_{\varphi \in \mathcal{G}} \varphi) \wedge \psi_{sk}$）。若成立则说明局部规则与全局规范逻辑冲突、不存在满足路径，立即回流 $\mathcal{F}_{\text{skill}}$ 修正局部规则，避免对矛盾规范做无效优化。初始生成技能一次通过率 89%，经 1 轮反馈升至 97%。
  2. **规划级行为验证（§4.2, Alg. 1, Tab. 2）：** 规划器 $\mathcal{F}_{\text{plan}}$ 基于语义契约与任务提示生成可执行计划 $\pi = \mathcal{F}_{\text{plan}}(\mathcal{C}_{sk}, \tau)$，通过算法 1 提取指令序列 $\sigma_0, \dots, \sigma_H$ 并借助 $L_{sk}$ 映射为命题轨迹，编译为 NuSMV 自动机 $\mathcal{A}_\pi$ 后检验 $\mathcal{A}_\pi \models \mathcal{G} \wedge \psi_{sk}$。
  3. **反例驱动的文本梯度更新（§4.2）：** 验证失败时，模型检测器返回反例轨迹（如违反 `G(person_observed -> X stop)`），VASO 将其转化为自然语言文本梯度 $g$，适配 LLM-AutoDiff 以冻结基础模型参数的方式迭代更新语义契约 $\mathcal{C}_{sk}^{(k+1)} = \mathcal{F}_{\text{skill}}(\mathcal{C}_{sk}^{(k)}, g, \tau)$。
- **物理平台评测与实验结论（§5, Tab. 1, App. B–D）：**
  - 实验配置：使用 `GPT-5-nano` 作为 $\mathcal{F}_{\text{skill}}$、`GPT-4o-mini` 作为 $\mathcal{F}_{\text{plan}}$，在 Clearpath Jackal 地面移动机器人（ROS 速度与 YOLO 检测 API，5 条全局/局部规范）与 PX4 四旋翼无人机（1 FPS NED 速度序列，6 条全局/局部规范）共 11 条 LTL 规范、20 个生成技能及 400 个规划上评测。
  - 核心指标（Table 1）：在安全性得分（SS）与仿真任务完成率（TC）上，手工命题对齐 VASO 达到 **97.2% SS / 85.3% TC**，自动命题对齐 VASO 达到 **96.8% SS / 86.5% TC**（性能与手工对齐差值在 1% 以内）；优于零样本基线 LLM+P（79.5 / 88.3）、ReAct（73.3 / 86.5）、LLM-Planner（90.3 / 89.3），以及学习类基线 DSPy（86.8 / 87.0）、RoboInstruct（82.5 / 67.0）、RLVF（94.3 / 82.5）。单技能优化样本少于 100 个、耗时低于 20 分钟（约为微调方案 RLVF 训练时间的 1/4），7 步内即超过 90% 安全率，而无结构化技能抽象的扁平提示优化基线 LAD-VF 在 10 步后仍停留在约 85%。
  - 跨提示复用（§5, App. B.1）：冻结优化后的 4 个代表性技能（`drive`、`find-object`、`bypass-object`、`fly-in-square-shape-traj`）并在每技能 5 个域内 + 5 个未见域外提示上生成各 5 次规划（共 200 份规划），在未见任务提示上取得 **92–100%** 安全率与 **80–96%** 任务完成率。
- **已披露局限与边界（§4.2, §7, App. C）：**
  1. 形式化保证严格依赖于由大模型自动生成的命题对齐函数 $L_{sk}$ 的忠实度；附录 C 失败案例表明，若 `_norm3` 漏掉 PX4 垂直速度分量 `v[2]`，会导致超速指令 `(0.8, 0.6, -0.5)`（合速度 1.12 > 1）在符号检查中被误判为通过。
  2. 当前仅支持顺序、非并发的单一技能验证，尚未覆盖多技能并发/交错执行及多智能体共享环境约束场景。

对 wiki 的映射：[VASO: Formally Verifiable Self-Evolving Skills for Physical AI Agents](../../wiki/entities/paper-agentic-2606-05395.md)
