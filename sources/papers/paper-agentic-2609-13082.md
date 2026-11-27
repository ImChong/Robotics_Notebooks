# Embodied-BenchForge: A Closed-Loop Agentic Workflow for Embodied Benchmark Construction

- **作者：** Baoyang Jiang, Fengchun Zhang, Leyuan Wang, Haotian Li, Yida Wang, Zhe Ji, Jinshan Lai, Xi Ren, Danyang Li, Zheng Yang, Jianwei Hu, Qiang Ma
- **机构：** QiYuan Lab、电子科技大学（University of Electronic Science and Technology of China）、北京邮电大学（Beijing University of Posts and Telecommunications）、Northeastern University、北京航空航天大学（Beihang University）、清华大学（Tsinghua University）
- **版本 / 载体：** arXiv:2609.13082v1 [cs.AI]（Submitted on 11 Sep 2026，arXiv HTML 全文版本）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2609.13082) / [arXiv HTML 全文](https://arxiv.org/html/2609.13082)
- **项目页 / 代码：** 未确认开源代码（附录 C.1 提及 "accompanying code snapshot"，但正文、arXiv 页面与当前检索材料均未给出可访问的 GitHub 仓库或独立项目主页 URL）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections 1–6、Tables 1–8、Figures 1–3、References，以及 Appendix A–E 含 Tables 9–32 与 Figures 4–8）
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[Embodied-BenchForge 实体页](../../wiki/entities/paper-agentic-2609-13082.md)
- **研究动机与核心定位：**
  - 现有具身基准与自动化任务生成方法（如 ProcTHOR、RoboCasa、MimicGen、RoboGen、GenSim、AutoBencher、BenchAgents、A2Eval 等）要么局限于特定仿真器/任务族，要么仅覆盖单阶段或缺乏产物级过程验证（process-level verification），导致上游局部缺陷向下游传播，终末过滤既浪费算力又容易丢弃已验证产物。
  - 本文提出 **Embodied-BenchForge**，将具身基准构建形式化为 **Closed-Loop Benchmark Synthesis（闭环基准合成）**，把前向产物合成与后向验证修复耦合在同一框架中。
- **核心架构与形式化表示：**
  1. **Skill-Orchestrated Artifact Synthesis（技能编排产物合成）：**
     - 每个产物表示为四元组 $a_i = \langle \tau_i, x_i, m_i, p_i \rangle$（类型 $\tau_i$、内容 $x_i$、元数据与验证状态 $m_i$、构建血缘 provenance $p_i$），产物依赖图 $G=(V,E)$ 记录依赖关系。
     - 每个构建操作封装为类型化技能 $s_k = \langle I_k, O_k, P_k, F_k \rangle$（输入类型、输出类型、执行前置条件、可执行后端：模型/确定性程序/仿真器接口/资源适配器/评分与验证器编译器），并通过层次化技能分解（Hierarchical Skill Decomposition）提供前向构建算子与后向验证修复算子（实现快照包含 64 个注册技能/卡片定义）。
     - 前向 5 阶段路径（Appendix D.1）：`S1 Intent Blueprinting` → `S2 Data Collection`（Acquire Raw Sources / Adapt Existing Benchmarks / Collect in Simulation）→ `S3 Evidence & State Structuring` → `S4 Benchmark Synthesis`（Template Design / Code Generation / Metric Production）→ `S5 Verification & Evaluation Reporting`。
  2. **Requirement-Guided Verification and Repair（需求引导的验证与修复）：**
     - 每类产物绑定需求契约 $\kappa_\tau = \{\phi_{\tau,1}, \dots, \phi_{\tau,n}\}$，涵盖确定性检查、基于执行的检查和辅助性模型检查（模型检查结果不得覆盖失败的确定性或执行检查）。
     - 质量门控（Quality Gates，图 2 与 Table 10）：`A. Grey-Batch Synthesis`（小批量灰度试产）→ OE-Track 分支（`B1. Invalid Item Screening`、`C1. Image-Dependency Blind Evaluation`）或 IE-Track 分支（`B2. Task Validity Check`、`C2. Task Completion Test`、`D2. Initial-State Check`）→ `E. Typical Model Evaluation`（11 个代表性模型面板）→ `F. IRT Diagnosis`（正则化 2PL 模型 $P(Y_{mi}=1\mid\theta_m)=\frac{1}{1+\exp[-a_i(\theta_m-b_i)]}$，High-Disc. 阈值 $a_i \ge 0.7$，辅以校正题总相关 $r_{i,-i}$ 与 $\mathrm{ReverseRisk}_i$ 审计）。
     - 血缘驱动局部修复（Appendix A.1.3）：定位最早因果失效节点 $v_f$，仅重算其下游闭包 $D(v_f)=\{u\in V\mid v_f \rightarrow^* u\}$。定义 6 类操作：Local field repair（局部字段修复）、Template re-binding（模板重绑定）、Candidate re-synthesis（候选重合成）、Skill re-execution（技能重执行）、Upstream rollback（上游回滚）与 Final rejection（终审拒绝）。
- **构建的基准套件（Table 2, Table 11, Table 22）：**
  - **Offline EQA Track（OE-Track）**：6 个基准各 10,000 条样本（共 60,000 条）：
    1. `SpatialHome-Task (SHT)`：ALFRED（Sim.），家用机器人，8 能力维度 / 6 题型 / 22 专家模板；
    2. `SpatialHome-Nav (SHN)`：Habitat（Sim.），移动机器人导航，7 维度 / 4 题型 / 13 模板；
    3. `SpatialDrive (SD)`：CARLA（Sim.），自动驾驶，8 维度 / 6 题型 / 80 模板；
    4. `SpatialArm (SA)`：LIBERO（Sim.），机械臂操作，9 维度 / 8 题型 / 203 模板；
    5. `SpatialUAV (SU)`：真实无标注航拍图像，无人机视角理解，9 维度 / 7 题型 / 260 模板；
    6. `SpatialQuadruped (SQ)`：TartanGround 数据集，四足/地面机器人，8 维度 / 8 题型 / 192 模板。
  - **Interactive Embodied Track（IE-Track）**：基于 AI2-THOR 的 28 个室内场景（8 厨房、7 客厅、7 卧室、6 浴室），构建 220 个可执行交互任务（Simple 110 / Hard 110，每类房间各 55 题；746 个实例化动作表达式；单回合上限 150 步），同时保留交互轨迹用于轨迹接地 EQA（计算 SR、EQA Accuracy、联合指标 SRL 及 Avg. Steps）。
- **核心实验与分析数据：**
  - **构建配置（Table 20, E.1.1）：** 工作流编排采用 `Qwen3.6-27B`，产物合成与模型辅助验证采用 `GPT-5.5 Pro`；运行硬件为 4×H100 80GB，vLLM 0.19.0（TP=4）。
  - **质量评估（Tables 3, 23, 24）：** 四位裁判模型（GPT-5.5、Claude Opus 4.7、Gemini 3 Flash、Qwen3.6-35B-A3B）在 8 个维度（UIA 12%、FSQ 12%、QAC 14%、EGC 18%、SGC 16%、AFC 10%、TSD 10%、SSC 8%）打分，6 个 OE-Track 基准 Overall 均分为 `88.34`（SQ）至 `90.75`（SA）。
  - **效率与成本（Tables 18, 19）：** OE-Track 构建 10,000 条样本平均耗时 `86.0 min`（范围 `38.0–160.0 min`），平均消耗 `11.45M` tokens（`1,145` tokens/item），池化吞吐量 `116.3 items/min`（SD 最快 263.2 items/min，SU 最慢 62.5 items/min，因真实航拍图 S3 阶段耗时 65.0 min）；IE-Track 构建并验证 20 个可执行任务平均耗时约 `1.5 小时`。S4 阶段占总 token 成本的 `52.8%–62.3%`，输入与输出 token 比例约为 `14.4:1`。
  - **模型评测与能力断层（Tables 4, 5, 25–30）：**
    - OE-Track（11 个模型 + 人类）：人类均分 `85.89`，GPT-5.5 `57.67`、Claude Opus 4.7 `56.61`、Gemini-3-Flash `51.97`、Qwen3.6-35B `49.86`（本地模型最高）。
    - IE-Track（220 任务）：人类 SR `97.73%` / EQA Acc `74.09%` / SRL `71.82%`；GPT-5.5 SR `83.18%` / EQA Acc `73.64%` / SRL `61.36%`；Claude Opus 4.7 SR `77.73%` / SRL `55.00%`；小模型如 Qwen3.5-4B/2B/0.8B 的 SR 均为 `0.00%`，但 EQA Acc 仍可达 `55.91%` / `46.82%` / `22.73%`，体现出“轨迹理解”与“闭环可执行能力”之间的显著鸿沟。
  - **技能复用（Tables 6, 15, 16）：** 单次运行平均调用 `41.0` 个技能，复用 `37.6` 个，总复用率 `91.6%`（仿真类 OE-Track 达 `97.5%`，SU `82.9%`，SQ `87.8%`，IE-Track `81.8%`）。
  - **消融实验（Tables 7, 8）：**
    - 系统级消融（OE-Track 均值）：完整系统 LLM Judge `89.07`、Human `91.28`、Judge–Human $\rho=0.87$、Valid Rate `93.7%`；移除 Verification and Repair 退化最大（Human 降 `23.25` 分至 `68.03`，Valid Rate 降至 `62.4%`）；移除 Source Grounding 降 `19.16` 分；移除 Evidence Skill 降 `17.20` 分。
    - 过程质控消融（SpatialUAV 1,000 条）：仅终末检查 Valid Rate 仅 `10.4%`（High-Disc. `14.6%`）；移除 Requirement Contracts Valid Rate 降至 `62.2%`；移除 Provenance-Guided Repair 虽质量接近（`88.74` vs `89.21`），但 token 消耗由 `2.62M` 增至 `3.93M`（增加 50%）。
  - **IRT 诊断稳定性（Table 32）：** 60,000 条 OE-Track 样本池化 High-Disc. 比例为 `42.5%`（95% bootstrap CI `[31.3, 43.7]%`），2PL 区分度与古典高低分组差值 $D_i$ 秩相关 $\rho_S=0.906$。
