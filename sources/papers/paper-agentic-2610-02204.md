# Reconstruct, Practice, Go Real: Guided Self-Improvement for Embodied Agents

- **作者：** Yen-Jen Wang, Haozhe Jiang, Shuying Deng, Haoru Xue, Weirui Ye, Rocky Duan, Nika Haghtalab, S. Shankar Sastry, Pieter Abbeel, Haozhi Qi
- **机构：** UC Berkeley、Amazon FAR、MIT、University of Chicago
- **版本 / 载体：** arXiv:2610.02204v1 [cs.RO]（2026-10-01，17 pages, 6 figures, 10 tables）
- **原文：** [arXiv HTML](https://arxiv.org/html/2610.02204) · [arXiv Abs](https://arxiv.org/abs/2610.02204)
- **项目页 / 代码：** [项目页](https://rpg-robot.github.io/) · 宣称将开源或未列链接（项目页标注 "Code coming soon"，截至 2026-11-27 未公开可运行源码仓库；抓取候选中的 `vision-language-kinematics/vision-language-kinematics.github.io` 仅为项目页模板来源，非本论文官方实现仓库）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、Sections I–VI、Appendices A–G，含 Table I–X、Fig. 1–6 说明）及项目页全文（含 38 个技能库分层清单、实机与仿真任务说明）
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Reconstruct, Practice, Go Real: Guided Self-Improvement for Embodied Agents](../../wiki/entities/paper-agentic-2610-02204.md)
- **核心问题与定位：** 针对具身智能体在 Code-as-Policy（执行高效但硬编码检查脆弱）与 Agent-as-Policy（视觉推理灵活但低层控制慢、易遗漏细节）之间的权衡，提出 **RPG（Reconstruct, Practice, Go Real）** 框架：在不更新底层大模型（MLLM）权重的前提下，从离线数据集抽取操作能力并构建仿真练习套件，通过特权状态执行对比与视频归因分析，持续演化共享符号技能库 $\mathcal{L}_r$ 与系统提示词 $\rho_r$，并在实机标定与冻结后零任务微调部署。
- **三阶段与六角色架构：**
  1. **Reconstruct（构建并冻结练习套件）：** `Constructor` 从离线数据集 $\mathcal{D}$（本文采用包含 193 个任务的 ABC 数据集）筛选源任务、查看 3–8 条轨迹视频与文本描述（单任务预算 7.5 分钟），在 MuJoCo 中基于双臂 YAM 机器人（双 6-DoF 臂 + 夹爪，14 维绝对关节位置指令）构建局部或类比仿真练习任务（Table I：整理太阳镜→合上眼镜盒；整理化妆品→关闭抽屉；卷毛巾→单次折叠）及结果评估器 `check_success`（练习开始前人工核验评估器），并在训练前将 CaP-X 的 9 个底层工具扩展为 15 项初始库 $S_0 = (\mathcal{L}_0, \rho_0)$。
  2. **Practice（特权执行对比、诊断与跨任务门控迭代）：**
     - `Runtime Agent`（仅观测部署可用 RGB-D 与本体感知）与 `Privileged Agent`（共享同一模型、提示词与技能库，额外获得物体位姿、尺寸、速度及布料角点等仿真真值状态）并行执行 5 个固定开发初始状态。
     - `Video Analyzer`（Fable 5.1, `xhigh` reasoning）对比两者轨迹及源数据集视频，定位首个可观测失败并提出具体修正建议。
     - `Implementor` 对每个未达 5/5 的开发任务至多生成 1 个独立候选修改 $\widetilde{S}$（可新增技能、修订技能或改写 $\rho_r$）。
     - **跨任务门控与合并（`Merger`）：** 在全部 22 任务 × 5 开发种子（110 回合）上验证候选，满足 $\Delta\widehat{J} > 0 \land \min_i \Delta\widehat{q}_i \geq -0.20$ 方可入围；按 $\Delta\widehat{J}$ 排序后贪心三路合并（每轮最多 8 次尝试，冲突由编码智能体消解），合并体 $S^+$ 必须再次通过同一跨任务门控方可更新 $S_{r+1} = S^+$，否则回滚。
  3. **Go Real（适配并冻结实机部署）：** 对最终系统与基线采用统一的坐标系标定和硬件适配（夹爪垫片偏置、空夹爪开度），不做按物体或按任务的特调；关闭 Privileged Agent 与 Video Analyzer，使用 `low` reasoning effort 降低实机推理时延。
- **关键实验数据核对：**
  - **主评测（Table II，22 任务 × 10 留出种子 = 220 回合）：** 仿真采用 Gemini 3.8 Flash（在线控制）+ Fable 5.1（离线修订），`xhigh` reasoning。15 轮后 RPG 达 **95.0%（209/220）**（CaP-X 组 98.6%，ABC 派生组 93.3%；第 1 轮为 28.6% [63/220]），显著优于 ASPIRE（**75.5%**, 166/220）、CaP-Agent0 + GPT-6 Astra Pro（**60.0%**, 132/220）、CaP-Agent0 + Fable 5.1（**49.1%**, 108/220）、CaP-Agent0 + Gemini 3.8 Flash（**48.6%**, 107/220）、CaP-Agent0 + Opus 5（**45.5%**, 100/220）以及 RATs（**41.8%**, 92/220）。15 轮内技能库从 15 项扩至 38 项（新增 23 项、修改 66 次）；R3→R4 仅改系统提示词使成功率从 43.2% 跃升至 73.6%（+30.5 个百分点），R4→R15 固定提示词纯靠技能库演进再提升 +21.4 个百分点。
  - **诊断输入消融（Table III，5 轮练习，220 回合）：** 完整 RPG **78.2%**（172/220）；移除 Video Analyzer 降至 **51.8%**（114/220，-26.4 pp）；移除 Privileged Agent 降至 **50.9%**（112/220，-27.3 pp）；两者均移除 **50.9%**（112/220）。
  - **纯技能库对比（Table IV，4 任务 × 10 配对种子 = 40 回合，固定 Runtime 模型与提示词）：** 修订前 **65.0%（26/40）** → 修订后 **92.5%（37/40）**（+27.5 pp）；若用特权脚本调用器测试，则从 19/40 升至 39/40。
  - **非直接反馈任务迁移（Appendix D, Table VII）：** 4 个不触发 Video Analyzer 诊断与候选生成（仅参与回归检验）的反馈留出任务中，`Two-arm handover` 从 R1 的 3/10 升至 R15 的 10/10，`Close sunglasses case` 从 0/10 升至 10/10，`Transfer object` 到 R4 即达 10/10（R15 为 9/10）；单独网格的 `Grasp plate` 在 R10–R11 达 10/10 后回落至 6/10，说明跨任务迁移存在非单调波动。
  - **实机验证（Table V，YAM 双臂实机，3 任务 × 10 次试验 = 30 次，`low` reasoning effort）：** RPG 达到 **30/30（100%）**（`Store ball in drawer` 10/10、`Fold towel` 10/10、`Transfer bowl` 10/10），而 CaP-Agent0 (GPT-6 Astra Pro) 为 14/30（4/10、9/10、1/10），CaP-Agent0 (Gemini 3.8 Flash) 为 3/30（3/10、0/10、0/10）。另展示 5 项零样本物理任务（Open Scissors、Uncap Marker、Pull out Tissue、Erase Whiteboard、Unscrew Bottle Cap）及 4 项附加定性演示。
- **核心局限：**
  1. **可变形物体操作仍是明显短板（Appendix F, Table IX）：** 仿真 `Fold towel` 在第 15 轮仅 3/10（第 8 轮峰值 5/10），收紧角点对齐容差至 1 cm 的 `Fold towel (hard)` 在全部 15 轮中始终为 **0/10**，暴露现有基于刚体几何/夹爪开度的技能难以可靠表征多层布料身份与持续接触状态。
  2. **API 限流与多任务合并冲突：** 每轮需对 22 个任务并发执行 Runtime 与 Privileged 轨迹并进行多轮全量回归门控，受限于模型 API 速率限制；技能耦合增加后，合并冲突与局部回归（如 R6、R14 总分微降 1 例）也随之上升。
