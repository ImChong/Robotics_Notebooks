# VABAR: A Variational Automation Benchmark for Agentic Robotics（arXiv:2611.06881）

> 来源归档（paper）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06881>（核对时仅有 v1）
- **PDF：** <https://arxiv.org/pdf/2611.06881>（9 页，已用 pdftotext 通读全文，Fig. 4 技能库为图片，另渲染读图）
- **版本 / 日期：** v1，2026-11-04（cs.RO）
- **作者：** Shuangyu Xie*, Kaiyuan Chen, Uksang Yoo, Xinyue Zhu, Meadow Shen, Jaimyn Drake, Eugen Solowjow, Lionel Gulich, Alexander Millane, Rafael Cathomen, Tuur Stuyck, Sangeeta Subramanian, Yichao Pan, Bruce McGowan, Michael Benedict, Linxi Fan, Yuke Zhu, Spencer Huang, Ken Goldberg（* 共同一作）
- **机构：** Siemens（Solowjow）、NVIDIA（Gulich 等 10 位）、UC Berkeley（前 6 位与 Goldberg）
- **项目页：** <https://berkeleyautomation.github.io/vabar-website/>（HTTP 200）
- **代码：** <https://github.com/BerkeleyAutomation/VABAR>（项目页标注 available by Nov 9；2026-11-06 核查时不可达）
- **入库日期：** 2026-11-06
- **线索来源：** [2026-11-06-seven-robotics-intake-draft.md](2026-11-06-seven-robotics-intake-draft.md)

## 核心摘录（人工转述，数字均取自原文）

- 提出 Variational Automation（VA）：介于固定自动化与通才机器人之间，任务目标持久、实例在已知范围内变化。
- 七条基准设计准则（§III-A）：与 agentic robotics 相关、应用相关、可及清晰、受控多样性、可重复、计算效率、耐久性。
- 形式化 T = (E, R, A, O, p_O, S)，实例 τ_i ~ p_O；变化须满足任务不变性、物理可行、机器人可行、难度可控、技能保持（§III-B）。
- 执行记录：s_t, o_t→ŝ_t, a_t, (s_{t+1}, C_t), e_{t+1}，四类失败映射（§III-C，Fig. 3）。
- 六类任务：Tool Sorting、Tool Hanging、Gear Meshing、Syringe Packing、Cable Insertion、Cable Routing；每任务两档难度（Table I）；55+ 资产；Franka 与双臂 YAM；Newton 与 Isaac Lab-Arena 两套实现；扩展 BDDL。
- 33 项技能（Fig. 4）：感知 12、抓取 8、运动 10、验证 3；相对 5 月 GaP 技能库新增 25、沿用 8。
- 基线（Table II，N = 30 实例 / 条件）：GaP(人工图) Newton、GaP(人工图) ILA、Claude Code（Claude Opus 5）、GaP 自动生成图。例：Gear Meshing L1 人工图 100.0 / GaP 70.0 / Claude Code 0.0。
- 失败来源（Table III）：低变化档以运动规划为主，杂乱与可变形任务感知 / 抓取劣化。
- Fig. 5：syringe jam 回合 t01 / t29 的物理状态检视。

## 与草稿登记文件的出入

- 草稿称"Graph-as-Policy 与 Isaac Lab Arena 的基线评测"，易读成两种方法；原文中 ILA 是**第二套仿真实现**，a-ii 列为同一份人工图（GaP Human）在 ILA 上的结果。
- 草稿"项目页 / 代码待确认"：项目页可达，仓库标注 Nov 9 公开，当前不可达。
- 草稿 v1 摘要口径与全文一致，未发现数字冲突。

## 对 wiki 的映射

- [paper-vabar-variational-automation-benchmark](../../wiki/entities/paper-vabar-variational-automation-benchmark.md)
- [variational-automation](../../wiki/concepts/variational-automation.md)
- [paper-gap-graph-as-policy](../../wiki/entities/paper-gap-graph-as-policy.md)
