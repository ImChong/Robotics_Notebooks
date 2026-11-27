# Uni-Skill: Building Self-Evolving Skill Repository for Generalizable Robotic Manipulation

- **作者：** Senwei Xie, Yuntian Zhang, Ruiping Wang, Xilin Chen（通讯作者：Ruiping Wang）
- **机构：** 中国科学院计算技术研究所（ICT，Key Laboratory of AI Safety of CAS）、中国科学院大学（UCAS）
- **版本 / 载体：** arXiv:2603.02623v1（cs.RO，2026-03-03 提交；arXiv 备注 Accepted to ICRA2026）；arXiv HTML 全文
- **原文：** https://arxiv.org/abs/2603.02623 ；全文 HTML：https://arxiv.org/html/2603.02623
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 元信息与论文 HTML 全文均未列出独立项目页、官方 GitHub 仓库或 SkillFolder 数据集下载链接）
- **核对范围：** 全文通读 arXiv HTML（Abstract、§I Introduction、§II Related Works、§III-A Skill-Aware Planning、§III-B Automatic Skill Evolution、§IV Experiment 含 §IV-A/B/C/D、Table I–IV、Fig. 1–6 说明及 References [1]–[41]）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 传统技能中心型操作框架（如 Code-as-Policies [CaP]、Instruct2Act、RoboCodeX）依赖固定原子技能库 $L_{\text{base}}$，一旦任务超出预设技能边界即无法执行；而既有动态扩技能方案仍需部署期人工示教或航点标注。另一方面，大规模无结构机器人视频（如 DROID）虽蕴含丰富动作知识，但缺少技能级切分与层次化索引。
- **技能感知规划（§III-A）：** 给定视觉观测 $O_t$、自由形式语言指令 $I_t$ 与基础技能描述 $L_{\text{base}}$，统一模态代码管道包含三个角色：充分性判别器 $\mathcal{E}(\{O_t, I_t\}, L_{\text{base}})$、技能生成器 $\mathcal{G}(\{O_t, I_t\}, L_{\text{base}})$（不足时生成扩展技能描述 $L_{\text{ext}}$）以及规划器 $\mathcal{P}$。规划器基于自扩充库 $\{L_{\text{base}}, L_{\text{ext}}\}$ 输出策略代码 $\{\pi_i, p_i\}_{i=1}^N$（式 1）。作者从视频演示构造 106K 运行时代码样本微调 VLM，使三者共享代码输出格式。
- **自动技能标注与 SkillFolder（§III-B）：** 使用 Gemini-2.0-Flash 担任三角色（Extractor 按物理转折关键帧提炼流程步骤；Descriptor 对照 $L_{\text{base}}$ 判定并合成技能描述；Aligner 结合 Number-It 整数帧标记对齐时间区间），从 350 小时 DROID 视频中自动切分出超过 10,000 条技能轨迹片段。参照 ImageNet/WordNet 思路，基于 VerbNet 构建四层技能树 SkillFolder：$\mathcal{N}_1$（106 个 VerbNet 动词类）→ $\mathcal{N}_2$（具体动词实例）→ $\mathcal{N}_3$（1,659 个以物体交互为中心的技能描述模板）→ $\mathcal{N}_4$（视觉落地技能片段叶节点），遍历失配时自动树扩展。
- **部署期少样本技能实现（§III-B）：** 通过 VerbNet 解析与层级匹配检索候选叶片段，以 CLIP 视觉相似度筛出视角与空间布局接近的参考片段，并过滤越界与长静默轨迹。将 6-DoF 轨迹投影为初始帧绿色 2D 航点参考 $\{\tau_e, \mathcal{O}_e\}$，辅以 VLM 提取的接触/航点语义约束 $\{\phi_c, \phi_w\}$，在网格化目标观测 $\mathcal{O}_i$（类似 MOKA）上由 GPT-4o 选出 2D 接触点与可变长航点序列并结合深度反投影至 3D（式 2）；再通过相邻航点构建局部坐标系 $R_{\text{local}}$，将源示范旋转矩阵迁移为技能局部旋转模式 $R_{\text{skill}}$ 并映射回目标全局坐标系 $R_{\text{tgt}}$（式 3），输出可执行 6-DoF 位姿序列。
- **实验与消融（§IV, Table I–IV）：** 所有基线（CaP-GPT-3.5 / CaP-GPT-4o / MOKA-GPT-4o）共享同一基础技能库（AnyGrasp + 自上而下放置）。
  - **RLBench 仿真（每任务 25 episode × 3 次重复）：** 8 个基础技能覆盖任务（Table I）上，Uni-Skill 零样本平均成功率 `0.42 ± 0.01`，高于 CaP（GPT-4o ICL）`0.28 ± 0.02` 与 CaP（GPT-3.5 ICL）`0.18 ± 0.01`。10 个基础技能外任务（Table II，含 Revolute/Prismatic/Flip/Unplug/Lift/Pour 六类）上，Uni-Skill 平均成功率 `0.41 ± 0.01`，高于 MOKA `0.10 ± 0.02`（+31.0 个百分点）与 CaP `0.01 ± 0.01`。
  - **Franka Emika 真机（Table III，8 类任务 × 10 次试验）：** Uni-Skill 平均成功率 `0.73`，高于 MOKA `0.39`（单任务含 Fold Cloth `0.70` vs `0.20`、Close Door `0.80` vs `0.10`、Stir Blocks `0.60` vs `0.00`）；论文摘要/引言概括为长时域任务提升 20.0%、未见技能提升 34.0%。注：Table III 中 CaP 的 Average 栏原文印作 `0.00`，但该列 8 项单任务数字分别为 `0.80 / 0.20 / 0.80 / 0.00 / 0.50 / 0.00 / 0.00 / 0.00`（算术均值约 `0.29`）。
  - **数据源对比与质量核查（Table IV, §IV-D）：** 相比 DROID（106 VerbNet 类、1,659 技能、成功率 `0.41 ± 0.01`），将管道适配到人手第一视角视频 `sth2sth`（搭配轻量手部检测）可得 122 类、1,432 技能，但因缺少动作标注与运动模糊，成功率降至 `0.29 ± 0.02`。闭环纠错将失败案例作为负向上下文示例，使 `Close Drawer` 与 `Seat Down` 成功率额外提升 12%。抽检 135 个 SkillFolder 片段：2 个语义失配、7 个遮挡或过短、0 个执行失败，经长度与边界过滤后低质样本低于 2%。
- **对 wiki 的映射：** [Uni-Skill 实体页](../../wiki/entities/paper-agentic-2603-02623.md)
