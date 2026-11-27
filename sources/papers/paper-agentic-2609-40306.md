# DynaHarness: A Dynamic Physical Harness for Self-Evolving Robot Agents

- **作者：** Haoyuan Deng、Jiebin Liu、Tengxiao Zhang、Langning Yan、Hongye Cao、Ziwei Wang
- **机构：** 南洋理工大学（Nanyang Technological University, NTU）、南京大学（Nanjing University, NJU）
- **版本 / 载体：** arXiv:2609.40306v1 [cs.RO]（2026-09-30 提交，37 页，19 图）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2609.40306)、[arXiv 摘要页](https://arxiv.org/abs/2609.40306)
- **项目页 / 代码：** [项目页](https://denghaoyuan123.github.io/Dynaharness_page/)、仓库地址 `https://github.com/Denghaoyuan123/DynaHarness`（**宣称将开源或未公开可访问仓库**：截至 2026-11-27 核查该 GitHub 仓库尚不可公开访问 / `exists: false`；论文 Reproducibility statement 说明提供了匿名补充材料包，包含实现、评测配置与脚本及运行记录；注：候选链接中出现的 `PhyAgentOS/PhyAgentOS-core` 为本文基线文献 Liu et al., 2026b 的仓库，非本文官方实现）
- **核对范围：** 全文通读 arXiv HTML v1（`paper_text.txt` 第 1–3744 行，135,125 字符），覆盖 Abstract、§1–5、Eq. 1–5、Table 1–3、Figure 1–8、References 及 Appendix A–I（含 Table 4–32、Figure 9–19）；并核对项目页文本快照 `project_page.txt`（第 1–166 行）与 `context.json` 中候选仓库核查记录。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心问题（§1, §2）

1. **双尺度时间错配：** 预训练 VLA（如 $\pi_{0.5}$）具备动作先验，但长程操作仍依赖高层语义规划与底层物理执行的协同；高层 VLM 规划以秒级粗粒度运行，而物理接触、进展与瞬态完成事件（如 *turn off the stove* 仅维持 0.55 s）发生在控制周期内。
2. **回合级失败缺乏诊断粒度：** 既有自改进智能体多基于任务成败修改程序或技能库，难以区分失败源于规划、接地（grounding）、能力执行、验证器还是恢复机制，也缺少回归准入门禁来防止局部修复破坏其他任务。

### 2. 系统架构、物理执行契约与自演化闭环（§3, Appendix A）

1. **多时钟分层与双脑分工（§3.1–§3.3）：**
   - 底层控制器运行于 **20 Hz**（真机伺服 100–1000 Hz 位于适配器后），底层安全/租约/预算检查运行于 **50 Hz**（速度包络 $2\text{ rad/s}$）；**快脑**（确定性物理治理）运行于 **2 Hz**（名义 0.5 s 周期，调度函数实测中位耗时 **0.046 ms**）；**慢脑**（`Qwen3-VL-4B-Instruct`）**按需触发**（实测中位耗时 **913 ms**）。
   - 慢脑读入原子快照 $(\ell, S, o_t, h_j)$ 输出建议 $z_j = \Pi(c_j) = (m_j, \alpha_j)$（仅命名能力 $m_j \in \mathcal{L}$ 与符号参数 $\alpha_j$，不含具体位姿，Eq. 1）。
   - 快脑经部分映射 $\gamma_k = G(z_j, c_j) \in \Gamma \cup \{\bot\}$ 解析物理参数 $\theta$、步数预算 $B$ 与租约 $\tau$（Eq. 2），对不可解析或预算不足的指令**显式拒绝** $\bot$ 并记录原因；在决策间通过布尔锁存 $v_k = v_{k-1} \vee \bigvee_{t \in \mathcal{T}_k} b_t$ 保留瞬态完成事件（Eq. 3）；决策继续、恢复、替换能力、切换 `vla_act` 或请求慢脑重规划。
2. **物理执行契约与能力库 $\mathcal{L}$（§3.4–§3.5）：**
   - 满足 **Grounded（已接地）、Bounded（有界）、Verifiable（可验证）、Attributable（可归因）** 四原则，所有观测、决策、拒绝原因与结果写入只增不减的 episode evidence store。
   - 能力库 $\mathcal{L}$ 统一封装确定性几何解析技能 $a_t = T_m(\theta, o_t)$、恢复技能（如 `keyframe_recovery`、regrasp、re-seat）及冻结 VLA 调用 `vla_act`（$a_t = \pi(o_t, \ell)$），使冻结 $\pi_{0.5}$ 受同一预算、租约与终止判定约束。
3. **有序失效归因与配对回归准入（§3.6–§3.7, Appendix A/H）：**
   - 离线诊断沿 $N=13$ 个有序层级（覆盖基础设施、上下文、规划、接地、分发、能力执行、验证与恢复）定位首个匹配故障层：$\lambda_i = A_N(E_i) = \min\{j \in \{1,\dots,N\}: d_j(E_i)=1\}$（Eq. 4），优先归因上游条件。
   - 依据归因复现故障并基于物理量（而非任务 ID 分支）提出修订 $\mathcal{L}' = \mathcal{L} \oplus \kappa$，通过配对回归门控（Eq. 5：总成功不降、harness 归因失败不增、裸策略原胜单元不退化、零污染 $x(\mathcal{L}')=0$）及全轮更广范围回归检查决定准入或回退。

### 3. 核心实验、迁移验证与消融结论（§4, Appendix B–H）

1. **LIBERO-Pro 主评测与受控基线对比（Table 1, §4.2）：**
   - 在 Goal-T、Goal-S、10-T、10-S 四单元共 800 局（开发种子 21–40）上，使用同一本地 `Qwen3-VL-4B` 上层模型与冻结公开 $\pi_{0.5}$ 策略：DynaHarness 达 **74.25%**（Goal-T 75.0% / Goal-S 81.0% / 10-T 76.0% / 10-S 65.0%），显著高于冻结 $\pi_{0.5}$ 的 **16.2%**（+58.0 pp）、PhyAgentOS 的 **20.3%**（GPT-4o-mini 为 21.3%）、EmbodiedSkills 的 **14.4%**、Harness VLA 的 **6.0%**、ENPIRE 的 **5.4%** 与 Zetta 的 **0.0%**。
2. **冻结后新状态迁移与未扰动/其他扰动基准（§4.2–§4.3, Table 2A, Table 8, Table 13–15）：**
   - 冻结最终系统（`088ef2ea`）后在 800 个新采样初始状态（Block C）上取得 **75.2%** vs 冻结 $\pi_{0.5}$ **17.5%**（+57.75 pp，473 vs 11 独胜，$p=3.13\times 10^{-124}$）；四个预注册快照（`stat23`→`stat28`→`stat39`→`Final`）开发集 60.00%→74.25%（+14.3 pp）几乎完整迁移至新状态 60.9%→75.2%（+14.4 pp，Spearman $\rho=1.00$）；在 261 个未触及官方状态（Block B，31 单元）上达 **77.0%** vs **16.5%**。
   - 标准无扰动 LIBERO（适配配置 `formal2`）达 **98.05%**（$\pi_{0.5}$ 复测 98.00%）；LIBERO-Plus（10,030 任务）达 **84.4%**（相机视角 66.5%、机械臂初始状态 76.1% 最低）。
   - 真机平台（UR7e 机械臂 + RealSense D435/D405 + Qwen 4B + SAM3，每任务 10 次随机位置测试，Table 3）：圈套柱 **90.0%**、叠杯 **80.0%**、面包入烤面包机 **70.0%**、抽屉放置 **70.0%**。
3. **机制消融与演化验证（§4.4–§4.7, Table 2B, Table 16–26, Appendix H）：**
   - **能力主导作用：** 移除 7 项几何解析接触技能后成功率由 **74.0%** 跌至 **16.6%**（-57.4 pp，接近裸 $\pi_{0.5}$）；仅移除 6 项恢复技能为 **73.1%**（$p=0.21$）；移除 VLA 调用为 **70.5%**（含 221 例基础设施/提议被拒标记）。770 局存档控制轨迹中 570 局（74%）未调用 VLA，成功率 **96.7%**，VLA 主要作为困难分支兜底。
   - **动态执行价值：** 保留同一技能库时，Full（**74.0%**）超越名义单步重规划 `A2static`（**63.9%**，+10.125 pp，$p=2.00\times 10^{-18}$）与冻结序列执行 `A2seq`（**63.8%**，+10.25 pp）；取消完成状态锁存使成功率从 73.1% 降至 69.6%（-28 净胜局，$p=6.2\times 10^{-5}$）。
   - **演化门控案例：** 针对抽屉抓手几何与重定位的两项物理量改进将 Goal 成功率由 75.25% 提至 78.0%（`goal_swap[0]` 从 55% 升至 100%）；而局部通过门控的候选方案在全轮测试中因误将灶台平面视作微波炉腔体导致 5 个单元共损失 79 局（`stat31`），经全局回归检出并回退后恢复。

### 4. 局限性与边界（Appendix I）

1. 泛化验证主要覆盖同任务单元的新初始状态，新任务族与新物理域尚需单独评测；LIBERO-Plus 为单次试验且缺少冻结策略对照。
2. 仿真中终止判定依赖 LIBERO 基准任务谓词，真机部署仍需为每个场景显式构造基于视觉分割（如 SAM3）或测量契约的完成判定与场景描述。
3. 未经演化的初始框架在种子 1–20 上仅 13.9%（低于裸 $\pi_{0.5}$ 的 17.1%），说明不成熟的快脑调度与缺失锁存会伤害基线策略。

对 wiki 的映射：[DynaHarness 论文实体](../../wiki/entities/paper-agentic-2609-40306.md)。
