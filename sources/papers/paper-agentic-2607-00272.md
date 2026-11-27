# ASPIRE: Agentic /Skills Discovery for Robotics

- **作者：** Runyu Lu\*、Yubo Wu\*、Ethan Kou\*、Haotian Lin\*、Letian Fu (Max Fu)、Wenli Xiao、Ajay Mandlekar、Yinzhen Xu、Guanya Shi、Ken Goldberg、Ang Chen、Mosharaf Chowdhury、Jan Kautz、Yuke Zhu†、Linxi "Jim" Fan†、Guanzhi Wang†（\*共同一作，†项目负责人）
- **机构：** NVIDIA（NVIDIA GEAR）、CMU、UC Berkeley、University of Michigan（UMich）、UIUC
- **版本 / 载体：** arXiv:2607.00272v2 [cs.RO]（2026-10-20 修订；v1 提交于 2026-06-30，41 页，19 图，26 表）
- **原文：** <https://arxiv.org/html/2607.00272>（[arXiv 摘要](https://arxiv.org/abs/2607.00272)，[官方项目页](https://research.nvidia.com/labs/gear/aspire/)）
- **项目页 / 代码：** 已开源（官方仓库 <https://github.com/NVlabs/ASPIRE>，Apache-2.0 协议；包含仿真 `iterative-debugging` 与 `evosearch` Agent Skills、CaP-Harness 接口规范、YAM 双臂真机 System 2 运行文档与任务示例；详见 [官方仓库归档](../repos/paper-agentic-2607-00272.md)）
- **核对范围：** 已通读 arXiv HTML 全文（§1 引言、§2.1–2.4 方法与 Algorithm 1、§3.1–3.9 实验与消融/案例研究、§4 相关工作、§5 局限、§6 结论、Appendix A–G 及 Tables 1–26），核对官方项目页全文及 `NVlabs/ASPIRE` 仓库目录抽样与 README 摘要。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [ASPIRE: Agentic /Skills Discovery for Robotics](../../wiki/entities/paper-agentic-2607-00272.md)

### 1. 研究动机与系统定位（§1）

- 传统机器人编程和早期代码即策略（Code-as-Policy）智能体面临两大瓶颈：
  1. **反馈粒度过粗：** 仅提供任务级成败或整段视频会让编码智能体难以区分失败究竟源于感知误检、运动规划碰撞、抓取不稳还是下游恢复失效。
  2. **跨任务经验清零：** 每次修好一个任务后，诊断与修复经验即被丢弃，系统解第 100 个任务时仍与第 1 个任务一样从零摸索。
- ASPIRE（*Agentic Skill Programming through Iterative Robot Exploration*）采用 **Coordinator–Actor** 架构：Coordinator 维护共享技能库与任务课程，为各任务派发独立 Actor（编码智能体）；Actor 不交换完整对话历史，而是将经闭环验证的修复蒸馏为可检索的紧凑文本/代码技能。

### 2. 三大核心模块与课程回溯（§2.1–2.4, Algorithm 1）

1. **闭环机器人执行引擎（Robot Execution Engine，§2.1）：**
   - 对每次感知、规划、抓取与控制原语调用记录完整 API、输入输出、返回状态码，以及调用前后的 RGB 关键帧、视觉叠加标注（overlays）、抓取候选、物体位姿与规划结果。
   - 通过仅暴露原语调用前后的关键证据而非完整视频，避免上下文被无关画面淹没。在 BEHAVIOR-1K `navigate-and-pick-up-radio` 案例中，Actor 通过逐原语 trace 发现感知已成功定位收音机，但 `navigate_to_pose` 因目标距离桌面边缘约 20 cm 碰撞缓冲区过近而反复返回 `PLANNING_ERROR`，进而针对性写入多角度接近（multi-angle approach）补丁并闭环验证。
2. **可扩展技能库与双阶段准入（Skill Library，§2.2）：**
   - 技能不预设固定分类，而是从验证修复中自然归纳（涵盖定位消歧、感知提示词、抓取约束、导航恢复、运动原语、场景推理、调试工作流等）。每条技能包含：失败特征（failure signature）、触发条件（when-to-apply guard）、修复策略与可选代码片段。
   - **双阶段准入检查：** Actor 提交结构化 findings（失败模式、已验证修复、可迁移技能提案）后，由全新的 **Verifier Actor** 仅凭提案在源任务调试子集上独立复现，Coordinator 校验 API 合规性（禁止越权调用非公开内部接口）后才晋升写入技能库。
3. **程序级进化搜索（Evolutionary Search，§2.3, Algorithm 1）：**
   - 针对单轨迹迭代调试易陷入局部修复循环的问题，在预算 $(K, T)$ 内每轮基于技能库和历史最优 Top-3 程序及残余失败 trace 提议 $K$ 个候选程序 $\\{P_i^k\\}_{k=1}^K$，在调试种子集 $S_{\text{dbg}}$ 上并行执行并更新历史归档 $\mathcal{H}$，达到阈值 $\theta$ 或耗尽 $T$ 轮后提取模式 $\mathcal{G}$，最终选出的最优程序 $P^\star$ **仅在 held-out 评估种子集 $S_{\text{eval}}$ 上评测一次**。
4. **课程调度与反向技能迁移（Curriculum & Backward Transfer，§2.4）：**
   - Coordinator 动态选择下一任务（填补技能库空白、测试现有技能、推进同套件任务）。全部任务结束后，对早期技能库尚小时得分较低的任务执行一次单独重跑（Backward Skill Transfer），使其受益于后续积累的完整技能库。

### 3. 主实验、零样本迁移与真机验证（§3.1–3.9, Tables 1–26）

- **实验协议与栈（§3.1–3.3）：** 主实验使用 Claude Code（Claude Opus 4.6，1M 上下文）在统一 CaP 框架中编写 Python 控制程序，评测 LIBERO-Pro（Object/Goal/Spatial 各 10 任务，学习种子 51–65、held-out 评估种子 1–50；位置扰动与指令改写扰动）、Robosuite（7 个接触丰富单/双臂任务，学习种子 101–125、held-out 100 试验种子 1–100）和 BEHAVIOR-1K（`soda-can` 与 `radio` 长时程移动操作，学习种子 26–35、增量块执行评估种子 1–25）。对比基线包括 CaP-Agent0（每评估种子现写程序+测试时重试）、OpenVLA、$\pi_0$、$\pi_{0.5}$ 与人类专家程序。
- **主评测结果（§3.4, Tables 5–7）：**
  - **LIBERO-Pro（Opus 4.6）：** 总体平均 72.0%（Pos 77.0%，Task 67.0%），对比最强基线（Object 轴宏平均领先 +77 pp、Goal +41.5 pp、Spatial +42.5 pp；CaP-Agent0 总均 18.0%，$\pi_{0.5}$ 总均 13.0%，OpenVLA 与 $\pi_0$ 为 0.0%）。
  - **Robosuite（Opus 4.6）：** 7 任务均值 81.0% vs CaP-Agent0 68.0%；双臂交接 `two_arm_handover` 从 20% 升至 92%，`cube_restack` 100%，`nut_assembly` 9%（Opus 5.5 升至 100%）。
  - **BEHAVIOR-1K（Opus 4.6）：** `soda-can` 导航 92% / 任务 88%（CaP-Agent0 84%/72%，人类 80%/72%）；`radio` 导航 100% / 任务 88%（CaP-Agent0 80%/56%，人类 88%/36%）。
- **零样本跨任务迁移（§3.5, Tables 8–9）：** 在 LIBERO-90 积累的技能库（$N \in \{0, 25, 50, 90\}$）零样本迁移到 LIBERO-Pro Long（无额外调试/重试），$N=90$ 总均 30.5%（Pos 22.6%，Task 38.3%），显著优于 CaP-Agent0（3.8%）和 $\pi_{0.5}$（5.0%），且随库规模 $N$（0→25→50→90：4.7%→13.7%→21.5%→30.5%）单调增长。
- **跨具身真机迁移（§3.6, Table 1）：** 将 Franka 仿真中习得的 3 条技能作为上下文指导迁移到不同具身和 API 的双臂 YAM 真机（GPT-5.5 via Codex）：`Put bowl on plate` 均 20/20 但总 token 8.65M→5.11M；`Lift soda can` 成功率 13/20 (65%)→19/20 (95%) 且总 token 61.94M→6.58M；`Open/push drawer` 从 0/20 (334.92M token) 提升至 11/20 (81.67M token)。
- **真机在线技能学习与 System 2 Harness（§3.7, Appendix D）：** 通过 MCP 桥接（Streamable HTTP + `observe`、`cap_localize_object`、`cap_move_to_pose` [Mink IK + cuRobo]、`cap_move_to_joints`、`cap_move_synchronized`、`cap_set_gripper`）让 GPT-6 Astra 在双臂 YAM 软夹爪上连续 3 个回合求解随机打乱魔方。第 1 回合 50 min 54 s 未解（完成 3 次面旋转），第 2、3 回合分别在 39 min（5 次转面）与 77 min 10 s（25 次转面）求解成功；单次转面耗时从 16 min 58 s 降至 7 min 48 s 再至 3 min 05 s，单次转面摊销 token 从 11.27M 降至 2.01M（费用从 $16.85 降至 $2.89）。
- **消融与模型缩放（§3.8–3.9, Tables 2–4, 10–26）：**
  - 仅执行引擎将 LIBERO-Pro 宏平均从 14% 提至 62%，叠加进化搜索达 72%（Table 10–11 详细分项：Pos 20%→62%→77%，Task 9%→61%→67%）。
  - 反向技能迁移（Table 2）：Opus 5 在 LIBERO-Pro Long 上对 5 个低分早期任务重跑后，整体从 74.6% (746/1000) 提升至 82.2% (822/1000)，其中两个抽屉任务从 0% 跃升至 52% 和 72%。
  - 12 模型对比（Table 18，LIBERO-Pro Goal Pos，空库无进化搜索）：从 GPT-6 Luna (14.8%, $0.83) 到 Claude Opus 4.6 (55.6%, $23.58)、Kimi K3 (78.4%, $58.19)、Qwen 3.8 Max (56.0%, $39.82)、GPT-6 Astra (94.4%, $35.37)、Claude Sonnet 5.5 (98.8%, $12.11, Pareto 最优)、Claude Opus 5.5 (99.2%, $18.29)。加入技能库与进化搜索后，Opus 5.5 在 LIBERO-Pro 80 任务达约 99%（Object 98.2%/100%、Goal 98.0%/99.4%、Spatial 99.4%/99.2%、Long 97.0%/98.8%），Robosuite 7 任务 100%，BEHAVIOR-1K 100%。

### 4. 局限性（§5）

1. 真机仍非完全无人值守自主学习（需人工提供初始启发文档、打乱魔方、复位环境与安全监控）。
2. 依赖冻结的强基础大模型（弱模型难以维持稳定诊断与修复）。
3. 受限于预定义机器人原语 API 表达边界。
4. 技能库缺少随规模增长的持续重验证、剪枝与任务感知检索排序机制（一次性准入后可能陈旧或误导）。
5. 调试与进化搜索循环消耗大量 LLM 调用与仿真/真机 rollout 成本。
