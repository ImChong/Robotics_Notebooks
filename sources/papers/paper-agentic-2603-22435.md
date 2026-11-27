# CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

- **作者：** Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, Guanya Shi, Jiajun Wu, Shankar Sastry, Yuke Zhu, Ken Goldberg, Linxi "Jim" Fan
- **机构：** NVIDIA; UC Berkeley; Stanford University; Carnegie Mellon University (CMU)
- **版本 / 载体：** arXiv:2603.22435v2 [cs.RO]（2026-03-23 提交 v1，2026-07-02 修订 v2）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2603.22435) / [arXiv Abs](https://arxiv.org/abs/2603.22435)
- **项目页 / 代码：** 宣称将开源或未列链接（截至 2026-11-27 核查：项目页 [capgym.github.io](https://capgym.github.io) 与论文均宣称 CaP-X 为开源开放访问平台且项目页含 `Code` 入口标识，但正文、项目页抓取文本及候选仓库列表中均未提供可核验的具体 GitHub 仓库 URL 与可运行代码入口；Appendix G 与 Appendix H.1 公开了 `S3` 原语类接口定义及 9 个自动合成技能函数片段）
- **核对范围：** arXiv:2603.22435v2 全文（Abstract、§1–§8、Table 1–9、Fig. 1–26 文字说明、Appendix A–K 全部内容）及项目页 `https://capgym.github.io` 抓取文本
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **整体架构（四组件）：**
  1. **CaP-Gym（§2）：** 基于 Gymnasium 接口的交互式代码执行环境，将底层物理仿真（或真机系统）与有状态 Python REPL 代码执行器绑定。统一封装 187 个操作任务（7 个 Robosuite 核心任务 + 130 个 LIBERO-PRO 任务 + 50 个 BEHAVIOR 移动操作任务）。底层感知原语采用无状态服务化接口（SAM3 语言/点提示分割、Molmo 2 开放词汇指点、OpenCV/Open3D 点云与定向包围盒），抓取与控制原语采用 Contact-GraspNet（`plan_grasp`）与 PyRoki 逆运动学/轨迹规划（`solve_ik`、`move_to_joints`）。
  2. **CaP-Bench（§3，Table 1）：** 在 7 个桌面单臂与双臂操作任务（Cube Lift、Cube Stack、Spill Wipe、Peg Insertion、Cube Re-stack、Two-Arm Lift、Two-Arm Handover；每层级每任务 100 次试验，Zero-Shot Pass@1）上评估 12 个主流开源与闭源模型。设计 S1–S4 单轮层级与 M1–M4 多轮层级，分别沿「原语抽象层级（高层人工宏 vs. 底层 API + 示例/无示例）」「交互轮次（单轮 vs. 多轮不重置环境）」与「视觉接地方式（纯文本执行流 vs. 原始 RGB 多模态直插 vs. 视觉差分模块 VDM）」三轴开展受控消融。
  3. **CaP-Agent0（§4，Appendix H）：** 面向机器人控制的免训练智能体框架，由三大机制组成：（a）**多轮 VDM**（默认采用 Gemini-3-Pro，首轮生成任务相关场景描述，后续每轮生成前后帧差分文本与完成度判断，避免直接插帧导致的跨模态对齐退化）；（b）**自动合成持久化技能库**（从 12 个模型在 S3 层的成功轨迹中正则抽取函数并由 Gemini-3-Pro 归纳出 9 个任务无关几何/坐标工具函数，见 Appendix H.1）；（c）**并行集成推理**（单模型 9 次变温采样或 `3M` 多模型集成：Gemini-3-Pro、Claude Opus 4.5、GPT-5.2 各 3 次不同温度采样，再由中心代码智能体综合并投票 `REGENERATE` / `FINISH`）。
  4. **CaP-RL（§5，Appendix E）：** 利用 CaP-Gym 的可验证环境奖励，采用 GRPO 对 Qwen2.5-Coder-7B-Instruct 进行在策略后训练（在特权状态层 S1 上每任务训练 50 步以避免感知噪声导致的信用分配模糊，再迁移至非特权感知层 S2 与 Franka Emika 真机）。
- **核心发现与关键数值：**
  - **人工基线 vs. 前沿模型（§3.3，Appendix K.1）：** 7 位具备 2+ 年机器人编程经验的作者在各层线下迭代调试单脚本达成 **88.5%** 单轮平均成功率（简单抓放耗时 <1 天，Peg Insertion 与 Two-Arm Handover 需 2–3 周/任务）；前沿模型单轮最高均值约 30%+，仍存在约 56 个百分点差距。
  - **抽象与多轮交互权衡（§3.3，Fig. 3–6）：** 提升原语抽象度（S4→S1）单调提升成功率，但限制表达力；多轮 `stdout`/`stderr`（M1）稳定优于单轮（S2）；直接把原始 RGB 图像插入上下文（M2）反而低于纯文本多轮（M1）；用 VDM 将视觉变化转为结构化文本（M3）在所有任务与模型上取得明显提升，且底层原语 + VDM（M4）即可超越高层单轮（S2）并与高层多轮（M3）持平。
  - **CaP-Agent0 表现（§4.1–4.3，Table 2–3，Table 5，Table 9）：** 在 7 个 CaP-Bench 任务中，有 4 个达到或超过人类专家单轮脚本，`3M` 平均成功率 **68.29%**（Cube Lift 97%、Cube Stack 98%、Spill Wipe 100%、Peg Insert 0%、Cube Restack 89%、Two Arm Lift 74%、Two Arm Handover 20%）。在 LIBERO-PRO 30 个受位置（Pos）与指令（Task）扰动任务上（Table 2），OpenVLA 与 $\pi_0$ 在两组扰动下均接近 0.00，$\pi_{0.5}$ 在 `libero-object`/`goal`/`spatial` 分别为 `0.17/0.01`、`0.38/0.00`、`0.20/0.01`，而免训练的 CaP-Agent0 达到 `0.22/0.18`、`0.26/0.17`、`0.12/0.14`。在 BEHAVIOR R1Pro 移动操作任务上（Table 3），Pick up Radio 任务成功率 56%（人类 36%、S3 24%），Pick up Soda Can 达 72%（人类 72%、S3 32%）。每轮代码生成耗时 6.8–23.8 s，M3 单次试验平均总耗时 113.6 s（Table 9）。真机已在 Franka Panda 与 AgiBot G1 上零样本验证 6 类复杂长程/推理任务（Appendix B.3）。
  - **CaP-RL 表现（Table 4，Appendix E）：** Qwen2.5-Coder-7B-Instruct 在 S2 仿真（$N=100$）上 Cube Lift 从 25% 升至 **80%**、Cube Stack 从 4% 升至 **44%**、Spill Wipe 从 30% 升至 **93%**；零样本部署到 Franka Emika 真机（$N=25$）达到 Cube Lift **84%**（基座 24%、人类 92%）与 Cube Stack **76%**（基座 12%、人类 84%），消除了“跳过抓取步骤/幻觉状态”错误并学会动态读取包围盒尺寸。
- **核心局限（Appendix A、J、H.4）：** 对需高频连续力/视觉伺服的接触密集型任务（如 Peg Insertion 在 `3M` 下仍为 0%）依然脆弱；LIBERO-PRO 中缺乏全场景碰撞感知运动规划会导致碰撞倾倒周边物体，且 SAM3 对相近语义罐体易混淆；在真机和长程任务中依赖高精度外部视觉与 VDM，推理延迟按“任务尝试级”（约 2 分钟/次）而非高频动作级计算。
- **对 wiki 的映射：** [CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation](../../wiki/entities/paper-agentic-2603-22435.md)
