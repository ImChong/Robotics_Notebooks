# Bridging Thought and Action: Taming Long-Horizon Instability in Open-Source LLM Agents with a MetaTool-Enhanced ROS Framework

- **作者：** Kazi Abrar Mahmud, Nilotpaul Kundu Dhurubo, Tamal Kirttonia, Sabbir Hossain Ujjal, Mohammad Ariful Haque
- **机构：** Bangladesh University of Engineering and Technology (BUET); University of Massachusetts Amherst
- **版本 / 载体：** arXiv:2609.13335v1 [cs.RO]（Submitted on 11 Sep 2026；8 pages, 6 figures；Submitted to IEEE Transactions on Cognitive and Developmental Systems (TCDS)）
- **原文：** https://arxiv.org/abs/2609.13335 （通读 HTML 全文：https://arxiv.org/html/2609.13335 ）
- **项目页 / 代码：** https://anonymous.4open.science/r/uh_bot_v1_sim-BDB1/README.md （宣称将开源或未列链接 — 论文 §IV-B 脚注 1 提供匿名仓库链接，但本次缓存抓取项目页仅返回 `"Anonymous Github"` 占位文本，`candidate_repos` 为空，未取回可核查的源码目录与可执行文件，代码可运行性未确认）
- **核对范围：** 通读 arXiv HTML 全文全部章节（Abstract、§I Introduction、§II Literature Review、§III Methodology [III-A~III-C]、§IV Evaluation [IV-A~IV-C]、§V Conclusion and Future Work、References [1]–[34]）与全部表格（Table I–VI）及图注（Fig. 1–6）；核对匿名项目页缓存 `project_page.txt`（仅返回 `"Anonymous Github"`）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Bridging Thought and Action (MetaTool-ROS-Agent)](../../wiki/entities/paper-agentic-2609-13335.md)

### 1. 研究动机与核心定位

- 基于 NASA JPL 的 ROS-Agent（ROSA）构建面向开源 LLM 的 ROS 智能体执行框架，解决开源 LLM 在多步工具调用中的长程推理不稳定问题（重复动作、递归死循环、上下文遗忘与指令漂移）。
- 在 Table I 中将自身与 ReAct、SayPlan、RoboGPT、CaP 对比，强调同时具备：显式规划缓冲区（Explicit Plan Buffer）、执行约束（Execution Constraint）、ROS 原生集成、开源 LLM 优化与真机验证。

### 2. 方法核对（§III）

- **智能体循环与工具空间（§III-A, Table II）：** 采用 ReAct 观察–推理–行动循环，系统提示词编码机器人本体与工具集。除 ROSA 原生工具（`rosnode`、`rostopic`）外，集成 8 个任务工具：`describe_image`、`RAG_query`、`face_expression`、`follow_me`、`get_goal_odometry`、`set_goal`、`publish_twist_to_move_base`、`stop_uh_v1`。
- **MetaTool 机制（§III-B）：**
  - 在任务执行前作为中间算子，根据自然语言指令 $q \in \mathcal{Q}$ 从工具集 $\mathcal{A} = \{a_1, \dots, a_m\}$ 中生成有序伪代码符号规划 $P = \langle a_{i_1}, a_{i_2}, \dots, a_{i_n} \rangle$。
  - 将 $P$ 外化写入 ROS-Agent 的 **scratchpad 记忆**并贯穿整个交互回合（从被动对话日志转化为可执行契约），每步工具输出均追加至同一 scratchpad。
  - 将工具选择策略从 $\pi(a_t \mid s_t)$ 约束为计划条件策略 $\pi_{\mathrm{meta}}(a_t \mid s_t, P)$，满足 $\pi_{\mathrm{meta}}(a_t \mid s_t, P) = 0\ \forall\, a_t \notin \mathcal{A}_P(t)$（其中 $\mathcal{A}_P(t) = \{a_{i_t}\}$ 为规划指定动作，除非触发显式重规划）。
- **硬件与部署配置（§III-C）：** 自定义移动交互机器人平台（集成感知、面部表情显示、机械臂手势与移动底盘），机载 Intel 主机 + NVIDIA GTX 1050 Ti GPU，自定义电机驱动融合 IMU 与轮式编码器里程计；感知与控制本地运行，LLM 推理通过 Groq API 调用。

### 3. 实验与评测核对（§IV）

- **任务等级（§IV-A）：**
  - `T1`（Simple Information Query）：设施规章与安全信息检索问答。
  - `T2`（Complex Contextual Navigation）：语义推理定位功能区（如电力系统设备实验室）并引导导航。
  - `T3`（Contextually Grounded Sequential Task）：长程串联任务，包括人物跟随（`follow_me`）、视觉地标记忆（如红色货架）、中间无关交互及后续回访记忆地标。
- **Gazebo 仿真消融（§IV-B, Table III & IV）：**
  - 4 个 Gazebo 场景（大办公室、医院楼层、带散落障碍物工厂、城市住宅），每任务 20 次独立试验。
  - 对比 `C1`（自适应提示词）→ `C2`（非持久提示规划）→ `C3`（持久 scratchpad 无执行约束）→ `C4`（完整 MetaTool：持久 scratchpad + 执行约束）。
  - 完整 `C4` 在 `T3` 上相较 `C1` 提升最显著：GPT-OSS-20B 从 6/20 (30%) 升至 11/20 (55%)，GPT-OSS-120B 从 8/20 (40%) 升至 14/20 (70%)，Kimi-K2 从 10/20 (50%) 升至 14/20 (70%)；GPT-4o 在 `T3` 从 `C1` 11/20 (55%) 升至 `C2/C3/C4` 15/20 (75%)，但在 `T1` 上 `C4`（18/20, 90%）略低于 `C3`（19/20, 95%）。注：论文 Table III 原表个别百分比存在排版笔误（如 `17/20 (80%)`、`18/20 (85%)`、`18/20 (95%)`），引用时按原表数值注明。
- **真机硬件效率评测（§IV-C, Table V）：** 基于 `GPT-OSS-120B` 在 `T2` 任务进行 40 次真机试验：
  - 重复工具调用率：ROS-Agent `19.0 ± 5.7%` → MetaTool `6.1 ± 2.2%`。
  - 无效/失败工具调用率：ROS-Agent `21.7 ± 6.5%` → MetaTool `8.1 ± 2.9%`。
  - 平均执行时间：ROS-Agent `89.51 ± 16.48 s` → MetaTool `84.23 ± 9.71 s`。
- **失败模式对比（§IV-C, Table VI）：** 在同平台 40 次物理试验（T1–T3，GPT-OSS-120B）上，将原作者重实现的 ReAct / CaP 与原版 ROSA、MetaTool 对比（失败类别非互斥）：
  - ReAct：错误/幻觉规划 17/40，导航失败 13/40，工具不可用/超时 10/40，总成功 17/40 (42.5%)。
  - CaP：错误/幻觉规划 13/40，导航失败 10/40，工具不可用/超时 13/40，总成功 20/40 (50.0%)。
  - ROSA：错误/幻觉规划 17/40，导航失败 10/40，工具不可用/超时 17/40，总成功 23/40 (57.5%)。
  - MetaTool (ours)：错误/幻觉规划 10/40，导航失败 7/40，工具不可用/超时 3/40，总成功 30/40 (75.0%)。

### 4. 局限与核查备注

- 仿真消融样本量有限（每任务 20 次），作者明确指出仅作为性能趋势参考而非统计学显著性验证。
- LLM 仍依赖云端 Groq API，未在 GTX 1050 Ti 机载端本地部署推理模型。
- 匿名仓库 `https://anonymous.4open.science/r/uh_bot_v1_sim-BDB1/README.md` 在当前抓取环境下未能获取实际代码文件，无法核验源码结构与可复现性。
