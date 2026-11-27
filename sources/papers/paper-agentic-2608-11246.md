# Towards the Harness of Embodied Agents

- **作者：** Qi Wang, Tianyi Wang, Chengyang Li, Shikun Ban, Yurun Chen, Yizhong Ge, Jason Qin, Chengtai Li, Wentao Zhu
- **机构：** 宁波东方理工大学（Eastern Institute of Technology, Ningbo / HAI Lab）
- **版本 / 载体：** Technical Report (July 2026); arXiv:2608.11246v1 [cs.AI]（2026-08-03 提交）
- **原文：** https://arxiv.org/abs/2608.11246 （HTML 全文：https://arxiv.org/html/2608.11246 ；项目页：https://eit-hai.github.io/thea ）
- **项目页 / 代码：** 项目主页 https://eit-hai.github.io/thea ；官方 GitHub 仓库 https://github.com/EIT-HAI/Thea （Apache-2.0；经 Step 2.5 核查判定为**已开源**：提供完整的 Thea Harness 编排运行时 `harness/`、仿真适配接口 `simulation/`、Feishu/Lark 交互通道 `lark/`、机器人移植模板 `examples/port-template/` 与 `./install.sh` / `./run.sh` 入口；README 注明部署者需自备模型 API 凭据、底层机器人 SDK 或仿真器、具体动作策略、感知/场景图后端及后置条件证据）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1 Introduction、§2 Why the Harness Works 公式 (1)–(3)、§3.1–3.7 System Architecture 及 Listing 1–7、§4.1–4.3 Bridging the Gaps 及 Table 1–2 / Listing 8–10、§5.1–5.4 Experiments 及 Table 3 / Fig. 8–10、§6 Related Work、§7 Discussion、References、Appendix A Tool Registry、Appendix B Context Structure / Listing 11、Appendix C.1–C.4 真机演示执行日志），并核对项目页文本与官方仓库 `EIT-HAI/Thea` README 及目录样本
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Towards the Harness of Embodied Agents (Thea) 实体页](../../wiki/entities/paper-agentic-2608-11246.md)

### 1. 核心命题与物理世界的两大结构性缺口（§1）
- 编码智能体（Coding Agents）的成功确立了 **Harness（执行脚手架/外围基础设施）** 范式：智能体的能力不仅取决于基础模型本身，还取决于围绕模型构建的工具接口、上下文组织与反馈闭环。
- 本文提出 **Thea**（源于希腊语 *thea*，“sight; view”），将编码智能体的 Harness 架构迁移到物理世界，使视觉语言模型（VLM）通过统一的工具调用循环编排导航、操作、感知与人机交互能力。
- 与软件环境不同，物理世界天然缺失两项编码智能体免费获得的闭环能力：
  1. **Gap 1 — 世界可读性（Readability of the World）**：代码库是结构化、符号化、持久化的文本；物理机器人接收 30 fps 的高维非结构化 RGB-D 流，每次只能看到局部瞬时切片，必须由系统构建持久可寻址的全局表征。
  2. **Gap 2 — 结果可验证性（Verifiability of Outcomes）**：软件进程会自动退出并返回 exit code、stderr 与测试 PASS/FAIL；VLA 等连续策略既无内置终止信号，也不会自动报告抓取是否成功或失败原因。
  3. **第三项具身差异（§4.3）**：编码智能体运行在统一的 shell/文件系统上，没有“身体”；物理机器人具有不同构型、传感器视角、基座运动学与工作空间，必须通过显式的具身配置文件（Embodiment Profile）加以描述。

### 2. 闭环可靠性的数学分析（§2）
- 设 $n$ 步任务中第 $i$ 步单次执行成功率为 $p_i$：
  - 开环执行成功率（Eq. 1）：$P_{\text{open}} = \prod_{i=1}^n p_i$，随步数 $n$ 呈几何衰减。
  - 若 Harness 为每步提供最多 $k$ 次“检测-重试”机会，评估器判定正确率为 $\alpha$（错误率为 $1-\alpha$），则触发重试的概率为 $r_i = p_i(1-\alpha) + (1-p_i)\alpha$。
  - 单步在 $k$ 次内取得真实成功并正确放行的概率（Eq. 2）：$p_i^*(\alpha, k) = p_i \alpha \cdot \frac{1 - r_i^k}{1 - r_i}$；任务总成功率（Eq. 3）：$P_{\text{harness}} = \prod_{i=1}^n p_i^*(\alpha, k)$。
  - 推论：当 $\alpha = 1$ 时，$p_i^* = 1 - (1-p_i)^k$（指数压缩失败率）；当 $\alpha = 0.5$ 时，$p_i^* = p_i(1-2^{-k}) \le p_i$（随机猜测无法超过单次尝试）；$k \to \infty$ 时上限受 $\alpha$ 严格制约，表明**高准确率评估器（$\alpha \to 1$）是物理 Harness 的首要工程瓶颈**。
  - 实际运行中由于失败诊断后可调整位姿/切换策略，且每步重规划可适应环境变化，上述独立同分布假设给出的是系统增益的下界。

### 3. 系统架构与编码智能体组件的物理化改造（§3）
- **Agentic Loop（§3.1, Listing 1）**：定义 `turn`（一次模型调用 + 工具执行）、`task`（单条指令直至模型不再输出 `tool_calls`）与 `session`（跨任务延续消息历史）。遵循**每轮最多调用一个工具（one tool call per turn）**的反应式原则，避免物理动作后果不可预测导致并发冲突。
- **Context Engineering（§3.2, Table 1, Listing 2, Appendix B）**：按生命周期将上下文严格划分为三层：
  - **Resident（驻留，任务开始加载）**：System Prompt、Memory（`MEMORY.md`）、Embodiment Profile、Tool Definitions（`name`、`description` 附带工具经验摘要、`inputSchema`；对模型隐藏 `POST_CONDITION`）。
  - **Refreshed（每轮刷新，决策前替换）**：Scene Graph Brief（全局尺度符号状态）与 Observations（局部尺度相机图像与四向 LiDAR 安全净空距离）。
  - **Accumulated（随轮次累积，超长时压缩）**：Instructions、Task Notes（单任务工作记事本，任务结束即过期）、Model Responses、Tool Results。
- **Tool Protocol（§3.3, Listing 3–4）**：所有能力遵循 MCP 风格的 `{name, description, inputSchema}`，并在工具文件内绑定隐藏的 `POST_CONDITION`。所有调用统一返回 `{"success": True, ...}` 或 `{"success": False, "reason": ...}`。轻量查询作为进程内 `BuiltinTool` 执行，导航与操作策略作为独立子进程通过 `MCPTool` 隔离执行；所有调用必须穿过不可绕过的确定性 `pre-hook`（如移动前净空检查）与 `post-hook`（如操作后自动触发评估器）。
- **Skills（§3.4, Listing 5）**：以目录 + `SKILL.md`（YAML 元数据 + Markdown 指南 + 可选资源）封装领域规约（如桌面整理规则），采用三层渐进式披露（名称/描述常驻，通过 `load_skill` 按需加载正文）。强调 **Tool（受契约/安全钩子/评估约束的行动能力）与 Skill（文本建议/工作规约）的严格边界**。
- **Memory（§3.5, Listing 6–7）**：分为单任务临时记事本 `Task Notes`、跨任务通用记忆 `MEMORY.md`（写入 Resident 上下文），以及按工具索引的 `tool_experience/<tool>.md`（汇总后追加到对应工具的 `description` 中）。任务结束时通过 `update_memory` 对已完结轨迹进行置信度筛选后落盘。
- **Safety（§3.6）**：物理世界无沙盒、无撤销，安全必须前置在确定性机制中：将四向障碍物净空距离注入 Refreshed 上下文，并在 `pre-hook` 中实时重测、阻断无可行方向的导航并将 `move_base` 平移量钳位到安全距离内；配合单步单工具、失败预算上限与 `query_user` 人工确认。
- **User Interaction（§3.7）**：提供阻塞式 `query_user`（歧义消解、危险动作确认，可附带候选对象 ref 与图像视角）与非阻塞式 `notify_user`（长时动作通知、异常告警、完成汇报）。

### 4. 填补物理缺口的三大机制（§4）
- **Scene Graph as Context（§4.1, Listing 8–9）**：
  - *持久世界状态*：以对象为中心的 3D 场景图，为对象分配稳定引用（如 `cup_2`），存储中心坐标、尺寸、置信度、时间戳与图像路径；支持容器节点（开关状态与内部物品）、机器人节点（位姿与抓持物）及 `on`/`inside`/`holding`/`near` 关系边。由感知观测与**经评估器确认后的执行结果**双路门控更新。
  - *决策接口*：每轮注入精简的 Scene Graph Brief，省略的细节由模型按需调用 `get_object_relations(obj)` 与 `get_image(obj)` 按 ref 检索。
- **Evaluation as Exit Codes（§4.2, Listing 10）**：
  - *评估时机*：端到端 VLA 策略无结束信号，在每个 action-segment 边界在线返回 `process`/`success`/`failure`（辅以步数上限）；Coding-as-Policy 自终止程序退化为终态 `success`/`failure`。
  - *独立性*：由 `post-hook` 自动触发，严格采用“推理盲审（reasoning-blind）”原则，仅读取当前观测与工具绑定 `POST_CONDITION`，不读取决策模型的推理过程或自报。
  - *输出格式*：返回 `status`、`evidence` 列表与 `failure_reason`，为智能体自适应重规划及 `tool_experience` 沉淀提供物理层“stderr”。
- **Embodiment Profile（§4.3, Table 2）**：模块化描述机体差异，包含三部分：Operational Envelope（底盘占地、移动自由度、末端可达空间）、Perception Configuration（传感器模态、模型可见相机视图）与 Base-Relative Positions（相机与初始夹爪相对基座中心的固定位姿）。

### 5. 实验配置、结果与局限（§5, §7, Appendix A–C）
- **机器人平台**：主定量实验在 **Astribot S1**（轮式双臂人形，含可动头部/躯干、全向移动底盘与双夹爪，共 25 DoF；感知接入 Astribot S1 SDK 与 SysNav）上进行；跨构型探索拓展至 **AgileX Cobot Magic**（双臂移动底盘）与 **Unitree G1**（配备 6-DoF BrainCo Revo 2 灵巧手）。
- **基础模型**：决策模型采用 **GPT-5.5**（`reasoning effort = high`），独立评估器采用 **Qwen3.7-Plus**。
- **对比基线（§5.1）**：端到端策略 ACT、LingBot-VLA-V2、$\pi_{0.5}$（每项任务基于 200 条示范训练/微调，其中 ACT 与 $\pi_{0.5}$ 同时作为 Thea 操作工具的底层策略）；编排基线 CaP-X（CaP-Bench M3 设置）与基于 SayCan 的分层规划（共享同一套工具）。
- **三难度实机任务（Table 3, Fig. 8）**：
  - L1 短程桌面抓取放置（20 次独立试验/方法）、L2 长程多物体清桌装箱（20 次/方法）、L3 跨工作台导航+取放递送（15 次/方法；ACT 非语言条件故未测 L3）。
  - 论文报告 Fig. 8 柱状图趋势：Thea 在 L1–L3 均取得最高成功率，L1 差距较小，L2–L3 随任务复杂度提升，端到端基线（无法感知中途失败）与开环编排基线成功率衰减明显，Thea 凭借闭环评估与重定位/重试拉开显著差距（注：HTML 正文未附 Fig. 8 各方法精确数值表）。
- **评估器准确率评测（§5.3, Fig. 9）**：在来自 Astribot S1、AgileX Cobot Magic 与 Unitree G1 的 90 个均衡采样轨迹检查点（成功 / 执行中 / 未恢复失败各 30 个，双人复核标注）上，Qwen3.7-Plus 评估器平均准确率达 **93.3%**；错误全部为单向假阳性（**6.7%** 的失败状态与 **13.3%** 的执行中状态被误判为成功，**0%** 的成功状态被误判）。代入 §2 模型（$\alpha=0.93, p=0.8, k=5$），5 步任务成功率由开环 **0.33** 提升至 Harness 下的 **0.91**。
- **涌现能力与演示日志（§5.4, Appendix C.1–C.4）**：在饮料取送（`query_user` 询问替代品）、柜体抽屉顺序搜索充电宝（主动感知）、桌面清理（`monitor_scene` + `tidy-workspace` 技能 + `tilt_head` + `notify_user`）与 Unitree G1 零食篮跨房间递送中验证了运行时工具组合、失败后 `move_base` 微调重抓与跨构型移植能力。
- **已知局限（§7）**：场景图受限于底层感知精度与跨帧关联误差；评估器仍存在假阳性/假阴性误差风险；每轮经由云端大模型推理存在延迟，尚未达到实时控制频率；物理力限与人机共融安全仍待加强。
