# ROSClaw: An OpenClaw ROS 2 Framework for Agentic Robot Control and Interaction

- **作者：** Irvin Steve Cardenas, Marcus Anthony Arnett, Natalie Catherine Yeo, Lucky Shah（arXiv 摘要页拼写为 Lucky Sah）, Jong-Hoon Kim
- **机构：** Advanced Telerobotics Research Lab, Kent State University; OpenDive Technologies; PlaiPin Inc
- **版本 / 载体：** arXiv:2603.26997v1 [cs.RO]（2026-03-27 提交）
- **原文：** https://arxiv.org/abs/2603.26997（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2603.26997）
- **项目页 / 代码：** https://anonymous.4open.science/r/ROSCLAW-IROS26-4B2C/（**宣称将开源或未列链接 / 仅匿名审稿代码包**：论文提供匿名审稿仓库链接，正式去匿名化 GitHub 仓库标注为 `https://github.com/[anonymized]` 待录用后发布；辅助元数据中的 `openclaw/openclaw` 与 `mgonzs13/llama_ros` 分别为底层代理运行时与相关对比基线，并非本论文官方仓库）
- **核对范围：** 通读 arXiv:2603.26997v1 全文（Abstract、Sections I–VII、Tables I–VII、Figures 1–4 题注与 References [1]–[27]）；项目页 `https://anonymous.4open.science/r/ROSCLAW-IROS26-4B2C/` 返回 HTTP 200 但渲染快照仅含占位标题 `Anonymous Github`，未获得本地可核对源码文件。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[ROSClaw 实体页](../../wiki/entities/paper-agentic-2603-26997.md)

1. **核心问题与系统定位（Sections I–II）**
   - 现有 LLM–ROS 集成常将模型提示、感知处理、动作执行与安全逻辑紧耦合在单一模型和单一机器人平台上，导致无法在受控界面下跨模型横向复现或比较行为差异。
   - ROSClaw 将 OpenClaw 开源自主代理运行时（[1]）与 ROS 2（[26]）连接起来，定位为**模型无关（model-agnostic）的执行层（executive layer）**，既是面向多形态机器人的实用代理基础设施，也是用于具身 AI 跨模型行为对比的可复现测量仪器。
   - Table I 将 ROSClaw 与 SayCan、Code as Policies（CaP）、ROSA、ROS-LLM、llama_ros 在模型无关性、记忆、安全钩子、能力发现、多模态、传输模式数与跨模型研究维度上作了特性对照。

2. **执行层契约与三项不变量（Section III-A, Table II）**
   - 契约形式化为四元组 $\mathcal{C} = \langle \mathcal{A}, \mathcal{O}, \mathcal{V}, \mathcal{L} \rangle$：类型化工具模式注册表 $\mathcal{A}$（可供性清单）、观测归一化器 $\mathcal{O}$、执行前校验器 $\mathcal{V}$ 与结构化审计日志器 $\mathcal{L}$。
   - 每步流程：$o_t = \mathcal{O}(\text{ROS state})$，智能体提出工具调用 $u_t$；校验器基于安全策略 $\mathcal{P}$ 与能力上下文 $\mathcal{M}$ 返回 $(d_t, r_t) = \mathcal{V}(u_t; \mathcal{P}, \mathcal{M})$，$d_t \in \{\texttt{ALLOW}, \texttt{BLOCK}\}$，$r_t$ 为结构化拒绝理由；仅 `ALLOW` 经传输层下发，`BLOCK` 将 $r_t$ 返回给模型触发重规划。每条决策以 $\ell_t = (t, o_t, u_t, d_t, r_t, y_t)$ 追加写入审计日志（被拦截动作记 $y_t = \varnothing$）。
   - 三项设计不变量：**I1（执行边界有界致动）**，对直接速度指令保证 $\|\mathbf{v}\| \le v_{\max} \wedge |\omega_z| \le \omega_{\max}$；**I2（接口不变性）**，固定机器人状态下所有后端收到同一 $(\mathcal{A}, \mathcal{O}, \mathcal{P})$；**I3（可审计性）**，在执行门控前记录 $\ell_t$，保留“企图但被拦截”的动作证据。作者明确声明这些是执行边界不变量而非形式化安全证明（避障与工作空间约束需额外层）。

3. **平台架构、8 个 ROS 2 工具与多模态归一化（Sections III-B–G, Table III）**
   - **OpenClaw 集成**：以 skill 插件形式挂载到 OpenClaw，统一转换不同供应商的原生 tool-calling 格式；每次试验开启全新会话以防跨试次记忆泄漏；滑窗保留系统提示词与最近 $k$ 轮（Claude/Gemini/Llama 4 为 1M 上下文，GPT-5.2 为 400k；实验任务 6–15 轮内完成，未触发截断）。
   - **动态能力发现与统一可供性注入**：ROS 2 发现节点周期性内省计算图并生成包含话题/服务/动作及消息类型的清单（如 TurtleBot3 的 `/cmd_vel`、`/odom`、`/scan`、相机话题及 $v_{\max}=1.0\text{ m/s}$、$\omega_{\max}=1.5\text{ rad/s}$）注入极简系统提示词（4 条行为规则）；在 8 个 L1 任务上使用 2 个改写提示词变体验证，完成率差异 $< 3\text{ pp}$。
   - **三种传输模式**：Mode A 本地 DDS（$<1\text{ ms}$）、Mode B rosbridge WebSocket（$5\text{--}10\text{ ms}$）、Mode C WebRTC P2P（$20\text{--}100\text{ ms}$）。相比 $1\text{--}3\text{ s}$ 的 LLM 推理耗时，传输延迟可忽略。
   - **8 个标准工具（Table III）**：`ros2_publish`、`ros2_subscribe`、`ros2_service`、`ros2_action`、`ros2_param_get`、`ros2_param_set`、`ros2_list_topics`、`ros2_camera`（返回 base64 帧）。
   - **安全包络与白名单**：`before_tool_call` 钩子实施速度限幅与接口白名单 $\mathcal{P}$（TurtleBot3 实验白名单仅开放 `/cmd_vel`、`Nav2 /navigate_to_pose` 与只读传感，禁用 `ros2_service` 与 `ros2_param_set`），另配独立 `/estop` 与可选 LiDAR 近障保护（$d_{\min}=0.35\text{ m}$）。
   - **双观测归一化模式**：(i) 原生多模态（直接向支持视觉的模型提供原始帧）；(ii) 桥接接地（使用 GPT-5.2-mini 以约 $0.4\text{ s/frame}$ 将图像转为固定 JSON 场景描述供文本模型或统一感知对照使用）。

4. **实验设计与核心数据（Sections IV–V, Tables IV–VII）**
   - **平台与模型**：TurtleBot3 Waffle Pi（轮式）、Unitree Go2 Pro（四足）、Unitree G1（人形）；4 个后端模型为 Claude Opus 4.6（`claude-opus-4-6`）、GPT-5.2（`gpt-5.2`）、Gemini 3.1 Pro（`gemini-3.1-pro-preview`）、Llama 4 Maverick（`Llama-4-Maverick-17B-128E-Instruct`，vLLM 自托管），采样参数 $\text{temperature}=0.7$、$\text{top-}p=0.95$。结构化任务跨模型类内相关系数 $\text{ICC}(2,1)=0.74$（95% CI $[0.58, 0.85]$），模型间方差约为模型内方差的 $3\times$。
   - **40 项任务套件**：20 项结构化任务（L1 直接指令 8 项、L2 上下文感知 7 项、L3 多步规划 5 项）、10 项开放行为任务（双盲评分者一致性 $\kappa=0.81$，400 对评分仅 7 处分歧）、10 项安全对抗任务。主实验在 TurtleBot3 上每条件重复 $N=10$ 次。
   - **移植摩擦与延迟（Sec. V-A）**：切换 LLM 后端改 1 个配置键（$<2\text{ min}$）；切换机器人无需修改 ROSClaw 源码，更新平台配置与白名单耗时 $10\text{--}15\text{ min}$；切换传输模式改 1 个标志（$<1\text{ min}$）。单轮平均 LLM 延迟 $1.3\text{--}2.3\text{ s}$（P95 $\le 3.8\text{ s}$），适合任务级深思规划而非 $<100\text{ ms}$ 反应式控制。
   - **与 ROSA 对等消融（Table IV，TurtleBot3 + GPT-5.2，20 结构化 + 10 安全任务，$N=10$）**：
     - ROSClaw（限幅可见）：完成率 **84.0%**，越界提议率 AR **9.0%**，每提示平均拦截次数 BP **0.18**，超速严重度中位数 SV **1.22**。
     - ROSClaw（限幅隐藏）：完成率 **82.5%**，AR **12.0%**，BP **0.27**，SV **1.25**。
     - ROSA（限幅可见）：完成率 **75.5%**，AR **14.0%**，BP **0.34**，SV **1.29**。
     - ROSA（限幅隐藏）：完成率 **73.0%**，AR **22.0%**，BP **0.52**，SV **1.34**。
   - **跨模型任务完成率（Table V，TurtleBot3 结构化任务，$N=10$）**：
     - Claude Opus 4.6：L1 $97.5\pm 5$、L2 $84.3\pm 12$、L3 $74.0\pm 15$、总计 **86.5%**。
     - GPT-5.2：L1 $93.8\pm 7$、L2 $81.4\pm 13$、L3 $66.0\pm 18$、总计 **82.3%**。
     - Gemini 3.1 Pro：L1 $91.3\pm 8$、L2 $77.1\pm 15$、L3 $62.0\pm 18$、总计 **79.0%**。
     - Llama 4 Maverick：L1 $83.8\pm 11$、L2 $62.9\pm 17$、L3 $46.0\pm 21$、总计 **66.8%**。
     - 统计检验：混合效应逻辑回归模型效应 $p<0.001$，Kruskal-Wallis $H(3)=14.7, p=0.002$。L1 里程计误差仅 $0.06\text{--}0.19\text{ m}$ 与 $4\text{--}12^\circ$。三大失败模式：参数格式错误（Llama 4 占 38%，Claude 占 8%）、动作类型选错（26%，集中于 Go2/G1）、拦截后重复重发死循环（总计 19%，Claude 4%，Llama 4 31%）。
   - **开放行为任务（Sec. V-D）**：“Shake and bake” 指令下 4 模型物理轨迹迥异；开放任务通过率（$\ge 2$ 分）为 Claude **72%**、GPT-5.2 **65%**、Gemini **58%**、Llama 4 **38%**。桥接 VLM 管线将视觉任务跨模型最佳-最差通过率差距从 34 pp 收窄至 18 pp。
   - **跨形态迁移（Table VI & Sec. V-E）**：TurtleBot3（20 项）/ Go2（14 项）/ G1（10 项）完成率依次为 Claude 86.5% / 83.0% / 76.5%、GPT-5.2 82.3% / 79.5% / 73.0%、Gemini 79.0% / 75.5% / 68.0%、Llama 4 66.8% / 61.5% / 54.0%；相对排序完全保留。G1 上 Llama 4 在 41% 失败试次中误用 `ros2_publish` 代替动作服务器，而 Claude 在 94% 适用任务中正确使用 `ros2_action`。
   - **安全越界分析（Table VII，TurtleBot3 100 提示/模型）**：Claude AR **14.0%**（95% CI $[8.4, 22.2]$，BP 0.32，SV 1.28）、GPT-5.2 AR **9.0%**（$[4.7, 16.4]$，BP 0.18，SV 1.22）、Gemini AR **31.0%**（$[22.5, 40.9]$，BP 0.78，SV 1.44）、Llama 4 AR **43.0%**（$[33.5, 53.0]$，BP 1.21，SV 1.57）；全部越界动作均被 100% 事前拦截（0 次执行）。Llama 4 与 GPT-5.2 相差 $4.8\times$，仅看三大商业前沿模型也存在 $3.4\times$ 差异（9%–31%，$p=4\times 10^{-4}$）。Go2（$N=5$）上 AR 为 16%/12%/34%/48%（$4.0\times$），G1（$N=5$）上为 20%/14%/38%/54%（$3.9\times$）。

5. **局限与后续工作（Section VI）**
   - 单步推理延迟（$1\text{--}3\text{ s}$）决定其仅适用于任务级规划，必须与底层高频控制器（如 Nav2、力控、步态控制）组合使用；当前无自动重试熔断器（replan loop-breaker）；默认校验器主要验证速度界与接口白名单，不含几何碰撞检查或形式化验证；$N=10$ 对细微组间差异（如 Claude 14% vs GPT-5.2 9%, $p=0.29$）统计检验力有限。
