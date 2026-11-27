# Agentic AI for Robot Control: Flexible but still Fragile

- **作者：** Oscar Lima, Marc Vinci, Martin Günther, Marian Renz, Alexander Sung, Sebastian Stock, Johannes Brust, Lennart Niecksch, Zongyao Yi, Felix Igelbrink, Benjamin Kisliuk, Martin Atzmueller, Joachim Hertzberg
- **机构：** German Research Center for Artificial Intelligence (DFKI, Cooperative and Autonomous Systems, Osnabrück); Osnabrück University (Institute of Computer Science)
- **版本 / 载体：** arXiv:2602.13081v1 [cs.RO]（Submitted on 13 Feb 2026，arXiv HTML 全文）
- **原文：** <https://arxiv.org/html/2602.13081>（摘要页：<https://arxiv.org/abs/2602.13081>）
- **项目页 / 代码：** 未确认开源代码（论文声明项目页为 <https://dfki-ni.github.io/AGENTS-MAKE-2026> 与提示词页 <https://dfki-ni.github.io/AGENTS-MAKE-2026/prompts>，但本批次抓查 `project_page_status=0` 未取得可解析项目页正文；候选 GitHub 链接 `https://github.com/DFKI-NI/symbolic` 系论文脚注 `https://github.com/DFKI-NI/symbolic_fact_generation` 的截断地址，且在 `candidate_repos` 中 `exists=false`，未确认本研究独立的可运行控制源码仓；论文另有提到用于部分复现的 Gazebo 仿真包 <https://pypi.org/project/mobipick-labs-docker-gui>）
- **核对范围：** `paper_text.txt` 全文（Abstract、Sec. 1–7、Table 1、Fig. 1–2 说明及 References，共 369 行 / 52,970 字符），完整核对多智能体编排与工具契约（Sec. 3 & 5）、八段式系统提示词结构（Sec. 4）、Mobipick 室内真机/仿真实验 1–4 与 Valdemar 户外农业仿真实验 5–7（Sec. 6）及失败模式与系统局限（Sec. 6–7）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Agentic AI for Robot Control: Flexible but still Fragile](../../wiki/entities/paper-agentic-2602-13081.md)

### 1. 研究问题与定位（Sec. 1–2）
- 核心矛盾在于把抽象规划连接至部分可观测、存在传感器噪声、状态陈旧及外生突发事件的物理机器人控制。
- 与直接从语言和原始感知映射到低层控制的端到端方法不同，该工作故意在高层技能与可验证状态谓词界面上运行，将几何定位、建图、抓握合成与运动规划留给底层经典机器人软件栈。

### 2. 多智能体架构与模型配置（Sec. 3, Table 1, Fig. 2）
- 由语音识别模块输出文本指令触发工作流，使用 OpenAI Agent Builder 配置并导出为 Python 代码，再绑定到机器人动作 API：
  - **Router**（`gpt-5-mini`，reasoning effort `minimal`，输出 `bool`，无工具）：将用户话语分类为对话型或可执行任务型。
  - **Chatbot**（`gpt-5-mini`，reasoning effort `minimal`，输出 `string`，工具：`get today date`, `get available locations`, `get semantic environment representation snapshot`, `speak`）：处理问答与闲聊。
  - **Planner and executor**（`o3`，reasoning effort `low`，输出 `string`，工具：`speak`, `act`, `reflect`, `get semantic environment representation snapshot`, `check for events`）：主控规划与执行智能体。
  - **Goal completion critic**（`o4-mini`，reasoning effort `low`，输出 `string`，无工具）：审查执行日志与原始目标，防止出现“只说不做”（verbal-only）失败模式；当任务物理不可行、多次受阻或需要人工确认时允许终止而非强制盲目重试。

### 3. 提示词工程与工具接口（Sec. 4–5）
- **提示词八部分结构（Sec. 4）：** （1）自然语言领域模型（形态、载荷/可达约束、可观与不可观谓词）；（2）语义环境状态快照示例；（3）操作指令与平台安全不变量（如 Mobipick 导航前收臂、放置前查空位；Valdemar 禁在棚内行驶、对接前到入口位姿）；（4）物体与动作 affordance 及启发式规则（减少动作数、高不确定性下先感知）；（5）动作目录与少样本规划-执行示例（先 `reflect` 再取状态、单步 `act`、检查事件）；（6）过往失败模式短记忆（减少重复调用事件检查或基于陈旧状态盲目放置等错误）。
- **工具契约（Sec. 5）：**
  - `reflect`：仅输出结构化自省用于调试与规划，不改环境。
  - `act(action, params)`：每次执行单个高层动作，`action` 取自 `{navigate, move_arm, perceive, pick, place, insert, listen, speak}`，`params` 为按位置解析的字符串列表（`speak` 在实际实现中作为 `act` 的一种模式而非独立工具）。
  - `get semantic environment representation snapshot`：读取传感器驱动的符号状态快照（如 Mobipick 的 `at(robot, location)`、`on(object, table_1)`、`in(object, box)`、`arm_posture(robot, observe)`；Valdemar 的最近已知地标/POI 及离散化电池状态 `okay` / `low` / `critical`）。
  - `check_events`：轮询自上次检查以来的离散文本外生事件（如急停触发、用户补充偏好、电量阈值变化）。

### 4. 实验验证与观察（Sec. 6）
- 原文明确说明 7 项实验为**定性概念验证（proof-of-concept demonstrations）**，而非穷尽定量评测，未汇报跨试验成功率或延迟统计表。
- **Mobipick 室内移动操作（仿真 + 真机，3 张桌子，螺丝刀/继电器/万用表/电钻/蓝盒子 + 未在提示词点名的漂白剂瓶）：**
  - 实验 1（常规任务：把万用表放进盒子并带到 table 2）：整体可完成，但暴露偶尔“盲放”（未先验证桌面空位）及路径冗余（先找齐两个物体再折返）。
  - 实验 2（歧义指令与事件中断：“pick any tool I can use to screw with”）：因先验与提示词 affordance 冲突，选中较难抓的小螺丝刀导致接近碰撞，用户按急停；通过 `check_events` 挂起等待，随后接纳用户改为电钻的口头偏好并成功抓取。
  - 实验 3（部分感知下的约束规划：把 table 3 含万用表的盒子搬到 table 2）：识别到“持有含物盒子时不可执行感知动作（会倒翻盒子）”，先去 table 2 感知空位再回 table 3 搬运。
  - 实验 4（长程压力测试：“transport all objects to table 2”）：复用蓝盒子多物同运，顺带收集了提示词未点名的漂白剂瓶；抓瓶时线缆拉紧触发急停后成功恢复，并在最后返回 table 1 复核；暴露放置前未强制刷新 table 2 局部陈旧状态的风险。
- **Valdemar 户外农业巡检（约 1 天开发工作量迁移并在仿真测试）：**
  - 实验 5（常规巡检 `poi_1`–`poi_5`）：核查对接状态与电量后出棚逐点扫描并返回对接，未出现问题。
  - 实验 6（低电量监测）：电量跌入 `low` 后中断路线返棚充电，恢复为 `okay` 后继续未完成航点；但未指定目标电量时在 80%、90%、100% 间呈非确定性波动，且**轮询机制无法抢占正在进行的长时间导航动作**。
  - 实验 7（不可能指令拒绝：“Fly to Paris”、“do a somersault”）：API 边界阻止非法动作执行并礼貌拒绝，但拒绝后偶尔出现“为取悦用户而提议同样不可执行的替代动作”的幻觉现象。

### 5. 核心结论与局限（Sec. 6–7）
- **灵活性：** 跨平台迁移主要只需重写提示词（领域模型、affordance 与动作目录）并重绑技能 API，支持开放词表、多语种同义表达及语音自省解释。
- **脆弱性：** 事件轮询非异步抢占、工具调用策略不稳定（如频繁连续调用 `check_events`）、安全约束需跨多个提示词区块冗余重复、远端模型推理带来延迟/费用/断网依赖，且长程下仍会出现非确定性与状态陈旧问题。
