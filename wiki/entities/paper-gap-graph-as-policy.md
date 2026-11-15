---
type: entity
tags:
  - paper
  - manipulation
  - agentic-robotics
  - code-as-policy
  - industrial-automation
  - simulation
  - nvidia
  - berkeley
  - cmu
status: complete
original:
  - ../../sources/papers/gap_arxiv_2607_05369_full_reading_2026_11_15.md
updated: 2026-11-15
arxiv: "2607.05369"
code: https://github.com/graph-robots/graph-as-policy
project: https://graph-robots.github.io/gap/
related:
  - ../concepts/variational-automation.md
  - ./paper-vabar-variational-automation-benchmark.md
  - ../methods/aspire.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../entities/nvidia-gear-lab.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../queries/real-robot-policy-autoresearch-harness.md
  - ./paper-robo-harness.md
  - ./ken-goldberg-agentic-robotics-goosebumps.md
sources:
  - ../../sources/papers/gap_arxiv_2607_05369.md
  - ../../sources/papers/robo_harness_arxiv_2607_18060.md
  - ../../sources/papers/gap_arxiv_2607_05369_full_reading_2026_11_15.md
  - ../../sources/sites/gap-graph-robots-project.md
  - ../../sources/repos/graph_robots_graph_as_policy.md
  - ../../sources/blogs/ken_goldberg_agentic_robotics_goosebumps_2026-09-18.md
  - ../../sources/papers/vabar_arxiv_2611_06881.md
summary: "GaP（arXiv:2607.05369，NVIDIA / UC Berkeley 等）：多 agent harness 将 VA 任务编译为有向计算图（MORSL 技能 + 静态类型检查），在 Isaac 仿真中排练自学习优化图，再以 edge 解释器持久执行；8 项 VA benchmark 上显著优于 VLA/TAMP/单 agent CaP，并可 staging VLA 获 >2× 增益。"
---

# GaP（Graph-as-Policy）

**GaP**（*A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks*，NVIDIA / UC Berkeley / CMU / Bosch，arXiv:[2607.05369](https://arxiv.org/abs/2607.05369)，[项目页](https://graph-robots.github.io/gap/)）提出 **Agentic Robotics** 新变体：**策略 = 有向计算图**（类 ROS/TAMP），由多 agent 编码 harness 从 **MORSL** 技能库合成、在仿真中自学习优化，再 **无 agent 地** 在 edge 设备上持久执行——旨在 **弥合传统工程方法的可靠性与 VLA 的泛化能力之间的鸿沟**。

## 一句话定义

**用多 agent 把自然语言 VA 任务编译成类型检查过的感知–规划–控制计算图，在仿真排练里自主改图直至可靠，然后像 ROS 节点图一样在真机上长期跑——既保留模块化可解释性，又接入 GraspGen 等 model-free 原语与 VLA staging。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GaP | Graph-as-Policy | 本工作：策略表示为有向计算图 + 多 agent 自学习 harness |
| VA | Variational Automation | 有界几何/位姿变化的持久自动化任务类 |
| MORSL | Modular Open Robot Skill Library | 开放机器人技能库，首发 51 项可组合技能 |
| CaP | Code as Policy | LLM 写可执行控制代码的范式；GaP 为其结构化图变体 |
| VLA | Vision-Language-Action | 端到端视觉-语言-动作模型；VA 上可靠性不足但可被 GaP staging |
| TAMP | Task and Motion Planning | 任务+运动联合规划；GaP 图结构的另一灵感来源 |
| ROS | Robot Operating System | 基于计算图的机器人中间件 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 加州大学伯克利分校（UC Berkeley）、NVIDIA、卡内基梅隆大学（CMU）、博世（Bosch） |
| 共同一作 | Kaiyuan Chen、Shuangyu Xie（Berkeley） |
| NVIDIA 署名 | Linxi "Jim" Fan、Yuke Zhu、Guanzhi Wang 等（与 [GEAR Lab](./nvidia-gear-lab.md) 网络重叠） |
| Berkeley PI | Ken Goldberg、S. Shankar Sastry |
| 仿真 | NVIDIA Isaac / Isaac Lab |
| 硬件 | Franka（LIBERO 衍生，任务 I–III）、UR5+力反馈（VA-IV）、双臂 Franka 洗箱仿真（VA-V） |
| 开源 | [graph-as-policy](https://github.com/graph-robots/graph-as-policy)（Beta，Apache-2.0）+ [open-robot-skills](https://github.com/graph-robots/open-robot-skills)；[项目页](https://graph-robots.github.io/gap/) |

## 为什么重要

- **明确 VA 任务刻度：** 提出 **FA / VA / GR** 三分法与 **8 项开放 benchmark**，使「工业可靠 vs 学习泛化」的讨论有 **可复现靶场**（见 [变体自动化](../concepts/variational-automation.md)）。
- **结构化 agentic robotics：** 相对单 agent **自由 Python [CaP](../methods/aspire.md)**，GaP 用 **图节点 + 静态验证** 抑制幻觉与「作弊指标」；消融显示 **去图或合并 agent → 0% 成功率**。
- **桥接 model-based 与 model-free：** MORSL 同时含 **ROS 过程** 与 **GraspGen** 等学习原语；**π₀.₅ w/ GaP** 用图式 staging 把 VLA 成功率 **>2×**——不是取代 VLA，而是 **用可解释图把 VLA 送进分布**。
- **仿真排练闭环：** 并行 task instance + 节点级状态/接触反馈 → **定位失败节点** → 自主改拓扑/参数；Make Popcorn **33% → 94%（sim）/ 90%（real）** 展示 **图级 self-learning** 可落地。
- **接近手工工程：** 双臂洗箱仿真 GaP **0.953** vs 手工图 **0.987**；真机 grocery fulfillment **25/25** vs TipTop **8/25**。

## 流程总览

```mermaid
flowchart TB
  spec["VA 任务描述\n自然语言 + 物体几何模型"]
  subgraph harness [多 Agent Harness]
    orch["Orchestration Agent\n语义分段"]
    skill["Skill Agents\n各段局部子图"]
    wire["聚合布线 + 静态类型检查"]
    orch --> skill --> wire
  end
  morsl["MORSL 技能库\n51 skills · ROS + GraspGen 等"]
  g0["初始计算图 G₀"]
  subgraph learn [Self-Learning Rehearsal]
    sample["采样 task instances\n来自信念空间 B"]
    sim["NVIDIA Isaac 并行排练"]
    fb["节点前后状态 + 接触反馈\n失败定位"]
    edit["编辑拓扑/参数"]
    sample --> sim --> fb --> edit
    edit -->|"iterate"| sim
  end
  gstar["优化图 G*"]
  edge["Edge 图解释器\n无 agent 持久执行"]
  robot["仿真 / 真机工位"]

  spec --> harness
  morsl --> harness
  harness --> g0
  g0 --> learn
  learn --> gstar
  gstar --> edge --> robot
```

## 核心机制

### Graph-as-Policy

策略 $\pi(a\mid\mathbf{x},\mathcal{T})$ 表示为有向图 $\mathcal{G}=(V,E)$：

- **节点 $V$：** 原子原语（取相机帧、感知推理、运动规划/执行、脚本路由）；具 **类型化 IO 签名**；可组合为 **skill**（自然语言规约的子任务段）。
- **边 $E$：** **数据边**（生产者输出 → 消费者输入）与 **控制边**（success/failure route、回环直至任务完成）。
- **执行：** 优化后的 $G^*$ 交给 **edge interpreter** 反复执行，**不依赖** 在线 LLM 推理。

项目页展示杂货打包实例：感知 → 抓取 → 搬运回环，失败路由汇聚 abort，成功至 done——**同一 JSON 图** 在仿真与真机回放。

### 多 Agent Harness

| 角色 | 职责 |
|------|------|
| **Orchestration Agent** | 将 VA 任务切分为 skill-aware 语义段 |
| **Skill Agents** | 为各段合成局部子图（仅见本段上下文） |
| **静态验证** | 全图类型与结构检查；生成与测试分离以降低「作弊」激励 |

依赖 frontier **coding tools**（Claude、Gemini 等）。与 [ASPIRE](../methods/aspire.md) 的差异：ASPIRE 优化 **Python 程序 + 技能库复利 + 进化搜索**；GaP 优化 **计算图拓扑 + 仿真排练**，更贴近 **ROS/TAMP 工程形态**。

### MORSL（Modular Open Robot Skill Library）

首发 **51** 项技能，覆盖感知、抓取规划、运动规划、2D/3D 视觉、ROS 翻译、验证等；每项声明 **图 IO、参数、前置条件** 的 agent 可读规约。技能库 **开放演进**（论文贡献之一）。

### Self-Learning Rehearsal

1. 从 $\mathcal{B}$ 并行采样 $N$ 个 task instance
2. 在 Isaac 中执行当前图，注册 **节点前后状态**
3. 用接触与状态差分 **定位失败节点**
4. Agent 编辑子图/参数；重复直至加权目标（成功率 + 吞吐）plateau

**Make Popcorn 案例：** 初图只会开关炉钮、抓锅失败 → 10 轮排练后抓取、搬运、放置三阶段改进 → sim **94%**、real **18/20**。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant User as 用户 CLI
  participant CLI as gap/cli/run.py
  participant Conn as 仿真或真机 connector
  participant Runtime as gap.runtime.execute
  participant Exec as WorkflowExecutor
  participant Tool as tool 或 script 节点
  User->>CLI: gap run graph --sim suite/task
  CLI->>Conn: 创建 connector
  CLI->>Runtime: execute(graph, connector, skills, trace_dir)
  Runtime->>Exec: 解释工作流
  Exec->>Exec: validate_workflow 与初始化 trace
  loop 图节点与条件路由
    Exec->>Tool: resolve_inputs 后分派节点
    Tool-->>Exec: 输出或异常
    Exec->>Exec: 记录结果与错误 trace
  end
  Runtime-->>CLI: 运行结果
  CLI->>Conn: 保存视频和成功指标后 close
```

## 工程实践

固定主仓、`open-robot-skills` 和仿真子模块版本，先执行 `gap run <graph> --validate-only`，再按官方 Quickstart 运行 LIBERO。正常执行会写工作流副本、`dag_trace.json` 与节点数据；失败时先看实际节点输入和错误，再决定修改图。`--checkpoints` 的 off/warn/raise 含义不同，单次运行的 SUCCESS 也不等于统计成功率。所读 CLI 与执行器见[实现核对](../../sources/papers/gap_arxiv_2607_05369_full_reading_2026_11_15.md)，本次未启动 GPU 或硬件。

## 实验要点

### 主对比（仿真 I–II，5500+ trials）

| 方法 | 大位姿变化列（示意） | 要点 |
|------|---------------------|------|
| **GaP** | **0.93–0.99** | 图 + 自学习 + MORSL |
| π₀.₅ / MolmoAct2 | **~0.15–0.20** | LIBERO 微调 VLA；VA 列可靠性断崖 |
| TipTop（TAMP+LLM） | **~0.22–0.46** | 开放词汇规划；真机 fulfillment 8/25 |
| CaP-X（单 agent） | **~0.01–0.22** | 无图自由 Python；VA 设定下极弱 |
| **π₀.₅ w/ GaP** | Pack varied **0.67** vs 裸 **0.17** | 图式 wrist camera staging + VLA handoff |
| **MolmoAct2 w/ GaP** | mixed_all **0.66** vs 裸 **0.20** | 同上 |

### 关键消融

- **无图单 LLM 吐 Python：** 成功率 **0%**（接口/语法不匹配终止）
- **多 agent 合并为单 LLM：** **0%**（结构验证全失败）
- **结论：** **图结构** 与 **多 agent 分工** 均为必要，非仅 prompt 工程

### 真机与工业向结果

- Grocery fulfillment：**25/25**（GaP）vs **8/25**（TipTop）
- USB-C 插线（UR5 + 力）：单图处理升/降序与奇偶口位，**121/130** 次插入成功；整段序列成功另计
- 工业洗箱仿真：GaP **0.953** vs 专家手工图 **0.987**
- **周期时间：** 杂货真机平均约 **67 s/instance**，尚未达到约 **7 s/instance** 的工业目标；需减 VLM 调用、加速 IK/规划

### 后续：在 VABAR 基准中作基线（arXiv:2611.06881）

作者大量重叠的 [VABAR](./paper-vabar-variational-automation-benchmark.md) 把 VA 做成六类任务 × 两档难度的仿真基准，GaP 是主要被测系统：

- **技能库：** 为六类任务给出 33 项技能，相对「5 月版 GaP 技能库」新增 25、沿用 8（感知 / 抓取 / 运动 / 验证）。
- **人工图 vs 自动生成图：** 二者共享感知、技能、规划与执行模块，GaP 自动生成的图在 Newton 上 12 个条件 **全部低于** 人工图——差距只来自图的组合与参数化。
- **对通用编码 agent：** 同一底层模型下，GaP 在 12 个条件中 11 个高于不带机器人 harness 的 Claude Code（挂工具 L2 两者均为 10.0）。
- **失败来源：** 低变化档以运动规划失败为主；杂乱与可变形任务中感知、抓取劣化。论文自述比较的是整体系统，无法拆分 harness 与技能库各自的贡献。

## 结论

**GaP 的赌注是把策略表示换成可静态验证的计算图：LLM 只出现在编译期与排练期，运行期交给无 agent 的 edge 解释器 — 用图级可解释性换 VA 任务上的持久可靠。**

- 消融说明 **结构本身是必要条件**，不是 prompt 技巧：去掉图或把多 agent 合并成单 LLM，成功率均为 **0%**。
- 把可用性推上去的是 **仿真自学习排练**：节点前后状态 + 接触反馈定位失败节点再改拓扑/参数，Make Popcorn 从 **33% → sim 94% / real 90%**。
- 与 VLA 是互补而非替代：图式 staging 让 π₀.₅ 在 Pack varied 上 **0.17 → 0.67**、MolmoAct2 在 mixed_all 上 **0.20 → 0.66**，角色是「用可解释图把 VLA 送进分布」。
- 差距与边界仍明确：benchmark 以 **准静态 pick-and-place** 为主、力控与可变形体覆盖少；洗箱仿真 **0.953** 仍略低于专家手工图 **0.987**；周期时间尚未达 **~7 s/instance** 的工业标准；Beta 代码的 API/schema 短期可能变动。
- 选型对照：**CaP** 是自由 Python、[ASPIRE](../methods/aspire.md) 优化程序 + 技能库进化、GaP 是 ROS 式图 + 仿真排练 — 判断依据是任务是否落在 [变体自动化](../concepts/variational-automation.md) 的 VA 假设内。

## 常见误区

- **误区 1：「GaP 是又一个 VLA。」** GaP 主策略是 **计算图**；VLA 出现在 **MORSL 原语** 与 **staging 末端**，角色是 **补泛化** 而非核心表示。
- **误区 2：「有 LLM 就不够工程可靠。」** 设计目标是 **编译期/agent 期** 用 LLM，**运行期** 仅 edge 解释器执行已验证图——对齐产线 **持久运行** 需求。
- **误区 3：「和 ASPIRE/CaP 重复。」** 三者同属 agentic 谱：**CaP** 自由代码；**ASPIRE** 程序 + 技能库 + trace/进化；**GaP** **ROS 式图** + **仿真排练改图** + **VA benchmark**——选型看任务是否在 **VA 假设** 内、是否需要 **图级可解释性**。

## 局限

- **先验与消融口径：** 原文 §5.2 说明 CaP-X 与 GaP 的工位先验不同，0% 消融结果仅适用于被测配置。杂货初始图未做自学习，不能把全部杂货增益归给排练；洗箱是仿真评测，不是真机产线验证。

- Benchmark 以 **准静态 pick-and-place** 为主；除线缆插入外少用力控/可变形体
- Harness 与 MORSL 规模增长后，需更强 **组合验证与维护** 工具
- Beta 代码已发布（2026-07-01）；API / workflow schema 在 2026-08-01 前可能变动

## 关联页面

- [VLA 纵深路线](../../roadmap/depth-vla.md)

- [变体自动化（VA）](../concepts/variational-automation.md) — FA/VA/GR 任务谱与形式化
- [VABAR](./paper-vabar-variational-automation-benchmark.md) — VA 仿真基准；GaP 作基线，自动生成图系统性低于人工图
- [ASPIRE](../methods/aspire.md) — GEAR 姊妹 **code-as-policy** 路线（技能库 vs 计算图）
- [VLA](../methods/vla.md) — GR 方向基线；GaP staging 的互补角色
- [RoboHarness](./paper-robo-harness.md) — 通过成功轨迹检索与 pairwise ranking SVM 选择下一策略支持域内的交接状态，区别于 GaP 的计算图 staging；论文上层 agent 单次 API 平均 4.6 秒，不能读成底层实时控制周期。（[来源复核归档](../../sources/papers/robo_harness_arxiv_2607_18060.md)）
- [Manipulation](../tasks/manipulation.md) — 操作任务栈中的 agentic / 工业分支
- [NVIDIA GEAR Lab](./nvidia-gear-lab.md) — 署名与 agentic 研究网络
- [Ken Goldberg：Agentic Robotics 范式位移](./ken-goldberg-agentic-robotics-goosebumps.md) — ICRA plenary / X 长文对 AR 总框架与产线部署叙事
- [Simulation Evaluation Infrastructure](../concepts/simulation-evaluation-infrastructure.md) — Isaac 内环排练

## 推荐继续阅读

- Chen et al., *GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks*, arXiv:2607.05369, 2026. <https://graph-robots.github.io/gap/>
- [ASPIRE](../methods/aspire.md) — 持续学习 code-as-policy 对照
- [Real-robot policy autoresearch harness](../queries/real-robot-policy-autoresearch-harness.md) — agentic 真机开发环

## 与其他工作对比

- 正文已给出与相邻路线 / baseline 的 **定性对照**；定量表格与 ablation 见原文（[参考来源](#参考来源)）。

## 参考来源

- [RoboHarness 原文与边界复核](../../sources/papers/robo_harness_arxiv_2607_18060.md)

- [GaP 论文归档](../../sources/papers/gap_arxiv_2607_05369_full_reading_2026_11_15.md)
- [GaP 项目页归档](../../sources/sites/gap-graph-robots-project.md)
- [graph-as-policy 代码索引](../../sources/repos/graph_robots_graph_as_policy.md)
- [VABAR 论文归档](../../sources/papers/vabar_arxiv_2611_06881.md) — GaP 作基线的后续基准
