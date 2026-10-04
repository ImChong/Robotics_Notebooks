---
type: entity
tags: [framework, evaluation, benchmark, open-source, physical-ai, vla, llm-agent, rerun, robocurve, ros, isaac-lab, unitree, g1, gr00t]
status: complete
updated: 2026-10-04
code: https://github.com/robocurve/inspect-robots
related:
  - ./robocurve.md
  - ./xpolicylab.md
  - ./isaac-lab-arena.md
  - ./isaac-lab.md
  - ./lerobot.md
  - ./unitree-g1.md
  - ./unitree-g1-software-stack.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../methods/vla.md
sources:
  - ../../sources/repos/robocurve_inspect_robots.md
  - ../../sources/repos/robocurve_inspect_robots_unitree_g1.md
  - ../../sources/sites/robocurve-org.md
summary: "Robocurve 的开源 Physical AI 评测框架：以 Task/Scene、Policy 与 Embodiment 契约组织仿真或真机 rollout；G1 通过独立 GR00T 适配器接入，当前范围聚焦站立双臂操作，不代表全身运动评测。"
---

# Inspect Robots

**Inspect Robots**（[GitHub](https://github.com/robocurve/inspect-robots)，[文档](https://docs.inspectrobots.org/)，MIT）是 [Robocurve](./robocurve.md) 发布的 Physical AI 评测框架。它借鉴 Inspect AI 的任务/策略评测组织方式，将机器人评测中的任务定义、被测策略和机器人/仿真接口拆分，使同一任务能在不同**兼容**的 Policy × Embodiment 组合上执行，并保留可审计的 rollout 记录。

## 一句话定义

**Inspect Robots 是一个插件化机器人评测 harness：Task 描述要测什么，Policy 产生动作，Embodiment 连接机器人或仿真；框架先检查契约，再执行 rollout、评分与留档。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 根据视觉与语言目标产生机器人动作的策略类型 |
| G1 | Unitree G1 Humanoid Robot | 本条目中的适配目标为站立 G1 双臂操作 |
| GR00T | Generalist Robot 00 Technology | 通过 GR00T PolicyServer 为 G1 适配器提供策略推理 |
| DDS | Data Distribution Service | G1 适配器使用的机器人通信中间件生态 |
| EvalLog | Evaluation Log | schema 版本化的评测记录，保存配置、结果、统计和样本 |
| RRD | Rerun Recording Data | Rerun 可视化录制文件；与主要 JSON EvalLog 分工不同 |

## 为什么重要

- **把跨模型评测从临时脚本变成显式契约。** VLA 通常带有动作维度、控制模式、坐标系、旋转表示、夹爪语义、相机和状态键等假设。Inspect Robots 把一部分假设声明到 Policy/Embodiment 空间中，在物理 rollout 前暴露维度或语义冲突。
- **把真机评测中的人机因素纳入数据模型。** 真机通常没有仿真的特权成功 oracle，重置、成功判定和墙钟频率也不同；框架提供操作者评分、故障记录和本体自有生命周期等 seam。
- **区分“跑模型”和“做 benchmark”。** 框架提供 Task/Scene/Scorer 与统一执行路径，任务集合可以由独立项目提供；装好 Inspect Robots 本身并不代表取得一组统一标准的 VLA 排名。
- **G1 示例暴露全身策略接入中的责任边界。** G1 adapter 将双臂策略接到 arm-sdk，支撑与平衡留给原 locomotion controller。理解这条边界有助于设计 safer VLA 实验，也避免把上肢 manipulation 分数误报成整机全身能力。

## 核心原理

### 1. 数据模型：测量对象、策略和本体分离

| 抽象 | 主要内容 | 跨评测的重要约束 |
|------|----------|----------------|
| **Task** | 一组 Scene、评测 horizon、epoch 数与 scorer | 定义任务和初始条件，不绑定特定本体结构 |
| **Scene** | 指令、可选 Target、初始化 seed | 同一 scene 要能被配对的本体实现 |
| **Policy** | Observation → ActionChunk；声明动作与所需观测空间 | 模型专属 resize/normalization/history 仍由 policy 负责 |
| **Embodiment** | 相机/状态观测、动作执行、控制频率、reset 与能力声明 | 负责仿真或真机接口、节奏与安全停止 |
| **Controller / Approver** | 管理 chunk 缓冲等时序策略；动作进入本体前可 pass/clamp/veto | 将 VLA 推理节奏与底层控制节奏分开 |
| **Scorer / Grader** | 从 TrialRecord/Target 计算分数；grader 可记录人或模型判断 | 评测判据应观察、可复现并明确标注来源 |

策略和本体的空间契约可包括动作 shape 与语义（控制模式、旋转表示、夹爪、坐标系）、策略必需相机/状态键及名称映射、控制频率和场景可实现性。明显不兼容可在 rollout 前以 CompatibilityError 终止。预检仍只是接口层保证，不会证明策略输出有意义或硬件安全。

### 2. 一次评测的数据流

~~~mermaid
flowchart LR
  task["Task: scenes + horizon + scorer"] --> check{"Compatibility preflight"}
  policy["Policy: VLA / LLM / code"] --> check
  embodiment["Embodiment: robot / simulator"] --> check
  check -->|"pass"| controller["Controller: chunk buffering"]
  controller --> approver["Approver: pass / clamp / veto"]
  approver --> rollout["Embodiment rollout: observe, act, step"]
  rollout --> record["TrialRecord: steps + transcript + latency"]
  record --> scorer["Scorer / Grader: recorded trial"]
  scorer --> eval["EvalLog JSON: metrics + samples + status"]
  rollout -. optional .-> rerun["Rerun stream / RRD"]
  check -->|"mismatch"| error["CompatibilityError before motion"]
~~~

流程关键点：

1. **先校验再运动**：检查动作与观测协议、场景、频率等声明。
2. **闭环执行**：本体产出观察；Controller 请求策略输出动作或 action chunk，Approver 再放行动作。
3. **记录后评分**：TrialRecord 保存步骤、策略 transcript、推理延迟等；Scorer 读取记录，避免评分器依赖瞬时环境状态。
4. **留下可复核产物**：EvalLog JSON 保存规范化配置、状态、样本与指标；Rerun 负责时间序列可视化。Rerun 反压时可丢帧或 step，不能替代 JSON 评测日志。

### 3. G1 + GR00T 接入示例：明确策略覆盖边界

Robocurve 的 [Unitree G1 adapter](https://github.com/robocurve/inspect-robots-unitree-g1) 是独立于核心仓的插件。README 描述的配置注册 g1_arms 本体和 gr00t policy，将策略与本体映射到 **16 维绝对关节位置**契约；G1 处于站立姿态，臂目标通过 rt/arm_sdk 发布并逐步混入已运行的全身控制器。

~~~mermaid
flowchart LR
  scene["Task Scene: language goal + initial state"] --> policy["GR00T PolicyServer"]
  camera["G1 head camera: D435i"] --> embodiment["g1_arms adapter: observe + arm-sdk"]
  state["G1 arm / hand state"] --> embodiment
  embodiment --> policy
  policy -->|"16-D absolute joint-position chunk"| preflight{"space + scene preflight"}
  preflight --> rollout["arm targets: rt/arm_sdk"]
  rollout --> controller["Existing full-body controller: balance, legs, waist"]
  controller --> trial["TrialRecord + scorer"]
~~~

**适用范围读法：** 这里的“全身控制器”指机器人原有的平衡、腿和腰控制仍在运行；Inspect Robots G1 适配器暴露的是双臂 manipulation 评测接口。它并未因此接管步态或完整 humanoid locomotion。具体策略契约、checkpoint tag、动作频率与夹爪配置见[G1 adapter 来源归档](../../sources/repos/robocurve_inspect_robots_unitree_g1.md)。

### Inspect AI 的概念映射

| Inspect AI | Inspect Robots | 读法 |
|------------|----------------|------|
| Model | Policy + Embodiment | 机器人动作取决于策略与身体接口 |
| Task = dataset + solver + scorer | Task = scenes + controller + scorer | 场景承载初态与指令 |
| Sample | Scene | 每条样本对应一个初始条件 |
| eval() → EvalLog | eval() → EvalLog | 统一运行并产出可复核日志 |

## 工程实践

### G1 VLA 评测配置清单

| 阶段 | 建议记录或验证 | 原因 |
|------|----------------|------|
| **固定基线** | 核心框架/adapter/SDK/GR00T 版本、checkpoint 与 embodiment tag | 代码 seam 与动作解码会变 |
| **核对契约** | 16-D 维度、绝对/相对动作、手型/夹爪映射、相机名与状态键 | shape 通过不等于动作含义一致 |
| **核对时序** | checkpoint 帧率、policy chunk 长度、adapter 的 control_hz 与发布频率 | 推理节奏、动作保持时长和底层 stream 率是不同概念 |
| **做物理预检** | 摄像头视角、DDS 网卡、GR00T 服务、关节/工作空间、急停 | dry-run 不接触硬件与服务 |
| **先低速试动** | 从观测姿态 seed target；验证单关节与 Dex 手极性；确认 weight ramp | 错误的目标语义或切换会产生突发运动 |
| **定义评分** | 可观测的成功判据、重复次数、人工 verdict 规则与 abstention 处理 | 真机通常没有仿真 oracle；分母和人工参与要透明 |
| **发布结果** | 每任务 success、epoch/reducer、失败与中止、延迟、场景/种子、人工评分比例 | 单一平均分会掩盖场景难度与失败来源 |

### 核心框架常用能力

| 需求 | 入口 |
|------|------|
| 安装核心与 Rerun | uv pip install "inspect-robots[rerun]" |
| 选本体、策略、默认配置 | inspect-robots setup |
| 查看注册组件 | inspect-robots list |
| 建立 task / 执行多场景评测 | Python API Task、Scene、eval() 或注册后的 CLI task |
| 汇总复跑 | inspect-robots inspect LOG.json、view logs/、summarize LOG.json |
| G1 静态接口预检 | 安装独立 adapter 后执行 inspect-robots-unitree-g1-preflight --dry-run |

不要将示例命令当成完整 benchmark：可比较结果仍需固定任务、场景生成方式、成功判据、episode horizon、epoch reducer 和 checkpoint。

## 局限与风险

- **Alpha API 与插件分仓**：核心仓当前标记 alpha；不同 rig、policy 服务端与底层 SDK 版本须显式锁定。
- **接口兼容性不等于语义正确**：shape、枚举和观测键检查无法识别“看似同维但绝对/相对坐标含义不同”这类错误。
- **G1 适配器不是全身 VLA 控制栈**：公开 README 将 locomotion controller 负责的 legs/balance/waist 与 G1 arm adapter 分开；其评测结果应标注为站立上肢任务。
- **预检不验证真机安全**：dry-run 不连接硬件或服务；action 相对性、手指极性、通信连通性和机械空间需单独现场核验。
- **评测质量取决于任务和评分**：框架不会自动让不同任务、rig、控制时长和 operator verdict 变得可比；任务集合通常来自独立仓。
- **日志与可视化职责不同**：EvalLog 是评分与审计主记录；Rerun 面向查看，慢连接时允许降级流数据。

## 关联页面

- [Robocurve（机构）](./robocurve.md) — 独立 Physical AI 能力评测机构
- [XPolicyLab](./xpolicylab.md) — 将多种 VLA 服务接入统一策略槽位
- [Isaac Lab-Arena](./isaac-lab-arena.md) — 仿真策略评测路径
- [LeRobot](./lerobot.md) — 另一套机器人数据与评测生态
- [Unitree G1](./unitree-g1.md) — 适配目标硬件平台
- [Unitree G1 软件服务栈](./unitree-g1-software-stack.md) — SDK2、DDS 与高层接口背景
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md) — 策略评测及仿真/真机校准位置
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md) — 闭环评测的仿真扩展思路
- [VLA 方法](../methods/vla.md) — 策略的输入输出与训练/推理语境
- [Sim2Real 评测 gap](../concepts/sim-vs-real-eval-gap.md) — 如何读仿真结论对真机的外推边界

## 参考来源

- [Inspect Robots 主仓归档](../../sources/repos/robocurve_inspect_robots.md)
- [Unitree G1 adapter 归档](../../sources/repos/robocurve_inspect_robots_unitree_g1.md)
- [Robocurve 站点归档](../../sources/sites/robocurve-org.md)

## 推荐继续阅读

- [Concepts：任务、策略、本体与兼容性](https://docs.inspectrobots.org/guide/concepts/)
- [Policies and embodiments：策略与机器人接口契约](https://docs.inspectrobots.org/guide/policies-and-embodiments/)
- [Writing a benchmark：Scene、Scorer 与 horizon](https://docs.inspectrobots.org/guide/writing-a-benchmark/)
- [G1 adapter README：安装、预检与真机风险](https://github.com/robocurve/inspect-robots-unitree-g1)
- [WorldEvals benchmark 仓](https://github.com/robocurve/worldevals)
