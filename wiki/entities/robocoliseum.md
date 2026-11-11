---
type: entity
tags: [agibot, benchmark, simulation, vla, evaluation, leaderboard, genie-sim, lerobot, manipulation]
status: complete
updated: 2026-11-11
related:
  - ./genie-sim-3.md
  - ./dexmal-dm05.md
  - ./robodojo.md
  - ./lerobot.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ../overview/agibot-release-category-02-sim-training-eval.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
original:
  - ../../sources/repos/genie_sim_robocoliseum.md
sources:
  - ../../sources/sites/robocoliseum.md
  - ../../sources/repos/genie_sim_robocoliseum.md
  - ../../sources/repos/genie_sim.md
summary: "RoboColiseum：智元 Genie Sim 在线仿真挑战赛，Challenge API 提交 instruction/spatial/manip/robust 四类任务，使用本地推理 Agent 的 WebSocket 隧道评测；已核对官方 skill、协议及 CoRobotPolicy 源码，官网另展示 Overall 聚合榜。"
---

# RoboColiseum（Genie Sim 仿真挑战赛）

**RoboColiseum**（[官网](https://robocoliseum.ai/)，[榜单](https://robocoliseum.ai/leaderboard)）是智元在 **[Genie Sim 3.0](./genie-sim-3.md)** 上运营的 **在线仿真挑战赛与排行榜**：参赛者下载 **LeRobot v2.1** 训练数据、在自有 GPU 上部署 VLA/操纵策略，经 **WebSocket 反向隧道** 接入托管网关，在 **G2_omnipicker** 等仿真任务上评测并上榜。**评测 job 每次只选一个 board；模型显示名来自已审核的账户资料，并非 POST 请求中的 `model_name`。**Challenge 文档、数据下载脚本与推理协议集成于 [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum)。

## 一句话定义

**四块独立能力榜 + 隧道式远程推理评测 + ModelScope 开放训练数据**，把 Genie Sim 仿真任务变成可对比、可复现的 **VLA 操纵策略公开竞技场**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略；本榜主要评测对象 |
| API | Application Programming Interface | Challenge 平台 REST + WebSocket 隧道接口 |
| WS | WebSocket | 参赛者 inference agent 与网关的长连接传输 |
| SR | Success Rate | 任务成功率；榜单以平台聚合 **score** 排序（具体协议以官方为准） |
| SFT | Supervised Fine-Tuning | 参赛者常用 LeRobot 数据对基座模型微调后提交 |

## 为什么重要

- **与 Genie Sim 训练栈闭环：** 同生态下的 **场景仿真 + Benchmark + 公开榜**，降低「只在自家 sim 里好看」的信息不对称。
- **远程推理、本地权重：** 平台不托管 checkpoint；选手自管 GPU 与模型，更贴近真实部署约束，也避免权重上传门槛。
- **四榜分拆能力维：** `instruction` / `spatial` / `manip` / `robust` **独立计分**，比仅观察单一排名更能暴露策略短板（**2026-09-15 历史快照**中，[DM0.5](./dexmal-dm05.md) 曾在四榜领先；不代表现时名次）。官网目前另提供 **Overall** 聚合展示，但 Challenge API 仍按四个 board 分别提交。
- **开放数据与 skill 文档：** ModelScope **GenieSim3.0-Dataset** 与 genie_sim 内 `challenge-*` skill 使「训→提→跑隧道→查分」路径可文档化复现。

## 核心原理

### 四块能力榜

| Board | 能力侧重（策展） |
|-------|------------------|
| `instruction` | 自然语言指令跟随 |
| `spatial` | 空间关系与布局理解 |
| `manip` | 双臂精细操作 |
| `robust` | 扰动与鲁棒性 |

各榜通过 `GET /api/challenge/leaderboard?board=<name>` 查询；**官方 Challenge job 的 `config.board` 仅接受 `instruction`、`spatial`、`manip`、`robust`。** 官网存在 **Overall** 展示榜，不能将其当成第五种可提交的 board，也不能用它替代各 board 的原始分数。

### 参赛者端到端流程

```mermaid
flowchart LR
  A[ModelScope<br/>LeRobot v2.1 数据] --> B[训练 / 微调 VLA]
  B --> C[challenge-login]
  C --> D[submit-job<br/>每日额度]
  D --> E[tunnel agent<br/>WebSocket]
  E --> F[Genie Sim 网关<br/>仿真 rollout]
  F --> G[poll-result / ranking]
```

1. **数据：** `challenge-download-datasets` → ModelScope `agibot_world/GenieSim3.0-Dataset`（instruction / manipulation / sim2real 等 suite）。
2. **提交：** `POST /api/challenge/job` 发送 `name` 与 `config.board`；显示用 `model_name` 来自审核通过的账户 profile，**不是提交字段**。每个 job 消耗一个每日提交 slot（实时额度通过 `GET /api/challenge/submission/quota` 查询），且一次只评一个 board。
3. **推理：** 提交返回数字 `JOB_ID`（仅供 HTTP 查询）、`JOB_UUID`（WebSocket dial）、`JOB_TOKEN`（job 级 Bearer 凭证）、`TUNNEL_ENDPOINT` 与 `PARALLELISM`；从推理仓运行 `./scripts/tunnel.sh <gpu> <job_uuid> <endpoint>`。每个 agent 进程对应一个 `agent_id` 和最多一个活动 session；并发上限由 `PARALLELISM` 控制。
4. **协议：** WebSocket tunnel 的二进制帧带 `session_id`；其 payload 是 msgpack JSON-RPC `infer`。G2 观测含 **head / hand_left / hand_right** 三相机 JPEG、14 维双臂、5 维腰部与 2 维夹爪状态、`prompt`。示例动作输出采用 `result` 包裹的手臂、夹爪及可选腰部 action chunk；**必须把 NumPy 数组转为原生列表再 msgpack 编码**，因仿真侧使用普通 msgpack 解包。详见官方 `challenge-inference-protocol/SKILL.md` 与 `corobotpolicy.py`。

### 与相关基准定位

| 基准 | 主要对象 | 与 RoboColiseum |
|------|----------|-----------------|
| **RoboColiseum** | Genie Sim 上 **四能力维** 仿真策略榜 | 本页 |
| [Genie Sim Benchmark](./genie-sim-3.md) | 智元五类能力仿真评测叙事 | 同仿真栈；RoboColiseum 为 **在线挑战赛产品化** |
| [RoboDojo](./robodojo.md) | sim+real 统一操纵评测 | 更重 **真机 RealEval** 与 XPolicyLab 适配；RoboColiseum 当前以 **Genie Sim 隧道仿真** 为主 |
| LIBERO / RoboTwin 等 | 固定任务套件仿真榜 | 可互补；RoboColiseum 强调 **智元 G2 本体 + 四榜分拆 + 隧道提交** |

## 源码运行时序图

下图只表示官方 [Challenge Skills](https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum)、[`CoRobotPolicy` 源码](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py) 与推理客户端协议可证实的**一次 job / 一条 agent 隧道**；未推断未公开的托管调度内部实现。

```mermaid
sequenceDiagram
    autonumber
    participant User as 参赛者
    participant API as Challenge API
    participant Gateway as Tunnel Gateway
    participant Agent as tunnel.sh 推理 Agent
    participant Model as 用户 VLA 策略
    participant Sim as Genie Sim CoRobotPolicy

    User->>API: 检查 quota 和账户 model_name 审核状态
    User->>API: POST /api/challenge/job 携带 name 与 config.board
    API-->>User: JOB_ID JOB_UUID JOB_TOKEN endpoint parallelism
    User->>Agent: 在本地 GPU 启动 tunnel.sh
    Agent->>Gateway: WebSocket dial 使用 JOB_UUID 和 job token
    Gateway-->>Agent: warmup
    Agent->>Model: 加载权重并预热
    Agent->>Gateway: ready
    Gateway-->>Agent: session_open
    loop 每次请求 / 动作 chunk
        Sim->>Sim: get_payload 采集 JPEG states prompt
        Sim->>Gateway: msgpack infer 数据
        Gateway->>Agent: 带 session_id 的二进制帧
        Agent->>Model: 解码观测并调用推理
        Model-->>Agent: 关节与夹爪 action chunk
        Agent-->>Gateway: 原生列表编码的 msgpack result
        Gateway-->>Sim: 返回对应 session 的结果
        Sim->>Sim: infer 解包 _parse_result 并缓存 action
        Sim->>Sim: act 逐步执行 action buffer
    end
    Gateway-->>Agent: session_close 然后 drain
    User->>API: GET /api/challenge/jobs 查询 JOB_ID 状态
    User->>API: GET /api/challenge/job/JOB_ID/result
    API-->>User: task scores 与 total
```

**依据与边界：** `challenge-run-agent` 定义 dial、warmup/ready、session、drain 和重连；`challenge-inference-protocol` 定义 `infer` / `result` 封包；`CoRobotPolicy.get_payload/infer/act` 确认观测封装、收包、动作缓存与回放。平台 API 的 `/result` 提供分数，**job 状态须从 `/jobs` 获取**，不能以 `/result.total == 0` 认定失败。官方没有发布独立的 pip Simulation SDK；本图不是声称实测跑通后端。若使用复现示例，`config.board=manip` 与数据下载 `suite=manipulation` 的名称需区分。

## 开源与工程实践

| 项 | 状态 / 要点 |
|----|-------------|
| **Challenge skill** | **已开源** — [genie_sim/skills/robocoliseum](https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum) |
| **训练数据** | **已开放** — ModelScope GenieSim3.0-Dataset（LeRobot v2.1） |
| **平台服务** | 托管 API + 榜单；无独立 RoboColiseum GitHub 仓 |
| **参赛者权重** | **不上传**；本地推理经隧道对接 |
| **官方 pip SDK** | **无** — 须自备 tunnel client；baseline 示例指向外部推理仓 |
| **环境状态** | `~/.simubotix-challenge.env` 持久化 token / job 元数据 |

### 公开榜快照（instruction board，2026-09-15 API）

| Rank | 组织 | 模型 | Score |
|------|------|------|-------|
| 1 | Dexmal | DM0.5 | 0.844 |
| 2 | 极佳科技 | GigaBrain-0.7 | 0.817 |
| 3 | 清华大学 | Z0-AE-Instruction | 0.812 |

> 四榜榜首与分数以 [官网榜单](https://robocoliseum.ai/leaderboard) 为准；本库 [DM0.5](./dexmal-dm05.md) 实体页链到 OpenDM 复现栈。

## 常见误区与局限

- **误区：有一个 `pip install robocoliseum`。** 官方 skill 明确 **无** 独立 Simulation SDK；协议以 genie_sim 源码与 skill 为准。
- **误区：提交 job = 上传模型文件。** 平台记录 `model_name` 与配置；**权重留在选手侧**，经隧道在线推理。
- **局限：** 榜单为 **仿真隧道评测**，与真机成功率仍有 gap；每日提交额度与并发 `PARALLELISM` 限制大规模 sweep。
- **局限：** Baseline checkpoint 命名与 board 强绑定，混用 checkpoint/board 会直接导致无效提交。

## 原文核对与复现边界

- **通读材料：** 已通读官方 Skills README、Quickstart、Benchmark task suites，以及 help / submit-job / run-agent / inference-protocol / poll-result / ranking / baseline-model 等官方说明；对 `corobotpolicy.py` 的 `get_payload`、`infer`、`_parse_result`、`act` 与数据下载脚本进行定点源码核对。详见 [来源归档中的通读记录](../../sources/repos/genie_sim_robocoliseum.md)。
- **易混名称：** `config.board=manip`，而 ModelScope 下载器的 suite 名称为 `manipulation`。训练数据按 suite 取用，不能据此推断 Challenge 允许使用 `manipulation` 作为提交 board。
- **动态信息：** 榜单、账户审批、每日 slot、服务端 task inventory 与实时评测分数可能变化，应以在线 API 和现场任务配置为准；本次没有实际提交 job、下载 checkpoint 或验证成功率。
- **项目区分：** 本页是智元 Robo**Coliseum** 在线仿真赛；请勿与 [RoboColosseum 实地物理智能评测论文](./paper-robocolosseum.md)（arXiv:2611.02497）混同。

## 关联页面

- [Genie Sim 3.0](./genie-sim-3.md) — 仿真平台与 Genie Sim Benchmark 叙事
- [Dexmal DM0.5（OpenDM）](./dexmal-dm05.md) — 2026-09 四榜领先的开源 VLA 栈
- [RoboDojo](./robodojo.md) — 另一类 sim+real 统一操纵公益榜
- [LeRobot](./lerobot.md) — 训练数据格式与 `lerobot-train` 生态
- [具身评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md)
- [agibot-go-3-preview](agibot-go-3-preview.md) — AGIBOT GO-3 Preview（manip 分榜第一的核对）

## 参考来源

- [官方 Challenge Skills README 与任务套件](https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum)
- [官方推理协议（SKILL.md）](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-inference-protocol/SKILL.md)
- [官方 CoRobotPolicy 源码](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py)

- [RoboColiseum 官网归档](../../sources/sites/robocoliseum.md)
- [Genie Sim RoboColiseum skills 归档](../../sources/repos/genie_sim_robocoliseum.md)
- [Genie Sim 3.0 源码归档](../../sources/repos/genie_sim.md)

## 推荐继续阅读

- [RoboColiseum 官网](https://robocoliseum.ai/)
- [Leaderboard](https://robocoliseum.ai/leaderboard)
- [Genie Sim — robocoliseum skills](https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum)
- [ModelScope GenieSim3.0-Dataset](https://www.modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)
