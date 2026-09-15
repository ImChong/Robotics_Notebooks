# RoboColiseum 官网与 Simulation Challenge 平台

> 来源归档（ingest）

- **标题：** RoboColiseum — Genie Sim 仿真挑战赛与在线榜单
- **类型：** site（官网 + Leaderboard + Challenge API）
- **URL：** <https://robocoliseum.ai/>
- **榜单：** <https://robocoliseum.ai/leaderboard>
- **机构：** 智元机器人（AgiBot）
- **代码集成：** <https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum>（Genie Sim 仓内 Challenge skill 集）
- **训练数据：** ModelScope [`agibot_world/GenieSim3.0-Dataset`](https://www.modelscope.cn/datasets/agibot_world/GenieSim3.0-Dataset)（LeRobot v2.1）
- **入库日期：** 2026-09-15
- **一句话说明：** 智元面向 **Genie Sim 3.0** 的 **在线仿真挑战赛平台**：四块能力榜（instruction / spatial / manip / robust）、WebSocket 隧道远程推理评测、ModelScope 训练数据与 checkpoint；参赛者经 API 提交 job 并在自有 GPU 上跑 inference agent。

## 开源核查（步骤 2.5，2026-09-15）

| 资源 | 状态 | 说明 |
|------|------|------|
| Challenge skill / 协议文档 | **已开源** | 集成于 [AgibotTech/genie_sim](https://github.com/AgibotTech/genie_sim) `source/geniesim_benchmark/skills/robocoliseum/`（`challenge-*` SKILL.md + 下载脚本） |
| 仿真后端 | **已开源** | 依托 Genie Sim 3.0 仿真栈（同仓） |
| 训练数据 | **已开放** | ModelScope `GenieSim3.0-Dataset`：instruction / manipulation / sim2real 等 **LeRobot v2.1** task suite |
| Baseline checkpoint | **已开放** | ModelScope `agibot_world/GenieSim3.0-Dataset` 下 `checkpoints/*`（如 `instruction_and_robust_pi05` 等）；示例推理仓为外部 ACoT-VLA fork |
| 平台 API / 榜单 | **在线服务** | `https://robocoliseum.ai/api/challenge/*`；无独立 RoboColiseum 代码仓，评测经托管网关调度 |
| 参赛者模型权重 | **不上传平台** | 提交 job 时指定 `model_name`；选手在本地/自有 GPU 经隧道提供推理，平台不托管 checkpoint 文件 |

## 四块能力榜（board）

| Board | 测什么（策展） |
|-------|----------------|
| `instruction` | 语言指令跟随 |
| `spatial` | 空间理解与布局推理 |
| `manip` | 精细操作执行 |
| `robust` | 扰动 / 鲁棒性 |

各榜 **独立计分与排名**；无跨榜「总分」。公开 API：`GET /api/challenge/leaderboard?board=<board>&page=1&per-page=20`。

## 参赛者管线（skill 路由）

```
下载 LeRobot 训练数据 → 训练/微调模型 → challenge-login
  → challenge-submit-job（消耗每日提交额度）
  → challenge-run-agent（WebSocket 隧道，PARALLELISM 路并发）
  → challenge-poll-result → challenge-ranking
```

- **观测协议：** msgpack JSON-RPC `infer`；三相机 JPEG（head / hand_left / hand_right）+ 双臂/腰/夹爪 state + `prompt`；默认 `robot_type=G2_omnipicker`（权威实现见 genie_sim `corobotpolicy.py`）。
- **状态文件：** `~/.simubotix-challenge.env`（`CHALLENGE_TOKEN`、`JOB_UUID`、`JOB_TOKEN` 等）。

## 对 wiki 的映射

- [wiki/entities/robocoliseum.md](../../wiki/entities/robocoliseum.md)
- [wiki/entities/genie-sim-3.md](../../wiki/entities/genie-sim-3.md)
- [sources/repos/genie_sim_robocoliseum.md](../repos/genie_sim_robocoliseum.md)
