# Genie Sim — RoboColiseum Challenge Skills

> 来源归档（ingest）

- **标题：** Genie Sim RoboColiseum Simulation Challenge Skills
- **类型：** repo（子路径）
- **机构：** 智元机器人（AgiBot）
- **主仓：** <https://github.com/AgibotTech/genie_sim>
- **子路径：** <https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum>
- **平台：** <https://robocoliseum.ai/>
- **入库日期：** 2026-09-15
- **通读：** 已完成
- **一句话说明：** Genie Sim 仓内 **RoboColiseum 仿真挑战赛** 的 agent skill 集：覆盖登录、提交 job、WebSocket 隧道推理、轮询结果、榜单查询、LeRobot v2.1 数据与 baseline checkpoint 下载，以及 obs/action 线协议说明。

## 开源状态（README / SKILL 核查，2026-09-15）

| 项 | 状态 |
|----|------|
| Skill 文档与脚本 | **已开源**（随 genie_sim 主仓发布） |
| 推理权威协议 | genie_sim `main/source/geniesim/benchmark/policy/corobotpolicy.py` |
| 示例 tunnel agent | 外部 ACoT-VLA `scripts/tunnel_agent.py`（baseline skill 引用） |
| 官方 Simulation SDK pip 包 | **无** — skill 明确「无官方 pip SDK」，参赛者自带推理客户端 |

## 子目录索引

| Skill 目录 | 作用 |
|------------|------|
| `challenge-help` | 总路由与端到端 stage map |
| `challenge-login` | 获取/刷新 `CHALLENGE_TOKEN` |
| `challenge-download-datasets` | `download_dataset.sh` → ModelScope LeRobot v2.1 task suite |
| `challenge-baseline-model` | clone 推理仓 + ModelScope checkpoint + 启动 baseline |
| `challenge-submit-job` | `POST /api/challenge/job`（消耗每日提交额度） |
| `challenge-run-agent` | `./scripts/tunnel.sh` WebSocket 反向隧道 |
| `challenge-inference-protocol` | obs/action msgpack 线协议（三相机 + 21-dof state 布局） |
| `challenge-poll-result` | 轮询 job 分数与状态 |
| `challenge-ranking` | `best-score` + per-board leaderboard |
| `challenge-troubleshoot` | 401 / Pending / agent 断连排障 |

## Checkpoint ↔ board 映射（baseline 示例）

| Job board | ModelScope checkpoint 名 |
|-----------|---------------------------|
| `instruction` | `instruction_and_robust_pi05` |
| `manip` | `manipulation_pi05` |
| `spatial` | `spatial_pi05` |

## 原文通读记录与核验边界（2026-11-11）

**完整阅读的官方原始文档**（AgibotTech/genie_sim，`main`）：

1. [Simulation Challenge Skills README](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/README.md)：跨 shell 状态、JOB_ID/JOB_UUID 区别、提交确认门禁、入口索引。
2. [Quickstart](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/quickstart.md) 与 [Benchmark task suites](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/benchmark-task-suite.md)：提交顺序、四榜 `board` 的合法值以及数据集 suite 名称。
3. [challenge-help/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-help/SKILL.md)、[challenge-submit-job/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-submit-job/SKILL.md) 与 [challenge-run-agent/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-run-agent/SKILL.md)：提交请求体、job-scoped token、WebSocket session 和重连协议。
4. [challenge-inference-protocol/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-inference-protocol/SKILL.md)、[challenge-poll-result/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-poll-result/SKILL.md)、[challenge-ranking/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-ranking/SKILL.md)、[challenge-baseline-model/SKILL.md](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-baseline-model/SKILL.md)：观测/动作封包、评分状态、榜单查询与基线运行入口。

**定点核查的源码**（不是声称通读整个仓库）：

- [`benchmark/policy/corobotpolicy.py`](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py) 的 `get_payload`、`infer`、`_parse_result`、`act`；`get_payload` 封装三相机 JPEG、关节状态与任务提示，`infer` 使用 msgpack 接收 `result`，`act` 使用 action buffer。
- [`challenge-download-datasets/scripts/download_dataset.sh`](https://github.com/AgibotTech/genie_sim/blob/main/source/geniesim_benchmark/skills/robocoliseum/challenge-download-datasets/scripts/download_dataset.sh)：仅接受 `instruction` / `manipulation` / `sim2real` 三种下载 suite，**不要混同** Challenge job 的 `manip` board。

**查证结果与局限：** 参赛 job 一次只对应一个 `board`；`POST /api/challenge/job` 只填写 `name` 与 `config.board` 等字段，`model_name` 由已审核的账户 profile 提供。数值型 `JOB_ID` 用于 HTTP 状态/结果查询，`JOB_UUID` 用于 WebSocket 隧道连接。官方网站展示的 Overall 属展示层聚合，不等同于第五个 Challenge `board`。本次未下载完整数据集、权重，也未提交真实挑战 job；平台实时配额、榜单及运行成功率不以静态文档断言。

## 对 wiki 的映射

- [wiki/entities/robocoliseum.md](../../wiki/entities/robocoliseum.md)
- [wiki/entities/genie-sim-3.md](../../wiki/entities/genie-sim-3.md)
- [sources/sites/robocoliseum.md](../sites/robocoliseum.md)
