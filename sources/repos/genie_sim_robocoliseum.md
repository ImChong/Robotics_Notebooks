# Genie Sim — RoboColiseum Challenge Skills

> 来源归档（ingest）

- **标题：** Genie Sim RoboColiseum Simulation Challenge Skills
- **类型：** repo（子路径）
- **机构：** 智元机器人（AgiBot）
- **主仓：** <https://github.com/AgibotTech/genie_sim>
- **子路径：** <https://github.com/AgibotTech/genie_sim/tree/main/source/geniesim_benchmark/skills/robocoliseum>
- **平台：** <https://robocoliseum.ai/>
- **入库日期：** 2026-09-15
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
| `manipulation` | `manipulation_pi05` |
| `spatial` | `spatial_pi05` |

## 对 wiki 的映射

- [wiki/entities/robocoliseum.md](../../wiki/entities/robocoliseum.md)
- [wiki/entities/genie-sim-3.md](../../wiki/entities/genie-sim-3.md)
- [sources/sites/robocoliseum.md](../sites/robocoliseum.md)
