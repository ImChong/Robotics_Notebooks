---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/ghrc_evaluation_2026.md
summary: "优必选 开源项目 GHRC_Evaluation_2026（评测）：评测基准与工具…"
institutions:
  - ubtech
---

# GHRC_Evaluation_2026

## 一句话定义

**GHRC_Evaluation_2026** 是 [优必选](https://github.com/UBTECH-Robot) 公开的 **评测** 开源项目：评测基准与工具

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 优必选 |
| 类别 | 评测 |
| 官方组织 | https://github.com/UBTECH-Robot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GHRC_Evaluation_2026` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`./run_eval.sh [task1 task2 task3 task4 | all]`（`HEADLESS` 默认 1）。脚本先 `docker run` 启动 infer 容器，执行 `python -m lerobot.scripts.ghrc_eval_infer`（配置 `eval_config/eval_infer.yaml`），用 `wait_for_infer_ready` 等待 8765（control）与 8766（stream）端口监听，再对每个任务 `docker run` 启动 sim 容器，执行 `python -m lerobot.scripts.ghrc_eval_sim`（配置 `eval_config/eval_sim.yaml`，经 `/isaac-sim/python.sh`）。

```mermaid
sequenceDiagram
    participant RE as run_eval.sh
    participant I as ghrc_eval_infer.py (SimInferContainer)
    participant A as PolicyAdapter (sim_eval/policy_adapter.py)
    participant WS as websocket_server.py / websocket_client.py
    participant S as ghrc_eval_sim.py (SimEvalContainer)
    participant R as robot (build_robot, Isaac Sim, Ubtech_sim)
    participant T as assertion_fn (sim_eval/task_assert.py)
    RE->>I: docker run infer 容器，python -m lerobot.scripts.ghrc_eval_infer
    I->>A: create_policy_adapter(adapter_type / adapter_class)，adapter.load(policy_path, device, config)
    I->>WS: SimInferWebSocketServer 注册 start/reset/episode_end/stop/observation 回调，ws_server.start()
    RE->>RE: wait_for_infer_ready（8765、8766 端口）
    RE->>S: docker run sim 容器，python -m lerobot.scripts.ghrc_eval_sim
    S->>R: build_robot(task, task_config_path)；robot.connect()
    S->>WS: EvalWebSocketClient.connect()
    S->>WS: send_start(episode_id)
    WS->>I: _on_start / _on_reset（挂起 policy reset）
    loop step < max_steps
        S->>R: get_observation()，flatten_obs
        S->>WS: send_observation(observation, task, episode_id, step)
        WS->>I: _on_observation 入队 observation_queue
        I->>A: adapter.predict(observation, InferenceContext)
        A-->>I: action
        I->>WS: broadcast_action(action, episode_id, step)
        WS-->>S: wait_for_action（超时 action_wait_timeout）
        S->>R: send_action(action_to_dict(action))
        S->>T: assertion_fn(...) 得到 is_success / terminal / reason / metrics
    end
    S->>WS: send_episode_end(...)；_reset_robot() 调 robot.reset()
    S->>S: save_summary / print_summary，send_stop()
```

图依据：GHRC_Evaluation_2026@44a8f2e，`README.md`、`run_eval.sh`、`eval_config/eval_infer.yaml`（头部）、`eval_config/eval_sim.yaml`（头部）、`src/lerobot/scripts/ghrc_eval_infer.py`、`src/lerobot/scripts/ghrc_eval_sim.py`；`sim_eval/` 下 websocket、assert、policy_adapter 模块与 `Ubtech_sim/` 仿真实现未读，仅按调用点标注。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GHRC_Evaluation_2026 源码归档](../../sources/repos/ghrc_evaluation_2026.md)（<https://github.com/UBTECH-Robot/GHRC_Evaluation_2026>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：UBTECH-Robot/GHRC_Evaluation_2026@44a8f2e，读过 README.md、run_eval.sh、eval_config/eval_infer.yaml 与 eval_sim.yaml（头部）、src/lerobot/scripts/ghrc_eval_infer.py、src/lerobot/scripts/ghrc_eval_sim.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [优必选 官方组织](https://github.com/UBTECH-Robot)
