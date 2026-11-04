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
  - ../../sources/repos/flexiv_trainer.md
summary: "非夕科技 开源项目 flexiv_trainer（VLA/操作模型）：面向非夕机器人组织数据、训练和Physical AI技能开发流程，把机器人接口接到策略训练与验证。…"
institutions:
  - flexiv
---

# flexiv_trainer

## 一句话定义

**flexiv_trainer** 是 [非夕科技](https://github.com/flexivrobotics) 公开的 **VLA/操作模型** 开源项目：面向非夕机器人组织数据、训练和Physical AI技能开发流程，把机器人接口接到策略训练与验证。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 非夕科技 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/flexivrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `flexiv_trainer` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`flexiv-trainer-server`（即 `flexivtrainer.api.app:run`，启动后在 `http://127.0.0.1:8000/` 打开 Web UI；遥操作需配置 `FLEXIV_TRAINER_TELEOP_ROBOT_PAIRS`）

```mermaid
sequenceDiagram
    autonumber
    participant UI as Web UI 浏览器
    participant APP as FastAPI 应用 api/app.py 与 routes
    participant RT as runtime/manager.py RuntimeManager
    participant TS as teleop/service.py TeleopService
    participant TDK as flexivtdk TransparentCartesianTeleopLAN 闭源
    participant TR as routes/training.py 与 jobs/train_policy.py TrainingService
    participant SP as 子进程 lerobot-train 或 python -m lerobot
    APP->>APP: run() 打印横幅, uvicorn.run(app, host, port)
    APP->>APP: lifespan 启动 device-warmup 后台线程 (warm_up_devices)
    UI->>APP: POST /teleop/start (routes/teleop.py start_teleop)
    APP->>RT: runtime.teleop.start(zero_ft_sensor)
    RT->>TS: TeleopService.start(zero_ft_sensor)
    TS->>TS: initialize() 若未构造则 TransparentCartesianTeleopLAN(robot_pairs)
    TS->>TDK: Init() 每次 Start 前调用
    TS->>TDK: 设置 follower 刚度 (_apply_follower_stiffness)
    TS->>TDK: Start()
    TS->>TDK: 对每个 pair 调用 Engage(idx, False)
    TS-->>UI: TeleopSnapshot (started=true)
    UI->>APP: POST /training/start (routes/training.py start_training)
    APP->>TR: runtime.training.start(dataset_root, output_dir, policy_type, ...)
    alt 需要 B-spline 数据转换
        TR->>TR: 后台线程 _collect_conversion 调用 _conversion_command
    end
    TR->>TR: _build_training_command: lerobot-train 或 sys.executable -m lerobot.scripts.lerobot_train
    TR->>SP: _spawn_job_process: subprocess.Popen(command, stdout PIPE, cwd, env)
    TR->>TR: 启动后台线程 _collect_logs 收集作业日志
    TR-->>UI: 作业状态 (running)
    UI->>APP: GET /training/status 轮询
    UI->>APP: POST /training/stop
    APP->>TR: stop() 终止训练子进程
    UI->>APP: POST /teleop/stop
    APP->>TS: stop()
    TS->>TDK: Stop()
```

图依据：flexiv-trainer@0588d77（仓库 flexiv_trainer），README.md、pyproject.toml（[project.scripts] 入口）、src/flexivtrainer/api/app.py、api/routes/teleop.py（节选）、api/routes/training.py（节选）、teleop/service.py（节选）、jobs/train_policy.py（节选）。TDK 与 lerobot 内部闭源或未读，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [flexiv_trainer 源码归档](../../sources/repos/flexiv_trainer.md)（<https://github.com/flexivrobotics/flexiv_trainer>）

- 源码核对（2026-11-02）：flexivrobotics/flexiv_trainer@0588d77，读过 README.md、pyproject.toml、src/flexivtrainer/api/app.py、api/routes/teleop.py（节选）、api/routes/training.py（节选）、teleop/service.py（节选）、jobs/train_policy.py（节选）；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [非夕科技 官方组织](https://github.com/flexivrobotics)
