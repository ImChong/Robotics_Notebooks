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
  - ../../sources/repos/x-trainer.md
summary: "越疆科技 开源项目 x-trainer（数据集/Benchmark）：为X-Trainer协作机械臂提供VR和手柄遥操作适配、数据采集与GPU仿真训练入口。…"
institutions:
  - dobot
---

# x-trainer

## 一句话定义

**x-trainer** 是 [越疆科技](https://github.com/embodied-dobot) 公开的 **数据集/Benchmark** 开源项目：为X-Trainer协作机械臂提供VR和手柄遥操作适配、数据采集与GPU仿真训练入口。

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
| 机构 | 越疆科技 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/embodied-dobot、https://github.com/Dobot-Arm |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `x-trainer` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python scripts/environments/teleoperation/teleop_se3_agent.py --task=LeIsaac-XTrainer-PickCube-v0 --teleop_device=xtrainerleader --num_envs=1 --device=cuda --enable_cameras --multi_view`（需 Isaac Sim 4.5 与 Isaac Lab 0.47.1 环境；`--teleop_device` 也可取 `bi_keyboard` 或 `xtrainer_vr`）。

```mermaid
sequenceDiagram
    autonumber
    participant S as scripts/environments/teleoperation/teleop_se3_agent.py main
    participant D as source/leisaac/devices/lerobot/xtrainer_leader.py XTrainerLeader
    participant AG as xtrainer_utils 的 BimanualAgent 与 DobotAgent
    participant A as devices/device_base.py Device.advance
    participant ENV as Isaac Lab 环境 gym.make 得到的 env 外部依赖

    S->>S: AppLauncher 启动 Isaac Sim
    S->>ENV: parse_env_cfg 与 gym.make 创建 task 环境
    S->>D: XTrainerLeader(env, left_disabled)
    D->>AG: 构造 BimanualAgent 左右 DobotAgent
    D->>D: 启动 _sampling_loop 采样线程 并做 EMA 平滑
    S->>ENV: env.reset()
    S->>D: teleop_interface.reset()
    loop simulation_app.is_running()
        S->>A: teleop_interface.advance()
        A->>D: input2action()
        D->>D: get_device_state 取最新 14 维关节状态
        D-->>A: 含 joint_state 与 motor_limits 的字典
        A->>ENV: cfg.preprocess_device_action 转为动作张量
        A-->>S: actions
        alt 收到 R 或 N 按键回调
            S->>ENV: env.reset 或 manual_terminate 标记成功
        else actions 为 None 即未开始
            S->>ENV: env.render()
        else 正常动作
            S->>ENV: env.step(actions)
        end
        S->>S: 录制开启时由 StreamingRecorderManager 写入 HDF5
        S->>S: RateLimiter.sleep 按 step_hz 节拍等待
    end
    S->>ENV: env.close() 与 simulation_app.close()
```

图依据：embodied-dobot/x-trainer@5862c3b，实际读过 `README.md`、`scripts/environments/teleoperation/teleop_se3_agent.py`（`main` 主循环）、`source/leisaac/leisaac/devices/lerobot/xtrainer_leader.py`（采样线程、`input2action`、`get_device_state`）、`source/leisaac/leisaac/devices/device_base.py`（`Device.advance`）。`xtrainer_utils` 与 Isaac Lab 本体未读（后者为外部依赖，`dependencies/IsaacLab` 在仓库中为子模块占位），图中仅据调用名标注；VR 分支的 Flask 子进程只在 `xtrainer_vr` 模式下出现，未画。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [x-trainer 源码归档](../../sources/repos/x-trainer.md)（<https://github.com/embodied-dobot/x-trainer>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：embodied-dobot/x-trainer@5862c3b，读过 README.md、scripts/environments/teleoperation/teleop_se3_agent.py、source/leisaac/leisaac/devices/lerobot/xtrainer_leader.py、source/leisaac/leisaac/devices/device_base.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [越疆科技 官方组织](https://github.com/embodied-dobot)
