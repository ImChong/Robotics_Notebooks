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
  - ../../sources/repos/leisaac.md
summary: "光轮智能 开源项目 LeIsaac（数据集/Benchmark）：在Isaac Lab中接入SO-101 Leader完成遥操作、数据采集、格式转换和后续策略训练，连接示范输入与机器人学习数据。…"
institutions:
  - lightwheel
---

# LeIsaac

## 一句话定义

**LeIsaac** 是 [光轮智能](https://github.com/LightwheelAI) 公开的 **数据集/Benchmark** 开源项目：在Isaac Lab中接入SO-101 Leader完成遥操作、数据采集、格式转换和后续策略训练，连接示范输入与机器人学习数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 光轮智能 |
| 类别 | 数据集/Benchmark |
| 官方组织 | https://github.com/LightwheelAI |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `LeIsaac` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（均需 Isaac Sim / IsaacLab 环境）：采集 `python scripts/environments/teleoperation/teleop_se3_agent.py --task <Task> --teleop_device so101leader --record`；转换 `python scripts/convert/isaaclab2lerobot.py`；评测 `python scripts/evaluation/policy_inference.py --task <Task> --policy_type <gr00tn1.5|gr00tn1.6|lerobot-*|openpi>`（需另起对应策略服务）。

```mermaid
sequenceDiagram
    autonumber
    participant U as 操作者 / 主臂
    participant T as teleop_se3_agent.py main
    participant ENV as gym.make(task) → ManagerBasedRLEnv / DirectRLEnv
    participant DEV as leisaac.devices（SO101Leader / Keyboard / Gamepad…）
    participant REC as StreamingRecorderManager / LeRobotRecorderManager
    participant CV as isaaclab2lerobot.py
    participant EV as policy_inference.py main
    participant POL as leisaac.policy.*ServicePolicyClient
    participant SRV as 外部策略服务 (GR00T / LeRobot / OpenPI)

    T->>T: AppLauncher 启动 Isaac Sim，parse_env_cfg(task)
    T->>ENV: env_cfg.use_teleop_device + gym.make
    T->>REC: --record 时替换 env.recorder_manager
    T->>DEV: 按 --teleop_device 实例化，add_callback("R"/"N")
    loop while simulation_app.is_running()
        T->>DEV: advance() 取动作
        U-->>DEV: 主臂关节 / 键盘 / 手柄输入
        T->>ENV: env.step(actions)
        ENV-->>REC: 记录演示（成功才导出）
        T->>T: RateLimiter.sleep(step_hz)
    end
    REC-->>CV: dataset.hdf5
    CV->>CV: HDF5DatasetFileHandler 读取，add_episode 写入 LeRobotDataset
    EV->>ENV: parse_env_cfg + gym.make，env.reset()
    EV->>POL: 按 --policy_type 构造客户端
    loop 每个评测回合
        EV->>POL: get_action(preprocess_obs_dict(obs))
        POL->>SRV: 请求动作块（ZMQ 或 WebSocket 服务）
        SRV-->>POL: actions
        EV->>ENV: 逐步 env.step(action) 至多 policy_action_horizon 步
        ENV-->>EV: terminated 记为成功，time_out 记为超时
    end
```

图依据：LightwheelAI/LeIsaac@24d3bcd，读过 README.md、scripts/environments/teleoperation/teleop_se3_agent.py、scripts/evaluation/policy_inference.py、scripts/convert/isaaclab2lerobot.py（仅头部与函数清单）、source/leisaac/leisaac/policy/base.py。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [LeIsaac 源码归档](../../sources/repos/leisaac.md)（<https://github.com/LightwheelAI/LeIsaac>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：LightwheelAI/LeIsaac@24d3bcd，读过 README.md、scripts/environments/teleoperation/teleop_se3_agent.py、scripts/evaluation/policy_inference.py、scripts/convert/isaaclab2lerobot.py（仅头部）、source/leisaac/leisaac/policy/base.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [光轮智能 官方组织](https://github.com/LightwheelAI)
