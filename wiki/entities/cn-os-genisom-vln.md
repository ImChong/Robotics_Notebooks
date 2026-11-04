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
  - ../../sources/repos/genisom_vln.md
summary: "智身科技 开源项目 genisom_vln（具身Agent/规划）：具身 Agent：任务规划与技能调度…"
institutions:
  - zsibot
---

# genisom_vln

## 一句话定义

**genisom_vln** 是 [智身科技](https://github.com/zsibot) 公开的 **具身Agent/规划** 开源项目：具身 Agent：任务规划与技能调度

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
| 机构 | 智身科技 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/zsibot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `genisom_vln` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（需三个 ROS2 节点与 MATRiX 仿真同时运行）：`python main.py --goal_type text --text_goal "green plant"`（或 `--goal_type ins_image --image_goal_path ./goals/bed.jpg`）；桥接节点：`ros2 run env_bridge env_bridge`、`ros2 run mc_sdk_bridge mc_sdk_bridge`。下图只展开 `main.py` 直接调用的路径，LLM/VLM 与建图细节见注。

```mermaid
sequenceDiagram
    participant M as main()（main.py）
    participant A as UniGoal_Agent（agents/zeroshot/unigoal/agent.py）
    participant V as MatrixEnv（envs/matrix_env.py）
    participant EB as env_bridge（bridge/.../env_bridge.py）
    participant MB as mc_sdk_bridge（bridge/.../mc_sdk_bridge.cpp）
    participant SIM as MATRiX 仿真 / SDK 话题
    M->>V: construct_envs(args) → MatrixEnv：PULL bind 5999，PUSH connect 5556
    M->>A: UniGoal_Agent(args, envs)
    M->>A: agent.reset()
    A->>V: envs.reset() → get_obs()
    SIM-->>EB: /image_raw/compressed、/image_raw/compressed/depth、/imu、/odom/mujoco_odom
    EB-->>V: ZMQ PUSH 5999（npz 打包 rgb/depth/pose/compass）
    V-->>A: obs_zsibot, info
    A-->>M: obs, rgbd, infos
    loop 探索与搜索（main.py while True）
        M->>M: BEV_map.mapping(obs, rgbd)；graph.update_scenegraph(obs)
        M->>A: agent.step(agent_input)
        A->>A: instance_discriminator() / get_action()（FMM 局部规划）
        A->>V: envs.step({'action': 1|2|3})
        V->>MB: ZMQ PUSH "1"前进 / "2"左转 / "3"右转（tcp 5556）
        MB->>MB: zmqLoop_ 接收字符，100Hz 持续发布
        MB->>SIM: sdk_pub_ publish SDKCmd（vx / yaw_rate）
        V->>V: 轮询 5999 直到位姿稳定（位移 <0.001 且偏航 <1°）
        EB-->>V: 稳定后的新观测
        V-->>A: obs, done, info
        A-->>M: obs, rgbd, done, info
        M->>M: graph.explore() 得到 global_goal，更新 goal_stack
    end
    SIM-->>MB: sdk_robotstate → highlevel_robotstate（ROS 话题回传）
```

注：`action=0` 在 `MatrixEnv.step` 中为长时间等待（`time.sleep`），未在图中展开；`debug_mode` 为真时不发送动作；文本/图像目标的 LLM/VLM 调用（`llms/vlmllm.py`）与 LightGlue 匹配未展开。

图依据：genisom_vln@f166429，读过 README.md、main.py、envs/matrix_env.py、agents/zeroshot/unigoal/agent.py（step/reset 及方法列表）、bridge/src/env_bridge/env_bridge/env_bridge.py（订阅话题与 ZMQ 推送）、bridge/src/mc_sdk_bridge/src/mc_sdk_bridge.cpp、bridge/src/mc_sdk_bridge/include/mc_sdk_bridge.h；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [genisom_vln 源码归档](../../sources/repos/genisom_vln.md)（<https://github.com/zsibot/genisom_vln>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：zsibot/genisom_vln@f166429，读过 README.md、main.py、envs/matrix_env.py、agents/zeroshot/unigoal/agent.py、bridge/src/env_bridge/env_bridge/env_bridge.py、bridge/src/mc_sdk_bridge/src/mc_sdk_bridge.cpp、bridge/src/mc_sdk_bridge/include/mc_sdk_bridge.h；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [智身科技 官方组织](https://github.com/zsibot)
