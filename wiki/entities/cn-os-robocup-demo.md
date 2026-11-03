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
  - ../../sources/repos/robocup_demo.md
summary: "加速进化 开源项目 robocup_demo（具身Agent/规划）：具身 Agent：任务规划与技能调度…"
institutions:
  - booster
---

# robocup_demo

## 一句话定义

**robocup_demo** 是 [加速进化](https://github.com/BoosterRobotics) 公开的 **具身Agent/规划** 开源项目：具身 Agent：任务规划与技能调度

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
| 机构 | 加速进化 |
| 类别 | 具身Agent/规划 |
| 官方组织 | https://github.com/BoosterRobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `robocup_demo` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：真机 `./scripts/start.sh`（依次 `ros2 launch vision launch.py`、`ros2 launch brain launch.py`、`ros2 launch game_controller launch.py`）；仿真用 `scripts/sim_start.sh`（brain）与 `scripts/start_vision.sh`（vision，`sim:=true`）。

```mermaid
sequenceDiagram
    participant GC as 裁判机 (UDP 广播)
    participant GCN as game_controller_node.cpp<br/>GameControllerNode::spin
    participant VN as vision_node.cpp<br/>VisionNode
    participant Det as YoloV8Detector<br/>(Inference)
    participant BR as brain.cpp<br/>Brain (main.cpp)
    participant BT as brain_tree.cpp<br/>BehaviorTree.CPP 行为树
    participant RC as robot_client.cpp<br/>RobotClient
    participant Rb as 机器人运控 (LocoApiTopic...Req)

    Note over BR: main.cpp：Brain::init() 加载 config.yaml 与 behavior_trees/game.xml；线程 t 以 100Hz 调 brain->tick()
    GC->>GCN: UDP 包 (recvfrom)
    GCN->>GCN: 校验长度/版本/IP 白名单, handle_packet
    GCN->>BR: 发布 /booster_soccer/game_controller
    BR->>BT: gameControlCallback 写入 gc_game_state 等黑板条目
    Note over VN: 订阅相机彩色/深度/内参话题
    VN->>Det: ColorCallback → ProcessData → detector_->Inference(color)
    Det-->>VN: 检测框 (按类别置信度后处理)
    VN->>BR: 发布 /booster_soccer/detection (Detections)
    VN->>BR: 发布 /booster_soccer/line_segments (分割后的场线)
    BR->>BR: detectionsCallback 分组为 Ball/Goalpost/Robot 等并更新 data
    loop 每个 tick (HZ=100)
        BR->>BR: updateMemory / handleSpecialStates / handleCooperation
        BR->>BT: tree->tick() → tree.tickOnce()
        BT->>BT: MainTree：按 control_state / gc_game_state 切换 CamFindAndTrackBall、StrikerDecide、Chase、Kick 等节点
        BT->>RC: client->setVelocity / moveHead / RLVisionKick
        RC->>Rb: call() 发布 RpcReqMsg (api_id + JSON body)
    end
```

图依据：BoosterRobotics/robocup_demo@fa4f31c，读过 README.md、scripts/start.sh、scripts/sim_start.sh、scripts/start_vision.sh、src/brain/src/main.cpp、brain.cpp（init/tick/各 Callback）、brain_tree.cpp、robot_client.cpp、src/brain/behavior_trees/game.xml、src/vision/src/vision_node.cpp、src/game_controller/src/game_controller_node.cpp。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [robocup_demo 源码归档](../../sources/repos/robocup_demo.md)（<https://github.com/BoosterRobotics/robocup_demo>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：BoosterRobotics/robocup_demo@fa4f31c，读过 README.md、scripts/start.sh、scripts/sim_start.sh、scripts/start_vision.sh、src/brain/src/main.cpp、brain.cpp、brain_tree.cpp、robot_client.cpp、behavior_trees/game.xml、src/vision/src/vision_node.cpp、src/game_controller/src/game_controller_node.cpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [加速进化 官方组织](https://github.com/BoosterRobotics)
