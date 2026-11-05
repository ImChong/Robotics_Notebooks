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
  - ../../sources/repos/gr00t-agilex.md
summary: "松灵机器人 开源项目 gr00t-agilex（VLA/操作模型）：VLA/策略接入：模型输出动作块驱动本体执行…"
institutions:
  - agilex
---

# gr00t-agilex

## 一句话定义

**gr00t-agilex** 是 [松灵机器人](https://github.com/agilexrobotics) 公开的 **VLA/操作模型** 开源项目：VLA/策略接入：模型输出动作块驱动本体执行

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
| 机构 | 松灵机器人 |
| 类别 | VLA/操作模型 |
| 官方组织 | https://github.com/agilexrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `gr00t-agilex` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口（两个进程，先起服务端）：
- 服务端：`uv run python gr00t/eval/run_gr00t_server.py --embodiment-tag ALOHA --model-path <CKPT> --device cuda:0 --host 0.0.0.0 --port 5555`（`tyro.cli(ServerConfig)` → `main()`）
- 客户端：`python examples/aloha/inference.py`（`main()` → `rclpy.init()` → `rclpy.spin(RosOperator)`），需 ROS2 相机与机械臂话题在线

```mermaid
sequenceDiagram
    participant SRV as run_gr00t_server.py main()
    participant PS as PolicyServer（ZMQ REP，gr00t/policy/server_client.py）
    participant GP as Gr00tPolicy（gr00t/policy/gr00t_policy.py）
    participant C as RosOperator（examples/aloha/inference.py）
    participant T as inference_process 线程
    participant PC as PolicyClient（ZMQ REQ）
    participant ROS as ROS2 话题（相机 / 关节 / 末端位姿 / 底盘）
    SRV->>GP: Gr00tPolicy(model_path, device, ...)
    SRV->>PS: PolicyServer(policy, host, port).run()（bind tcp 5555）
    C->>C: RosOperator.__init__：create_subscription / create_publisher
    ROS-->>C: 相机彩色与深度、arm_joint_state、arm_end_pose、odom、instruction 回调
    C->>C: model_inference 线程：make_policy(args)
    C->>PC: PolicyClient(host, port, timeout 30s)
    PC->>PS: ping()（连通性测试）
    PS-->>PC: 应答
    loop 30Hz 控制循环（create_rate(30)）
        C->>T: 满足 check_frame() 且无在途推理时启动 inference_process
        T->>C: get_frame() 取最新帧
        T->>T: 组装 observation：state（左右臂与夹爪）、video（left / front / right）、language
        T->>PC: policy.get_action(observation, options)
        PC->>PS: REQ：endpoint get_action，observation，options
        PS->>GP: get_action → check_observation → _get_action → check_action
        GP-->>PS: action, info
        PS-->>PC: REP：action, info
        PC-->>T: action_pred, info
        T->>C: 写入 inference_actions 与 inference_timestep（action chunk）
        C->>C: 按 t - t_start 取 raw_action，拆分各关节与夹爪
        C->>ROS: arm_joint_state_ctrl 发布 JointState / arm_end_pose_ctrl 发布 PoseStamped / robot_base_vel_ctrl 发布 Twist
    end
```

注：图中为 `asynchronous_inference` 分支；控制循环开始前有 `input("Please press any key to start inference:")` 的人工确认。`pika`、`aloha_eef` 等其他目录的 wrapper 与 `Gr00tPolicy` 内部模型前向未展开。

图依据：gr00t-agilex@83ce744，读过 README.md、gr00t/eval/run_gr00t_server.py（main、服务端构造）、gr00t/policy/server_client.py（PolicyServer.run、PolicyClient._get_action、call_endpoint）、gr00t/policy/gr00t_policy.py（类与 _get_action 签名，仅 grep）、examples/aloha/inference.py（make_policy、inference_process、RosOperator 的订阅与发布、main，经 git show 读取）；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [gr00t-agilex 源码归档](../../sources/repos/gr00t-agilex.md)（<https://github.com/agilexrobotics/gr00t-agilex>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：agilexrobotics/gr00t-agilex@83ce744，读过 README.md、gr00t/eval/run_gr00t_server.py、gr00t/policy/server_client.py、gr00t/policy/gr00t_policy.py（grep）、examples/aloha/inference.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [松灵机器人 官方组织](https://github.com/agilexrobotics)
