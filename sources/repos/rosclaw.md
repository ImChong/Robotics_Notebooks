# RosClaw（代码仓库）

> 来源归档

- **标题：** RosClaw — ROS2 meets OpenClaw
- **类型：** repo
- **组织：** PlaiPin
- **链接：** <https://github.com/PlaiPin/rosclaw>
- **入库日期：** 2026-08-22
- **一句话说明：** OpenClaw 扩展 + ROS2 插件层：通过 IM（WhatsApp/Telegram/Discord/Slack）用自然语言发布话题、调用服务、发送 Action；含 rosbridge TypeScript 客户端、Docker 演示栈与能力发现节点。
- **沉淀到 wiki：** [`wiki/entities/rosclaw.md`](../../wiki/entities/rosclaw.md)

## 开源状态

**已开源**（Apache-2.0）。README（2026-08-22 核查）注明项目 **major re-architecture / 拆仓迁移中**；功能与包结构以 `main` 为准。

| 包 / 节点 | 说明 |
|-----------|------|
| `@rosclaw/rosbridge-client` | rosbridge WebSocket 客户端（可独立使用） |
| `@rosclaw/openclaw-plugin` | OpenClaw 扩展：ROS2 工具、安全钩子、`/estop` |
| `@rosclaw/openclaw-canvas` | 实时仪表盘（Phase 3） |
| `rosclaw_discovery` | ROS2 能力自动发现（Python） |
| `rosclaw_msgs` | 自定义 msg/srv |
| `docker/` | ROS2 + rosbridge + Gazebo Compose 演示 |

**Agent 工具（README）：** `ros2_publish`、`ros2_subscribe_once`、`ros2_service_call`、`ros2_action_goal`、`ros2_param_get/set`、`ros2_list_topics`、`ros2_camera_snapshot`；命令 `/estop` 紧急停止。

**快速演示：** `pnpm install && pnpm build`；`cd docker && docker compose up`；OpenClaw 配置 `ws://localhost:9090`。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [rosclaw](../../wiki/entities/rosclaw.md) | 实体页 |
| [openclaw](../../wiki/entities/openclaw.md) | 上游控制平面 / agent harness |
| [roboclaw](../../wiki/entities/roboclaw.md) | 文内对比的 SJTU MINT 具身助手 |
| [ros2-control](../../wiki/entities/ros2-control.md) | ROS2 控制器与硬件接口生态 |

## 来源博文

- [古月居：用自然语言控制 ROS2 机器人的完整技术方案](../blogs/wechat_guyue_rosclaw_ros2_natural_language.md)
