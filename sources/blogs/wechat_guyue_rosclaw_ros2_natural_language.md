# 用自然语言控制 ROS2 机器人的完整技术方案

> 来源归档（blog / 微信公众号）

- **标题：** 用自然语言控制 ROS2 机器人的完整技术方案
- **类型：** blog
- **作者：** 古月居（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/eCpu9ykouejUiekxcWPjhw
- **发表日期：** 2026-08-22（frontmatter）
- **入库日期：** 2026-08-22
- **抓取方式：** [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`--no-images`；`playwright==1.49.1`）
- **一句话说明：** 古月居长文拆解 **RosClaw**（OpenClaw × ROS2 插件）：三层架构、三种部署模式（同机 DDS / 局域网 rosbridge / 云端 WebRTC）、ROS2 工具集与安全验证；并对比上海交大 MINT **RoboClaw** 的跨本体具身理解路线。

## 核心摘录（归纳，非全文）

### RosClaw 定位

- **项目：** [PlaiPin/rosclaw](https://github.com/PlaiPin/rosclaw) — 通过 WhatsApp、Telegram、Discord、Slack 等 IM 用自然语言控制 ROS2 机器人；OpenClaw Gateway + RosClaw Plugin + rosbridge / 本地 DDS / WebRTC。
- **开源状态（仓库核查 2026-08-22）：** **已开源**；Apache-2.0；README 注明 **major re-architecture / 拆仓迁移中**，以 GitHub 为准。
- **技术栈：** TypeScript monorepo（pnpm 9+、Node 20+）；`@rosclaw/rosbridge-client`、`@rosclaw/openclaw-plugin`；ROS2 侧 `rosclaw_discovery`（Python）、`rosclaw_msgs`；Docker Compose 演示栈（Gazebo + rosbridge）。

### 三层架构

| 层 | 职责 |
|----|------|
| 消息接入 | OpenClaw Gateway 统一 WhatsApp/Telegram/Discord/Slack |
| AI Gateway | 会话、记忆、LLM 意图理解、工具注册（RosClaw Plugin） |
| ROS2 | DDS / Nav2 / MoveIt2 / 传感器与执行器；经 rosbridge 或本地 rclnodejs |

### 三种部署模式（文内叙事）

| 模式 | 拓扑 | 传输 | 典型场景 |
|------|------|------|----------|
| A 同机 | OpenClaw 与 ROS2 同计算单元 | 本地 DDS（rclnodejs） | 边缘 AGV、服务机器人、Jetson |
| B 局域网 | OpenClaw 与机器人在同一 LAN | WebSocket → `rosbridge_server` | 实验室开发、多机测试 |
| C 云端 | OpenClaw 在云/VPS，机器人在远端 NAT 后 | WebRTC 数据通道 + 信令 | RaaS、远程运维、多租户 |

### ROS2 工具集（Agent Tools）

- `ros2_publish` — 话题发布（如 `/cmd_vel`）
- `ros2_subscribe_once` — 单次读话题（状态查询）
- `ros2_service_call` — 同步服务调用
- `ros2_action_goal` — Nav2 等长时 Action（文内 Phase 2；上游 README 已列）
- 上游 README 另列：`ros2_param_get/set`、`ros2_list_topics`、`ros2_camera_snapshot`
- **`/estop`** — 绕过 AI 的紧急停止（零速度 + 取消 Action）

### 安全机制

- 工具执行前 **SafetyValidator**：速度上限、工作空间边界、导航目标校验
- `/estop` 在消息预处理阶段拦截，直接发零 `cmd_vel`

### RoboClaw（上海交大 MINT，对比章节）

- **仓库：** [MINT-SJTU/RoboClaw](https://github.com/MINT-SJTU/RoboClaw) — **早期开源**；具身智能助手，强调 **跨本体 / 跨环境 / 跨任务** 能力迁移。
- **与 RosClaw 分工（文内）：** RosClaw 偏 **交互界面**（IM → ROS2）；RoboClaw 偏 **本体理解**（熟悉校准、能力抽象、训练辅助）。
- **四层（文内）：** 助手层 → 具身层（本体建模、空间建联、能力抽象、熟悉校准、训练辅助）→ 执行层（ROS2）→ 载体层（仿真/真机）。

### 应用案例（文内，未独立验证）

- TurtleBot3 Gazebo Docker 演示
- 工业 AGV 车队（模式 B、多机上下文切换）
- 高校远程科研机器人（模式 C、权限分级与审计）

## 对 wiki 的映射

- [rosclaw](../../wiki/entities/rosclaw.md)（**新建**）
- [roboclaw](../../wiki/entities/roboclaw.md)（**新建**）
- 交叉更新：[openclaw](../../wiki/entities/openclaw.md)、[ros2-control](../../wiki/entities/ros2-control.md)

## 可信度与使用边界

- 本文为 **古月居公众号技术解读**，部分代码片段为文内示例，可能与当前 `main` 分支逐行不一致；**以 [PlaiPin/rosclaw](https://github.com/PlaiPin/rosclaw) README 为准**。
- 模式 C WebRTC、`rosclaw_agent` 等细节以仓库实际发布为准；README 当前主路径强调 rosbridge + Docker 演示。
- RoboClaw 为 **early stage**；文内「与 RosClaw 组合」为方向性叙述，非已发布集成。
- 微信 CDN 图片未纳入 wiki 正文；工业 AGV 等案例为文内叙述，无第三方佐证。

## 当前提炼状态

- sources 归档：**完成**
- wiki 实体：**RosClaw / RoboClaw 已升格**
