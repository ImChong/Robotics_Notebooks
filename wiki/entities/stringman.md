---
type: entity
project_id: stringman
project: https://neufangled.com/stringman
code: https://github.com/nhnifong/cranebot3-firmware
tags: [robotics, manipulation, teleoperation, imitation-learning, open-source]
status: draft
updated: 2026-11-01
related:
  - ./lerobot.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/sites/stringman.md
  - ../../sources/repos/stringman.md
summary: "Stringman 是 Neufangled 的房间尺度绳索驱动机器人，结合悬挂夹爪、外部控制器和可选 MCP 接口执行抓取实验。"
---

# Stringman：房间尺度绳索机器人

Stringman 是悬挂在房间上方的四线绳索驱动并联机器人，用于遥操作、模仿学习和抓取搬运实验。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CDPR | Cable-Driven Parallel Robot | 多根绳索共同驱动末端的并联机器人 |
| MCP | Model Context Protocol | 使智能体发现与调用工具的接口协议 |
| IMU | Inertial Measurement Unit | 测量末端惯性信息 |
| IL | Imitation Learning | 用示范数据训练策略 |

## 流程总览

```mermaid
flowchart TD
  cameras["锚点与夹爪相机"] --> host["外部运动控制器"]
  operator["手柄或控制界面"] --> host
  agent["AI 智能体"] --> mcp["可选 MCP 接口"]
  mcp --> host
  host --> motors["两组卷线电机"]
  motors --> gripper["四线悬挂夹爪"]
  gripper --> cameras
  host --> policy["LeRobot 策略和示范采集"]
  policy --> host
```

两组主动锚点分别收放两根绳，另两个角使用被动导环；相机跟踪标记，夹爪有压力传感、IMU 和测距。系统通过目标识别、抓取模型与放置流程组合完成指定任务。

## 源码运行时序图

以下是已核实的 MCP 入口与运动任务交互，设备内部控制细节仍待逐函数核实。

```mermaid
sequenceDiagram
  actor User as 用户
  participant CLI as stringman-headless
  participant Obs as observer
  participant Bridge as McpBridge
  participant Agent as MCP 智能体
  User->>CLI: 启动并传入 --mcp
  CLI->>Obs: 调用 observer.main
  Obs->>Bridge: 注册 McpBridge
  Agent->>Bridge: get_state 和 get_image
  Bridge-->>Agent: 状态与图像
  Agent->>Bridge: move_to 或 grip
  Bridge->>Obs: 创建异步运动任务
  Bridge-->>Agent: 返回任务已启动
  Agent->>Bridge: wait 查询运动结果
  Bridge-->>Agent: 返回结果与事件
```

## 工程实践与局限

- **范围与负载：** 官方标称动态负载 2 kg，取决于位置和速度。产品页对最大房间尺寸存在不同口径，应结合房间与安装条件核实。
- **自主能力：** 预设流程结合目标识别与抓取，并将物品放到指定位置；官方明确称仍为原型，尚不具备家电级可靠性。家具可能挡住绳索，当前没有自动避障。
- **MCP：** 可读取状态、图像，发起运动、抓取和停止；代码执行工具需单独启用。工具启动动作后需等待结果，不能把接口接通等同于已解决自主整理。
- **成本：** 2026-11-01 产品页显示成品 795 美元，安装、外部主机、运费与税费等需另行确认。本地 AI 推理需要加速硬件。
- **开源：** 官方控制软件仓库公开且声明 Apache-2.0；打印与构建文档公开，硬件各部分许可应单独核对。

## 核对状态

本页依据产品页、使用指南及控制器、MCP 桥接源码编写；完整硬件与系统架构文档及设备侧低层调用链尚未通读核实，未在真机运行。此条目暂为草稿。

## 关联页面

- [LeRobot](./lerobot.md)
- [Manipulation](../tasks/manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)

## 参考来源

- [官方资料归档](../../sources/sites/stringman.md)
- [官方代码归档与版本](../../sources/repos/stringman.md)

## 推荐继续阅读

- [MCP 操作指南](https://github.com/nhnifong/cranebot3-firmware/blob/79f2b695f885e2b09a0a7d1e0fac3546431e9bc1/src/nf_robot/host/mcp_bridge/GUIDE.md)
