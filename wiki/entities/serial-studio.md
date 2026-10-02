---
type: entity
tags: [software, visualization, debugging, telemetry, embedded, serial, can, mqtt]
status: complete
updated: 2026-10-02
related:
  - ./plotjuggler.md
  - ./foxglove-studio.md
  - ../concepts/uart-serial-communication.md
  - ../concepts/can-bus-protocol.md
  - ../queries/robot-policy-debug-playbook.md
sources:
  - ../../sources/repos/serial-studio.md
  - ../../sources/sites/serial-studio.md
summary: "Serial Studio 把硬件遥测转成实时仪表盘；适合串口电机与传感器调试。GPL 核心支持 UART/BLE/网络与 CSV，CAN/Modbus/MQTT、会话数据库和输出控件属于专有 Pro。"
---

# Serial Studio

**Serial Studio** 是跨平台硬件遥测仪表盘：将设备数据解析成字段，显示曲线、仪表与传感器状态，帮助机器人开发者观察电机和通信链路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| UART | Universal Asynchronous Receiver/Transmitter | 微控制器串口数据接口 |
| CAN | Controller Area Network | 常见电机与工业设备总线 |
| CSV | Comma-Separated Values | Quick Plot 输入与基础导出格式 |
| DBC | Database CAN | 定义 CAN 报文与信号的数据库格式 |
| FFT | Fast Fourier Transform | 观察信号频谱的变换 |
| GPL | GNU General Public License | 核心源码采用的开源许可 |

## 为什么重要

机器人上电调试常需要同时看关节目标角、实测角、电流、温度和 IMU。Serial Studio 通过配置仪表盘减少临时编写上位机的工作，尤其适合 STM32、ESP32 或串口桥接的单模块验证。它补充本库以 ROS 日志为主的调试工具链。

## 核心原理

| 环节 | 输入 / 处理 | 输出 |
|---|---|---|
| 接入与切帧 | UART、BLE、网络或已授权工业驱动接收字节；按帧边界识别数据 | 完整遥测帧 |
| 解析 | Built-In 模板或 JavaScript / Lua 将帧解为 datasets | 具名数值字段 |
| 变换 | 对字段缩放、标定、滤波、换算单位 | 可读物理量与派生量 |
| 展示 | 项目文件定义 groups、datasets、widgets；Quick Plot 可直接画 CSV | 曲线、仪表、FFT、IMU 等 |
| 记录与复盘 | GPL 导出 CSV；Pro 增加 SQLite 会话与 MDF4 等 | 可供离线分析的日志 |

接收数据和发送控制是两条不同能力：Pro 输出控件可通过脚本发送设备命令；遥测界面本身不承担机器人底层实时闭环。

## 工程实践

### 单关节调试示例（工程建议）

1. 控制器保留本地电机闭环，通过独立遥测通道发送 `t_ms,q_des_rad,q_rad,dq_rad_s,current_A,temp_C`。
2. 首先用 Quick Plot 发送纯数值行，例如 `1000,0.20,0.18,0.04,1.2,35.0`，确认字段顺序；正式项目在编辑器里命名字段并明确单位。
3. 同屏画目标角和实测角；为电流、温度配置仪表；利用字段变换计算 `q_des - q`。
4. 调参时导出 CSV，在 Python 中按设备时间戳比较跟踪误差、超调和电流峰值；保存项目文件以复用布局。
5. 多电机时按链路带宽选择遥测降采样，记录序号判断丢帧。100 Hz 遥测只是示例配置，不代表 GUI 或 API 提供 1000 Hz 实时闭环保证。

### 功能与许可边界（截至 2026-10-02）

| 需求 | 可用路径 |
|---|---|
| UART / BLE / 网络曲线和 CSV | GPL 核心可自行编译 |
| CAN / DBC、Modbus、MQTT、OPC UA | Pro 驱动与导入功能 |
| 同时接多个设备、按钮/滑块发送命令 | Pro |
| SQLite 会话、MDF4、报告 | Pro |
| 开源状态 | **部分开源**；Pro 源码可见但非 GPL，官方二进制按专有许可发布 |

源码构建要求 Qt 6.9+、C++20 与 CMake 3.20+，默认构建为 GPL 核心。安装包和构建步骤见上游 README；功能应对照实际安装版本，master 的描述可能先于稳定版。

### 与已有工具的分工

| 任务 | 本库建议入口 |
|---|---|
| MCU、串口传感器、定制硬件仪表盘 | Serial Studio |
| ROS topic / rosbag 与多曲线对齐 | [PlotJuggler](./plotjuggler.md) |
| ROS 图像、点云与场景复盘 | [Foxglove](./foxglove-studio.md) |
| 策略部署中观测、动作与时序排查 | [机器人策略调试 Playbook](../queries/robot-policy-debug-playbook.md) |

此表是按本库已有能力归纳的选型建议，不是同一数据集上的性能排名。

## 局限与风险

- **CAN 接入不等于 MIT 协议即插即用**：实际电机帧、量纲、端序和缩放仍需按设备说明解析；DBC 导入还要求匹配的数据库。
- **接收时间不等于采样时间**：USB、缓冲与网络会引入延迟；记录设备时间戳与序号后再分析抖动和丢帧。
- **图形显示频率不等于控制频率**：控制环运行在 MCU 或实时进程，桌面仪表盘用于观测和调试。
- **源码可见不代表全部开源**：核心为 GPL-3.0-or-later，Pro 文件是专有许可；官方安装包与自行构建 GPL 版的功能和许可不同。
- 本次核查没有验证原生 ROS bag 导入；ROS 日志分析优先沿用本库现有工具链。

## 关联页面

- [UART 串口通信](../concepts/uart-serial-communication.md) — 串口接线、波特率与数据帧。
- [CAN 总线协议](../concepts/can-bus-protocol.md) — 电机总线与上位机解码的边界。
- [PlotJuggler](./plotjuggler.md)、[Foxglove](./foxglove-studio.md) — 调试工具分工。
- [机器人策略调试 Playbook](../queries/robot-policy-debug-playbook.md) — 部署排查路径。

## 参考来源

- [Serial Studio 仓库归档](../../sources/repos/serial-studio.md) — README、构建要求与许可核查。
- [Serial Studio 官方站点核查](../../sources/sites/serial-studio.md) — 项目入口、GPL / Pro 能力边界。

## 推荐继续阅读

- [Getting Started](https://github.com/Serial-Studio/Serial-Studio/blob/master/doc/help/Getting-Started.md)
- [Pro vs Free](https://github.com/Serial-Studio/Serial-Studio/blob/master/doc/help/Pro-vs-Free.md)
- [官方示例](https://github.com/Serial-Studio/Serial-Studio/tree/master/examples)
