# Eclipse Mosquitto

> 来源归档

- **标题：** Eclipse Mosquitto — MQTT Broker / Client Library
- **类型：** repo
- **链接：** https://github.com/eclipse/mosquitto
- **官网 / 文档：** https://mosquitto.org/ · https://mosquitto.org/documentation/
- **最近复核：** 2026-09-17
- **许可证：** Eclipse Public License 2.0（broker/libmosquitto）；部分组件 EDL 1.0
- **一句话说明：** 最流行的开源 **MQTT Broker** 与 C 客户端库之一；支持 MQTT 3.1.1/5.0、TLS、WebSocket、桥接；本地调试与 CI 冒烟的默认参考实现。
- **沉淀到 wiki：** 是 → [`wiki/concepts/mqtt-protocol.md`](../../wiki/concepts/mqtt-protocol.md)

## 为什么值得保留

- 读 OASIS 规范时需要 **可运行 Broker** 验证 CONNECT/QoS/Will；Mosquitto 是工程界默认选择。
- 机器人栈中 **PlotJuggler MQTT 插件**、**ESP32 固件示例**、**Wokwi Wi-Fi 仿真** 均常指向 Mosquitto 或兼容 Broker。

## 开源核查（步骤 2.5）

| 项 | 状态 |
|----|------|
| 源码 | **已开源** — GitHub `eclipse/mosquitto` |
| 发布 | 官网 Download + 各发行版包（Debian/Ubuntu `mosquitto` 等） |
| MQTT 5 | 现代版本支持 v5；旧环境需确认 `--protocol-version` |

## 核心摘录

### 组件

| 组件 | 用途 |
|------|------|
| **mosquitto** | Broker 守护进程；监听 1883（明文）/ 8883（TLS） |
| **libmosquitto** | C 客户端库（pub/sub API） |
| **mosquitto_pub / mosquitto_sub** | CLI 测试工具 |
| **mosquitto.conf** | ACL、桥接、持久化、listener 配置 |

### 典型本地调试（与规范对照）

```bash
# 终端 1：启动 broker（默认 1883）
mosquitto -v

# 终端 2：订阅 QoS 1
mosquitto_sub -t 'robot/+/telemetry' -q 1 -v

# 终端 3：发布
mosquitto_pub -t 'robot/g1/telemetry' -q 1 -m '{"joint_temp":[42.1]}'
```

### 与机器人工程的关系

- **适用：** 板端状态上报、远程监控、非硬实时 HMI、仿真（Wokwi）连公网 Broker。
- **不适用：** 500 Hz–1 kHz 关节力矩环——延迟与 Broker 单点不符合 [实时运控中间件指南](../../wiki/queries/real-time-control-middleware-guide.md)。

## 对 wiki 的映射

- [mqtt-protocol](../../wiki/concepts/mqtt-protocol.md)
- [plotjuggler](../../wiki/entities/plotjuggler.md)（MQTT DataStreamer 插件）
- [wokwi](../../wiki/entities/wokwi.md)（ESP32 MQTT 仿真）

## 推荐继续阅读（外部）

- Mosquitto [Man pages](https://mosquitto.org/man/mosquitto-8.html)
- Eclipse [MQTT 5 support notes](https://mosquitto.org/documentation/migrating-to-2-0/)

## 当前提炼状态

- [x] 开源核查
- [x] wiki 映射
