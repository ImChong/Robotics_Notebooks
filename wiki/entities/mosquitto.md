---
type: entity
tags: [middleware, mqtt, broker, iot, open-source, embedded, eclipse, software]
status: complete
updated: 2026-09-17
related:
  - ../concepts/mqtt-protocol.md
  - ./plotjuggler.md
  - ./wokwi.md
sources:
  - ../../sources/repos/mosquitto.md
summary: "Eclipse Mosquitto 是主流开源 MQTT Broker 与 C 客户端库：支持 3.1.1/5.0、TLS、WebSocket 与桥接；本地调试 MQTT 协议与 PlotJuggler/ESP32 联调的事实参考实现。"
---

# Eclipse Mosquitto

**Eclipse Mosquitto** 是最广泛部署的 **开源 MQTT Broker** 之一，附带 **libmosquitto** C 库与 `mosquitto_pub` / `mosquitto_sub` CLI。读 OASIS MQTT 规范时的默认 **可运行对照实现**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MQTT | Message Queuing Telemetry Transport | OASIS IoT pub/sub 协议 |
| TLS | Transport Layer Security | 加密传输层 |
| CLI | Command-Line Interface | 命令行工具 |
| ACL | Access Control List | Broker 主题级访问控制 |
| EPL | Eclipse Public License | Mosquitto 主许可证 |

## 为什么重要

- **协议验证**：CONNECT、QoS 1/2 握手、Will、保留消息均可在本机复现。
- **机器人周边**：[PlotJuggler](./plotjuggler.md) MQTT 插件、[Wokwi](./wokwi.md) ESP32 仿真、边端 Agent 上报常对接 Mosquitto 或兼容 Broker。

## 核心信息

| 项 | 内容 |
|----|------|
| 机构 | 伊克利普斯基金会（Eclipse Foundation） |
| 仓库 | [github.com/eclipse/mosquitto](https://github.com/eclipse/mosquitto) |
| 许可证 | EPL-2.0 / EDL-1.0 |
| 协议版本 | MQTT 3.1.1、5.0（2.x 系） |
| 默认端口 | 1883（TCP）、8883（TLS） |

## 工程实践

```bash
sudo apt install mosquitto mosquitto-clients   # Debian/Ubuntu
mosquitto -v
mosquitto_sub -t 'robot/#' -v
```

配置见 `/etc/mosquitto/mosquitto.conf`：listener、password_file、aclfile、bridge 到云端 Broker。

## 局限与风险

- **单进程 Broker** 非集群 HA；生产应评估 EMQX/HiveMQ/云 IoT 或 Mosquitto 桥接拓扑。
- **非运控中间件**：勿用于 1 kHz 关节环，见 [MQTT 协议](../concepts/mqtt-protocol.md)。

## 关联页面

- [MQTT 通信协议](../concepts/mqtt-protocol.md)
- [PlotJuggler](./plotjuggler.md)
- [Wokwi](./wokwi.md)

## 源码运行时序图

入口：`mosquitto -c mosquitto.conf`（主函数在 `src/mosquitto.c` 的 `main`；`-c` 指定配置文件，可加 `-v` 等参数）

```mermaid
sequenceDiagram
    participant MAIN as src/mosquitto.c main
    participant CFG as src/conf.c config__parse_args
    participant SEC as src/security_default.c 与 plugin 安全接口
    participant LST as src/listeners.c listeners__start
    participant MUX as src/mux_epoll.c mux__handle
    participant MLOOP as src/loop.c mosquitto_main_loop
    participant NET as src/net.c 与 src/read_handle.c
    participant CONN as src/handle_connect.c
    participant PUB as src/handle_publish.c
    participant SUB as src/handle_subscribe.c 与 src/subs.c

    MAIN->>CFG: config__init 与 config__parse_args 读取配置
    MAIN->>MAIN: drop_privileges、pid__write、db__open、log__init
    MAIN->>MAIN: plugin__load_all，mosquitto_security_init
    MAIN->>MAIN: plugin_persist__handle_restore，sys_tree__init，mux__init
    MAIN->>LST: listeners__start 创建监听 socket 并注册到 mux
    MAIN->>MAIN: bridge__start_all，signal__setup
    MAIN->>MLOOP: mosquitto_main_loop(listensock)
    loop while g_run
        MLOOP->>MLOOP: retain__expiry_check，keepalive__check，session_expiry__check，will_delay__check
        MLOOP->>MUX: mux__handle，即 epoll_wait
        alt 监听 socket 可读
            MUX->>NET: net__socket_accept，再 mux__new 注册新客户端
        else 客户端 socket 可读写
            MUX->>MLOOP: loop_handle_reads_writes
            MLOOP->>NET: packet__read 读入报文
            MLOOP->>NET: handle__packet 按命令类型分派
            alt CONNECT
                NET->>SEC: mosquitto_security_auth_start 认证
                SEC-->>CONN: 认证通过
                CONN->>CONN: connect__on_authorised，处理会话接管与 session
                CONN->>NET: send__connack，db__message_write_queued_out
            else PUBLISH
                NET->>PUB: handle__publish
                PUB->>SEC: mosquitto_acl_check 发布权限检查
                PUB->>PUB: handle__accepted_publish，plugin__handle_message_in
                PUB->>PUB: db__message_store 存储消息，并按订阅投递（subs.c）
                PUB->>NET: send__puback 或对应确认
            else SUBSCRIBE
                NET->>SUB: handle__subscribe
                SUB->>SEC: mosquitto_acl_check 订阅权限检查
                SUB->>SUB: sub__add，retain__queue 投递保留消息
                SUB->>NET: send__suback
            end
        end
        MLOOP->>NET: 有待发报文时 packet__write（EPOLLOUT）
    end
```

图依据：eclipse/mosquitto@83f2620，读过 `src/mosquitto.c`（`main` 的初始化顺序）、`src/loop.c`（`mosquitto_main_loop` 的循环体）、`src/mux_epoll.c`（`mux__handle` 的 accept 与读写分派）、`src/read_handle.c`（`handle__packet` 的 switch）、`src/handle_connect.c`（`connect__on_authorised` 与认证调用）、`src/handle_publish.c`（`handle__publish` 的 ACL 检查与 `handle__accepted_publish`）、`src/handle_subscribe.c`（`handle__subscribe` 的 ACL、`sub__add`、`retain__queue`）；`subs.c` 的投递细节与 `security_default.c` 仅按函数名检索，未逐行通读；未运行代码。

## 参考来源

- [Eclipse Mosquitto 仓库归档](../../sources/repos/mosquitto.md)
- 源码核对（2026-11-02）：eclipse/mosquitto@83f2620，读过 `src/mosquitto.c`、`src/loop.c`、`src/mux_epoll.c`、`src/read_handle.c`、`src/handle_connect.c`、`src/handle_publish.c`、`src/handle_subscribe.c`；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [Mosquitto 官方文档](https://mosquitto.org/documentation/)
