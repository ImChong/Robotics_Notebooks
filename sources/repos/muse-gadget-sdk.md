# facebookincubator/muse-gadget-sdk（Muse Gadgets SDK）

> 来源归档（repo）

- **标题：** Muse Gadgets
- **代码：** <https://github.com/facebookincubator/muse-gadget-sdk>
- **类型：** ESP32 固件与 Linux 设备 SDK
- **License：** Apache-2.0（少数第三方文件按各自上游许可）
- **首次入库：** 2026-10-03
- **关联项目页：** [gadgets.muse.ai](../sites/muse-gadgets.md)

## 一句话摘要

公开 SDK 把 ESP32 板卡或 Linux / Raspberry Pi 主机配对为 Muse 外设：ESP32 侧可接显示屏、按钮和音频；Linux 侧可扩展命令、文件操作、传感器桥接与本地 HTTP 服务。

## 代码入口

| 模块 | 入口 | 用途 |
|------|------|------|
| ESP32 Device SDK | <https://github.com/facebookincubator/muse-gadget-sdk/tree/main/esp32> | 构建和刷写面向多种 ESP32 板卡的固件，可接屏幕、按键、麦克风 / 扬声器等 |
| Linux Device SDK | <https://github.com/facebookincubator/muse-gadget-sdk/tree/main/linux> | 将 Linux 主机配对为设备，提供命令扩展、文件访问和 Muse 消息发送接口 |
| 项目说明 | <https://github.com/facebookincubator/muse-gadget-sdk/blob/main/README.md> | 配对前置条件、许可证与设备 SDK 概览 |

## 工程与安全边界

- 配对依赖 SDK token 与 Muse 手机应用；每次设置会建立加密会话。ESP32 文档提醒设备配对没有厂商身份验证，不能防止主动中间人攻击。
- Linux 命令以安装账户权限运行。官方文档明确指出：若该账户能使用 sudo，Muse 执行命令时也能使用 sudo；安装脚本应先阅读，生产主机宜用专用低权限账户、限制命令与网络访问。
- SDK token 会随 ESP32 固件部署；官方建议支持的板卡启用 NVS 加密，以保护闪存中的 Wi-Fi 凭据与 token。
- 公开代码范围是设备端 SDK / 固件；未见项目页声称开放 Muse 助手模型或云端服务实现。

## 对 wiki 的映射

- [Muse Gadgets 实体页](../../wiki/entities/muse-gadgets.md)
- [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)
- [项目页归档](../sites/muse-gadgets.md)
