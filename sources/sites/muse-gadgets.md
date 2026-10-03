# gadgets.muse.ai（Muse Gadgets 官方项目页）

> 来源归档（site / project page）

- **标题：** Muse Gadgets
- **类型：** 官方项目页 / 开源设备生态
- **官方入口：** <https://gadgets.muse.ai/>
- **代码：** <https://github.com/facebookincubator/muse-gadget-sdk>
- **入库日期：** 2026-10-03
- **一句话说明：** Meta 的 Muse 助手设备扩展入口，提供 ESP32 与 Linux SDK，让自制屏幕、按钮、传感器和本地服务接入 Muse。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 官网 | <https://gadgets.muse.ai/> |
| SDK 与固件 | <https://github.com/facebookincubator/muse-gadget-sdk> |
| ESP32 SDK | <https://github.com/facebookincubator/muse-gadget-sdk/tree/main/esp32> |
| Linux SDK | <https://github.com/facebookincubator/muse-gadget-sdk/tree/main/linux> |
| SDK Token 设置 | <https://gadgets.muse.ai/settings/sdk-tokens> |
| SDK 条款 | <https://gadgets.muse.ai/sdk-terms> |

## 开源核查

- 官网将 ESP32 与 Linux Device SDK 链接到公开 GitHub 仓库；仓库包含设备 SDK 与固件，许可证为 Apache-2.0（少数第三方文件保留其上游许可证）。
- 每台设备配对需要 SDK token，并通过 Muse 手机应用的 Developer mode 添加设备。官网文案称设备 SDK 与固件按原样提供且不附带保证。
- 这是设备接入与助手工具调用项目，不是机器人策略、ROS 驱动或具身数据集；本库只将其作为边缘设备交互和权限边界案例留档。

## 对 wiki 的映射

- [Muse Gadgets 实体页](../../wiki/entities/muse-gadgets.md)
- [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)
- [官方 SDK 仓库](../repos/muse-gadget-sdk.md)
