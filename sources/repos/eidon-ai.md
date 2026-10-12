# Eidon AI 开源仓库

- **类型：** repo collection（可穿戴采集硬件 / 手指追踪 / 仿真与可视化）
- **官方入口：** <https://github.com/Eidon-AI>
- **项目页：** [Eidon AI Robotics Data](../sites/eidon-ai.md)
- **核查日期：** 2026-10-11
- **开源状态：** 官方归档页列出三个公开仓库；按仓库和组件分别核对许可证。
- **沉淀到 Wiki：** [Eidon Tracker + POV 数据集](../../wiki/entities/eidon-tracker-pov-dataset.md)

## 仓库入口

| 仓库 | 作用与可复用内容 | 许可/边界 |
|------|------|------|
| [eidon-tracker](https://github.com/Eidon-AI/eidon-tracker) | 电池供电 IMU→Bluetooth Tracker；含固件、设备/外壳文件和 Web 端姿态可视化。README 给出 XIAO nRF52840 Sense、BNO08x、PlatformIO 构建流程。 | README 标注 MIT；硬件组装、校准和数据协议以仓库说明为准。 |
| [eidon-glove](https://github.com/Eidon-AI/eidon-glove) | 16 个 Hall-effect 传感器追踪手指关节；含 ESP32-C3 固件、PCB/多路复用器、可打印结构件和 BLE HID 主机端工具。 | README 要求逐组件核对许可证；并注明其参考 Project-Homunculus、finger-tracker 和 glove-v3。 |
| [eidon-sim](https://github.com/Eidon-AI/eidon-sim) | 浏览器端设备接入和可视化工具：WebHID 读取 Tracker/Glove，显示骨架和向量链，通过七轴求解器生成手臂关节角；静态 Vite/TypeScript/Three.js 应用。 | README 标注 MIT；这是运动学/可视化工具，不等于机器人控制器或原采集后端。 |

## 官网提到但不是独立公开仓库的服务

- **Eidon Sym：** 官网称历史数据管线曾用 VLM 进行每日质量控制、过滤、自动标记并导出仿真兼容格式；官网将 Eidon Sim 说明为其开源后继项目。应按 Eidon Sim 仓库当前文档理解其实际功能。
- **iOS/Android 采集 App：** 官网说该 App 曾用于设备配对、视频/传感器同步和纯视频采集，但所依赖的后端已关闭；官方归档页未提供一个可替代该后端的运行服务。

## 对 Wiki 的映射

- [Eidon Tracker + POV 数据集实体页](../../wiki/entities/eidon-tracker-pov-dataset.md)
- [Eidon 项目页归档](../sites/eidon-ai.md)
- [数据集卡与使用说明](../datasets/eidon-tracker-pov.md)
