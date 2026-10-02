# Serial Studio 官方项目页

- **类型：** site / project-page
- **URL：** <https://serial-studio.com/>
- **代码：** <https://github.com/Serial-Studio/Serial-Studio>；[仓库归档](../repos/serial-studio.md)
- **入库日期：** 2026-10-02
- **一句话说明：** 官方硬件遥测仪表盘入口，展示机器人、电机、IMU、工业总线的可视化与记录应用。

## 项目页开放核查

| 核查项 | 结论 |
|---|---|
| GitHub 入口 | 官方页列出公开仓库链接 |
| 开放程度 | **部分开源**：GPL 核心与专有 Pro 模块共存；依据上游 LICENSE.md 的逐文件 SPDX 规则 |
| 核心接口 | UART、TCP/UDP、BLE 标为 FREE / GPL |
| 工业接口 | CAN、Modbus、MQTT、OPC UA 等标为 Pro |
| 官方下载 | 含 Pro 的官方二进制，不能等同于自行编译的 GPL 核心 |
| 训练数据/模型 | 本条为工程工具，不涉及训练权重或数据集发布 |

## 页面要点

- 接收设备字节流，切帧、解析、生成仪表盘，同时记录数据供复盘。
- 机器人场景展示关节角、编码器、电流与 IMU；可用 CSV 快速验证数据链路。
- 工具提供项目编辑器和 API，适合构建调试台；付费能力以当前功能矩阵及安装版本为准。

## 关联资料

- [仓库核查与许可](../repos/serial-studio.md)
- [Serial Studio 知识页](../../wiki/entities/serial-studio.md)
