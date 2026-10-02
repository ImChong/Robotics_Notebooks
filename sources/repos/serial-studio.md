# Serial Studio：硬件遥测与实时仪表盘

- **类型：** repo
- **URL：** <https://github.com/Serial-Studio/Serial-Studio>
- **项目页：** <https://serial-studio.com/>；[站点核查](../sites/serial-studio.md)
- **维护者：** Alex Spataru
- **入库日期：** 2026-10-02
- **核查分支：** master；功能描述来自入库时 README，稳定发布版本可能不同。
- **一句话说明：** 用项目文件定义硬件数据格式，把串口、网络及工业协议遥测变成实时曲线、仪表和可导出日志。

## 原始资料入口

- [README](https://github.com/Serial-Studio/Serial-Studio/blob/master/README.md)
- [许可说明](https://github.com/Serial-Studio/Serial-Studio/blob/master/LICENSE.md)
- [GPL / Pro 功能矩阵](https://github.com/Serial-Studio/Serial-Studio/blob/master/doc/help/Pro-vs-Free.md)
- [上手指南](https://github.com/Serial-Studio/Serial-Studio/blob/master/doc/help/Getting-Started.md)
- [示例](https://github.com/Serial-Studio/Serial-Studio/tree/master/examples)
- [发布下载](https://github.com/Serial-Studio/Serial-Studio/releases/latest)

## 核查结论

**部分开源：GPL-3.0-or-later 核心可自行构建；Pro 模块源码可见但属于专有许可。** 文件 SPDX 声明决定许可，不能把整个仓库或官方二进制视为 GPL。默认 GPL 构建不包含 Pro 模块；官方二进制含 Pro，适用 EULA 与 14 天试用。

| 能力 | GPL 核心 | Pro |
|---|---|---|
| 接入 | UART、BLE、TCP/UDP/WebSocket/HTTP | MQTT、Modbus、CAN、OPC UA 等工业驱动；多设备 |
| 解析 | Built-In 模板、JavaScript、Lua；逐数据集变换 | 供应商寄存器表、CAN DBC 导入 |
| 显示 | 曲线、仪表、FFT、IMU、GPS 等 | 3D/XY、瀑布图、图像、Canvas、输出控件 |
| 记录 | CSV 导出 | SQLite 会话录制回放、MDF4、会话报告 |

## README 摘录归纳

1. Quick Plot 直接接受逗号分隔数据；Project Editor 定义 groups、datasets、widgets，适合固化机器人调试布局。
2. 解析器将原始帧转为数据集；逐字段变换可执行缩放、标定、滤波与单位转换。
3. 桌面端跨 Windows、macOS、Linux 与 Raspberry Pi；源码构建使用 C++20、Qt 6.9+、CMake 3.20+。
4. TCP、gRPC 与 MCP 提供自动化接口；接口能力不代表实时控制时限保证，也不解除 Pro 功能许可边界。
5. CAN/DBC 是 Pro 能力；MIT 电机协议需按实际设备定义解码，不能仅凭支持 CAN 推断直接兼容。

## 对 wiki 的映射

- [Serial Studio 实体页](../../wiki/entities/serial-studio.md)：输入输出机制、单关节示例、工具分工与许可边界。
- [PlotJuggler](../../wiki/entities/plotjuggler.md)：ROS 时序分析与硬件遥测仪表盘的选型关系。
