# NavBot-EN01 官方代码与硬件资料｜Official Firmware and Hardware Files

- **类型：** repo
- **仓库：** <https://github.com/fuwei007/Navbot-EN01>
- **固定版本：** `3139f2fb32b38d10611feb905e16f8aa012f2f2e`
- **配套资料：** [作者安装图文与产品页](../sites/navbot-en01-official.md)
- **入库日期：** 2026-11-25
- **核对范围：** 仓库 README、硬件文件树、BOM 工作簿、EasyEDA 工程归档目录、Arduino 草图及网页、网络、协议、舵机和显示任务源码。未编译、刷写、导入 EDA 工程或组装。
- **通读：** 已完成

## 文件与入口

- [机械目录](https://github.com/fuwei007/Navbot-EN01/tree/3139f2fb32b38d10611feb905e16f8aa012f2f2e/Hardware/RebotModel)：`OriginalRobotModel.stp`、STL、碳纤维面板图、铝件 CNC 图及模型、`BOM.xlsx`。保留上游目录实际拼写，不表示已验证装配配合。
- [PCB 目录](https://github.com/fuwei007/Navbot-EN01/tree/3139f2fb32b38d10611feb905e16f8aa012f2f2e/Hardware/PCB)：`.epro` 容器内可见 4 个 PCB 文件及原理图页；作者以 EasyEDA 导入。没有验证板卡修订匹配、电气规则或可制造性。
- [固件](https://github.com/fuwei007/Navbot-EN01/tree/3139f2fb32b38d10611feb905e16f8aa012f2f2e/Src/wl_pro_robot)：从 `wl_pro_robot.ino` 进入；README 指定 Arduino IDE、ESP32 Dev Module 和 Arduino-ESP32 2.0.3。依赖应以仓库版本为起点；OLED 表情文件是独立的 SPIFFS 资源，不是平衡算法。
- `basic_web.h` 与 `wifi.cpp`：设备提供内嵌 HTTP 网页，浏览器用 WebSocket 发送 JSON；AP 是机器人热点，STA 是接入已有的网络。`wl_pro_robot.ino` 的本地网页事件调用 `RobotProtocol::parseBasic()`，后者更新 `wrobot` 和滚转偏置。
- `wl_pro_robot.ino` / `Servo_STS3032.cpp`：主循环先处理网页，再更新 IMU，计算平衡、转向和腿姿，随后执行左右轮 `loopFOC()` / `move()`。腿位置用总线同步写入。源码把轮电机配置为 torque 模式下的 voltage 控制，不等同于闭环电流或实测轮端力矩。
- README 署名感谢 [MuShibo / Li Yufeng 的 Micro-Wheeled_leg-Robot](https://github.com/MuShibo/Micro-Wheeled_leg-Robot)。未见 EN01 顶层许可证，依赖目录的许可证不自动覆盖项目自有代码、机械图、PCB 或照片；知识库仅保存摘要与链接。
- `BOM.xlsx` 的电池购物条目为 3.7 V，固件电机驱动的供电参数设为 8 V，电量提示从 7.2 V 附近分段；作者装配文另给出与仓库 README 不同的电池接口。上述值不能合成为一套安全的接线或采购方案。

## 对 wiki 的映射

- [NavBot-EN01 实体](../../wiki/entities/navbot-en01.md)
