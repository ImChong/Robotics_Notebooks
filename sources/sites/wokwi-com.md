# Wokwi（wokwi.com）

> 来源归档

- **标题：** Wokwi — Online Electronics Simulator
- **类型：** site（在线嵌入式仿真平台）
- **来源：** Wokwi Ltd.
- **链接：** https://wokwi.com/
- **文档：** https://docs.wokwi.com/
- **入库日期：** 2026-06-24
- **一句话说明：** 浏览器端 **嵌入式电子电路仿真器**：支持 Arduino、ESP32、STM32、Raspberry Pi Pico 等 MCU 与大量传感器/外设；提供 Wi-Fi 仿真、逻辑分析仪、GDB 调试、VS Code/CLion 插件与 **wokwi-cli** CI 集成；ESP-IDF 官方列为第三方仿真工具。
- **沉淀到 wiki：** [Wokwi](../../wiki/entities/wokwi.md)、[电机底软通信总览](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

---

## 平台定位（官网 / 文档摘要）

- **不是** 机器人物理引擎（MuJoCo / Isaac 类），而是 **MCU + 外围电路 + 总线协议** 的在线 bring-up 与教学仿真。
- 个人使用免费；商业与团队功能见 [Pricing](https://wokwi.com/pricing)。
- 与 [Tiny Tapeout](https://tinytapeout.com/) 合作：可在浏览器设计数字电路并流片（偏芯片设计教育，与机器人固件 bring-up 不同层）。

---

## 支持硬件（docs：Supported Hardware）

| 架构 | 代表 MCU / 板卡 |
|------|-----------------|
| **AVR** | ATmega328P（Arduino Uno/Nano）、ATmega2560（Mega）、ATtiny85 |
| **ESP32** | Xtensa：ESP32 / S2 / S3；RISC-V：C3 / C5* / C6 / C61 / H2 / P4 / S31* |
| **STM32** | STM32C031、STM32L031、STM32F103C8 |
| **Pi Pico** | RP2040（双核 Cortex-M0+） |

\* C5、S31 为 alpha；P4 为 beta。

另仿真大量 **传感器、显示器、舵机驱动、I2C/SPI/UART 外设** 与社区 **Chips API** 自定义元件。

---

## 差异化能力（文档 Unique Features）

| 能力 | 说明 |
|------|------|
| **Wi-Fi 仿真** | 模拟项目可连互联网；支持 MQTT、HTTP、NTP 等（IoT / 遥测原型） |
| **虚拟逻辑分析仪** | 抓取 UART / I2C / SPI 等数字波形并在本机分析 |
| **GDB 调试** | 浏览器内 Web GDB；Arduino / Pico 高级调试 |
| **SD 卡仿真** | 代码读写文件系统；付费用户可上传二进制资产 |
| **VS Code 集成** | [Wokwi for VS Code](https://marketplace.visualstudio.com/items?itemName=wokwi.wokwi-vscode) 本地编辑 + 仿真 |
| **CLion 插件** | Espressif IDE 2.x+ 内置 Wokwi；ESP-IDF 官方文档推荐 |
| **wokwi-cli + CI** | YAML 场景自动化测试、截图回归；`pytest-embedded-wokwi` 对接 ESP-IDF 测试框架 |
| **Wokwi Classroom** | 高校教学授权与课程项目模板 |

---

## 与机器人栈的关系

| 场景 | 价值 |
|------|------|
| **固件 bring-up** | 在焊板前验证 I2C IMU、UART 日志、PWM 舵机、编码器读数 |
| **FOC / 电机原型** | 与 [SimpleFOC](../../wiki/entities/simplefoc.md) 等同栈：Arduino / ESP32 / STM32 上跑 `loopFOC()` 与总线外设 |
| **IoT 遥测** | ESP32 Wi-Fi + MQTT 模拟机器人状态上报，无需实物热点环境 |
| **教学** | 开源人形（如 InMoov 的 Arduino 栈）与创客课程的低门槛入口 |
| **与 PiL 对照** | 轻量 **固件在环** 冒烟测试；**不能** 替代 MuJoCo + 生产固件 CAN 抖动的 [处理器在环 Sim2Real](../../wiki/concepts/processor-in-the-loop-sim2real.md) |

---

## 外部权威引用

- [ESP-IDF：Wokwi 第三方工具](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/third-party-tools/wokwi.html)
- [Wokwi Docs — Welcome](https://docs.wokwi.com/)
- [GDB Debugging](https://docs.wokwi.com/gdb-debugging)
- [Supported Hardware](https://docs.wokwi.com/getting-started/supported-hardware)
