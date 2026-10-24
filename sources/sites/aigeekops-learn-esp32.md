# Learn ESP32 · ESP32 交互学习站

> 来源归档

- **类型：** site（ESP32 交互课程）
- **入口：** <https://aigeekops.github.io/learn-esp32/>
- **官方源码：** <https://github.com/aigeekops/learn-esp32>
- **许可证：** MIT（应用源码；第三方依赖与素材依各自声明）
- **核查日期：** 2026-10-24
- **一句话说明：** 面向 ESP32-DevKitC V4 / ESP32-WROOM-32E 的浏览器交互学习站，以 3D 板卡、参数调整、接线图和 Arduino C++ 示例讲解 16 个基础实验。
- **相关知识页：** [Wokwi](../../wiki/entities/wokwi.md)（用于对照固件仿真与交互式课程）

## 课程范围

课程从板卡、LED、串口、按钮、ADC、PWM 和温度传感器开始，继续讲 I²C OLED、蜂鸣器、舵机、状态机告警，再到 Wi-Fi、HTTP、网页控制与综合温度告警项目。README 指定 Arduino-ESP32 3.3.12、`esp32:esp32:esp32` 板型和 115200 baud 串口；OLED 示例固定依赖 Adafruit SSD1306、GFX 与 BusIO 版本。

## 开源状态与边界

**已开源：** 对应 GitHub 仓库公开，标注 MIT。课程页面运行参数化教学模型，不执行任意 C++，不连接真实 ESP32 或电脑 Wi-Fi，也不求解面包板电气连通。项目 README 记录了 16 份默认固件编译通过；这是作者提供的、有日期的编译记录，不代表真实硬件接线与运行已验证。

该站适合作为 ESP32 入门课程与固件代码样例。若需要逐周期验证 MCU 外设、编译运行固件或做自动化测试，应使用目标硬件或适合的固件仿真工具，不能把该站的动画模拟等同于电路仿真。

## 一手来源

- [Learn ESP32 官方课程站](https://aigeekops.github.io/learn-esp32/)
- [官方源码仓库 README](https://github.com/aigeekops/learn-esp32/blob/main/README.md)
- [课程定义与 16 课内容](https://github.com/aigeekops/learn-esp32/blob/main/docs/CURRICULUM.md)
- [开发验收记录与验证边界](https://github.com/aigeekops/learn-esp32/blob/main/docs/DEVELOPMENT_REPORT.md)
- [硬件、框架与课程参考资料](https://github.com/aigeekops/learn-esp32/blob/main/docs/REFERENCES.md)
