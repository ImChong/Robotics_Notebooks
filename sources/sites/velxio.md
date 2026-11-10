# Velxio — 官方平台与订阅信息

> 来源归档（官方站点；2026-11-10 核对）

- **标题：** Velxio — Arduino & Embedded Board Emulator
- **类型：** site（浏览器端嵌入式电路仿真）
- **项目主页：** https://velxio.dev/
- **官方文档：** https://github.com/davidmonterocrespo24/velxio/tree/master/docs
- **定价：** https://velxio.dev/pricing
- **代码：** https://github.com/davidmonterocrespo24/velxio
- **作者一手文章：** https://developer.espressif.com/blog/2026/07/velxio-browser-based-esp32-simulation/
- **核对范围：** 官方首页主要功能与硬件目录、完整 Pricing 方案表、作者发表于 Espressif Developer Portal 的技术文章（核心工作原理、ESP32 例程、Agent 与自部署章节）、仓库 README 与 docs/ARCHITECTURE.md；以 2026-11-10 网页和 GitHub 当前状态为准。
- **通读：** 已完成
- **开源结论：** **核心开源，官网商业增强层非全部开源**。上游仓库以 AGPL-3.0 提供仿真内核、前后端和基础编译桥；账户、Agent、部分高级/付费板卡能力采用私有 overlay。自部署能力不等同官网全部板卡。
- **沉淀到 wiki：** [Velxio](../../wiki/entities/velxio.md)

## 官方能力与技术路线

- **代码执行：** Arduino C++ / ESP-IDF / MicroPython / Python（按板卡区分），由真实工具链产生固件，而不是只解释脚本。
- **CPU：** AVR8、ARM Cortex-M、Xtensa、RISC-V、ARM Cortex-A 五个处理器家族；AVR 与 RP2040 走浏览器内模拟器，ESP32 与 Linux 单板走后端 QEMU，统一在画布展示 GPIO/串口等事件。
- **电气：** ngspice 编译成 WASM，根据导线/元件生成网表并计算模拟节点电压；ADC 与仪表从计算节点获取结果。
- **示例：** LED、I2C/SPI 传感器、OLED/TFT、ESP32 Wi-Fi/MQTT。作者的一手文章演示 ESP32 Weather Station（BMP280 + DHT22 + ILI9341），不是独立实测基准。

## 数量口径与风险

- 官网页面截至入库时显示 **35 boards / 6 CPU architectures**，将 Linux 也列为 architecture；**Linux 是操作系统，不是 CPU ISA**。
- GitHub 最新 README（master @ 8a254109afe4cd17e91fb0aeaabf14ab42c46c4a）显示 **40 boards / 5 CPU families**。官方两处数字不一致，不能当成精确的当前兼容清单。
- 官方价格以 **年付折算每月** 显示：Free $0；No AI $3/月（$36/年）；Maker $5.75/月（$69/年）；Pro $15.75/月（$189/年）。月付实际价格、增值税及随时间变化须在结账页复核。
- Free 提供基础编译仿真及有限 AI 额度；Maker 提供 Agent/Tutor，Pro 才提供私有项目、BOM 导出与 GitHub Sync。免费版某些板卡与 ESP32 运行时间有限制。
- 有效性边界：仿真验证外设协议与算法逻辑，不等于实机时序、电磁兼容、模拟寄生参数和制造验证。

## 一手资料

- [Velxio 首页](https://velxio.dev/)
- [Velxio 官方定价](https://velxio.dev/pricing)
- [Espressif Developer Portal — 创作者 David Montero Crespo 的技术介绍（2026-07-31）](https://developer.espressif.com/blog/2026/07/velxio-browser-based-esp32-simulation/)
- [GitHub README](https://github.com/davidmonterocrespo24/velxio/blob/master/README.md)
- [源码归档](../repos/velxio.md)
