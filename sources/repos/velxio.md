# Velxio — 官方开源仓库与架构核对

> 来源归档（官方代码；2026-11-10）

- **标题：** Velxio: Arduino & Embedded Board Emulator
- **类型：** repo（MCU 固件执行 + 数模混合电路仿真）
- **代码：** https://github.com/davidmonterocrespo24/velxio
- **项目：** https://velxio.dev/
- **许可：** AGPL-3.0（并有商业授权说明 COMMERCIAL_LICENSE.md）
- **核查版本：** `master@8a254109afe4cd17e91fb0aeaabf14ab42c46c4a`（2026-11-10）
- **核对范围：** 完整阅读 README.md、docs/ARCHITECTURE.md 与 docs/emulator.md；追查 frontend/src/App.tsx、frontend/src/pages/EditorPage.tsx、frontend/src/components/editor/EditorToolbar.tsx、frontend/src/simulation/AVRSimulator.ts、frontend/src/simulation/Esp32Bridge.ts、frontend/src/store/useSimulatorStore.ts，及 backend/app/main.py、backend/app/api/routes/compile.py、backend/app/api/routes/simulation.py、backend/app/services/arduino_cli.py 的关键运行入口与分支。未逐行审阅所有源码，也未实际启动 Docker。
- **通读：** 已完成
- **开源状态：** **基础仿真栈已开源，商业扩展未全部开源**；仓库 README 和 FastAPI 的 simulation.py 说明 Raspberry Pi Linux、STM32 的 QEMU 扩展由 `app.pro_boards` 私有包提供；账户/AI 等由 `app.pro` 私有 overlay 提供。Pico W 的部分联网能力也依赖服务端 overlay。
- **沉淀到 wiki：** [Velxio](../../wiki/entities/velxio.md)

## 核心实现：按真实入口追踪

1. **页面入口：** `frontend/src/App.tsx` 路由 `/editor` 到 `EditorPage`；它包含 `EditorToolbar`、`SimulatorCanvas`、`SerialMonitor`。
2. **编译入口：** `EditorToolbar.tsx` 引用 `services/compilation` 的 `compileCode`；`backend/app/main.py` 挂载 `/api/compile`，`backend/app/api/routes/compile.py` 的 `compile_sketch` 接收多文件及 board_fqbn；`ArduinoCLIService.compile` / ESP-IDF 路线生成 .hex/.bin/.uf2。
3. **运行入口：** 前端 `useSimulatorStore` 管理每板运行；AV​​R 用 `AVRSimulator` / avr8js，RP2040 用 rp2040js；ESP32 `Esp32Bridge.ts` 通过 WebSocket 向 `backend/app/api/routes/simulation.py` 的 `simulation_websocket` 发送 `start_esp32`，由 `esp_lib_manager` 或 `esp_qemu_manager` 启动 QEMU 并回传 `gpio_change`、`serial_output`。
4. **组件与电路：** `PinManager` 和 `PartSimulationRegistry` 将 MCU GPIO / 总线事件映射到画布；`NetlistBuilder` 把连线组织成 SPICE 网表，ngspice-WASM 求解电压，ADC 与模拟仪表读取结果。
5. **边界：** 微控制器级仿真与数模协同，非 MuJoCo/Isaac 刚体接触仿真；CPU 仿真也不能代替实机 EMI、硬实时总线和 ADC 噪声测量。

## 官方配置与能力范围

| 运行位置 | 处理器/实现 | 对应板卡示例 |
|---|---|---|
| 浏览器 Web Worker | avr8js / RP2040 JS | Arduino Uno/Nano/Mega/ATtiny85、Pico |
| 后端 QEMU | Xtensa LX6/LX7、RISC-V RV32 | ESP32、ESP32-S3、ESP32-C3 |
| 官网付费扩展 | ARM Cortex-M、Cortex-A 的 QEMU | STM32、Raspberry Pi Linux（不随基础开源 Docker 完整提供） |

README 目前声称 40 块板、5 个 CPU 家族；官网仍显示 35 块板和 6 类架构，数字应注明来源和时间，按实际板型验收。

## 启动与复现

README 给出的开源 Docker 镜像为 `ghcr.io/davidmonterocrespo24/velxio:master`，默认映射 `3080:80`；启动前请看完整 README 中编译缓存 volumes、镜像依赖和受支持板卡限制。**未在本次入库环境运行镜像**，因此不声称实测性能或兼容性。

## 代码与文档依据

- [README](https://github.com/davidmonterocrespo24/velxio/blob/master/README.md)
- [Architecture](https://github.com/davidmonterocrespo24/velxio/blob/master/docs/ARCHITECTURE.md)
- [Emulator Architecture](https://github.com/davidmonterocrespo24/velxio/blob/master/docs/emulator.md)
- [EditorToolbar 源码](https://github.com/davidmonterocrespo24/velxio/blob/master/frontend/src/components/editor/EditorToolbar.tsx)
- [FastAPI 入口](https://github.com/davidmonterocrespo24/velxio/blob/master/backend/app/main.py)
- [编译路由](https://github.com/davidmonterocrespo24/velxio/blob/master/backend/app/api/routes/compile.py)
- [仿真 WebSocket 路由](https://github.com/davidmonterocrespo24/velxio/blob/master/backend/app/api/routes/simulation.py)
- [官方项目页归档](../sites/velxio.md)
