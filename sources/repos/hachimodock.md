# HachimoDock 官方软件与固件 / Official software and firmware

- **类型：** repo
- **代码：** https://github.com/YizhengWw/HachimoDock
- **项目页：** https://oshwhub.com/eda_gqvzlprk/project_cnbmkbjc
- **版本：** 7fbe729aa52911da5fdd7e8c3c0578d4a4d8f60f（2026-11-07 获取）；根 README / package.json 为客户端 0.1.201，根 README 配套固件 R181；子目录 README / BUILD 的版本说明仍为 0.1.199 / R179，不混作同一发布包证明。
- **核对范围：** 完整阅读根 README、LICENSE、THIRD_PARTY_NOTICES、pc/README.md、firmware/BUILD.md；阅读 pc/package.json、pc/src-tauri/src/main.rs、lib.rs 的 run 初始化与 start_usb_state_forwarder 状态转发路径，以及 firmware/main/pet_p4_main.c 的 app_main、UART 接收、行组装、协议 worker、渲染循环，pet_p4_protocol.c 的 JSON 分发与 handle_state_topic。不是全仓审计，未编译或实机验证。
- **通读：** 已完成
- **开放状态：** 软件与固件源码公开可获取，属于非商业源码可见；不是标准自由开源许可。未发现需要另建模型权重或数据集节点的独立发布。
- **许可：** HachimoDock Project License 1.0（2026-09-04）；第三方材料保留其原许可；MiLoCo 实现另受 Xiaomi Miloco License 约束。
- **入库日期：** 2026-11-07
- **沉淀到 wiki：** [HachimoDock](../../wiki/entities/hachimodock.md)

## 技术要点

电脑运行 Agent、语音服务与桌面管理端；ESP32-P4 承担显示、输入、音频及有界组件运行。Tauri 2 + React 客户端支持 Windows x64 和 macOS Apple silicon。公开版本不提供服务 Key，语音、对话、生成服务需用户配置。

客户端入口为 `pc/src-tauri/src/main.rs::main` → `pet_manager_tauri_lib::run`，启动 USB manager、heartbeat 和状态 forwarder。固件入口 `firmware/main/pet_p4_main.c::app_main` 初始化状态、存储、显示及接收任务；`uart_rx_task` 接收字节，行组装后入队；`protocol_worker_task` 调用 `pet_p4_handle_line_with_reply` 或配置对应的普通分发路径。`state/` 消息更新状态，主循环构建 view model、复制渲染快照并在状态锁外渲染。

## 官方运行入口

- PC：进入 `pc/`，`npm ci`；按 pc/README.md 准备原生运行依赖后 `npm run dev`。`npm run dev:web` 只有 UI，不能当作设备与 Agent 功能验证。
- 固件：Python 3.10+、PlatformIO Core 6.1.19；进入 `firmware/`，按芯片运行 `python -m platformio run -e esp32_p4_evboard` 或 `esp32_p4_evboard_v3`。
- 首次烧录用匹配芯片的完整镜像；v1 / v3 不能互刷。普通 OTA 不改变旧分区，完整烧录会清数据。

## 来源与交叉索引

- [固定版本 README](https://github.com/YizhengWw/HachimoDock/blob/7fbe729aa52911da5fdd7e8c3c0578d4a4d8f60f/README.md)
- [固定版本固件入口](https://github.com/YizhengWw/HachimoDock/blob/7fbe729aa52911da5fdd7e8c3c0578d4a4d8f60f/firmware/main/pet_p4_main.c)
- [硬件页面归档](../sites/hachimodock.md)
