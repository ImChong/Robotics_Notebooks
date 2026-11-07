# PyLabRobot 官方文档站（docs.pylabrobot.org）

> 来源归档

- **标题：** PyLabRobot documentation（stable）
- **类型：** site
- **链接：** https://docs.pylabrobot.org/stable/
- **页面：** [首页](https://docs.pylabrobot.org/stable/) · [Supported Machines](https://docs.pylabrobot.org/stable/user_guide/machines.html) · [Using the Visualizer](https://docs.pylabrobot.org/stable/user_guide/machine-agnostic-features/using-the-visualizer.html) · [Installation](https://docs.pylabrobot.org/stable/user_guide/getting-started/installation.html)
- **代码：** https://github.com/PyLabRobot/pylabrobot（MIT，见 [pylabrobot 仓库归档](../repos/pylabrobot.md)）
- **论文：** [pylabrobot_device_2023.md](../papers/pylabrobot_device_2023.md)
- **抓取日期：** 2026-11-07（文档站随开发更新，machines 表为动态生成，数据以抓取日为准）
- **核对范围：** 完整读了首页、Supported Machines（全表逐行含支持程度，约 950 行文本，其中 Mettler Toledo 型号子行逐条为 WIP）、Using the Visualizer（全文）、Installation（安装与依赖分组部分）；Liquid Handling 总览页只有 "TODO: remove this page"，无内容。**文档站其余大量页面（各机器教程、API 参考、贡献者指南等）未逐页阅读**，架构细节改由仓库源码静态阅读补足
- **通读：** 未完成

## 要点

### 首页
- 自述为 "hardware- and operating system-agnostic, pure Python SDK for Automated & Autonomous Laboratories"。
- 说明 PyLabRobot 不同于 PyHamilton：后者是仅 Windows 的 Hamilton VENUS 接口。引用格式给出 Device 论文 DOI。

### Supported Machines（machines.html）
- 表列厂商 / 设备 / 能力 / 支持程度 / 驱动代（v0 或 v1）/ 链接 / 维护者。
- 支持程度定义：**WIP**（含等待首次硬件验证的型号）、**Basic**（核心功能可用、已并入 main）、**Mostly**（多数能力可用，仍缺已知命令）、**Full**（≥90% 能力，带文档）。
- **v0 驱动**位于 `pylabrobot.legacy`，正在迁移到 v1。
- 液体处理机（能力含 liquid handling）逐台：Hamilton STAR / STARlet——Full，v0；Hamilton Vantage——Mostly，v0；Hamilton Nimbus——Mostly，v0；Hamilton Prep——WIP；Opentrons OT-2——Mostly，v1；Opentrons Flex——Basic，v1（带备注星号）；Tecan EVO——Basic，v0。
- 其他设备共覆盖读板器、离心机、热循环仪、储存、摇床、机械臂（Brooks PreciseFlex、UFACTORY xArm 6、PAA KX2）、泵、天平等数十种，多数 Full / Mostly / WIP 不一；Mettler Toledo MT-SICS 天平含上百个型号子行，几乎全为 WIP。
- 表由 `docs/_static/devices.json` 生成，新增机器只需加一个 JSON 对象。

### Using the Visualizer
- Visualizer 可视化一个 `Resource`（如 `LiquidHandler`）及其状态；配合不需要真机的 `LiquidHandlerChatterboxBackend` 即可模拟机器人行为，也可与真机并用。
- 流程：建 `LiquidHandler(backend=LiquidHandlerChatterboxBackend(), deck=STARLetDeck())` → `await lh.setup()` → `vis = Visualizer(resource=lh)`；`await vis.setup()` 后在 `127.0.0.1:2121` 起 websocket 服务、`127.0.0.1:1337` 起文件服务，浏览器打开后者。
- 往 deck 上 `assign_child_resource` 即自动出现；用 `fill()`、`set_tip_state()`、`set_well_volumes()` 配置初始状态；孔的不透明度与相对最大体积的装满程度成正比。
- 须全局开启 `set_tip_tracking(True)` 与 `set_volume_tracking(True)`，可视化才跟踪 tip 与体积；之后 `pick_up_tips`、`aspirate`、`dispense`、`return_tips` 与 CoRe 96 的 `*96` 方法的状态变化会反映到页面；`await vis.stop()` 收尾。
- 文档页自身仍有 "Setting up a connection with the robot: TODO"。

### Installation
- 需要 Python ≥ 3.10；`pip install pylabrobot`；按通信方式装可选依赖组：serial、usb（Hamilton STAR/STARlet、Tecan EVO 固件）、ftdi、hid、modbus、opentrons、sila 等。

## 对 wiki 的映射

- [PyLabRobot 项目节点](../../wiki/entities/pylabrobot.md)：受支持液体处理机清单、Visualizer 工程实践、安装步骤。
