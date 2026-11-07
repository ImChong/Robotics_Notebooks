# PyLabRobot 仓库（PyLabRobot/pylabrobot）

> 来源归档

- **标题：** PyLabRobot/pylabrobot
- **类型：** repo
- **链接：** https://github.com/PyLabRobot/pylabrobot
- **许可：** MIT（`LICENSE`，Copyright (c) 2018 dgretton；README 称 "open-source and free to use"）
- **核对版本：** `git clone --depth 1`，HEAD `1f7d93615cad2f6ac24664926716417b740af224`（2026-11-06），`pylabrobot/version.txt` = `1.0.0b1`，Python ≥ 3.10；静态阅读，**未运行任何代码**
- **论文：** [pylabrobot_device_2023.md](../papers/pylabrobot_device_2023.md)；**文档：** [pylabrobot_docs.md](../sites/pylabrobot_docs.md)
- **入库日期：** 2026-11-07
- **核对范围：** 读了 README、CHANGELOG 开头、CITATION.cff、`pyproject.toml` 关键字段、目录结构；源码重点读了 `legacy/liquid_handling/liquid_handler.py`（`__init__`、`setup`、`aspirate` 到 backend 的调用与 tracker 提交 / 回滚段）、`legacy/machines/machine.py`、`legacy/liquid_handling/backends/backend.py`（抽象方法表）、`hamilton/base.py`（`send_command`、`_write_and_read_command`）、`STAR_backend.py`（`setup`、`aspirate` 后半段、`aspirate_pip`）、`opentrons_backend.py`（`setup`、`aspirate` 片段）、`tecan/EVO_backend.py`（方法列表）、`chatterbox.py`、`visualizer/visualizer.py`（构造、回调与 setup 入口）、`hamilton/star/design.md`；**未读** 全部 4 万余行 STAR 固件命令实现、其余设备驱动、测试细节
- **通读：** 未完成

## 目录结构（HEAD）

- `pylabrobot/legacy/`：v0 驱动与前端（`liquid_handling/`、`machines/`、`plate_reading/` 等）；`pylabrobot/liquid_handling/__init__.py` 与 `pylabrobot/machines/__init__.py` 只是带 DeprecationWarning 的转发到 `legacy`。
- `pylabrobot/resources/`：`Resource`、`Deck`、`Carrier`、`Container`、`ItemizedResource`、`Plate`、`TipRack`、`VolumeTracker`（`volume_tracker.py`）、各厂商 labware。
- `pylabrobot/hamilton/`（v1 风格：`star/device.py`、`star/driver/master.py` 的 `STARDriver`、`features/`、`simulator.py`）、`pylabrobot/opentrons/`（`ot2/`、`flex/`、HTTP `api.py`）。
- `pylabrobot/io/`（`usb.py`、`serial.py`、`ftdi.py`、`http.py`、`socket.py`、`sila/`、`validation.py`、`capture.py`）；`pylabrobot/visualizer/`、`visualizer3D/`；`rpc/`；`events/`。
- 另有 `docs/`（Sphinx）、`AGENTS.md`、`CONTRIBUTING.md`、`CHANGELOG.md`、`CITATION.cff`。

## 后端 / 机器抽象（真实模块）

- `legacy/machines/machine.py`：`Machine(SerializableMixin, ABC)`，`setup()` 调 `backend.setup()` 后置 `_setup_finished`；`need_setup_finished` 装饰器；支持 `async with`。
- `legacy/liquid_handling/liquid_handler.py`：`LiquidHandler(Resource, Machine)`，`__init__(backend, deck, ...)`，`setup()` 先 `backend.set_deck`、`set_heads`，再 `Machine.setup`，为每个通道建 `TipTracker`；`pick_up_tips` / `drop_tips` / `aspirate` / `dispense` 及 `*96`、`transfer`、`return_tips`。
- `legacy/liquid_handling/backends/backend.py`：`LiquidHandlerBackend(MachineBackend, ABCMeta)`，抽象方法 `pick_up_tips`、`drop_tips`、`aspirate`、`dispense`、`pick_up_tips96`、`drop_tips96`、`aspirate96`、`dispense96`、`pick_up_resource`、`move_picked_up_resource`、`drop_resource`、`num_channels`、`can_pick_up_tip`。
- 具体后端：`backends/hamilton/STAR_backend.py`（`STARBackend(HamiltonLiquidHandler, ...)`，`hamilton/base.py` 里经 `USB` 的 `io.write` 发固件命令字符串）、`vantage_backend.py`、`nimbus_backend.py`、`opentrons_backend.py`（`OpentronsOT2Backend`，经 Opentrons HTTP run API）、`tecan/EVO_backend.py`（`EVOBackend`）、`chatterbox_backend.py`（`LiquidHandlerChatterboxBackend`，只打印）、`serializing_backend.py`。
- 标准数据类：`legacy/liquid_handling/standard.py`（`Pickup`、`Drop`、`SingleChannelAspiration`、`SingleChannelDispense`、`PickupTipRack`、`MultiHeadAspirationPlate` 等）。
- 论文的 `SimulatorBackend` / `WebsocketBackend` 在 HEAD 的 `.py` 中检索不到；浏览器可视化现为 `pylabrobot/visualizer/visualizer.py` 的 `Visualizer`（websocket 端口默认 2121、文件服务默认 1337，订阅资源的 assign / unassign 与 state 更新回调）。

## 入口与运行方式

- 库为主，无单一训练 / 推理脚本；入口是 Python 中构造 `LiquidHandler(backend=..., deck=...)` 并 `await lh.setup()`（README 示例同此）。另有 `pylabrobot/cli.py`。
- 安装：`pip install pylabrobot`，硬件另装 extras（见官方文档归档）。

## 对 wiki 的映射

- [PyLabRobot 项目节点](../../wiki/entities/pylabrobot.md) 的"源码运行时序图"与"工程实践"。
