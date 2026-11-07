# PyLabRobot：硬件无关的液体处理机器人接口（Device 2023 / bioRxiv 预印本）

> 来源归档

- **标题：** PyLabRobot: An Open-Source, Hardware Agnostic Interface for Liquid-Handling Robots and Accessories（期刊版题名 *...Hardware-Agnostic Interface...*）
- **类型：** paper
- **作者：** Rick P. Wierenga（莱顿大学 / MIT Media Lab）、Stefan Golas、Wilson Ho、Connor Coley、Kevin M. Esvelt（MIT）
- **期刊：** Device 1(4): 100111（2023），DOI [10.1016/j.device.2023.100111](https://doi.org/10.1016/j.device.2023.100111)；Cell Press / ScienceDirect 页面对本环境返回 403（反爬），**期刊正式版未能取到**
- **实际阅读版本：** bioRxiv 预印本 v1，2023-07-15 发布，CC-BY 4.0，DOI [10.1101/2023.07.10.547733](https://doi.org/10.1101/2023.07.10.547733)，PDF 20 页（取自 Europe PMC 列出的 Unpaywall 开放链接 `https://www.biorxiv.org/content/biorxiv/early/2023/07/15/2023.07.10.547733.full.pdf`）
- **口径提示：** 任务说明里写的 bioRxiv 编号 `547337` 与实际编号 `547733` 不一致，以 Europe PMC / 预印本页眉的 `547733` 为准
- **代码：** https://github.com/PyLabRobot/pylabrobot（MIT 许可，见 [pylabrobot 仓库归档](../repos/pylabrobot.md)）
- **官方文档：** [pylabrobot_docs.md](../sites/pylabrobot_docs.md)
- **入库日期：** 2026-11-07
- **核对范围：** bioRxiv v1 预印本全部 20 页逐页通读（摘要、图摘要、引言、结果各小节、Fig.1–5、Table 1–3、Discussion、Methods 各小节、致谢与参考文献）；**未读** Device 期刊正式版（无法获取），期刊版相对预印本的修订内容未核实
- **通读：** 已完成

## 核心摘录

1. **两层架构（§Hardware Agnostic Interface，Fig.1）：** 用户实例化接口层的 `LiquidHandler` / `PlateReader` 等并配一个对应硬件的 backend；接口层负责命令、甲板布局与资源状态并校验位置与冲突；执行层 backend 把命令译成固件指令或 HTTP 请求。同一类机器的 backend 都继承同一个抽象基类（ABC）。
2. **原子命令而非整份脚本：** 通过原子低层命令，可在 REPL / IPython / Jupyter 里逐条实时发命令，作者与"整份脚本从头跑到尾"的协议式做法对照。
3. **包结构（Table 1）：** liquid_handling、liquid_handling.backends、liquid_handling.liquid_classes、plate_reading、resources、server、utils。
4. **资源模型（§Deck Layout and Labware Model，Fig.2）：** `Resource` 以 name 唯一标识，用有向有根树（arborescence）存 children / parent；位置是相对父节点的笛卡尔偏移，绝对位置惰性地沿树累加；`Deck`、`Carrier`、`Container`（含 `Trough`、`Well`，带体积追踪器）、`ItemizedResource`（`Plate`、`TipRack`，"A1" 式索引）；资源可 JSON 序列化 / 反序列化。
5. **标准化操作（§Standardized operations，Table 2）：** 四个基础命令 aspirate / dispense / pick_up_tips / drop_tips，对应 `LiquidHandler` 与 `LiquidHandlerBackend` 的同名方法与四个 dataclass（Pickup、Drop、Aspiration、Dispense）；96 通道头用 `-TipRack` / `-Plate` 变体；另有 Move。组合操作（discard、transfer）在 `LiquidHandler` 层实现，backend 职责最小化。
6. **Tracker 与事务（§Monitoring and parallelization）：** tip tracker 与 volume tracker 在命令发往硬件前校验；采用"校验入队 → 顺序执行 → 全部成功才提交，否则回滚"的事务模式；用 `asyncio` 并行（Fig.4 示例并行执行移液序列与 `backend.refill_pump()`）。
7. **模拟器（§Simulator，Fig.5）：** 浏览器端模拟器，`SimulatorBackend` 是 `WebsocketBackend` 子类，可本地运行或用托管的 simulator.pylabrobot.org；对任何已有甲板模型的机器人都可用，新机型只需写机器特有约束。另有图形化 labware 布局编辑器。
8. **应用演示（§Applications）：** Art Studio（Opentrons 上用水彩打印 12×8 图案）、Game of Life（用结晶紫染色孔作内存、经读板器读出）、PLR-RLP 位置标定与路径示教工具（键盘 / GUI / 游戏手柄）。
9. **LLM 助手（§LLM Assistant，Table 3）：** 给 GPT-3.5 若干 PyLabRobot 示例代码，再把自然语言提示转成机器人代码；Table 3 给出 4 条提示与输出（如 "aspirate 20uL from the first and last wells in column 2"）。论文未给出量化成功率，只是示例展示；Discussion 提到 LLM 智能体可借此"自主设计并执行实验"，属展望。
10. **Methods：** mypy 静态类型、PyLint、pytest（执行层单元测试断言用户输入生成一致的机器级命令，无需硬件）、Sphinx 文档、PyPI 自动发布、论坛 forums.pylabrobot.org（"20+ posts per day"，作者表述）。
11. **后端与平台验证（Methods）：** STAR backend 直接经 USB 与固件通信（PyUSB），在 Windows（libusbK）、macOS（libUSB）、Raspberry Pi OS 上验证；EVO backend 在 Windows、macOS 验证；Opentrons backend 用 Opentrons HTTP API，并通过 `assigned_resource_callback` / `unassigned_resource_callback` 镜像甲板；Hamilton 与 Opentrons 的 labware 互用（"Hamilton resources on an Opentrons robot and vice versa"）验证了硬件无关性。BMG 读板器亦已支持（Discussion）。

## 局限（作者自述与本次核对）

- 验证主要是单元测试 + 演示，**论文没有性能 / 精度（如移液 CV）的定量对比表**（本次通读未见）。
- "有些开发仍需物理机器人"，靠制造商与同行评审建立信任（Methods > Code simulation）。
- 论文模拟器（SimulatorBackend / WebsocketBackend）在当前仓库 HEAD 的 Python 源码中已检索不到，现用 Visualizer + ChatterboxBackend（见官方文档与仓库归档）。

## 对 wiki 的映射

- [PyLabRobot 项目节点](../../wiki/entities/pylabrobot.md)：架构（接口层 / 执行层 / 资源树 / tracker 事务）、实验与评测（演示与 Table 3）、与智能体实验室的关系。
