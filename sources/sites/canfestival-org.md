# CanFestival 官网（canfestival.org）

> 来源归档（site）

- **标题：** CanFestival — Free software CANopen framework
- **类型：** site
- **来源：** CanFestival 项目站
- **链接：** https://canfestival.org/
- **子页：** [Code](https://canfestival.org/code) · [Documentation](https://canfestival.org/doc) · [Applications](https://canfestival.org/apps)
- **代码：** 官网 Code 页列出多源；社区主线 Mercurial 为 <http://hg.beremiz.org/canfestival>；现代 CMake/Git 维护仓见 [sources/repos/canfestival.md](../repos/canfestival.md)（[beremiz/canfestival](https://github.com/beremiz/canfestival)）
- **入库日期：** 2026-08-12
- **一句话说明：** 自 2001 年起开源的 ANSI-C、平台无关 CANopen® 栈官网：可作 Master/Slave，面向 PC、实时工控机与微控制器；运行时 LGPLv2、配套工具 GPLv2。
- **沉淀到 wiki：** [wiki/entities/canfestival.md](../../wiki/entities/canfestival.md)

## 为什么值得保留

- 仓库已有 **CANopen / CiA 402 概念与选型总览**（[cia_canopen_overview](cia_canopen_overview.md)、[motor_drive_firmware_bus_protocols](../courses/motor_drive_firmware_bus_protocols.md)），但缺少可落地的 **开源 CANopen 协议栈实体**；CanFestival 是工业自动化与科研机床/PLC 场景常见的免费实现。
- 官网明确区分 **运行时 LGPL**（可与专有代码链接）与 **工具 GPL**（Objdictedit / objdictgen），对关节驱动器/主站选型的许可边界很关键。
- Applications 页给出 Beremiz/PLC、六轴 CANopen 伺服 + 实时 Linux CNC 等案例，可与人形/工业臂「L2=CANopen」路线对照。

## 开源核查（步骤 2.5，2026-08-12）

| 项 | 结论 |
|----|------|
| 开放程度 | **已开源**（多仓库分叉并存；官网 Code 页列 hg / Bitbucket / 商业支持分支） |
| 官网 Code | 主仓描述为 `hg.beremiz.org/canfestival`（「quite lazily updated」）；另有 Ingelibre、ICA、JaFojtik 等克隆 |
| Git 活跃镜像 | [beremiz/canfestival](https://github.com/beremiz/canfestival)（CMake、Python 3 objdictgen；近期仍有推送） |
| 许可 | Runtime **LGPLv2**；developer tools **GPLv2**（官网 Documentation 明示；生成的 OD C 代码不受 GPL/LGPL 覆盖） |
| 社区通道 | SourceForge 邮件列表：<http://sourceforge.net/mail/?group_id=29577> |

## 核心摘录

### 1) 定位与许可

- **要点：** ANSI-C、平台无关 CANopen 栈；同一代码基可构建 **Master 或 Slave**；目标平台含 PC、Real-time IPC、Microcontroller。2001 年开源，依赖社区贡献成长。
- **对 wiki 的映射：** [canfestival](../../wiki/entities/canfestival.md)、[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

### 2) 工具链与对象字典

- **要点：** **Objdictedit**（GUI）与 **objdictgen**（CLI：`python objdictgen/objdictgen.py Node.od Node.c`）从对象字典描述生成 C；另提供 **CANopen Shell** 作主/从命令行调试。运行时通过 OS 定时器与可动态加载的 CAN 接口绑定。
- **对 wiki 的映射：** [canfestival](../../wiki/entities/canfestival.md)、[can-bus-protocol](../../wiki/concepts/can-bus-protocol.md)

### 3) 驱动与平台矩阵（文档页摘要）

- **要点：** 原生支持 Linux / Xenomai / Win32；CAN 接口含 SocketCAN、Peak、can4linux、IXXAT、VScom、AnaGate 等；裸机示例含 AVR、（文档称 HCS12 曾 broken）。PDF 手册可从旧 Automforge 树下载（链接可能陈旧）。
- **对 wiki 的映射：** [canfestival](../../wiki/entities/canfestival.md)、[can-vs-ethercat-joint-bus](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

### 4) 应用案例（Applications）

- **要点：** Smarteh MC8 Open PLC（双 CAN + Beremiz）；Micropral Xenomai ARM PLC；LAAS×FESTO 六自由度定位机（六台 CANopen 伺服 + 工控机作 CANopen CNC，20 ms 时间精度需求）；Peak 开源咖啡机演示。
- **对 wiki 的映射：** [canfestival](../../wiki/entities/canfestival.md)、[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

## 推荐继续阅读（外部）

- 官网 Documentation：<https://canfestival.org/doc>
- CiA CANopen 概览：<https://www.can-cia.org/can-knowledge/canopen/>
- Beremiz 自动化工作台：<http://www.beremiz.org>

## 当前提炼状态

- [x] 官网首页 / Code / Doc / Apps 摘录完成
- [x] 开源状态与 Git 维护仓交叉归档
- [x] 升格 wiki 实体页
