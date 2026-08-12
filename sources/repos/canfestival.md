# CanFestival（beremiz/canfestival）

> 来源归档（repo）

- **标题：** CanFestival — open-source CANopen stack + object dictionary tools
- **类型：** repo
- **来源：**
  - 项目站：<https://canfestival.org/>（见 [sources/sites/canfestival-org.md](../sites/canfestival-org.md)）
  - Git / CMake 维护仓：<https://github.com/beremiz/canfestival>
  - 历史 Mercurial 主线（官网 Code 页）：<http://hg.beremiz.org/canfestival>
- **Stars / 活跃度（beremiz/canfestival，核查日 2026-08-12）：** ~14 stars；`pushed_at` 2026-07-10；default branch `default`；许可证 SPDX `LGPL-2.1`
- **入库日期：** 2026-08-12
- **一句话说明：** ANSI-C CANopen 协议栈（Master/Slave）与 Python 对象字典编辑/代码生成工具；现代入口以 Beremiz 组织的 CMake 仓为准，官网仍索引多处历史克隆。
- **沉淀到 wiki：** [wiki/entities/canfestival.md](../../wiki/entities/canfestival.md)

## 为什么值得保留

- 机器人集成「工业 CANopen + CiA 402」时需要**可链接的开源主站/从站栈**；本仓补齐 wiki 在协议概念之外的工程实体。
- README 给出可复现的 CMake 构建、SocketCAN/`virtual` 驱动与 `objdictgen` 流水线，适合 Linux 主站原型与嵌入式移植对照。
- 与 [SimpleFOC](simplefoc_arduino_foc.md) 互补：后者是电流环算法层；CanFestival 是 **L2 CANopen 通信层**。

## 开源状态

- **已开源**：完整 C 栈 + `objdictgen/` 工具 + `examples/`。
- **分叉现实**：官网列出 Automforge/hg、Ingelibre 分支、多家 Bitbucket 克隆；选型时应固定 commit，并优先评估 [beremiz/canfestival](https://github.com/beremiz/canfestival) 的 CMake 路径。
- **许可边界（与官网一致）：** 运行时库 LGPL；工具 GPL；**由 objdictgen 生成的节点 C 代码**声明不受 GPL/LGPL 覆盖（见官网 Doc）。

## 构建与入口（README）

```sh
mkdir build && cd build
cmake .. -DCF_TARGET=unix -DCF_CAN_DRIVER=virtual -DCF_TIMERS_DRIVER=unix
make
# 示例：
# cmake .. -DCF_BUILD_EXAMPLES=ON -DCF_ENABLE_LSS=ON
```

| 产物 | 说明 |
|------|------|
| `libcanfestival.a` | 核心 CANopen 栈 |
| `libcanfestival_<target>.a` | 平台/定时器驱动 |
| `libcanfestival_can_<driver>.so` | 可选动态 CAN 驱动（`CF_ENABLE_DLL_DRIVERS`） |

常用 CMake 选项：`CF_TARGET`（unix/windows）、`CF_CAN_DRIVER`（virtual/socket/peak/…）、`CF_TIMERS_DRIVER`（unix/windows/xeno）、`CF_ENABLE_LSS`、`CF_BUILD_EXAMPLES`。

### 对象字典工具

```sh
cd objdictgen && python3 objdictedit.py
python3 objdictgen/objdictgen.py MyNode.dcf MyNode.c
```

依赖：Python 3、wxPython 4（GUI）。字典存为扩展 EDS 风格的 **DCF**（含 `[CanFestivalNode]` 等节）。

### 示例（需 `-DCF_BUILD_EXAMPLES=ON`）

| 示例 | 备注 |
|------|------|
| `CANOpenShell` | 交互式主/从 Shell |
| `DS401_Master` / `DS401_Slave_Gui` | DS-401 主从 |
| `TestMasterSlave` / `TestMasterSlaveLSS` | 同进程主从；LSS 需 `CF_ENABLE_LSS` |

## 目录结构（README）

```
src/            核心栈（C）
drivers/        平台、定时器、CAN 驱动
include/        公共头文件
objdictgen/     OD 编辑与代码生成（Python）
examples/       示例应用
doc/            文档
```

## CAN 驱动矩阵（README 摘要）

| Driver | 平台 | 说明 |
|--------|------|------|
| `virtual` | 全平台 | 进程内 pipe，测试用 |
| `socket` | Linux | SocketCAN |
| `peak` / `kvaser` / `anagate` / `ixxat` / `vscom` | Windows（⚠️） | README 标注：随新 CMake 体系**过时未测**，可能无法直接构建 |

## 对 wiki 的映射

- 实体： [wiki/entities/canfestival.md](../../wiki/entities/canfestival.md)
- 协议总览： [wiki/overview/motor-drive-firmware-bus-protocols.md](../../wiki/overview/motor-drive-firmware-bus-protocols.md)
- CAN 概念： [wiki/concepts/can-bus-protocol.md](../../wiki/concepts/can-bus-protocol.md)
- 关节总线对比： [wiki/comparisons/can-vs-ethercat-joint-bus.md](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

## 当前提炼状态

- [x] README / 构建入口摘录
- [x] 与官网多源分叉关系说明
- [x] 升格 wiki 实体
