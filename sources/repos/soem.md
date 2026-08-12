# SOEM（OpenEtherCATsociety/SOEM）

> 来源归档（repo）

- **标题：** SOEM — Simple Open EtherCAT Master
- **类型：** repo
- **来源：**
  - GitHub：<https://github.com/OpenEtherCATsociety/SOEM>
  - 项目页：<https://openethercatsociety.github.io/>（见 [sources/sites/openethercatsociety-github-io.md](../sites/openethercatsociety-github-io.md)）
  - 文档（需登录）：<https://docs.rt-labs.com/soem>
- **Stars / 活跃度（核查日 2026-08-12）：** ~2046 stars；~902 forks；`pushed_at` 2026-06-12；default branch `master`；最新 release **v2.0.0**（2025-07-11）；语言 C；topics：`c` / `ethercat` / `industrial-automation` / `protocol-library` / `soem`
- **许可：** 双许可 — **GPLv3** + **商业许可**（`LICENSE.md`；产品闭源通常需联系 sales@rt-labs.com）
- **入库日期：** 2026-08-12
- **一句话说明：** 用户态、轻量 EtherCAT MainDevice 库（CMake ≥3.28，project VERSION 2.0.0）；Linux/Windows/RT-Kernel 一等公民，contrib 覆盖 RTEMS/VxWorks/macOS 等；含 CoE/FoE/SoE/EoE/DC 与 samples。
- **沉淀到 wiki：** [wiki/entities/soem.md](../../wiki/entities/soem.md)

## 为什么值得保留

- 机器人知识库在 EtherCAT 选型中反复提到 **SOEM vs IgH**，但此前无独立实体页；本仓是科研/原型阶段最常用的 **开源用户态主站**。
- README 明确「库而非独立应用」+ 实时嵌入式定位；`samples/ec_sample` 展示 DC 同步周期线程，可与 [主站优化指南](../../wiki/queries/ethercat-master-optimization.md) 对照。
- 与 [CanFestival](canfestival.md) 互补：后者是 CAN 上的 CANopen；SOEM 是以太网上的 CoE 主站。

## 开源状态

- **已开源**：完整 `src/` 协议栈 + `osal/` / `oshw/` 平台层 + `samples/` + CMake 构建。
- **姊妹项目：** [SOES](https://github.com/OpenEtherCATsociety/SOES)（从站栈）；组织另有实验仓 `soem-ng`。
- **许可边界：** GPLv3 适合开源项目；闭源商业产品按 `LICENSE.md` 需商业许可——选型时务必写进合规清单。

## 构建与入口（README / CMake）

```sh
git clone https://github.com/OpenEtherCATsociety/SOEM.git
cd SOEM
cmake -S . -B build -DSOEM_BUILD_SAMPLES=ON
cmake --build build
# 示例（Linux，需 root 或 CAP_NET_RAW 访问网卡）：
# ./build/samples/slaveinfo/slaveinfo eth0
# ./build/samples/ec_sample/ec_sample eth0
```

| 产物 / 选项 | 说明 |
|-------------|------|
| `libsoem` | 核心主站库（`src/ec_*.c`） |
| `SOEM_BUILD_SAMPLES` | 默认 ON：构建 `samples/` |
| `EC_DEBUG` | 调试输出 |
| `EC_MAXSLAVE` 等 | CMake CACHE 可调缓冲区/从站上限 |

官方文档站需登录；公开复现以仓库 README + samples 源码为准。

## 目录结构（仓库树摘要）

```
include/soem/     公共头（soem.h → ec_main / ec_dc / ec_coe …）
src/              协议实现（ec_base/main/dc/coe/foe/soe/eoe/config/print）
osal/             OS 抽象（linux / win32 / rtk）
oshw/             网卡驱动 nicdrv（linux / win32 / rtk）
contrib/          社区平台（rtems / vxworks / macosx / erika / intime …）
samples/          可运行示例
cmake/            平台与 ENI 辅助
```

## samples（README 对齐）

| 示例 | 作用 |
|------|------|
| `slaveinfo` | 扫描总线、打印从站与可选 SDO/映射 |
| `ec_sample` | 周期过程数据 + DC 同步 PI（默认 1 ms 周期） |
| `eepromtool` | EEPROM 工具 |
| `eni_test` | ENI XML 测试 |
| `eoe_test` / `firm_update` / `simple_ng` | EoE、固件更新、精简示例 |

## 对 wiki 的映射

- 实体： [wiki/entities/soem.md](../../wiki/entities/soem.md)
- 协议： [wiki/concepts/ethercat-protocol.md](../../wiki/concepts/ethercat-protocol.md)
- 主站调优： [wiki/queries/ethercat-master-optimization.md](../../wiki/queries/ethercat-master-optimization.md)
- 选型对比： [wiki/comparisons/can-vs-ethercat-joint-bus.md](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)、[wiki/comparisons/ethercat-vs-ethernet-ip.md](../../wiki/comparisons/ethercat-vs-ethernet-ip.md)
- 底软总览： [wiki/overview/motor-drive-firmware-bus-protocols.md](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

## 当前提炼状态

- [x] README / CMake / samples / LICENSE 摘录
- [x] 项目页交叉归档
- [x] 升格 wiki 实体
