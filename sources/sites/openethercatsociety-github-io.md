# Open EtherCAT Society（SOEM / SOES 项目页）

> 来源归档（site）

- **标题：** Open EtherCAT Society — Home of SOEM and SOES
- **类型：** site
- **来源：** OpenEtherCATsociety GitHub Pages
- **链接：** https://openethercatsociety.github.io/
- **代码：** [OpenEtherCATsociety/SOEM](https://github.com/OpenEtherCATsociety/SOEM)（主站）；姊妹仓 [SOES](https://github.com/OpenEtherCATsociety/SOES)（从站）
- **文档：** README 指向 <https://docs.rt-labs.com/soem>（需 RT-Labs 账号登录）；项目页另有 Reference manuals 入口
- **维护方：** [RT-Labs](http://www.rt-labs.com)（组织 blog 同址）
- **入库日期：** 2026-08-12
- **一句话说明：** SOEM（Simple Open EtherCAT Master）与 SOES（Slave）的官方介绍页：嵌入式友好的轻量 EtherCAT 栈、支持 CoE/VoE/DC/SoE，并列出 RT-Labs 参考设计平台。
- **沉淀到 wiki：** [wiki/entities/soem.md](../../wiki/entities/soem.md)

## 为什么值得保留

- wiki 已有 [EtherCAT 协议](../../wiki/concepts/ethercat-protocol.md) 与 [主站优化 query](../../wiki/queries/ethercat-master-optimization.md)，但缺少 **可链接的开源主站栈实体**；本页是 SOEM/SOES 官方入口，补齐「用户态主站」工程锚点。
- 明确列出支持的应用层 Profile（CoE / VoE / DC / SoE）与嵌入式参考平台，便于与 IgH 内核主站对照选型。
- 贡献需签 CLA；商业文档与培训指向 RT-Labs——许可与支持边界对产品化选型关键。

## 开源核查（步骤 2.5，2026-08-12）

| 项 | 结论 |
|----|------|
| 开放程度 | **已开源** — GitHub 完整 C 库 + `samples/` |
| 主仓 | [OpenEtherCATsociety/SOEM](https://github.com/OpenEtherCATsociety/SOEM)（~2.0k stars；default `master`；最新 release **v2.0.0**，2025-07-11） |
| 从站姊妹仓 | [OpenEtherCATsociety/SOES](https://github.com/OpenEtherCATsociety/SOES)（~832 stars） |
| 许可 | **双许可**：GPLv3 + 商业许可（见仓库 `LICENSE.md`）；产品化闭源常需购买商业许可 |
| 文档站 | `docs.rt-labs.com/soem` **需登录**；公开信息以 GitHub README + 本项目页为主 |

## 核心摘录

### 1) 定位

- **要点：** SOEM / SOES 是面向嵌入式市场的小型 EtherCAT 主站 / 从站栈；可作为库嵌入应用，而非独立应用程序。
- **对 wiki 的映射：** [soem](../../wiki/entities/soem.md)、[ethercat-protocol](../../wiki/concepts/ethercat-protocol.md)

### 2) 支持的 Profile / 特性

- **要点：** CoE（CANopen over EtherCAT）、VoE、Distributed Clocks、SoE（SERCOS over EtherCAT）。
- **对 wiki 的映射：** [soem](../../wiki/entities/soem.md)、[ethercat-master-optimization](../../wiki/queries/ethercat-master-optimization.md)

### 3) 参考设计平台（摘录）

- **主站（SOEM）：** Freescale i.MX53、Blackfin 5xx/6xx、Intel、Infineon XMC47/48。
- **从站（SOES）：** K10/K60 + ET1100、Zynq + ET1815、XMC43/48、LAN9252 等。
- **对 wiki 的映射：** [soem](../../wiki/entities/soem.md)、[motor-drive-firmware-bus-protocols](../../wiki/overview/motor-drive-firmware-bus-protocols.md)

## 推荐继续阅读（外部）

- GitHub SOEM：<https://github.com/OpenEtherCATsociety/SOEM>
- GitHub SOES：<https://github.com/OpenEtherCATsociety/SOES>
- RT-Labs：<http://www.rt-labs.com>

## 当前提炼状态

- [x] 项目页与开源状态核查
- [x] 升格 wiki 实体
- [x] 配套 sources/repos/soem.md
