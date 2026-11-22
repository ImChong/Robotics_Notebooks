# LinuxPTP (IEEE 1588 PTP / gPTP) 官方实现文档与标准笔记

> 来源归档（official docs / implementation；2026-11-22 核对并通读）

- **标题：** LinuxPTP Project — IEEE Standard 1588 Precision Time Protocol (PTP) Implementation for Linux
- **类型：** site / open-source spec implementation
- **源码仓库：** <https://github.com/richardcochran/linuxptp>（[README.org](https://raw.githubusercontent.com/richardcochran/linuxptp/master/README.org)）
- **标准：** IEEE Std 1588 (PTP) / IEEE 802.1AS-2011 & 802.1AS-2020 (gPTP)
- **入库日期：** 2026-11-22
- **通读：** 已完成
- **核对范围：** `richardcochran/linuxptp` 官方 `README.org` 全文（含 Linux `SO_TIMESTAMPING` / PTP Hardware Clock (PHC) / `clock_adjtimex` 接口、OC/BC/TC 时钟角色、802.1AS-2020 CMLDS、L2/UDPv4/UDPv6 传输与配置文件支持）。
- **一句话说明：** `linuxptp` 是 Linux 核内 PTP Hardware Clock（PHC）与 `SO_TIMESTAMPING` 接口的标准实现，实现了 Ordinary/Boundary/Transparent Clock 及 IEEE 802.1AS gPTP。

## 核心摘录（面向 wiki 编译）

1. **硬件时间戳与 PHC：** 通过 Linux Socket `SO_TIMESTAMPING` 捕获网卡/PHY 硬件时间戳，并通过 `clock_gettime` 与 `clock_adjtimex` 伺服本地 PTP Hardware Clock（PHC），消除了用户态与协议栈软件抖动。
2. **时钟拓扑角色：** 支持 Ordinary Clock（OC）、Boundary Clock（BC）与 Transparent Clock（TC），以及点对点一步（P2P one-step）与多域 IEEE 802.1AS-2020 Common Mean Link Delay Service（CMLDS）。
3. **传输协议：** 支持 Raw Ethernet Layer 2、UDP/IPv4 与 UDP/IPv6，并内置汽车/工业/gPTP 等标准 Profile。

## 对 wiki 的映射

- [`wiki/concepts/clock-synchronization-algorithms.md`](../../wiki/concepts/clock-synchronization-algorithms.md)
