# RFC 1112 IP 组播标准与 LCM UDP 多播实现

> 来源归档（standard / repo；2026-11-22 核对并通读）

- **标题：** RFC 1112: Host Extensions for IP Multicasting (S. Deering, 1989) & Lightweight Communications and Marshalling (LCM)
- **类型：** specification / repo
- **规范原文：** <https://www.rfc-editor.org/rfc/rfc1112.txt>（S. Deering, Stanford University, August 1989）
- **LCM 仓库：** <https://github.com/lcm-proj/lcm>（[README.md](https://raw.githubusercontent.com/lcm-proj/lcm/master/README.md)）
- **入库日期：** 2026-11-22
- **通读：** 已完成
- **核对范围：** RFC 1112 全文第 1–7 节与附录（IP 组播“best-efforts”语义、不保证可靠送达与保序、IGMP 主机组成员管理）及 `lcm-proj/lcm` 官方 `README.md` 全文。
- **一句话说明：** RFC 1112 定义了 IP 组播的“尽最大努力（best-efforts）、不保序、不重传”传输语义，LCM 在此之上用无中心守护进程的点对点 UDP 多播提供低时延发布-订阅消息总线。

## 核心摘录（面向 wiki 编译）

1. **Best-Efforts 传输语义（RFC 1112 §2）：** 组播数据报与常规单播 IP 具有相同的尽最大努力可靠性——“not guaranteed to arrive intact at all members of the destination group or in the same order relative to other datagrams”，即天然存在丢包、迟到、乱序与多订阅者一致性偏差。
2. **LCM 去中心化设计（`lcm-proj/lcm` README）：** 面向实时高带宽低时延系统，直接采用基于 UDP Multicast 的对等通信（“No centralized database or hub -- peers communicate directly；No daemons”），并提供跨 C/C++/Python 等语言的自动编组（marshalling）与日志回放。

## 对 wiki 的映射

- [`wiki/formalizations/udp-multicast-dynamics.md`](../../wiki/formalizations/udp-multicast-dynamics.md)
