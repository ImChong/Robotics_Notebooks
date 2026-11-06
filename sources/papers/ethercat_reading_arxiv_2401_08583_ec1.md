# 脊柱手术机器人 EtherCAT 控制框架论文通读（IgH 主站 + PREEMPT_RT 实测）

> 来源归档（原文通读 · 支撑 `wiki/concepts/ethercat-protocol.md`）

- **标题：** Development of Control Framework for Spine Surgery Robot Using EtherCAT
- **作者：** Veysi Adin, Chunwoo Kim（KIST，韩国）
- **类型：** paper / fieldbus / real-time
- **原文：** <https://arxiv.org/abs/2401.08583>（2026-11 通过 arXiv PDF v1 获取全文，共 2 页）
- **一句话说明：** 用开源 IgH EtherLab 主站 + RT-PREEMPT Linux 做 1 kHz 多轴控制，并以 3 小时 108 万个时序样本实测周期、抖动与执行时间。
- **核对范围：** 通读全文 2 页：摘要、§I 引言、§II 实现（主站、实时 Linux）、§III 实验（图 1 连接图、表 I 软硬件配置、表 II 实测结果，已按渲染页核对表格数值）、§IV 结论与参考文献列表。参考文献 [4]–[8] 仅看题录，未阅读；ETG 官网技术页因访问受限未获取，EtherCAT 通用规格仅采用本文转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I）：** 医疗机器人需兼顾 IEC 60601 / 62304 / 80601-2-77 等标准，物理集成需要高速、可靠、确定性的通信；EtherCAT 使用标准以太网卡做 IO，用户与商用驱动器 / IO 板日益增多。
- **EtherCAT 特性（§II-A，转述 ETG）：** 周期时间典型小于 100 µs，抖动小于 1 µs；上层协议有 CoE、FoE 等；主站需运行 EtherCAT 主站软件，本文用开源 IgH EtherLab（v1.5.2）。
- **实时 Linux（§II-B）：** 双内核（Xenomai、RTAI）实时性能更好，但考虑实现时间、可移植性、社区支持等，选单内核 RT_PREEMPT；系统为 Xubuntu 20.04，内核 5.10.73-rt54，任务优先级 98。
- **硬件（图 1、表 I）：** 主站为 Advantech 嵌入式工控机（i7-6600U、8GB、Intel I219 网卡）；从站为 3 台 Maxon EPOS4 50/5 伺服驱动器（各 TxPDO 14 B + RxPDO 14 B，共 28 B）加 1 个自制 EasyCAT 从站（IO，TxPDO 4 B + RxPDO 4 B），接 2 个限位开关与急停；控制频率 1 kHz。
- **控制方式（§III）：** 手柄输入经 USB 在第二线程读取，传给 1 kHz 实时通信线程，以同步周期速度模式下发期望速度。
- **实测（表 II）：** 运行 3 小时、每 10 ms 采样一次，共 1.08 百万样本。周期 1000.017 ± 7.035 µs（最小 / 最大 870.388 / 1148.032）；抖动均值 2.125 ± 6.706 µs，最大 148.032 µs；单周期执行时间均值 11.479 ± 7.039 µs，最大 231.767 µs。作者结论为适用于实时应用。
- **未来工作（§IV）：** 计划引入机器人中间件，按组件化软件工程重构。

## 对 wiki 的映射

- [`wiki/concepts/ethercat-protocol.md`](../../wiki/concepts/ethercat-protocol.md)
