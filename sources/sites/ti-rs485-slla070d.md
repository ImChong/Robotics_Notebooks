# TI 应用笔记：RS-422 与 RS-485 标准概览（SLLA070D）

> 来源归档（standard 解读 / 厂商应用笔记）

- **标题：** RS-422 and RS-485 Standards Overview and System Configurations（TI 文献号 SLLA070D，2002-06 起，2010-05 修订）
- **类型：** site / application note（PDF）
- **链接：** <https://www.ti.com/lit/an/slla070d/slla070d.pdf>
- **入库日期：** 2026-11-07
- **核对范围：** 全文用 pdftotext 提取后读：第 1 章导言、第 2 章 RS-422/RS-485 电气规格（含 U.L. 概念、驱动/接收测试条件）、第 3 章故障保护、第 4 章端接与接地概述、第 5 章拓扑（菊花链/总线+支线/点对点）、第 6 章 RS-422 与 RS-485 比较表；第 4 章的各端接方式实验波形（图 16–28）只读图注与结论，未逐图分析；TIA/EIA-485-A 标准本体付费，未读。同名后缀的 SLLA272（RS-485 Design Guide）曾下载过目录与总线结构章节，但**未读完，不作为本归档依据**。
- **通读：** 已完成
- **更正（对 [uart_rs485_serial_embedded.md](../courses/uart_rs485_serial_embedded.md) 的核对）：** 该归档把 TI 文献号 **SLLA383** 写作「UART-to-RS-485 Interface」，实际 2026-11-07 取得的 `slla383b.pdf` 是 **LIN Protocol and Physical Layer Requirements**（LIN 总线应用笔记），与 RS-485 无关；RS-485 依据请改看本归档。
- **沉淀到 wiki：** [CAN vs EtherCAT 关节总线选型](../../wiki/comparisons/can-vs-ethercat-joint-bus.md)

## 核对摘录

- **标准身份：** RS-485 官方名为 TIA/EIA-485，标题 *Electrical Characteristics of Generators and Receivers for Use in Balanced Digital Multipoint Systems*，由 TIA TR-30.2 子委员会修订；RS-422 是单驱动多接收（最多 1 个驱动 + 10 个接收）的 simplex multidrop。
- **多点：** RS-485 一个时刻只允许一个驱动器发送（半双工），总线上最多 **32 个单位负载（U.L.）**；0.5 U.L. 的收发器可挂 64 个。
- **line contention：** 若发送前未禁能其他驱动器（图例中的 D3），两个驱动器同时使能就会出错数据；RS-485 器件必须有限流（短路电流限 250 mA 量级）与热关断保护。
- **电气（表 4）：** 驱动器差分输出 1.5 V ≤ |VOD| ≤ 5 V（负载 ≥60 Ω）；接收器灵敏度 ±200 mV；共模范围 −7 V 到 +12 V；最大理论长度 1200 m；数据率 >10 Mbps（TI 器件至 50 Mbps）。
- **端接：** RS-485 一般两端都要端接，端接阻抗取电缆特性阻抗（约 90–120 Ω）；支线（stub）越长反射越严重，应尽量短；拓扑为菊花链（骨干 + 短支线）。
- **故障保护（failsafe）：** 总线空闲（所有驱动器高阻）时差分电压不确定，接收器可能被噪声误触发，需内部或外部 failsafe 偏置。
- **对照 CAN 的依据：** 本文**只定义电气层**，未涉及标识符仲裁、帧格式或错误处理，这些须由上层协议（如主从轮询）自行规定；对照 [cia_can_knowledge_can_classic_and_hs.md](cia_can_knowledge_can_classic_and_hs.md)。
