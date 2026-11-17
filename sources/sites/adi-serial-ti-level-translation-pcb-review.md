# ADI 串行接口与 TI 电平转换官方资料复核

- **类型：** 官方技术资料 / 原文复核
- **核对日期：** 2026-11-16
- **核对范围：** ADI/Maxim Fundamentals of RS-232 Serial Communications（页面日期 2001-03-29）完整正文、信号表、器件表及图注；TI SN74LVC1T45 产品页的完整 Product details、Features、Description 技术正文。未读取 TIA/JEDEC 标准本体或产品页链接的全部数据手册。
- **通读：** 已完成

## 原文与版本

- [ADI RS-232 tutorial](https://www.analog.com/en/resources/technical-articles/fundamentals-of-rs232-serial-communications.html)；本次读取材料 SHA-256：`6ecf601dcf706aa0c3b26afeae6aaebdb590dd550f6531d4284f05921288c42f`。
- [TI SN74LVC1T45](https://www.ti.com/product/SN74LVC1T45)；本次读取材料 SHA-256：`3cb38ce1220af891b2cf1aaa09374c2c5f04037a40e14357c2a755aa5780f2a5`。

## 核对结论

- UART 负责并串转换和起始/停止/校验位；RS-232 收发器完成 TTL/CMOS 与正负摆幅之间的电平转换及极性转换。
- RS-232 接收端负电压表示 mark/逻辑1，正电压表示 space/逻辑0；DTE/DCE、数据线和握手线的角色应分别核对。电缆限制关联电容，不能只给固定距离。
- ADI 历史文章中的规范速率与器件扩展速率不能混用；页面正文与表格部分口径不同，且 slew rate 的 V/ms 文本不作为本库新数值依据，实际以所选器件当代数据手册为准。
- SN74LVC1T45 是方向受控、单比特双电源转换器，A/B 电源可独立在 1.65–5.5 V；DIR 以 VCCA 为参考，不是自动识别方向的通用双向串口器件。
- Ioff 用于部分断电时防止反灌；任一电源接地时两端口高阻。VCC isolation 是电源隔离功能命名，不等于电气安全所说的电隔离屏障。

## 对 wiki 的映射

- [rs-232-serial-interface](../../wiki/concepts/rs-232-serial-interface.md)
- [ttl-serial-logic-level](../../wiki/concepts/ttl-serial-logic-level.md)
- [uart-serial-communication](../../wiki/concepts/uart-serial-communication.md)
