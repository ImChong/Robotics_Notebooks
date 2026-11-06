# CiA 官网 CANopen / CAN FD / CAN HS 三页通读（电机底软通信协议视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/motor-drive-firmware-bus-protocols.md`）

- **标题：** CANopen / CAN FD: The basic idea / CAN HS transmission（CAN in Automation）
- **作者：** CAN in Automation (CiA)
- **类型：** site / fieldbus / official-docs
- **原文：**
  - <https://www.can-cia.org/can-knowledge/canopen/>
  - <https://www.can-cia.org/can-knowledge/can-fd-the-basic-idea/>
  - <https://www.can-cia.org/can-knowledge/can-hs-transmission/>
  （2026-11 直接抓取页面全文）
- **一句话说明：** CiA 对 CANopen 分层与对象字典、CAN FD 双速率与帧扩展、CAN HS 物理层电平与布线的公开说明，是本页「物理总线 × 应用协议」两层划分的一手依据。
- **核对范围：** 三个页面的正文全部读完（含 CAN FD 页的帧结构图注与 CiA 601 系列文档列表的 Scope 说明、CANopen 页的 CiA 301 Scope 与 Profile 编号列表）。CiA 301 / CiA 402 / CiA 601 等规范本体需登录获取，未阅读；页面末尾的 CAN Newsletter 文章标题仅列出未逐篇阅读；EtherCAT/CoE、DroneCAN、厂商私有帧、Modbus 等协议未在此三页中，不由本归档覆盖。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **CANopen 分层（CANopen 页）：** CANopen CC 基于 CAN CC，CANopen FD 基于 CAN FD；包含高层协议与 Profile 规范。CANopen CC 基于 ISO 11898-1 数据链路层，位时序由 CiA 301 规定，速率 10–1000 kbit/s；寻址以 11 位 CAN-ID 为主，也支持 29 位；物理层默认 ISO 11898-2，但不排除其他物理层。
- **标准化通信对象：** 提供 COB 用于时间关键过程、配置与网络管理；协议包括 SDO、PDO、NMT、特殊功能与错误控制协议。开发者无需处理位时序与验收滤波等 CAN 硬件细节。
- **设备三部分：** 协议栈、应用软件、对象字典；对象字典以 16 位索引（4 位十六进制）引用，1000h–1FFFh 为通信参数，2000h–9FFFh 为应用参数，其中 2000h–5FFFh 为厂商专有、6000h–9FFFh 为标准化参数。
- **Profile 与互操作：** 设备 / 接口 / 应用 Profile 使设备具备「即插即用」，同时允许厂商扩展；CiA 402 列于设备 Profile 编号中（页面未展开其内容）。
- **CAN FD 动机与双速率（CAN FD 页）：** 2011 年 Bosch 与车企合作开发；突破经典 CAN 两个限制——速率超 1 Mbit/s、载荷由 8 字节扩至 64 字节。仲裁段速率受拓扑限制（≤1 Mbit/s），数据段受收发器特性限制；仅单节点发送时可提速，ACK 位前需重新同步。
- **吞吐：** 仲裁与数据段速率比 1:8 时，考虑头部与 CRC 变长，吞吐约为经典 CAN 的 6 倍。
- **帧格式：** FDF 位隐性表示 CAN FD 帧、显性表示 CAN CC 帧；BRS 位隐性时启用第二速率，在 BRS 采样点切换；ESI 位隐性表示 error passive。CRC 为 17 位（载荷 ≤16 字节）或 21 位（>16 字节），另有 3 位格雷码 stuff-bit 计数器 + 奇偶位 + 固定填充位。FD 帧不支持远程帧；11 位 ID 用 FBFF，29 位用 FEFF；CAN CC 与 CAN FD 同在 ISO 11898-1:2024。非 ISO 的 CAN FD 控制器不具备上述附加保护。
- **CAN HS 物理层（CAN HS 页）：** ISO 11898-2；最高 1 Mbit/s；隐性时 CAN_H / CAN_L 均约 2.5 V，显性时 CAN_H≈3.5 V、CAN_L≈1.5 V；显性差分 0.9–2.0 V，隐性 −1.0–0.5 V。因仲裁机制，速率越高允许线越短，理论 1 Mbit/s 约 40 m，实际更短；线缆延迟 ≤5 ns/m；推荐线型拓扑、两端 120 Ω 终端（可用 2×60 Ω 分裂终端加电容）。低功耗与选择性唤醒收发器并入 ISO 11898-2:2016，系统基础芯片（SBC）集成收发器与稳压 / 看门狗等。

## 对 wiki 的映射

- [`wiki/overview/motor-drive-firmware-bus-protocols.md`](../../wiki/overview/motor-drive-firmware-bus-protocols.md)
- [`wiki/concepts/can-fd.md`](../../wiki/concepts/can-fd.md)
- [`wiki/concepts/can-bus-protocol.md`](../../wiki/concepts/can-bus-protocol.md)

## 参考来源

- <https://www.can-cia.org/can-knowledge/canopen/>
- <https://www.can-cia.org/can-knowledge/can-fd-the-basic-idea/>
- <https://www.can-cia.org/can-knowledge/can-hs-transmission/>
