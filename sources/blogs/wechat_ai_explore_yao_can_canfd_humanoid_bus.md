# 一文读懂CAN与CAN FD：从车载诞生到人形机器人底层总线

> 来源归档（blog / 微信公众号）

- **标题：** 一文读懂CAN与CAN FD：从车载诞生到人形机器人底层总线
- **类型：** blog
- **作者：** AI探索Yao（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/UvjlH1bCsZwNHC2_z12cBg
- **发表日期：** 2026-07-20
- **入库日期：** 2026-08-12
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [`sources/raw/wechat_ai_explore_yao_can_canfd_humanoid_bus_2026-07-20.md`](../raw/wechat_ai_explore_yao_can_canfd_humanoid_bus_2026-07-20.md)
- **一句话说明：** 从车载 CAN 2.0 / CAN FD 发展史与帧结构出发，给出人形**分层通信**选型：EtherCAT 主干承载大功率关节与视觉，CAN FD 末端分支承载灵巧手、轻载关节、分布式传感与 BMS/STO。
- **项目页 / 源码：** 无独立项目页；科普综述，无配套开源仓（步骤 2.5 不适用）。

## 核心摘录（归纳，非全文）

### 1) 经典 CAN → CAN FD 时间线（文内）

| 节点 | 要点 |
|------|------|
| 1983–1986 | Bosch 启动并发布 CAN：多设备共线、差分抗干扰、硬件优先级仲裁 |
| 1993 | ISO 11898：链路层 + 高速/低速物理层拆分 |
| CAN 2.0A/B | 11 bit / 29 bit ID；载荷 **≤8 B**；全程单速率 **≤1 Mbit/s** |
| 2011–2015 | CAN FD：仲裁段兼容经典时序，数据段提速；载荷 **≤64 B**；纳入 ISO 11898-1:2015 |

### 2) 协议对照（文内参数）

| 参数 | CAN 2.0 | CAN FD |
|------|---------|--------|
| 单帧载荷 | 8 B | 64 B |
| 速率 | 全程 ≤1 Mbit/s | 仲裁 ≤1 Mbit/s；数据段常见至 5–8 Mbit/s |
| CRC | 15 bit | 17/26 bit 自适应 |
| 兼容 | — | FD 设备可收经典帧；反之不行 |

### 3) 人形落地：分层总线（本文主贡献）

文中称全尺寸量产人形常见 **分层架构**：

- **主干 EtherCAT**：髋/膝等大功率关节、视觉/图像等高带宽路径。
- **末端 CAN FD 分支**：五指灵巧手、腕/颈/踝等轻载关节、足底力/IMU/碰撞传感、BMS 与急停 STO、诊断/OTA。

选型直觉（文内对比）：

| 对照 | 为何末端常选 CAN FD |
|------|---------------------|
| vs TTL 串口 | 多主共线 + 硬件仲裁 + 差分抗干扰；单节点故障隔离 |
| vs 经典 CAN | 64 B 少分包；文称同等数据量总线负载可降约 60%，利于维持 ~1 kHz |
| vs EtherCAT | MCU 原生 CAN FD 成本低；狭小腔体总线布线更易；末端无图像/点云时带宽够用 |

### 4) 可信度边界

- 第三方科普归纳，非 Bosch/CiA 官方白皮书；具体波特率、线长与「量产标配」表述需以整机 BOM 与 ISO/CiA 为准。
- 「总线负载降低 60%」「维持 1 kHz」等量级依赖帧长、节点数与调度，不能直接外推到任意机型。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [CAN FD](../../wiki/concepts/can-fd.md) | 主升格补充：人形分层末端总线与场景表 |
| [CAN 总线（经典）](../../wiki/concepts/can-bus-protocol.md) | 历史与 8 B 瓶颈对照 |
| [CAN vs EtherCAT 选型](../../wiki/comparisons/can-vs-ethercat-joint-bus.md) | 分层架构选型读法 |
| [电机底软协议总览](../../wiki/overview/motor-drive-firmware-bus-protocols.md) | 组合套餐：EtherCAT 主干 + CAN FD 分支 |
| [EtherCAT 协议基础](../../wiki/concepts/ethercat-protocol.md) | 主干侧对照 |

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
- [x] 原始抓取落盘
