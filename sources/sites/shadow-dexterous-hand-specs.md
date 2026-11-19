# Shadow Dexterous Hand 官方产品页与技术规格书（E 系列 & Lite 系列）

> 来源归档

- **标题：** Shadow Dexterous Hand (Electric) Technical Specification (Sep 2025) & Lite Series Technical Specification (Sep 2025)
- **类型：** site + official specification PDFs
- **发布方：** Shadow Robot Company
- **原始链接：** <https://www.shadowrobot.com/dexterous-hand-series/>
- **规格版本：** Electric 系列规格书（Sep 2025）与 Lite / Extra Lite / Super Lite 系列规格书（Sep 2025）
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 通读 `shadow_e_spec.txt`（577 行，Electric 旗舰手完整机构、电机、感知、接口与 NDA 边界规格）、`shadow_lite_spec.txt`（487 行，Lite / Extra Lite / Super Lite 减配家族规格）及 `shadow_hand_site.txt` 官网总览
- **一句话说明：** Shadow Robot Company 官方发布的 Dexterous Hand 系列工程规格文档，详述 24 关节 / 20 主动自由度肌腱驱动旗舰手（及 Lite/Extra Lite/Super Lite 变体）的 Smart Motor 双环控制架构、1 kHz EtherCAT 总线接口、Hall 位置/肌腱张力/STF 触觉传感配置及闭源固件 NDA 边界。
- **沉淀到 wiki：** [Shadow Hand](../../wiki/entities/shadow-hand.md)

---

## Shadow Dexterous Hand (Electric) 核心规格（Sep 2025）

| 维度 | 参数详情 |
|------|----------|
| **运动学结构** | **5 指共 24 个关节 / 20 个主动可控自由度（DoF）**：食指/中指/无名指各 3 主动 DoF + 4 关节（远端指节机械耦合 `J1 <= J2`）；小指 4 主动 DoF + 5 关节（含掌部弯曲 `LF5` 0–45°）；拇指 5 DoF（`TH1`–`TH5`）；手腕 2 DoF（`WR1`–`WR2`） |
| **执行器与驱动架构** | 前臂内集成 20 台 **Maxon DC 电机**，经成对对抗肌腱拉动关节，组成模块化 **Smart Motor** 单元 |
| **整机重量与负载** | 含前臂驱动箱重 **4.3 kg**；包络抓持最大负载 **4 kg**；空载全行程运动频率约 **1.0 Hz** |
| **底层保护与控制接口** | 每台 Smart Motor 内置 PIC 微控制器（PIC18Fxx80 / PIC32），以 **5 kHz** 运行电流/力矩与 PID 安全保护内环（支持零力背驱选项）；**上位机不直接开放用户底层关节力矩控制**，用户通过 **100 Mbps EtherCAT** 以 **1 kHz** 运行外环位置/混合指令控制 |
| **位置感知** | 12 位 **Hall 效应关节角编码器**（采样 **1000 Hz**，分辨率约 **0.2°**），外加前臂 1 颗 IMU |
| **力与触觉感知** | **40 个肌腱张力（load）传感器**（采样 **500 Hz**，分辨率约 **30 mN**，出厂为未标定差分读数）；可选 **STF（Shadow Tactile Fingertips）**：每指尖 **17×3 DoF Hall 效应触觉阵列**（12 位 ADC，**1000 Hz**，标配拇指+食指 2 枚，最多可扩展至全部 5 指） |
| **供电** | **48 V DC @ 2.5 A** |
| **历史系列说明** | 早期历史产品曾包含气动（Pneumatic）手，现行 2025 年官方主力规格书面向电动 Smart Motor + EtherCAT 系列 |

---

## Shadow Hand Lite 家族规格对比（Sep 2025）

| 型号 | 手指构型 | 主动 DoF / 总关节 | 重量 | 抓持负载 | 手腕自由度 | 供电 | 备注 |
|------|----------|-------------------|------|----------|------------|------|------|
| **Lite** | 4 指（3 指 + 拇指） | 13 DoF / 16 关节 | 2.4 kg | 4 kg | 无 | 24 V @ 2.5 A | 相比 E 系列约降功耗 35% |
| **Extra Lite** | 3 指 | 10 DoF / 12 关节 | 2.1 / 1.8 kg | — | 无 | 24 V | 轻量化三指手 |
| **Super Lite** | 2 指 | 7 DoF / 8 关节 | 1.8 / 1.2 kg | — | 无 | 24 V | 双指高精度夹持构型 |

---

## 工程边界与开放状态

- **硬件与底层固件闭源：** Smart Motor 板载 PIC 固件、电路原理图与 EtherCAT 从站固件均受 **NDA（保密协议）** 保护，不对外部用户开放源码。
- **ROS 描述与消息栈开源：** 模型 URDF/xacro 与消息接口位于开源仓库 `shadow-robot/sr_common`（详见配套仓库归档）。

---

## 对 wiki 的映射

- 实体页：[Shadow Hand](../../wiki/entities/shadow-hand.md)
- 配套代码源：[shadow-robot-sr-common.md](../repos/shadow-robot-sr-common.md)
