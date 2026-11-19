# Wonik Robotics Allegro Hand V5 用户手册与官网规格

> 来源归档

- **标题：** Allegro Hand V5(F4)/(F4) Plus User's Manual (v1.2.1) & Wonik Robotics 官网
- **类型：** site + official manual (PDF)
- **发布方：** Wonik Robotics
- **原始链接：** <https://www.allegrohand.com/>
- **规格/手册版本：** `AllegroHandV5(4F)UserManual_v1.2.1.pdf`（2025-03-26）与官网产品页（含 V5 / V5 Plus / V5 Sense / V6 V / V6 F 概览）
- **入库日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** 通读 `AllegroHandV5(4F)UserManual_v1.2.1.pdf` 全文（1321 行，含机械/电气/接口参数表、CAN 2.0 协议帧 ID、USB-C 固件烧录流程与 V3/V4 兼容性说明）及 `allegrohand.com` 产品线概览
- **一句话说明：** Wonik Robotics 发布的 Allegro Hand V5(4F) / V5(4F) Plus 官方硬件手册与通信协议规范，明确 4 指 16 自由度关节配置、减速比与堵转力矩、500 Hz CAN 2.0 电流力矩控制指令及可选气压指尖触觉传感器参数。
- **沉淀到 wiki：** [Allegro Hand](../../wiki/entities/allegro-hand.md)

---

## 核心硬件规格（V5(4F) vs V5(4F) Plus）

根据 `AllegroHandV5(4F)UserManual_v1.2.1.pdf`：

| 参数项 | Allegro Hand V5(4F) | Allegro Hand V5(4F) Plus |
|--------|---------------------|--------------------------|
| 构型与自由度 | 4 指（食指、中指、无名指 + 拇指）× 4 关节 = 16 个独立主动自由度 | 同左（16 DoF） |
| 驱动方式 | 直流电机 + 直连/减速传动，电流（`[mA]`）力矩控制 | 同左 |
| 减速比 | 288.35 : 1（全关节） | 食/中/无名指第 2 关节（非拇指）为 576.7 : 1，其余 288.35 : 1 |
| 堵转力矩（Stall Torque） | 0.92 Nm | 非拇指第 2 关节 1.84 Nm，其余 0.92 Nm |
| 额定/标称力矩（Nominal） | 0.23 Nm | 非拇指第 2 关节 0.46 Nm，其余 0.23 Nm |
| 最大抓持负载（Payload） | 10 kg | 15 kg |
| 整手质量 | 1000 g | 1024 g |
| 关节编码器分辨率 | 0.088 deg（弧度换算 `pos_data * pi * 0.088 / 180`） | 同左 |
| 通信总线与周期 | CAN 2.0（1 Mbps），500 Hz 控制/反馈回路（2 ms 周期） | 同左 |
| 供电需求 | 24 V DC / 5 A（120 W） | 同左 |
| 推荐 CAN 适配器 | PEAK PCAN-USB（`IPEH-002021` 无隔离 / `IPEH-002022` 光电隔离） | 同左 |
| 指尖触觉传感器（可选） | 全向硅胶封装气压传感器，量程 30–125 kPa（以 101.3 kPa 为零点基准），精度 6 Pa | 同左 |

> 官网产品目录同时披露后续扩展型号：**V5 Sense**（9-DoF 相关感知版本）、**V6 V** 与 **V6 F** 系列。

---

## 通信协议与软件兼容性要点

1. **纯电流力矩接口：** 手指控制器固件**仅接收电流型目标力矩指令（单位 `[mA]`）**，位置 PD 闭环、重力补偿与抓握力控制均由上位机 PC 实时计算（500 Hz，2 ms 周期）。
2. **标准 CAN 帧 ID（手册定义）：**
   - `0x040` / `0x041`：Servo ON / Servo OFF（上电/断电伺服使能）
   - `0x060`–`0x063`：Set Torque Finger 1–4（下发 4 指共 16 关节 `[mA]` 力矩指令）
   - `0x020`–`0x023`：Position Finger 1–4（回传 16 位整型编码器位置）
   - `0xF0` / `0xF2`：指尖气压触觉传感器数据（单位 Pa）
   - `0xEE`：错误/异常状态帧
3. **固件与版本兼容性：** V5 通过内部隐藏 USB-C 接口及 Y-modem 协议（串口 `115200` 8N1）升级固件；**V3/V4 时代上位机软件与 V5 不兼容**（通信速率、传感器与机械减速比均有变化）。
4. **ROS 2 驱动简化：** 与早期依赖 PEAK 专有 `pcan` 内核驱动（`/dev/pcan*`）的 V4 流程不同，V5 的 ROS 2 方案直接利用 Linux 主线内核自带 `peak_usb` 驱动与 **SocketCAN**（`can0` 接口），免去额外内核编译。

---

## 对 wiki 的映射

- 实体页：[Allegro Hand](../../wiki/entities/allegro-hand.md)
- 配套代码源：[wonik-allegro-hand-ros2.md](../repos/wonik-allegro-hand-ros2.md)
- 关键纠偏：确保准确描述 V5/V5 Plus 为 **4 指 16 自由度**、负载 10 kg / 15 kg、500 Hz CAN 2.0 回路及可选气压触觉传感器，避免沿用 V3/V4 旧数字。
