# LunaDrive: A Delay-Compensated High-Voltage GaN FET-Based Motor Driver for Dynamic Robots with Flat BLDC Motors（arXiv:2609.21818）

> 来源归档（ingest）

- **标题：** LunaDrive: A Delay-Compensated High-Voltage GaN FET-Based Motor Driver for Dynamic Robots with Flat BLDC Motors
- **类型：** paper / hardware / motor-driver / GaN / FOC / flat-BLDC / dynamic-robot
- **arXiv abs：** <https://arxiv.org/abs/2609.21818>
- **PDF：** <https://arxiv.org/pdf/2609.21818>
- **项目页：** <https://woodrobo.github.io/lunadrive/> — 归档见 [`sources/sites/lunadrive-woodrobo-github-io.md`](../sites/lunadrive-woodrobo-github-io.md)
- **机构：** 东京大学（The University of Tokyo）机械情报学系 / JSK — Sota Yuzaki、Temma Suzuki、Hiromi Tada、Masanori Konishi、Kento Kawaharazuka、Kei Okada
- **venue：** IROS 2026
- **推广：** [X @nikhilkr](https://x.com/nikhilkr/status/2102067592785740003)（社区传播链，非一手技术页）
- **入库日期：** 2026-09-23
- **一句话说明：** **70 mm × 8.51 mm** 盘式 **GaN FET** 无框 BLDC 驱动：96 V（峰值 150 V）下连续 **30 A**（散热片）、峰值 **80 A**、电频率 **3110 Hz**；**编码器 DAEC + MCU 延迟补偿** 使超额定电压高速 FOC 稳定；线驱动跳跃负载实验验证动态机器人场景。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://woodrobo.github.io/lunadrive/> | PDF / arXiv / Video；无 GitHub |
| 目标电机 | CubeMars RO80 | 48 V 额定、21 极对、5040 rpm 空载 |
| 对照驱动 | Elmo Gold Solo Twitter | 同电压带 compact 商业驱动对比 |
| 应用 | 线轴高速负载提升 | 8.1 kg、0.5 m 线长、LiPo 24S 100 V |

## 摘要级要点

- **动机：** 动态人形/四足常用 **高功率 flat BLDC**，但商用 servo 多 **≤48 V**，限制最高转速；超压驱动需 **高耐压 + 大电流** 驱动器，Si MOSFET 难兼顾，GaN 有潜力但 **可贴电机背面** 的 compact 方案少。
- **硬件：** 控制器板（隔离 6 层 1 oz + PIC32MK + AS5147U 磁编 + CAN FD）+ 驱动板（2 oz **8 层** + EPC2305 GaN ×3 半桥）；直径 70 mm、厚 8.51 mm；106 颗 MLCC 按 **有效面电容密度** 选型。
- **控制：** 25 kHz FOC + SVM；PC↔驱动 1 kHz USB-CAN FD；MCU 间 2 kHz SPI；**θ′ = θ + Td·ωe** 延迟补偿 + 编码器 **DAEC**。
- **关键数（论文）：** 96 V 无补偿最高 ~5760 rpm → 有补偿 **8890 rpm（3110 Hz 电频率）**；散热片连续 30 A；峰值 80 A（T-Motor U13II 负载）。
- **对比：** 同厚度维度上连续/峰值电流优于 Gold Solo Twitter（85–100 V 档）。

## 核心摘录（面向 wiki 编译）

### 1) 延迟补偿动机

- RO80：**p = 21** 极对 → 超压时电频率 ~**3000 Hz**，远超常见 BLDC 驱动推荐 ~700 Hz。
- 系统延迟（编码器 + MCU + 驱动级）在高 **ωe** 下造成 d–q 轴耦合，无补偿时过流保护触发。

### 2) 热与电流

| 散热 | 连续电流（96 V, θe=0 保守测法） |
|------|--------------------------------|
| 无散热片 | 13 A |
| Flat heat sink | 28 A |
| Finned heat sink | 30 A |
| 峰值（0.5 s） | 80 A |

### 3) 开源状态（项目页，2026-09-23）

| 组件 | 状态 |
|------|------|
| 论文 PDF / arXiv / 演示视频 | 已公开 |
| PCB 设计 / 固件 / BOM | **截至入库日项目页未列 GitHub** |
| 商业产品 | 未发布 |

## 对 wiki 的映射

- 新建：[paper-lunadrive](../../wiki/entities/paper-lunadrive.md)
- 交叉：[field-oriented-control](../../wiki/concepts/field-oriented-control.md)、[joint-encoder-selection](../../wiki/concepts/joint-encoder-selection.md)、[paper-low-cost-modular-actuator-katz](../../wiki/entities/paper-low-cost-modular-actuator-katz.md)

## 当前提炼状态

- [x] arXiv + 项目页核查（步骤 2.5）
- [x] 开源：未列代码仓库
- [ ] 若作者发布硬件/固件再补 `sources/repos/`
