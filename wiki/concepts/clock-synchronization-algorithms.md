---
type: concept
tags: [real-time, clock-sync, ptp, ethercat, hardware, networking]
status: complete
updated: 2026-05-02
related:
  - ./ethercat-protocol.md
  - ./lcm-basics.md
  - ./ros2-basics.md
  - ../formalizations/control-loop-latency-modeling.md
  - ../formalizations/udp-multicast-dynamics.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/papers/sim2real.md
summary: "时钟同步算法：把 PTP（IEEE 1588）与 EtherCAT 分布式时钟（DC）抽象成一套主时钟选举 + 偏置/漂移估计 + 周期校正的统一框架，解释多板卡运控中如何把跨节点抖动压到亚微秒级。"
---

# 时钟同步算法 (Clock Synchronization Algorithms)

**时钟同步算法** 解决一个看似简单、却在多板卡运控里反复折腾人的问题：**两台机器的时间到底差多少，怎么把这个差距持续压到与控制环路相比可忽略的水平？** 在人形机器人里，IMU 在一块板、关节驱动器在另一条总线、视觉感知挂在第三个 NUC，如果时钟不同步，融合出来的状态就会带上一个"网络抖动"形状的虚假项，直接污染 [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md) 与 [UDP 组播动力学](../formalizations/udp-multicast-dynamics.md) 中的所有测量。

## 为什么单靠 NTP 不够

家用与服务器场景常用的 **NTP** 在跨广域网下精度约 $\sim 1\sim 10\ ms$，最好的 LAN 配置也只能到 $\sim 100\ \mu s$。这个量级对**视频通话**够用，对 **1 kHz 力矩闭环**完全失格——它本身已经接近一个控制周期。机器人需要的是**亚微秒到纳秒级**的同步，于是引出两套主流方案：

- **PTP (IEEE 1588)**：跑在普通以太网上的软/硬件协议，常见精度 $100\ ns\sim 1\ \mu s$。
- **EtherCAT 分布式时钟 (DC)**：跑在 EtherCAT 总线上的特化方案，精度通常 $< 100\ ns$。

两者在数学结构上**高度同构**，理解了一边，另一边就是工程参数差异。

## 共同的三段式抽象

任何实用的时钟同步算法都可以拆成三个阶段：

1. **主时钟选举 (Master Selection)**：决定整个域里谁是时间基准。PTP 用 **BMCA**（Best Master Clock Algorithm），按 priority1 → class → accuracy → variance → priority2 → clockIdentity 的字典序投票；EtherCAT DC 默认把第一个有 DC 能力的从站当作参考时钟（Reference Clock），由主站显式指定。
2. **偏置与漂移估计 (Offset & Skew Estimation)**：周期性交换时间戳，估出本地时钟与主时钟的瞬时偏差 $\theta(t)$ 与频率比 $\gamma(t)$。
3. **校正注入 (Discipline)**：把 $\theta, \gamma$ 反馈给本地振荡器或软件钟，通常用 **PI 控制律** 调整频率，避免硬跳变。

这三段对应工程现场最常见的三类故障：选主抖动（频繁切换 GM）、估计噪声（链路非对称）、校正过冲（PI 增益太大引起振荡）。

## PTP 的两次握手时间戳

PTP 通过四个时间戳估计单跳偏置（**delay request-response** 机制）：

- 主站在 $t_1$ 发出 `Sync`，从站在 $t_2$ 收到；
- 从站在 $t_3$ 发 `Delay_Req`，主站在 $t_4$ 收到；
- 主站把 $t_4$ 通过 `Delay_Resp` 告诉从站。

在**链路对称**假设下，单向传播延迟与时钟偏置分别为：

$$
d = \frac{(t_2 - t_1) + (t_4 - t_3)}{2}, \qquad \theta = \frac{(t_2 - t_1) - (t_4 - t_3)}{2}
$$

工程上的关键改进：

- **One-step Sync**：把 $t_1$ 直接写在 `Sync` 帧里，省去 `Follow_Up`，减少抖动；
- **硬件时间戳 (HWTS)**：把 $t_1\sim t_4$ 在 PHY 层打戳，去掉协议栈抖动；
- **Transparent Clock (TC)**：交换机在转发时把驻留时间累加到帧里，消除多跳路径的抖动累积；
- **Boundary Clock (BC)**：交换机本身做一次同步、再以主时钟身份向下游发 Sync，限制误差线性累积。

实测同步精度近似为

$$
\sigma_\theta \approx \sqrt{\sigma_{\text{HWTS}}^2 + \tfrac{1}{2}\sigma_{\Delta d}^2}
$$

其中 $\sigma_{\Delta d}$ 是上下行延迟非对称性的标准差——它是 PTP 的真正天花板，对 PHY 与交换机选型最敏感。

## EtherCAT DC：把同步放在总线物理层

EtherCAT 走另一条路：既然总线是确定性环网，不如让所有从站在 ASIC 里维护一个 64 位 DC 时钟，由主站做一次性测量后下发偏置。其工作流程：

1. 主站发出一帧 `BRW(0x900)`，途经每个 DC 从站时打两个时间戳（来时、去时）；
2. 报文绕环回来，主站读出每段 **传播延迟差**，计算每个从站相对参考从站的偏置 $\theta_i$；
3. 主站通过 `FPWR` 把 $\theta_i$ 写回从站的 `0x920` 寄存器，从此本地 DC 时钟在硬件层补偿；
4. 主站周期性发 **DC Sync 报文** 维护漂移，带宽几乎可忽略。

由于偏置是用**同一帧的来回时间戳**算的，链路非对称在这里几乎被消去；又因为补偿做在 ASIC 里，软件抖动完全不参与。这两点合起来解释了为什么 EtherCAT DC 在 100 个轴的人形机器人上仍能稳定到 $< 100\ ns$，而同样数量的 PTP 节点在普通交换机上只能做到 $\sim 1\ \mu s$。

## gPTP / 802.1AS：汽车与机器人共用的子集

**gPTP (IEEE 802.1AS)** 是 PTP 的严格子集，强制使用：

- 单一同步域、单一 BMCA；
- 必须开启 peer-to-peer Path Delay 机制（`Pdelay_Req/Resp`），逐跳测延迟；
- 全网交换机都必须是 TC 或 BC，不允许"哑交换"穿透。

它牺牲了 PTP 的灵活性，换来**确定性最坏情况误差**，被 TSN（Time-Sensitive Networking）做基础。机器人侧一旦上 TSN 交换机，gPTP 就成为默认选项。

## 工程对比表

| 方案 | 典型精度 | 硬件依赖 | 适用场景 |
|------|---------|---------|---------|
| NTP | $1\sim 100\ ms$ | 无 | 日志时间戳、跨广域网 |
| PTP (软件 TS) | $10\sim 100\ \mu s$ | 普通网卡 | 一般感知融合 |
| PTP (硬件 TS + BC/TC) | $100\ ns\sim 1\ \mu s$ | 1588 网卡 + 1588 交换机 | 跨板卡视觉/IMU 融合 |
| gPTP / 802.1AS | $100\ ns$ 量级 | TSN 交换机 | 车规、TSN 总线 |
| EtherCAT DC | $< 100\ ns$ | EtherCAT ASIC | 人形 / 工业关节运控 |

实际部署里，一台人形机器人常常**同时跑两套**：关节驱动走 EtherCAT DC、上层感知节点走硬件 PTP，两者再通过主站板卡的 `phc2sys` 做一次跨域桥接。

## 与控制环路的耦合

时钟同步质量直接进入两个上层模型：

- **延迟测量去偏**：[控制环路延迟建模](../formalizations/control-loop-latency-modeling.md) 中的 $T_{\text{bus}}$、$T_{\text{compute}}$ 都依赖跨节点时间戳相减，$\sigma_\theta$ 会以加性方差形式直接抬高总抖动；
- **多订阅者一致性**：[UDP 组播动力学](../formalizations/udp-multicast-dynamics.md) 里 $\Delta_{ij,k}$ 的极值分布只有在跨节点时钟同步好之后才有物理意义，否则把时钟漂移当成网络抖动。

工程经验法则：**$\sigma_\theta$ 必须 $< T_{\text{ctrl}} / 100$**，否则同步误差会成为闭环带宽的隐形上限。

## 常见误区

- **误把 NTP 当作"够用"**：在 1 ms 控制周期下，NTP 的 100 µs 精度已等价于一次完整环路抖动，会污染所有 deadline 统计。
- **忘记关闭 Linux NTP/chrony**：开了 PTP 又留着 NTP，两个守护进程会互相打架，时钟出现锯齿状跳变。
- **不开 HWTS**：软件时间戳的抖动至少是 microsecond 级，PTP 的硬件优势全部归零。
- **跨域桥接精度损失**：`phc2sys` 单方向的同步精度通常只有 $\sim 1\ \mu s$，把 EtherCAT 的 ns 级精度桥到 PTP 域时务必注意。
- **链路非对称未补偿**：单模 / 多模光纤、不同长度双绞线都会引入恒定偏置，PTP 算出来的 $\theta$ 会带上系统误差，需要离线测量后填到 `asymmetry` 寄存器。

## 学这个方法时最该盯住的点

1. **同步是"估计 + 控制"两步走**：先估偏置/漂移，再用 PI 把振荡器拉过去，光估不控会漂回去。
2. **链路对称是 PTP 的隐含信仰**：一旦不成立（光纤不等长、上下行队列不同），精度立刻退化到 $\sigma_{\Delta d}$ 主导。
3. **EtherCAT DC 的强项在物理层**：抖动主要来自 ASIC 时钟稳定度，与软件几乎无关。
4. **跨域永远是最弱一环**：把 PTP 桥到 DC，或把 ROS 2 节点接到 PTP 域，都要单独测最终精度。

## 关联页面

- [EtherCAT 协议基础](./ethercat-protocol.md)
- [LCM 基础](./lcm-basics.md)
- [ROS 2 基础](./ros2-basics.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)
- [UDP 组播动力学](../formalizations/udp-multicast-dynamics.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
- IEEE Std 1588-2019, *Standard for a Precision Clock Synchronization Protocol for Networked Measurement and Control Systems*.
- IEEE Std 802.1AS-2020, *Timing and Synchronization for Time-Sensitive Applications*.
- EtherCAT Technology Group, *EtherCAT Distributed Clocks: System Description*.
- Mills, D. L., *Computer Network Time Synchronization: The Network Time Protocol on Earth and in Space*, 2nd ed., CRC Press, 2010.

## 推荐继续阅读

- linuxptp 项目文档（`ptp4l`, `phc2sys` 配置示例）
- IgH EtherCAT Master 手册：DC 配置与 `ec_slave_config_dc()` API
