---
type: formalization
tags: [real-time, communication, middleware, lcm, networking, math]
status: complete
updated: 2026-05-01
related:
  - ./control-loop-latency-modeling.md
  - ../concepts/lcm-basics.md
  - ../concepts/ros2-basics.md
  - ../comparisons/ros2-vs-lcm.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/papers/sim2real.md
summary: "UDP 组播动力学：把 LCM 等基于 UDP 多播的中间件抽象成丢包-迟到-乱序-不一致四类随机过程，给出丢包率、订阅者间一致性偏差与控制端可用观测窗口之间的可计算约束。"
---

# UDP 组播动力学 (UDP Multicast Dynamics)

**UDP 组播动力学** 把 [LCM](../concepts/lcm-basics.md) 这类“即发即弃”的多播中间件，抽象成一个发送方 → 多个接收方的随机过程网络。它不研究单条消息怎么走，而是回答四个工程问题：**丢多少包？谁先收到？大家收到的内容是否一致？数据老化到什么程度还能拿来闭环？** 把这些问题写成公式，意味着抖动不再只是“真机感觉卡”，而是可以套到 [控制环路延迟建模](./control-loop-latency-modeling.md) 里去定量预算。

## 为什么要形式化

UDP 没有重传，没有 ACK。LCM 的设计哲学就是“最新一帧最重要，丢了就丢了”。但当一台人形机器人挂着 IMU、关节编码器、力矩驱动器、视觉感知，全部以 1 kHz 走 UDP 多播时，几个问题马上跳出来：

- 同一条 IMU 报文，控制进程和日志进程谁先收到？差多少？
- 多播交换机带宽打满时丢包率会从 $10^{-5}$ 突变到 $10^{-2}$，控制器是否还稳？
- 状态估计器在 1 ms 内没收到新一帧编码器，是该外推还是该退化？

不写下来，就只能凭经验调 socket buffer。写下来之后，**丢包率 $p$、抖动 $\sigma$、订阅者一致性偏差 $\Delta$、控制可用窗口 $W$** 之间的换算就是几个不等式。

## 多播信道的四类随机事件

设发送方在 $t_k$ 时刻发出第 $k$ 条报文。对接收方 $i \in \{1, \dots, N\}$，把每条报文映射到四个互斥事件：

| 事件 | 含义 | 标记 |
|------|------|------|
| **成功 (delivered)** | 在截止时间 $t_k + D$ 前收到 | $S_{i,k} = 1$ |
| **丢包 (lost)** | 永远未收到 | $L_{i,k} = 1$ |
| **迟到 (late)** | 收到但超过 $D$，按业务相当于丢 | $T_{i,k} = 1$ |
| **乱序 (reordered)** | 收到，但发生在第 $k+1$ 条之后 | $R_{i,k} = 1$ |

满足 $S_{i,k} + L_{i,k} + T_{i,k} + R_{i,k} = 1$（迟到与乱序可重叠时按业务最严格的那类计）。所有形式化都围绕这四个指示变量的分布展开。

## 丢包率：Gilbert-Elliott 两态模型

UDP 丢包通常**不是独立同分布**，而是“突发性”——交换机短时拥塞会一连丢十几帧。最简也最常用的模型是 **Gilbert-Elliott** 两态马尔可夫链，状态 $X_k \in \{G, B\}$，其中：

- $G$：表示链路处于「畅通」状态（Good），丢包率极低；
- $B$：表示链路处于「拥塞 / 突发丢包」状态（Bad），丢包率显著抬高。


$$
P = \begin{bmatrix} 1 - p_{GB} & p_{GB} \\ p_{BG} & 1 - p_{BG} \end{bmatrix}, \qquad
\Pr(L_{i,k} = 1 \mid X_k) = \begin{cases} \varepsilon_G & X_k = G \\ \varepsilon_B & X_k = B \end{cases}
$$

其中 $\varepsilon_G \ll \varepsilon_B$。稳态丢包率为

$$
\bar{p} = \pi_G \varepsilon_G + \pi_B \varepsilon_B, \qquad \pi_B = \frac{p_{GB}}{p_{GB} + p_{BG}}
$$

平均突发长度 $\mathbb{E}[B] = 1/p_{BG}$。这是工程上选择**冗余间隔**的关键量：状态估计器要能跨越 $\sim 3 \mathbb{E}[B]$ 个连续丢包窗口，否则会出现“估计断片”。

## 端到端延迟：迟到的尾部分布

成功送达的一条报文，单接收方延迟可分解为

$$
T_{i,k} = T_{\text{send}} + T_{\text{switch}} + T_{\text{recv}} + T_{\text{stack}}
$$

- $T_{\text{send}}$：发送方内核协议栈到网卡；
- $T_{\text{switch}}$：交换机存储转发 + 多播复制 ($\propto N$ 订阅者数量)；
- $T_{\text{recv}}$：接收方网卡 → socket buffer；
- $T_{\text{stack}}$：用户态 `recvmsg` + 反序列化。

每段都右偏，整体仍是各分量的卷积。**业务上的“迟到”** 是 $T_{i,k} > D$，其概率

$$
p_{\text{late}}(D) = \Pr(T_{i,k} > D) = 1 - F_T(D)
$$

可以由经验直方图直接读出。把 $p_{\text{late}}(D)$ 加到丢包率上，就是控制器实际“看不到第 k 帧”的总概率：

$$
p_{\text{miss}}(D) = \bar{p} + (1 - \bar{p}) p_{\text{late}}(D)
$$

## 多订阅者一致性偏差

多播的真正价值在于“一发多收”，但不同接收方的栈延迟不同，于是出现**一致性偏差**。定义订阅者 $i, j$ 在第 $k$ 条报文上的偏差

$$
\Delta_{ij,k} = T_{i,k} - T_{j,k}
$$

最关心的是组内最大偏差

$$
\Delta_k^{\max} = \max_{i,j} |T_{i,k} - T_{j,k}|
$$

经验上 $\Delta_k^{\max}$ 服从 Gumbel 型极值分布，其期望随订阅者数 $N$ 缓慢增长 $\sim \sigma\sqrt{2\ln N}$。这意味着：把日志、可视化、状态估计三类订阅者放在同一组播组上，调度抖动会被“最慢那个”放大。**工程对策**：把硬实时订阅者（控制环）单独走一个组播组或共享内存，绝不和 GUI/日志混播。

## 数据老化与控制可用窗口

控制器在 $\tau$ 时刻拿到的最新观测时间戳为 $t_{\text{last}}(\tau)$，定义**数据年龄 (Age of Information, AoI)**：

$$
A(\tau) = \tau - t_{\text{last}}(\tau)
$$

在 UDP 多播下 $A(\tau)$ 是一个分段线性的锯齿过程：每来一帧瞬时跌到 $T_{\text{stack}}$，无新帧时以斜率 1 累加。稳态平均年龄

$$
\mathbb{E}[A] = \frac{1}{f_{\text{ctrl}}} \cdot \frac{1}{1 - p_{\text{miss}}} \cdot \tfrac{1}{2} + \mathbb{E}[T_{\text{stack}}]
$$

控制器可用观测窗口 $W$ 是 $A(\tau) \leq A_{\max}$ 的时间占比，要求

$$
\Pr(A(\tau) > A_{\max}) \leq \delta
$$

把它代回 [控制环路延迟建模](./control-loop-latency-modeling.md) 中的相位裕度约束 $\omega_c \cdot (T_{\text{loop}} + A_{\max}) \leq \pi/6$，就把“多播丢包率”和“可用控制带宽”串起来了。

## 工程可观测量

落地这套形式化，通常归结为四组测量：

1. 在每条报文里写入发送时间戳，接收侧用 `clock_gettime(CLOCK_MONOTONIC_RAW)` 算 $T_{i,k}$，按窗口统计 99 / 99.9 分位；
2. 用序列号检测漏号，估计稳态丢包率 $\bar{p}$ 与突发长度分布；
3. 双订阅者交叉日志，计算 $\Delta_{ij,k}$ 的极值分布；
4. 控制环里持续记录 AoI，绘 $A(\tau)$ 锯齿曲线，找 deadline miss 的根因。

`lcm-spy`、`tcpdump -G` 与 `ss -mu` 是这套测量的最低成本工具组合。

## 方法局限性

- **Gilbert-Elliott 是二阶近似**：实际交换机拥塞往往涉及多状态 / 自相似，长尾比两态模型重；
- **独立性假设过强**：发送方阻塞时四类事件高度耦合（一旦 socket buffer 溢出，丢包与延迟同时爆涨）；
- **不刻画时钟漂移**：跨板卡的 $\Delta_{ij,k}$ 测量必须先做 PTP 同步，否则把时钟偏移当成网络抖动；
- **未考虑应用层去重**：状态估计器若做了 hold-last-sample，等价改变了 $A_{\max}$ 的物理含义。

## 学这个方法时最该盯住的点

1. **丢包是突发的**：用 $\bar{p}$ 看不出风险，必须看突发长度分布；
2. **迟到等于丢包**：硬实时里 $D$ 之后到达的帧已经没用，全算入 $p_{\text{miss}}$；
3. **AoI 比延迟更接近控制语义**：控制器关心“手里这条数据多老”，而不是“这条数据走了多久”；
4. **订阅者越多，最慢者主导**：组播组的设计要和实时性等级对齐，不能图方便混播。

## 关联页面

- [控制环路延迟建模](./control-loop-latency-modeling.md)
- [LCM 基础](../concepts/lcm-basics.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [ROS 2 vs LCM 选型对比](../comparisons/ros2-vs-lcm.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
- Huang, A. S., Olson, E., Moore, D. C., *LCM: Lightweight Communications and Marshalling*, IROS 2010.
- Gilbert, E. N., *Capacity of a Burst-Noise Channel*, Bell System Technical Journal, 1960.
- Kaul, S., Yates, R., Gruteser, M., *Real-Time Status: How Often Should One Update?*, INFOCOM 2012.

## 推荐继续阅读

- RFC 1112 / 4604 (IGMP 与 SSM 多播)
- Tanenbaum, *Computer Networks* (5th ed.) 第 4 章：多址访问与多播
