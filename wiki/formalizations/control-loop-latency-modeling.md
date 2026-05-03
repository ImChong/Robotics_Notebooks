---
type: formalization
tags: [real-time, control, hardware, latency, middleware, math]
status: complete
updated: 2026-04-30
related:
  - ../concepts/ethercat-protocol.md
  - ../concepts/lcm-basics.md
  - ../concepts/ros2-basics.md
  - ../concepts/sim2real.md
  - ../concepts/clock-synchronization-algorithms.md
  - ../queries/real-time-control-middleware-guide.md
  - ./udp-multicast-dynamics.md
sources:
  - ../../sources/papers/sim2real.md
summary: "控制环路延迟建模：把一次力矩闭环拆成传感、总线、计算、调度、执行五段独立随机变量的累加，并给出端到端延迟分布、抖动与控制带宽之间的可计算约束。"
---

# 控制环路延迟建模 (Control Loop Latency Modeling)

**控制环路延迟建模** 把一次「读传感 → 算策略 → 写力矩」的闭环拆解成若干段独立的延迟分量，用随机变量与卷积刻画端到端时延及抖动，从而把"机器人为什么在真机上抽搐"这种工程现象变成一个可定量分析、可优化的数学问题。

## 为什么要形式化

在仿真里，控制器假设当前观测 $o_t$ 和下一时刻的力矩指令 $\tau_{t+1}$ 之间没有延迟。但在真机上：

- 传感数据通过总线到达 CPU 需要时间；
- 策略推理本身要 $\sim\!\text{ms}$ 级算力；
- 内核调度可能让控制线程被抢占；
- 力矩指令再经过一次总线才能到达驱动器。

如果不把这条链路写成一个公式，工程师就只能凭"感觉"调控制频率；写下来之后，**最坏情况延迟 $T_{\max}$、控制带宽 $\omega_c$、抖动方差 $\sigma^2$** 之间的换算关系就一目了然。

## 端到端延迟的加性分解

设一次控制环路的总延迟为 $T_{\text{loop}}$，按物理路径分解为 5 段：

$$
T_{\text{loop}} = T_{\text{sense}} + T_{\text{bus,up}} + T_{\text{compute}} + T_{\text{sched}} + T_{\text{bus,dn}}
$$

| 分量 | 含义 | 典型量级 |
|------|------|---------|
| $T_{\text{sense}}$ | 传感器自身的采样、滤波、ADC 转换 | $50\ \mu s\sim 1\ ms$ |
| $T_{\text{bus,up}}$ | 从驱动器/IMU 经 EtherCAT/CAN 到 CPU | $0.25\sim 2\ ms$ |
| $T_{\text{compute}}$ | 策略前向 / WBC QP / 状态估计 | $0.5\sim 10\ ms$ |
| $T_{\text{sched}}$ | 内核调度抖动 + 上下文切换 | $0.05\sim 5\ ms$ |
| $T_{\text{bus,dn}}$ | 力矩指令从 CPU 经总线写回驱动器 | $0.25\sim 2\ ms$ |

每一项都是随机变量，其分布通常**右偏**（minor 多数样本接近均值，少数长尾）。把它们近似为相互独立，则总延迟分布是各分量分布的卷积：

$$
p_{T_{\text{loop}}}(t) = \big(p_{\text{sense}} \ast p_{\text{bus,up}} \ast p_{\text{compute}} \ast p_{\text{sched}} \ast p_{\text{bus,dn}}\big)(t)
$$

期望和方差可加：

$$
\mathbb{E}[T_{\text{loop}}] = \sum_i \mu_i, \qquad \mathrm{Var}[T_{\text{loop}}] = \sum_i \sigma_i^2
$$

实时系统真正关心的是 99% 或 99.9% 分位 $T_{\text{loop}}^{(p)}$，它必须严格小于一个控制周期 $T_{\text{ctrl}} = 1/f_{\text{ctrl}}$，否则就会发生 **deadline miss**。

## 总线延迟：报文长度模型

EtherCAT、CAN、UDP 等总线的单段延迟可以写成

$$
T_{\text{bus}} = T_{\text{access}} + \frac{L_{\text{frame}}}{B} + T_{\text{prop}}
$$

其中 $T_{\text{access}}$ 是介质访问等待（CSMA、DC 同步窗等），$L_{\text{frame}}$ 是报文长度（bit），$B$ 是链路带宽（bit/s），$T_{\text{prop}}$ 是物理传播延迟。对于一台 100 关节人形机器人，$L_{\text{frame}}$ 在 EtherCAT 下约 $1500$ 字节，100 Mbit/s 链路上传输部分约 $120\ \mu s$，再加上 $T_{\text{access}}$ 与从站处理才得到 [EtherCAT 协议](../concepts/ethercat-protocol.md) 实测 $250\ \mu s$ 的循环。

## 计算延迟：批量与缓存命中

策略推理的计算延迟近似为

$$
T_{\text{compute}} = \alpha N_{\text{ops}} + \beta(1 - h_{\text{cache}}) + T_{\text{kernel-launch}}
$$

- $N_{\text{ops}}$：网络浮点操作数；$\alpha$ 与 GPU 实际算力相关；
- $h_{\text{cache}} \in [0, 1]$：缓存命中率；缓存抖动会把均值抬高 $\beta$；
- $T_{\text{kernel-launch}}$：每次 CUDA / NPU 启动的固定开销，对小模型主导。

这个公式直接说明了为什么在真机上"小网络 + 高频推理"通常优于"大网络 + 低频推理"：$T_{\text{kernel-launch}}$ 被强行摊到每帧，与 $N_{\text{ops}}$ 不可加。

## 调度抖动：PREEMPT_RT 的形式化角色

在标准 Linux 中，$T_{\text{sched}}$ 的尾部由非自愿上下文切换、SoftIRQ、ksoftirqd 等支配，呈现重尾分布。开 [PREEMPT_RT](../queries/real-time-control-middleware-guide.md) 后，调度延迟近似为：

$$
T_{\text{sched}}^{(p)} \approx T_{\text{wcl}} = \max\Big( \text{IRQ-off windows} \Big) + T_{\text{wakeup}}
$$

其中 $T_{\text{wcl}}$（worst-case latency）可由 `cyclictest` 长时间运行测得。CPU 隔离 (`isolcpus`, `nohz_full`, `rcu_nocbs`) 的目标就是把上式中 IRQ-off 窗口压到只剩硬件中断必经的几十微秒。

## 端到端延迟与控制带宽的关系

把控制环路看作一个采样系统，**延迟 $T_{\text{loop}}$ 在频域上等效为一个全通相位滞后**：

$$
H_{\text{delay}}(j\omega) = e^{-j\omega T_{\text{loop}}}
$$

要保证闭环稳定，常用经验法则是相位裕度 $\geq 60^\circ$，即

$$
\omega_c \cdot T_{\text{loop}} \leq \tfrac{\pi}{6} \quad\Longleftrightarrow\quad \omega_c \leq \frac{\pi}{6\, T_{\text{loop}}}
$$

也就是说，端到端延迟 $5\ ms$ 对应的可用控制带宽上限约 $\omega_c \approx 105\ \text{rad/s} \approx 17\ \text{Hz}$。**抖动 $\sigma$ 也必须计入**：把 $T_{\text{loop}}$ 替换为 $\mathbb{E}[T_{\text{loop}}] + 3\sigma$ 即可拿到一个保守的可用带宽估计。

## 与采样定理的连接

控制周期 $T_{\text{ctrl}}$ 与端到端延迟必须满足：

$$
T_{\text{loop}}^{(p)} < T_{\text{ctrl}} < \frac{1}{2 \omega_c / (2\pi)}
$$

左半部分保证不丢周期（实时约束），右半部分保证不混叠（Nyquist）。这两个不等式结合在一起就给出了"控制频率不是越高越好"的形式化解释：当 $T_{\text{ctrl}}$ 接近 $T_{\text{loop}}^{(p)}$ 时，deadline miss 概率指数级增长，反而不如降一档频率换取低抖动。

## 工程检查清单

把上述形式化落到实物上，最终通常归结为四组测量：

1. 用示波器或 DC 时间戳测量 $\mathbb{E}[T_{\text{bus}}]$ 与最坏情况；
2. 在控制循环里打 `clock_gettime(CLOCK_MONOTONIC_RAW)` 时间戳，统计 $T_{\text{compute}}$ 的 99 分位；
3. 用 `cyclictest --priority=98 --duration=1h` 测 $T_{\text{sched}}$ 长尾；
4. 把上面三组分布做卷积或直接把对应时间戳相加，得到 $T_{\text{loop}}$ 的经验分布并和 $T_{\text{ctrl}}$ 对比。

## 方法局限性

- **独立性假设过强**：真实系统中计算延迟与调度延迟会强相关（高负载同时拖累两者），卷积只是一阶近似。
- **加性分解漏掉了反馈耦合**：一些 OS 调度策略会在丢周期后追赶下一帧，使得短期内"看起来正常"但相位仍偏。
- **频域线性近似**：$e^{-j\omega T}$ 假设 $T$ 是常数，对显著抖动的系统可能要走 LMI / 时变系统稳定性分析。

## 学这个方法时最该盯住的点

1. **延迟必须按物理段拆**：抖动来自哪一段，对应的工程对策完全不同（总线、内核、模型）。
2. **均值不重要，分位数才重要**：硬实时只看 $T^{(99.9\%)}$，看均值会被尾部坑死。
3. **延迟 ↔ 带宽 ↔ 频率三角**：先固定能容忍的延迟，再反推可用带宽与控制周期，不要反过来定频率。

## 关联页面

- [EtherCAT 协议基础](../concepts/ethercat-protocol.md)
- [LCM 基础](../concepts/lcm-basics.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [Sim2Real](../concepts/sim2real.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [UDP 组播动力学](./udp-multicast-dynamics.md)
- [时钟同步算法](../concepts/clock-synchronization-algorithms.md)

## 参考来源

- [sources/papers/sim2real.md](../../sources/papers/sim2real.md)
- Liu, J., *Real-Time Systems*, Prentice Hall, 2000.
- Buttazzo, G. C., *Hard Real-Time Computing Systems*, Springer, 2011.
- Khalil, H. K., *Nonlinear Systems* (Sec. 时间延迟系统稳定性), 3rd ed.
