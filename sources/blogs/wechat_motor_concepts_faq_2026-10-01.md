# 电机相关概念常识的原理

> 来源归档（blog / 微信公众号）

- **标题：** 电机相关概念常识的原理
- **类型：** blog
- **作者：** 待核实（WebFetch 未解析公众号 nick_name）
- **原始链接：** https://mp.weixin.qq.com/s/wHQkEv6UdfJd2gE2dTJuLw
- **入库日期：** 2026-10-01
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文（本环境无 `wechat-article-for-ai` 快照）
- **原始抓取落盘：** [`sources/raw/wechat_motor_concepts_faq_2026-10-01.md`](../raw/wechat_motor_concepts_faq_2026-10-01.md)
- **一句话说明：** 32 问电机入门 FAQ：电磁转矩与反电动势、TN/功率、绕组稳态方程、极对与同步转速、有刷/BLDC/PMSM/异步/步进/伺服差异、FOC 与 V/F、三环与有感无感、额定与工作制、负载特性与惯量匹配、无人机/EV/工业机器人/风机等选型读法。
- **步骤 2.5（开源核查）：** 科普文，**无**项目页或代码仓库。
- **沉淀到 wiki：** [`wiki/concepts/electric-motor-fundamentals.md`](../../wiki/concepts/electric-motor-fundamentals.md)

## 核心摘录（归纳）

### 电磁与机械量

- 转矩来自安培力；旋转产生反电动势 $E \propto \omega$，决定空载最高转速；启动时 $E=0$ 需限流。
- 机械功率 $P=T\omega$；工程式 $T=9550 P/n$（$P$ kW，$n$ rpm）。
- 稳态：$U = IR + L di/dt + E$；电角度 = 机械角 × 极对数 $p$；$n=60f/p$。

### 机型对比（选型锚点）

| 类型 | 控制/波形 | 典型优点 | 典型缺点 |
|------|-----------|----------|----------|
| 有刷 DC | 换向器，调压/PWM | 简单便宜 | 刷磨损、效率低 |
| BLDC | 六步/方波，梯形反电势 | 寿命长、效率高 | 需位置/无感 |
| PMSM | 正弦反电势，FOC | 转矩脉动小、精度高 | 驱动复杂 |
| 异步 | V/F 或矢量 | 坚固、大功率 | 低速差、效率略低 |
| 步进 | 开环脉冲 | 定位简单 | 高速扭矩掉、易丢步 |
| 伺服 | PMSM+编码器+驱动器 | 高动态高精度 | 成本高 |

### 控制栈

- **FOC：** $i_d/i_q$ 解耦，转矩电流 $i_q$；需电流采样与转子位置。
- **V/F：** 开环恒磁通，风机水泵类。
- **三环：** 电流（内）→ 速度 → 位置（外），带宽由内向外递减。
- **有感 vs 无感：** 零速满转矩/精定位选霍尔或编码器；高速成本敏感可选反电势/磁链估计。

### 选型步骤（文内）

1. 负载特性（摩擦、重力、惯量、运动曲线）→ 2. 峰值/有效转矩与最高转速 → 3. 初选电机 → 4. 惯量比 $J_L/J_M \lesssim 3\sim5$ → 5. 热与工作制 S1/S2/S3 → 6. 电源、驱动、安装与环境。

## 对 wiki 的映射

- 概念页：[electric-motor-fundamentals](../../wiki/concepts/electric-motor-fundamentals.md)（新建）
- 交叉深化：[FOC](../../wiki/concepts/field-oriented-control.md)、[TN 曲线](../../wiki/concepts/motor-torque-speed-curve.md)、[人形 101 电机节](../../wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md)
