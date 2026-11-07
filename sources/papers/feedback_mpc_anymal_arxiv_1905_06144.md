# Feedback MPC：力矩控制腿足机器人的反馈 MPC（通读）

> 来源归档（原文通读 · 支撑 `wiki/queries/mpc-tuning-guide.md`）

- **标题：** Feedback MPC for Torque-Controlled Legged Robots
- **作者：** Ruben Grandia, Farbod Farshidian, René Ranftl, Marco Hutter（ETH RSL / Intel Labs）
- **类型：** paper
- **链接：** <https://arxiv.org/abs/1905.06144>（PDF：<https://arxiv.org/pdf/1905.06144>）；IROS 2019
- **版本 / 日期：** arXiv v2，2019-08-09
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、§I 引言、§II SLQ / 松弛障碍函数 / 约束 Riccati 推导大意、§III 频域整形代价、§IV 实验（仿真反馈 MPC、摩擦锥附近的反馈增益、频率整形、真机负载与动态步行）与 §V 结论；约束 Riccati 的推导细节与参考文献仅浏览。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题：** 力矩级全身 MPC 的更新率不足以应对模型误差；常用做法是外挂高频跟踪器。本文改用 DDP/SLQ-MPC 的 **反馈策略**（Riccati 增益）填补低 MPC 更新率与执行器指令率之间的缺口。
- **约束：** 用 **松弛障碍函数** 处理摩擦锥不等式（扰动二阶锥，固定 $\mu=0.5,\delta=0.1$ 时所有约束严格满足）；不加锥约束时最优解会在后腿产生负接触力。
- **频率整形代价：** 对输入施加高通滤波器 $r_i(\omega)=(1+\beta_i j\omega)/(1+\alpha_i j\omega)$，抑制高频增益以尊重执行器带宽；实验取 $\alpha=0.01$，接触力输入 $\beta=0.2$、关节速度输入 $\beta=0.1$；所有实验用 **对角代价**。没有它，系统在全支撑相下也会失稳（作者报告）。
- **更新率实验：** 控制频率 400 Hz；前馈 MPC 降到 20 Hz 更新会出现明显的更新处不连续，反馈策略下基本恢复；两种方式都在带 10% 质量误差的仿真里比较。真机：ANYmal 动态步行用 **1.0 s 滚动时域**、MPC 更新约 **15 Hz**；负载 5.7 kg（约总质量 15%）实验中反馈策略无力矩跳变。
- **局限：** 仅验证了 ANYmal 的若干步态（动态步行、pace、trot）；频率整形参数依赖执行器带宽，原文未给出自动整定方法。

## 对 wiki 的映射

- [`wiki/queries/mpc-tuning-guide.md`](../../wiki/queries/mpc-tuning-guide.md)

## 参考来源

- Grandia, R., Farshidian, F., Ranftl, R., Hutter, M. (2019). Feedback MPC for torque-controlled legged robots. IROS 2019. <https://arxiv.org/abs/1905.06144>
