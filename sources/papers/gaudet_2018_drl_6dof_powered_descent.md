# Deep Reinforcement Learning for Six Degree-of-Freedom Planetary Powered Descent and Landing

> 来源归档（ingest）

- **标题：** Deep Reinforcement Learning for Six Degree-of-Freedom Planetary Powered Descent and Landing
- **作者：** Brian Gaudet, Richard Linares, Roberto Furfaro
- **论文：** <https://arxiv.org/abs/1810.08719>（v1，2018-10-20，37 页）
- **出版物：** arXiv 版未印期刊信息；期刊版信息本次未核对
- **类型：** arXiv 预印本
- **入库日期：** 2026-11-06
- **核对范围：** 用 pdftotext 抽取全文后通读；表 5、表 6 部分符号错乱（姿态限值、重力随机化的正负号）。
- **通读：** 已完成
- **一句话说明：** 用 PPO 训练火星 6-DoF 着陆策略，直接输出四台可节流推力器的推力指令，测试 10,000 次着陆的位置误差在米级。

## 核心摘录

1. **算法。** PPO，策略与价值网络为四层 tanh 前馈网络（未用循环网络）；终端奖励与塑形奖励用不同折扣率，因此放弃了 GAE。
2. **观测与动作。** 观测 `[v_error, q, ω, r_z, t_go]`；动作是四台推力器的推力（每台 1000–5000 N）。湿质量 2000 kg，Isp 225 s，火星重力 3.7114 m/s²。作者指出这种推力器布置 "does not allow any direct control of the rotational velocity around the z-axis."
3. **奖励与终止。** 跟踪一个指向着陆点上方 15 m、竖直 −2 m/s 的速度场，加控制代价、姿态约束惩罚与着陆奖励；高度低于零或违反姿态约束即终止回合。
4. **随机化。** 训练：质量 ±5%、重力 ±2%、惯量扰动；测试：每回合加均值 U(−100,100) N、标准差 100 N 的外力噪声。没有发动机故障。
5. **结果。** 训练 300,000 回合（结束时仍在提升）；测试 10,000 回合：纵向 0.4 ± 0.7 m、横向 −0.1 ± 0.7 m（最差 −5.2 m）；燃料 291 kg，对比 3-DoF DR/DV 制导 279 kg；与 GPOPS 开环最优（3-DoF）相比文中写 "250 kg, as compared to 290 kg ... 18 percent higher"（按数字算约 16%，原文如此）。策略推理 "<1ms"。
6. **局限（作者自述）。** 无最优性与稳定性保证，硬约束只能靠终止和惩罚近似，训练"does not converge every time"，推力矢量有小幅振荡；目标是最终从原始传感器直接映射到执行器。

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md) — 学习型路线
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 6
