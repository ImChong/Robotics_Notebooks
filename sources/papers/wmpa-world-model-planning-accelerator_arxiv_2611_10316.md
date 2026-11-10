# WMPA：世界模型规划加速器（arXiv:2611.10316）

> 来源归档（paper）

- **标题：** The World Model Planning Accelerator (WMPA): Hardware-Algorithm Co-Design for Embedded AI in Robotics
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10316>
- **PDF：** <https://arxiv.org/pdf/2611.10316v1>（cs.AR，v1，2026-11-06）
- **代码：** https://github.com/Prk10/world-models-theoretical-min（论文第 I 节称含 profiling 与 roofline 代码；本次未能访问，README 与目录均未核实）
- **作者与机构（以论文首页为准）：** Prerana Ramkumar、Fazal Hameed（美国沙迦大学 AUS，计算机工程系）；Suhas Suresh Bharadwaj（BITS Pilani 迪拜校区，电气与电子工程系）
- **入库日期：** 2026-11-10
- **一句话说明：** 从信息论与 CMOS 动态能耗推出世界模型潜空间 rollout 的算术与能耗下界，在 T4 GPU 上量出约 5.9×10^10 倍差距，并据此提出概念性近存计算加速器 WMPA；40 倍降功耗是设计目标，不是综合或流片结果。
- **通读：** 已完成
- **核对范围：** 论文 v1 PDF 全文 6 节（I 引言、II 理论背景、III 理论分析、IV 经验 profiling 与热仿真、V 加速器架构、VI 结论）及参考文献表逐段读完，含图 1–4 的图注与图内文字；论文无附录 / 补充材料。图 1–3 的曲线本体只读了图注与正文数字，未逐点核对。代码仓库无法访问，未读。公式 (6)–(11) 的数值代入已自行复算（6.4e-17 J/门、1.024e-14 J/步、6.55e-12 J/次、0.3873/6.55e-12≈5.91e10）。

## 开源状态

- 论文称 profiling 与 roofline 代码公开；GitHub 页面与 raw README 本次访问返回 403/404，**未核实**。WMPA 本身没有 RTL 或综合产物。

## 核心摘录

1. **问题（第 I 节）：** 世界模型 f: Z×A→Z 在 MPC / MCTS 中被连续调用 H 步、B 条并行轨迹；rollout 是深度串行链，宽脉动阵列空转，带宽成为约束。
2. **理论下界（第 III 节）：** 假设 A1–A5（潜状态马尔可夫、转移有最小条件熵 δ、信息经过至少一次门求值、覆写不可逆、CMOS 开关能耗）下，每步门数 ≥ I(Zt;Zt+1)（式 6），MAC 数 ≥ I/b（式 7），CMOS 动态能耗 ≥ I·αC·Vdd²（式 9）。Landauer 界（命题 2）论文自称依赖模型、「不评估」。
3. **数值（第 III-D 节）：** 32×32 类别潜变量熵上限 160 bit；TSMC 12nm 参数 Vdd=0.8V、Cgate≈1fF、α=0.1；单步下限 1.024×10^-14 J，B=32、H=20 规划调用下限约 6.55×10^-12 J。
4. **实测（第 IV 节）：** DreamerV3 风格 RSSM（512 单元 GRU，32×32 类别潜变量，PyTorch，权重不是训练得到的），Tesla T4，B=32，H∈{5,10,15,20}，pynvml 50ms 采样；H=20 平均 0.3873 J/次（10 次，σ=5.923 mJ），板级 68.19 W，对 CMOS 下限差距 5.91×10^10（H 更短时 6.08×10^10）。
5. **roofline（第 IV-B 节）：** 算术强度 0.38–15.55 FLOPs/byte，低于 T4 / A100 / H100 的 ridge 点 203.5 / 153.0 / 295.2，全部落在访存受限区 R1。
6. **稀疏性代理（第 IV-C 节）：** 用 softmax 温度衰减的合成压力测试，GRU 状态稀疏 14.0%，特征投影 29.0%；论文明确这是合成代理，不是训练策略的测量。
7. **热仿真（第 IV-D 节）：** 3D 瞬态非线性热传导（k(T)=k300(300/T)^γ，k300≈152，γ≈1.35），1mm×1mm×100µm 硅 tile，1.5mm 铜扩散层，h_eff=23,850 W/m²K，理想散热；基线热流 6.17×10^6 W/m²（来自把板功率 9% 分给 tile 的假设），峰值 556.07 K；目标 WMPA 为 306.52 K。论文自述这些「只证明相对可行性，不是绝对工作温度」。
8. **WMPA 架构（第 V 节）：** 四部件 NMCU（近存计算，设计目标减少 80% 片外流量）、EEH（熵估计硬件，按预测熵选精度）、VPRU（FP16 / INT8 / INT4 变精度单元，合成代理下 40% 步走 FP16、60% 走 INT8 或 INT4）、LSS（写时复制潜状态暂存，2MB SRAM，估计减少 60–70% 冗余读，CACTI 7.0 在 22nm 建模后线性缩放到 12nm 得 0.68mm²、0.87ns）。40 倍降功耗（68.19 W→1.70 W）是设计目标。
9. **论文自承的三项局限（第 VI 节）：** 精度调度依赖合成熵坍缩代理；40× 不是综合结果；热模型在铜扩散层后假设理想散热。

## 与小站（具身智能小站 2026-11-10）说法的核对

- 「从理论能耗下界和 GPU 分析出发，提出近存计算与可变精度的世界模型规划架构」：与摘要及第 V 节一致。
- 「40 倍降功耗是设计目标，并非已经流片验证的收益」：与第 V-C 节「as a design target and not as a derived or measured result」一致，且比「未流片」更弱：论文连综合都没做。

## 对 wiki 的映射

- [paper-wmpa-world-model-planning-accelerator](../../wiki/entities/paper-wmpa-world-model-planning-accelerator.md)
