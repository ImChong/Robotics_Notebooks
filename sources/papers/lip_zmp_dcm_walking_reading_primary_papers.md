# LIP / ZMP / DCM 步态简化模型：四篇一手论文通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/lip-zmp.md`、`wiki/formalizations/zmp-lip.md`、`wiki/concepts/capture-point-dcm.md`）

- **标题：** 四篇开放获取的步态简化模型论文（均用 LIP / ZMP / DCM 做在线步行规划与控制）
- **类型：** paper 合集 / locomotion / simplified-model
- **入库日期：** 2026-11-07
- **原文（均已下载全文并用 pdftotext 提取）：**
  1. Herdt, Diedam, Wieber, Dimitrov, Mombaur, Diehl, *Online Walking Motion Generation with Automatic Foot Step Placement*, Advanced Robotics 24(5-6), 2010. HAL 开放版 <https://inria.hal.science/inria-00391408/document>（19 页）。
  2. Romualdi, Dafarra, Hu, Pucci, *A Benchmarking of DCM Based Architectures for Position and Velocity Controlled Walking of Humanoid Robots*, Humanoids 2018. <https://arxiv.org/abs/1809.02167>（v2，8 页）。
  3. Shafiee, Romualdi, Dafarra, Andrade Chavez, Pucci, *Online DCM Trajectory Generation for Push Recovery of Torque-Controlled Humanoid Robots*, Humanoids 2019. <https://arxiv.org/abs/1909.10403>（v2，8 页）。
  4. Khadiv, Herzog, Moosavian, Righetti, *Step Timing Adjustment: A Step toward Generating Robust Gaits*, Humanoids 2016. <https://arxiv.org/abs/1610.02377>（v2，8 页）。
- **一句话说明：** 从 LIP 方程、ZMP 与 CoP 的关系出发，依次给出线性 MPC 步行（Herdt）、DCM 三层控制架构基准（Romualdi）、DCM + 步位 / 步时 QP 推扰恢复（Shafiee、Khadiv）。
- **核对范围：** 四篇正文均从摘要读到结论，含各自的模型推导（Herdt 式 (1)–(24)、Romualdi 式 (1)–(19)、Shafiee 式 (1)–(38)、Khadiv 式 (1)–(17)）、实验设置与结果文字。图中逐点曲线、Herdt §4 中 CoP 约束矩阵的逐元素形式、Romualdi / Shafiee 的全身 QP 增益取值与参考文献未逐项核对。**未取得也未阅读**本库 `footstep_and_balance.md` 所列的 Kajita 2003 预览控制、Kajita 2001 3D-LIPM、Pratt 2006 Capture Point、Koolen 2012 可捕获性、Englsberger 2015 DCM 原论文（IEEE / SAGE 付费墙，开放镜像不可达）；这些工作的内容只通过上述四篇的引述和描述得知，wiki 中凡出自这些原文而本归档未核实的陈述须标「二手引述」。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **LIP 与 ZMP（Herdt §2；Romualdi §II-C）：** 质心被约束在高度 h（z0）的水平面内，CoP（此处等同 ZMP）近似为 z = x − (h/g) ẍ（Herdt 式 (5)），即 ẍ = ω²(x − r_zmp)，ω = √(g/z0)（Romualdi 式 (1)）。Herdt 用分段常值 jerk 的三阶积分器做预测模型，采样 T = 0.1 s、N = 16（预测 1.6 s，约两步）。
- **DCM 定义与分解（Romualdi 式 (2)–(3)；Shafiee 式 (8)–(11)）：** ξ = x + ẋ/ω；DCM 动力学 ξ̇ = ω(ξ − r_zmp)，CoM 动力学 ẋ = ω(ξ − x)。LIP 因此分成稳定部分（CoM 追 DCM）与不稳定部分（DCM 以正实部极点 ω 发散）。ZMP 与 DCM 的关系 r_zmp = ξ − ξ̇/ω。三维下 ξ = x + bẋ，b = √(z0/g)，用虚拟排斥点 VRP：ξ̇ = (ξ − r_vrp)/b；若质心角动量恒定，VRP 在地面的投影等于 ZMP（Shafiee §II-C）。Romualdi 称 DCM 可视为 Capture Point 向三维的推广，前提是 VRP 与 eCMP 高度差恒定。
- **DCM 参考轨迹（Romualdi 式 (4)–(6)、Shafiee 式 (12)–(13)）：** 单支撑内 ξ(t) = r_zmp + e^{ω(t−t_i)}(ξ_eos − r_zmp)，从末步反向递推各步终点；双支撑内用三次多项式保证 DCM 为 C¹，从而 ZMP 连续。纯单支撑规划的缺点是 ZMP 参考不连续，导致接触力与关节力矩不连续。
- **DCM 控制律（Romualdi 式 (7)–(8)；Shafiee 式 (14)–(15)）：** r_ref = ξ_ref − ξ̇_ref/ω + Kp(ξ − ξ_ref) + Ki∫(ξ − ξ_ref)，Kp > I 时误差动力学渐近稳定；该律不对消不稳定动力学，但不保证 ZMP 留在支撑多边形内，也不看未来参考。
- **DCM 预测控制（Romualdi 式 (9)–(12)）：** 以分段常值 ZMP 离散化 ξ_{k+1} = e^{ωT}ξ_k + (1 − e^{ωT}) r_k，加线性 ZMP 支撑多边形约束 A r ≤ b，解 QP。
- **iCub 对比结果（Romualdi 表 I）：** 最高直行速度 —— 预测 + 速度控制 0.19 m/s，预测 + 位置控制 0.20 m/s，瞬时 + 速度控制 0.22 m/s，瞬时 + 位置控制 0.41 m/s。作者发现预测控制在 0.41 m/s 因期望 ZMP 变化过快而跌倒，比瞬时控制对 ZMP 跟踪误差更不鲁棒；位置控制的足端跟踪优于速度控制（数值微分噪声）。
- **线性 MPC 自动落脚（Herdt §3–§6）：** 把步位作为 QP 决策变量，CoP 约束为支撑多边形，目标只调节 CoM 速度即可；HRP-2 仿真中每 0.8 s 一步，QP 平均 < 1 ms（QL / qpOASES）；CoP 离边界留 3 cm 余量；"安全优先"—— 被推时宁可偏离期望速度也保持 CoP 可行；横向交叉腿受限时一步一步交替导致平均横向速度只有指令的约 2/3。
- **推扰恢复（Shafiee 摘要 + §IV–§V）：** 在单支撑中用 ZMP 的指数插值 r(t) = A e^{−t/b} + B 使最终 DCM 偏差 γ_T 与步时 σ = e^{T/b} 的关系线性化，步位、步时、DCM 终值统一成一个 QP（代价偏离名义轨迹）；iCub（33 kg）仿真 0.28 m/s 下承受 150 N、0.05 s 的盆骨侧推，平均步宽增 6 cm、步时较名义 0.53 s 缩短约 0.12 s，整套架构平均 < 3 ms（qpOASES）。作者自述局限：只用 DCM 简化模型，不考虑角动量，大扰动可能需要对名义轨迹做过大修改；只在仿真验证。
- **步时调整（Khadiv §II–§IV）：** DCM 偏移 b = ξ0 − u0 = ξT − uT；用变量替换 τ = e^{ω0 T} 把含步时的 LIP 终值方程化为线性等式，得到小规模 QP；60 kg、质心高 0.8 m 的 LIPM，限制步长 ±50 cm、步宽 10–40 cm、步时 0.2–0.8 s，在 t = 1.4 s 受 325 N、0.1 s 推力：固定步时无法恢复、DCM 发散，步位 + 步时联合调整可恢复。作者指出：点足 / 被动踝机器人无法靠 CoP 调节 DCM，步位（和步时）调整是更有效的稳定手段；包含步时的多步 MPC 会失去线性，本方法只看下一步以保持凸。
- **作者对历史工作的归属（Khadiv 引言，二手）：** Kajita 等提出 ZMP 预览控制；Wieber 把它改写为 MPC；Takenaka 等用 LIPM 发散部分生成 DCM 轨迹；Hof 以 XCoM 名义、Pratt 以 Capture Point 名义早已使用同一量；Englsberger 等用 DCM 控制不稳定部分。

## 对 wiki 的映射

- [`wiki/concepts/lip-zmp.md`](../../wiki/concepts/lip-zmp.md)
- [`wiki/formalizations/zmp-lip.md`](../../wiki/formalizations/zmp-lip.md)
- [`wiki/concepts/capture-point-dcm.md`](../../wiki/concepts/capture-point-dcm.md)

## 参考来源

- 见上文四篇原文链接；相关二手索引归档见 [`footstep_and_balance.md`](footstep_and_balance.md)。
