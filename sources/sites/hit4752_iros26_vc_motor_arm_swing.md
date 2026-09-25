# Dynamic Humanoid Arm-Swing Motion（hit4752.github.io / IROS 2026）

- **类型：** 项目页
- **URL：** <https://hit4752.github.io/projects/202609_iros26-arm-swing/>
- **论文：** *Dynamic Humanoid Arm-Swing Motion via Trajectory Optimization Leveraging Speed–Torque Mode Switching of a Variable Chain Motor* — IEEE/RSJ IROS 2026
- **作者：** Hiromi Tada, Takuma Hiraoka, Jin Hirai, Kunio Kojima, Kei Okada（东京大学）
- **收录日期：** 2026-09-25

## 一句话

在 **可变链电机（VC motor）** 肘关节上，用 **多模式速度–扭矩可行域并集** 约束做轨迹优化，生成可 **动态切换电气模式** 的人形甩臂动作，并在 **JAXON** 真机验证峰值末端速度 **9.01 m/s**。

## 开源状态（2026-09-25 步骤 2.5）

| 项 | 结论 |
|----|------|
| **代码** | **确认未开源** — 项目页仅提供摘要、方法图、对比表与实验视频下载；页内无 GitHub / Zenodo 链接 |
| **arXiv** | 截至入库日 **未列** 预印本链接 |
| **关联作者页** | <https://github.com/hit4752/>（个人主页与多篇项目链，**无** 本篇独立仓库） |
| **硬件前作** | IROS 2025 VC 电机开发论文 DOI：<https://doi.org/10.1109/iros60139.2025.11246199> |

## 页面要点（项目页）

- **VC motor：** 四个小电机单元 + 专用电路在 **串联 / 并联** 绕组连接间切换，即时改变速度–扭矩特性（高扭矩 ↔ 高速）。
- **轨迹优化：** 关节速度 \(v\) 与扭矩 \(\tau\) 约束为 **三模式可行 speed–torque 区域的并集**；优化中 **不引入离散模式变量**；执行时按优化轨迹的 speed–torque 剖面逐步选模式。
- **约束实现：** 对 \(|v|\)、\(|\tau|\) 用各模式 \(v_{\max,i},\tau_{\max,i}\) 与斜率 \(\alpha\) 的 **smooth max/min（LSE）** 近似。
- **实验：** JAXON 左肘换 VC 电机；对比固定高扭矩 / 中间 / 高速三模式；模式切换 (d) 相对最快固定模式 (c) 优化轨迹末端峰值 **+7.7%**，真机 **9.01 m/s**（固定 (c) 为 8.24 m/s）。

## 交叉链接

- 论文摘录：[iros26_vc_motor_arm_swing_tada.md](../papers/iros26_vc_motor_arm_swing_tada.md)
- 硬件前作摘录：[iros25_variable_chain_motor_tada.md](../papers/iros25_variable_chain_motor_tada.md)
- Wiki：[可变链电机（VC motor）](../../wiki/entities/variable-chain-motor.md)、[IROS 2026 动态甩臂轨迹优化](../../wiki/entities/paper-iros26-vc-motor-dynamic-arm-swing.md)
