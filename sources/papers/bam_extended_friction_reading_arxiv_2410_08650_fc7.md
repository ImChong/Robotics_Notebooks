# BAM 扩展摩擦模型论文通读（摩擦补偿视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/friction-compensation.md`）

- **标题：** Extended Friction Models for the Physics Simulation of Servo Actuators
- **作者：** Marc Duclusaud, Grégoire Passault, Vincent Padois, Olivier Ly
- **类型：** paper / actuator-modeling / sim2real
- **原文：** <https://arxiv.org/abs/2410.08650>（2026-11 通过 arXiv PDF v4 获取全文，pdftotext 抽取；式号以 v4 为准）
- **一句话说明：** 把摩擦表述为阻止运动的力矩预算 τf^m，提出 M1–M6 递进摩擦模型，在摆锤台架上用 CMA-ES 辨识，并在 MuJoCo 2R 臂上验证相对 Coulomb–Viscous 的提升；其辨识结果可直接作为摩擦前馈项 τ̂f 的参数来源。
- **核对范围：** 通读摘要与 §I–§VII 全文（§II 台架动力学与 drive/backdrive 图、§III 模型族、§IV 电压/电流控制律与算法 1、§V 辨识、§VI 实验、结论与局限）及参考文献列表；论文无附录。论文是仿真侧摩擦建模，并未实现"实机前馈补偿"，该用法为 wiki 页面的映射（推断）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **Coulomb–Viscous（§II-B，式 2–3）：** τf = −Kv θ̇ − sign(θ̇)Kc；θ̇=0 时摩擦为零且换向处不连续，与实际不符；MuJoCo、Isaac Gym 仍采用该模型。
- **摩擦即预算（式 4–5）：** 令下一步速度为零所需力矩 τf^stop = −(J/Δt·θ̇ + τm + τe)，实际摩擦 = clip(τf^stop, [−τf^m, τf^m])。
- **drive/backdrive 图（§II-C，Fig.3）：** 摆锤台架手动移到静区极限位置读 τdrive/τbackdrive；MX-106、eRob80:100 的图表明摩擦随负载增大（load-dependent）。
- **模型族（§III）：** M2 Stribeck（5 参数）；M3 负载相关 Kl|τm−τe|（3 参数）；M4 Stribeck+负载相关（7 参数）；M5 方向性（9 参数，可表达自锁蜗杆 Kes>1）；M6 二次项（11 参数，谐波减速器）。
- **执行器模型（§IV）：** 电压控制 τm = (kt/R)U − (kt²/R)θ̇（drive/brake 假设）；电流控制 I 限幅于 Iemf、Iheat，τm = kt·I；kt、R、Jm 未知时一并辨识。物理引擎中通过在线更新静摩擦与粘摩擦实现，τe 用上一步值近似。
- **辨识（§V）：** 四类轨迹（加速振荡、慢振荡叠小振荡、缓升缓降、抬起释放；释放时电流置零或释放 H 桥以排除反电动势与粘滞混淆）；CMA-ES（optuna），MAE 为目标；75% 辨识 / 25% 验证；重复 3 次结果一致，约 4000 次迭代、5 分钟内收敛。
- **结果（§VI）：** MX-64、MX-106、eRob80:50、eRob80:100 各约 100 条 6 秒日志；最佳模型相对 M1 的 MAE 降低倍数 2.93 / 2.02 / 1.51 / 2.34。Dynamixel 上 M1→M5 持续改善、M6 无改善；eRob80:100 在 M6 最佳；eRob80:50 在 M3 之后无明显改善。2R 臂：Dynamixel 上 M4 最优、MAE 不到 M1 的一半；M5 在辨识集更好但 2R 臂无提升（作者推测过拟合）；eRob80 上 M6 最优，同样降到一半以下。
- **局限（§VII-A）：** 未考虑温度、径向力、驻留时间（dwell time）；未来工作为在物理引擎中实现并面向人形与 RL 迁移。开源实现：github.com/Rhoban/bam。

## 对 wiki 的映射

- [`wiki/concepts/friction-compensation.md`](../../wiki/concepts/friction-compensation.md)
- [`wiki/concepts/joint-friction-models.md`](../../wiki/concepts/joint-friction-models.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2410.08650>
