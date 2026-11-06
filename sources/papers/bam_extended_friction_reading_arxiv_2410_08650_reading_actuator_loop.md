# BAM 扩展摩擦模型论文通读（执行器驱动链选型视角）

> 来源归档（原文通读 · 支撑 `wiki/queries/actuator-drive-chain-selection-loop.md`）

- **标题：** Extended Friction Models for the Physics Simulation of Servo Actuators
- **作者：** Marc Duclusaud, Grégoire Passault, Vincent Padois, Olivier Ly
- **类型：** paper / actuator-modeling / sim2real
- **原文：** <https://arxiv.org/abs/2410.08650>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext 抽取）
- **一句话说明：** 把摩擦表述为可阻止运动的力矩上界，提出 M1–M6 递进扩展摩擦模型（Stribeck、负载相关、方向性、二次项），在摆锤台架上用 CMA-ES 辨识，并在 MuJoCo 2R 臂上验证相对 Coulomb–Viscous 的提升。
- **核对范围：** 通读全文 §I–§VII（含 §II 台架动力学与 drive/backdrive 图、§III 模型族、§IV 电压/电流控制律与仿真算法 1、§V 辨识轨迹与优化、§VI 实验结果、结论与局限）及参考文献列表；该论文无附录。仅覆盖驱动链第③层（显式摩擦辨识），未阅读其他三层（EDA、FOC、EtherCAT）的一手资料，这些层的内容仅依赖 wiki 既有页面。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I–§II）：** MuJoCo、Isaac Gym 仍用 Coulomb–Viscous（M1，式 2–3）；该模型在 θ̇=0 时摩擦为零且方向切换处不连续；RL 常用的低增益控制下仿真与真机偏差显著（Fig.1）。
- **台架动力学（式 1）：** τm + τe(θ) + τf = Jθ̈，J = ml² + Jm，Jm = N²Jr（转子惯量已知时）否则作为待辨识参数。
- **摩擦即上界（式 3–4，算法 1）：** 摩擦被限幅在 [−τf^m, τf^m]，用「下一步速度归零所需力矩」τf,stop = −(J/Δt·θ̇ + τm + τe) 裁剪。
- **drive/backdrive 图（§II-C，Fig.3）：** 在摆锤台架上手动移到静区极限位置读 τdrive/τbackdrive；MX-106 与 eRob80:100 的图显示摩擦随受力增大，即负载相关，Coulomb–Viscous 无法表达。
- **模型族（§III）：** M2 Stribeck（5 参数，式 5）；M3 负载相关 Kl|τm−τe|（式 6）；M4 Stribeck+负载相关（7 参数，式 7）；M5 方向性（9 参数，Km/Ke，式 8）；M6 二次项（11 参数，用于谐波减速器，式 9）。
- **执行器模型（§IV）：** 控制律 + 电机模型；电压控制 τm = (kt/R)U − (kt²/R)θ̇（式 12，drive/brake 假设）；电流控制 I 受 Iemf 与 Iheat 限幅、τm = kt·I（式 13–14）；kt、R、Jm 未知时一并辨识。物理引擎里通过在线更新静/粘摩擦实现，τe 用上一步值近似。
- **辨识（§V）：** 4 类轨迹（加速振荡、慢振荡叠小振荡、缓升缓降、抬起释放；后者释放电流或 H 桥以排除反电动势与粘滞的混淆）；MAE 作为目标，CMA-ES（经 optuna）；日志 75% 辨识 / 25% 验证，重复 3 次收敛一致。
- **结果（§VI）：** 四个执行器（MX-64、MX-106、eRob80:50、eRob80:100）各约 100 条 6 秒日志；验证集 MAE 相对 M1 降低倍数：MX-64 2.93、MX-106 2.02、eRob80:100 2.34、eRob80:50 1.51。Dynamixel 上 M1 到 M5 持续改善、M6 无改善；eRob80:100 在 M6 最佳；eRob80:50 在 M3 之后无明显改善。MuJoCo 2R 臂上 M4（Dynamixel）与 M6（eRob80）的 MAE 约为 M1 的一半以下；M5 辨识阶段更好但 2R 臂上无提升，作者推测过拟合。
- **局限（§VII-A）：** 未考虑温度（需热模型与测温）、径向力、驻留时间（dwell time）；未来工作为在物理引擎中实现并面向人形。

## 对 wiki 的映射

- [`wiki/queries/actuator-drive-chain-selection-loop.md`](../../wiki/queries/actuator-drive-chain-selection-loop.md)
- [`wiki/entities/paper-bam-extended-friction-servo-actuators.md`](../../wiki/entities/paper-bam-extended-friction-servo-actuators.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2410.08650>
