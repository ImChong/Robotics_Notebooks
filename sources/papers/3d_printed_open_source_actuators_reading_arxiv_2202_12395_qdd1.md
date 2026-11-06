# 3D 打印开源 QDD 执行器论文通读（开源 QDD 选型视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/open-source-qdd-actuator-projects.md`）

- **标题：** Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion
- **作者：** Karthik Urs, Challen Enninful Adu, Elliott J. Rouse, Talia Y. Moore（University of Michigan）
- **类型：** paper / actuator / qdd
- **原文：** <https://arxiv.org/abs/2202.12395>（2026-11 通过 arXiv PDF v1 获取全文，pdftotext 转文本）
- **一句话说明：** 用 T-Motor RI50 + moteus r4.5 + 全 3D 打印传动（7.5:1 行星、15:1 Wolfrom 双边驱动）做出单个材料成本低于 200 美元的 QDD 执行器，并系统表征机械、电气、热、效率与 57 小时寿命。
- **核对范围：** 通读全文 §1–§5（引言、§2 设计、§3 表征方法、§4 结果、§5 讨论）、表 1–3、图注及参考文献列表；该文 arXiv v1 无附录。未阅读其引用的电机选型预印本 [18] 与设计网站 embirlab.com/3dpactuator 上的 CAD/固件，电机参数仅取论文转述值。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（§1–§2）：** 面向 8-15 kg 四足，目标 5-15 N·m、15-35 rad/s、输出惯量 1e-4 至 1e-2 kg·m²；只考虑 1:1 到 30:1 的直驱/行星/双边驱动刚性传动，以便 3D 打印。
- **电机选择（§2.1，表 1）：** 内转子 T-Motor RI50（约 62 美元）：KT=0.105 N·m/A、KM=0.12、Jm=9.01e-6 kg·m²、193 g；虽 KT/KM 较低，但可用更高减速比补偿并保持较低反射惯量。
- **机械（§2.2）：** 非标件全部 3D 打印；多数为 FDM-PLA（约 3.5 美元），近电机处为 SLA 高温树脂（约 16 美元）；轴承约 15 美元，紧固件约 8 美元，驱动器 moteus r4.5 为 84 美元；齿轮 30° 压力角；齿轮轴沿 Z 轴打印使切向力落在强层面，并用金属销与压缩预紧防分层；整机约 14 小时打印。
- **热设计（§2.3、§4.3）：** 被动散热不足，改用两个独立的离心风扇（各约 5 美元）导风过绕组；热阻由无风扇的约 8 C/W 级降到总 Rth 2.27 C/W，约 3.5 倍冷却优势，持续电流/力矩近乎翻倍；绕组须低于 100 C。
- **表 2 关键指标：** 7.5:1 行星——峰值 7.5 N·m(20 s)/19.1 N·m(2 s)、连续 4.4 N·m、最高 21 rad/s、482 g；15:1——15/38.2 N·m、连续 8.8 N·m、11 rad/s、536 g；对照 Mini Cheetah 6:1 连续 6.9 N·m、40 rad/s、480 g。峰值力矩受热限制，最大实测力矩更低。
- **动力学（§4.2）：** 反射惯量符合 Ja=N²Jm：7.5:1 为 6.37e-4，15:1 为 2.22e-3 kg·m²；一阶模型 VAF 92-98%。
- **效率（§4.4）：** 高速低扭矩效率高，低速高扭矩因 i_q² 铜损而降低；低速高扭矩区 15:1 更高效。
- **寿命（§4.5）：** 两台 7.5:1 以 Mini Cheetah 步态数据回放 57 小时（约 42 万步），无失效；效率先降约 3.5% 再回升，背隙增长约 26 mrad（约 250 微弧度/小时），阻尼无明显趋势。
- **控制（§4.6）：** 电流环上升时间 2.65 ms、半功率带宽 102 Hz（受高电感限制，低于同类）；位置环带宽 11-30 Hz。
- **开源性（致谢）：** 设计发布于 embirlab.com/3dpactuator；论文正文未给出 GitHub 地址。
- **局限（作者自述与观察）：** 15:1 Wolfrom 传动效率天然偏低；最高转速低于 Mini Cheetah；仅测 7.5:1 的寿命；未测冲击/跌落。

## 对 wiki 的映射

- [`wiki/comparisons/open-source-qdd-actuator-projects.md`](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- [`wiki/entities/paper-3d-printed-open-source-actuators-legged.md`](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2202.12395>
