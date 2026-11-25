# Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion

> 来源归档

- **标题：** Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion
- **类型：** paper
- **作者：** Karthik Urs, Challen Enninful Adu, Elliott J. Rouse, Talia Y. Moore（University of Michigan）
- **链接：** https://arxiv.org/abs/2202.12395
- **文内发布地址：** `www.embirlab.com/3dpactuator`
- **入库日期：** 2026-07-25（2026-11-21 完成 `arXiv:2202.12395v1` 全文通读与参数/实验表核对）
- **通读：** 已完成
- **一句话说明：** 面向 8–15 kg 腿式机器人的两种 3D 打印 QDD 执行器：T-Motor RI50 + SLA-HT/FDM-PLA 7.5:1 行星与 15.04:1 Wolfrom 双边驱动，搭配双离心风扇风冷与 moteus r4.5 驱动；系统测电参、热二阶模型、4 象限效率、带宽及 57 h / 42 万步态循环寿命。
- **开源状态：** 论文宣称机械/电气/软件完全开源（文中给出 `www.embirlab.com/3dpactuator`），GitHub 搜索未见统一官方主仓，驱动侧采用开源 **moteus r4.5**。
- **沉淀到 wiki：** [paper-3d-printed-open-source-actuators-legged](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md)

---

## 核心贡献与实测指标（原文 PDF 精读）

1. **RI50 电机与两套传动对比（Table 1 & 2）**：
   - 选型电机：T-Motor RI50（Wye 接，$R_\phi=705\pm3\,\text{m}\Omega$，$L_e=2559\pm2\,\mu\text{H}$，$K_T=0.105\pm0.002\,\text{N}\cdot\text{m/A}$，$K_B=0.094\pm0.002\,\text{V}\cdot\text{s/rad}$，$K_M=0.118\pm0.003\,\text{N}\cdot\text{m/W}^{1/2}$，$J_m=9.01\times10^{-6}\,\text{kg}\cdot\text{m}^2$，$m=193\,\text{g}$）。
   - **7.5:1 行星（FDM PLA 30° 压力角）**：总质量 **482 g**（Ø90×70 mm），$K_{Ta}=0.79\,\text{N}\cdot\text{m/A}$，$K_{Ma}=0.88\,\text{N}\cdot\text{m/W}^{1/2}$，$J_a=0.64\,\text{g}\cdot\text{m}^2$，2 s 峰值 **19.1 N·m** / 20 s 峰值 **7.5 N·m** / 连续 **4.4 N·m**，最高速 **21 rad/s**。
   - **15.04:1 Wolfrom 双边驱动**：总质量 **536 g**（Ø90×90 mm），$K_{Ta}=1.58\,\text{N}\cdot\text{m/A}$，$K_{Ma}=1.76\,\text{N}\cdot\text{m/W}^{1/2}$，$J_a=2.2\,\text{g}\cdot\text{m}^2$，2 s 峰值 **38.2 N·m** / 20 s 峰值 **15 N·m** / 连续 **8.8 N·m**，最高速 **11 rad/s**。
   - 单执行器 BOM **<$200 USD**（RI50 `$62` + moteus r4.5 `$84` + 轴承 `$15` + 紧固销轴 `$8` + SLA-HT `$16` + FDM PLA `$3.5` + 双离心风扇 `$10`）。
2. **电阻自校准热传感与风冷方案（Table 3, §4.3）**：以线间电阻变化标定绕组温度 $T_w = 139.564 R_{ll} - 177.826$（$R^2=0.99$，~0.51%/°C）；加装双 30×10 mm 离心风扇风道后，二阶热模型总热阻从无风扇 **7.77 °C/W**（$R_{WH}=1.09, R_{HA}=6.68$）降至 **2.27 °C/W**（$R_{WH}=0.828, R_{HA}=1.44$），**约 3.5× 散热增益**，使连续电流与连续力矩近翻倍（22 A 仅维持 2 s、9.2 A 可维持 20 s）。
3. **动态与控制带宽（§4.2, §4.6）**：40 Hz 以下系统辨识相干度 >0.9，VAF 达 92%–98%；moteus `iq` 电流阶跃上升时间 **2.65±1.88 ms**，电流环 -3 dB 带宽 **102±1 Hz**（受高电感 $L_e$ 限制），位置环带宽 **11–30 Hz**。
4. **57 小时 / 42 万步态循环寿命（§4.5）**：回放 106 min MIT Mini Cheetah 实测步态载荷 57 h（~420k strides），无结构失效；前 8–10 h 因塑料微粉摩擦正功效率先降 3.5%，随后磨合回升（总体仅约 2% 净降幅）；总背隙差从 56 mrad 增至 82 mrad（+26 mrad 总增量，单执行器线性增长 **250±30 µrad/hr**）。

## 对 wiki 的映射

- 论文实体页：[paper-3d-printed-open-source-actuators-legged](../../wiki/entities/paper-3d-printed-open-source-actuators-legged.md)
- 对比页：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 驱动：[moteus](../../wiki/entities/moteus.md)
