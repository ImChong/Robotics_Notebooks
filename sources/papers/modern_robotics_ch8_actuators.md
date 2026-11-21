# Modern Robotics Chapter 8 (Sec. 8.9) & Chapter 11 (Sec. 11.8) — 电机、减速比与反射惯量

> 来源归档（textbook / primary source）

- **标题：** *Modern Robotics: Mechanics, Planning, and Control* — Section 8.9 Actuators & Section 11.8 Torque Control
- **作者：** Kevin M. Lynch, Frank C. Park
- **版本：** Cambridge University Press preprint (May 2017, 642 pp.)
- **官方入口：** <http://hades.mech.northwestern.edu/images/7/7f/MR.pdf>
- **入库日期：** 2026-11-21
- **核对范围：** 下载 2017 preprint PDF 完整文件并逐页核对 Ch. 8 Sec. 8.9.1–8.9.3（DC Motor 等效电路、TN 曲线、减速与反射惯量 Eq. 8.103–8.111）及 Ch. 11 Sec. 11.8（直接驱动、齿轮传动、应变片与 SEA 力矩控制）。
- **通读：** 已完成
- **一句话说明：** 机器人学教材对永磁直流/等效无刷力矩电机 TN 直线公式、连续/间歇工作区、齿轮效率传递与转子反射惯量 $G^2 I_{\text{rotor}}$ 的标准第一性推导。
- **沉淀到 wiki：** [motor-torque-speed-curve](../../wiki/concepts/motor-torque-speed-curve.md) · [motor-torque-current-curve](../../wiki/concepts/motor-torque-current-curve.md) · [armature-modeling](../../wiki/concepts/armature-modeling.md) · [motor-design-workflow](../../wiki/overview/motor-design-workflow.md)

---

## 核心原文摘录（Ch. 8 Sec. 8.9.1–8.9.3）

### 1. DC 电机基本关系与 TN 曲线（Sec. 8.9.1）

- 力矩–电流与反电动势–角速度同常数（SI 制下数值相等）：$\tau = k_t I$，$V_b = k_e w$（$k_t = k_e$）。
- 忽略电感稳态电压方程 $V = I R + k_t w$，消去 $I$ 得 **转矩–转速曲线**：
  $$w = \frac{V}{k_t} - \frac{R}{k_t^2}\tau$$
- 最大电压 $V_{\max}$ 下的边界点：
  - **空载转速：** $w_0 = V_{\max}/k_t$（$\tau=0$）；
  - **堵转力矩：** $\tau_{\text{stall}} = k_t V_{\max}/R$（$w=0$）。
- 受电阻发热限制，持续运行区域（continuous operating region）须满足 $|\tau| \le \tau_{\text{cont}}$（由最大允许连续电流 $I_{\text{cont}}$ 与绕组热极限决定），短时峰值允许进入间歇区（intermittent operating region）。

### 2. 齿轮传动与反射惯量（Sec. 8.9.2–8.9.3）

- 理想无损减速比 $G$（$w_{\text{motor}} = G w_{\text{gear}}$）下，$\tau_{\text{gear}} = \eta G \tau_{\text{motor}}$（含效率因子 $\eta < 1$）。
- 电机转子惯量 $I_{\text{rotor}}$ 折算到关节输出端为 **表观/反射惯量（apparent/reflected inertia）**：
  $$I_{\text{reflected}} = G^2 I_{\text{rotor}}$$
  高减速比下，转子经 $G^2$ 放大后常支配关节等效惯量与碰撞冲击负载（Eq. 8.103–8.111 在 Newton-Euler 关节动力学中显式计入该项）。

### 3. 力矩控制路径（Ch. 11 Sec. 11.8）

- 直驱可由电流直接估算力矩但受限力矩密度；高减速比齿轮引入非线性摩擦、背隙与柔度，高精度力矩控制往往需输出端应变片或串联弹性执行器（SEA）。

## 对 wiki 的映射

- [motor-torque-speed-curve](../../wiki/concepts/motor-torque-speed-curve.md)
- [motor-torque-current-curve](../../wiki/concepts/motor-torque-current-curve.md)
- [armature-modeling](../../wiki/concepts/armature-modeling.md)
- [motor-design-workflow](../../wiki/overview/motor-design-workflow.md)
