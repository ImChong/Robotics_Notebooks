# SSRM（稳态响应法）电机/关节机械参数辨识一手论文簇

> 来源归档（ingest）

- **标题：** Steady-State Response Method (SSRM) for motor / joint mechanical parameter identification
- **类型：** paper（经典方法簇 + 工程归纳）
- **来源：** Journal of the Franklin Institute / Automatica / Applied Mechanics Reviews / Syroco 1988 / 工程博客
- **入库日期：** 2026-09-21
- **最后更新：** 2026-09-21
- **一句话说明：** 通过规定输入、等响应进入稳态、再按简化方程反推参数；机械侧常用 $T=J\dot\omega+B\omega+T_f$，可辨识粘性 $B$、库仑 $T_c$、恒定偏置与转动惯量 $J$（后者需配合动态段或 TRM）。
- **沉淀到 wiki：** 是 → [`wiki/methods/ssrm-steady-state-response-method.md`](../../wiki/methods/ssrm-steady-state-response-method.md)

## 开源状态（步骤 2.5）

| 资料 | 代码 | 结论 |
|------|------|------|
| Swamy 1965 | 无 | **确认未开源**（理论文） |
| Armstrong–Dupont–Canudas de Wit 1994 | 无官方仓 | **确认未开源**（综述） |
| Elhami & Brookfield 1997 | 无官方仓 | **确认未开源** |
| Specht & Isermann 1989 | 无官方仓 | **确认未开源** |
| 自由度 FreeDof 动力学辨识文 | 无项目页 | **步骤 2.5 不适用**（工程方法文） |

## 术语说明

- **SSRM** = **Steady-State Response Method**（稳态响应法）：在暂态衰减后读取响应，用稳态方程反推参数。
- **TRM** = **Transient Response Method**（瞬态响应法）：利用启动/加速段的动态方程（如 $J\dot\omega$ 占主导）辨识。矿用变频器、PMSM 电气参数文献中常与 SSRM 成对出现（直流稳态测 $R_s$，瞬态测 $L$/$R_r$）；本簇聚焦 **机械侧** $J,B,T_f$。
- 中文测控教材亦用「稳态响应法」指 **频率逐点扫正弦、等输出稳态后读幅相** 的 FRF 辨识（与 SSRM 哲学一致，见推荐继续阅读）。

## 核心论文摘录（MVP）

### 1) The steady state response of a servosystem taking stiction and coulomb friction into consideration（Swamy, 1965）

- **链接：** <https://doi.org/10.1016/0016-0032(65)90002-5>
- **期刊：** Journal of the Franklin Institute
- **核心贡献：** 早期在伺服系统分析中 **显式把稳态响应与 stiction、库仑摩擦** 联立；说明摩擦会在稳态力矩平衡里留下可测指纹，是后续「恒速测摩擦」路线的理论前驱。
- **对 wiki 的映射：**
  - [SSRM（稳态响应法）](../../wiki/methods/ssrm-steady-state-response-method.md)
  - [Joint Friction Models](../../wiki/concepts/joint-friction-models.md)

### 2) Friction in Servo Machines: Analysis and Control Methods（Armstrong-Héélouvry, Dupont, Canudas de Wit, 1994）

- **链接：** <https://doi.org/10.1115/1.3111082>
- **期刊：** Applied Mechanics Reviews
- **核心贡献：** 伺服机器摩擦 **综述**：建模（Coulomb / viscous / Stribeck）、分析工具与补偿。归纳 **准静态 / 恒速稳态** 下从力矩–速度关系分离库仑与黏性，以及 **恒力矩起动阈值** 测静摩擦；是机器人/伺服工程里 SSRM 实验设计的英文经典入口。
- **对 wiki 的映射：**
  - [SSRM（稳态响应法）](../../wiki/methods/ssrm-steady-state-response-method.md)
  - [Joint Friction Models](../../wiki/concepts/joint-friction-models.md)
  - [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)

### 3) Sequential identification of coulomb and viscous friction in robot drives（Elhami & Brookfield, 1997）

- **链接：** <https://doi.org/10.1016/s0005-1098(96)00183-5>
- **期刊：** Automatica
- **核心贡献：** 机器人驱动 **分步辨识**：先在 **恒定低速稳态** 下估计库仑与黏性（$\dot q$ 已知、$\ddot q\approx 0$ 时方程退化为仿射），再处理动态段。给出「顺序固定参数、避免纠缠」的正式表述，与 SSRM 核心流程一致。
- **对 wiki 的映射：**
  - [SSRM（稳态响应法）](../../wiki/methods/ssrm-steady-state-response-method.md)
  - [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)

### 4) On-line identification of inertia, friction and gravitational forces applied to an industrial robot（Specht & Isermann, 1989）

- **链接：** <https://doi.org/10.1016/b978-0-08-035742-3.50041-1>（Syroco 1988 论文集）；另见 <https://doi.org/10.1016/s1474-6670(17)54613-3>
- **核心贡献：** 工业机器人 **在线** 辨识惯量、摩擦与重力：利用不同运动段让某项在方程中 **消失或占主导**（准静态段 → 摩擦/重力；加速段 → 惯量），是「换工况分离参数」的工业早期实例。
- **对 wiki 的映射：**
  - [SSRM（稳态响应法）](../../wiki/methods/ssrm-steady-state-response-method.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

### 5) Sim2Real 动力学辨识：实验设计（自由度 FreeDof, 2026-08-12）

- **链接：** <https://mp.weixin.qq.com/s/B_sH9VNRxB6GCTJwnx6esQ>
- **归档：** [`sources/blogs/wechat_freedof_sim2real_dynamics_identification.md`](../blogs/wechat_freedof_sim2real_dynamics_identification.md)
- **核心贡献：** 中文工程语境下把 SSRM 流程写透：**多档恒速** → $\tau$–$\dot q$ 图读 $b,\tau_c$；**恒扭矩起动** → 阈值与半差/半和分离摩擦与偏置；强调 **不能用稳态位置误差反推摩擦**。与 [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md) ② 级实验一一对应。
- **对 wiki 的映射：**
  - [SSRM（稳态响应法）](../../wiki/methods/ssrm-steady-state-response-method.md)
  - [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)

## 相关但不同域的 SSRM 用法（交叉参考）

| 领域 | 典型 SSRM 测什么 | 代表文献 |
|------|------------------|----------|
| 异步/PMSM 电气参数 | 定子电阻 $R_s$、互感 $L_m$（直流/恒压频比稳态） | 工控自动化 2021：<http://www.gkzdh.cn/fileGKZDH/journal/article/gkzdh/2021/8/gkzdh-2021-8-96.html> |
| 测试系统动态特性 | 一阶/二阶系统时间常数、$\omega_n,\zeta$（逐频点正弦稳态） | 测控教材「4.1.5 稳态响应法」 |
| **本 wiki 聚焦** | $J,B,T_c,T_{\mathrm{bias}}$ 机械参数 | 上文 1–5 |
