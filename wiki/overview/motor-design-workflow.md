---
type: overview
tags: [motor, actuator, hardware, humanoid, design, foc, simulation]
status: complete
updated: 2026-11-28
original:
  - ../../sources/blogs/wechat_zanehub_frameless_torque_motor_toolchain_2026-11-28.md
  - ../../sources/sites/simplefoc_documentation.md
  - ../../sources/sites/femm_info.md
  - ../../sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md
  - ../../sources/papers/modern_robotics_ch8_actuators.md
related:
  - ../concepts/motor-torque-speed-curve.md
  - ../concepts/motor-torque-current-curve.md
  - ../comparisons/motor-em-simulation-software.md
  - ../comparisons/open-source-torque-motor-em-design.md
  - ../concepts/field-oriented-control.md
  - ../formalizations/field-oriented-control-derivation.md
  - ./humanoid-actuator-102-technology-map.md
  - ./humanoid-hardware-101-integrated-actuators.md
  - ../entities/simplefoc.md
  - ../entities/bavaria-direct-winding-calculator.md
  - ../entities/femm.md
  - ../entities/pyleecan.md
  - ../entities/ironless-qdd-actuator.md
  - ../entities/cadenkraft-ironless-axial-flux-motor.md
  - ../entities/pygeartrain.md
  - ../concepts/joint-module-self-development-workflow.md
  - ../queries/joint-module-engineer-motor-design-literacy.md
  - ../concepts/halbach-array.md
  - ../entities/paper-halbach-permanent-multipole-magnets.md
sources:
  - ../../sources/blogs/wechat_zanehub_frameless_torque_motor_toolchain_2026-11-28.md
  - ../../sources/sites/ansys_motor_cad_electric_machine_design.md
  - ../../sources/personal/motor_curves_and_em_simulation_faq.md
  - ../../sources/sites/simplefoc_documentation.md
  - ../../sources/sites/femm_info.md
  - ../../sources/personal/open_source_torque_motor_em_design_curator.md
  - ../../sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md
  - ../../sources/blogs/wechat_zanehub_joint_module_engineer_motor_design_2026-11-22.md
summary: "关节电机从指标定义、电磁/热仿真、样机台架到 FOC 驱动验证的标准迭代流程；人形选型需同时满足 TN/TI 曲线与连续散热边界；开源路径挂接 Ironless/PYLEECAN/FEMM；Caden 轴向 Part 1 作 Halbach 手算入门。"
---

# 电机设计流程（规格 → 仿真 → 样机 → 控制）

> **本页定位**：把 **永磁关节电机** 从需求到可交付模组的主干步骤串成一条可交叉引用的流程；工具细节见 [电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)，读图见 [TN](../concepts/motor-torque-speed-curve.md) / [TI](../concepts/motor-torque-current-curve.md) 曲线页，控制环见 [FOC](../concepts/field-oriented-control.md)。

## 一句话定义

电机设计不是「先画线圈再试转」，而是 **指标驱动、电磁–热–机械耦合迭代**，最终以 **TN/TI 台架曲线 + FOC 电流环** 证明连续与峰值能力都达标。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TN | Torque-Speed (curve) | 转矩-转速特性曲线 |
| TI | Torque-Current (curve) | 转矩-电流特性曲线 |
| FEA | Finite Element Analysis | 有限元分析（电磁/结构/热） |
| FOC | Field-Oriented Control | 磁场定向控制 |
| MTPA | Maximum Torque Per Ampere | 单位电流最大转矩策略 |
| PMSM | Permanent Magnet Synchronous Motor | 永磁同步电机，关节常见 |
| IPM | Interior Permanent Magnet | 内置式永磁，弱磁区设计常见 |

## 为什么重要

- **人形/腿足关节** 同时需要 **峰值爆发**（起跳、抗冲击）与 **连续行走**（散热）；设计流程必须在早期就把 [峰值:持续](../overview/humanoid-actuator-102-decision-species.md) 与 [热学现实](../overview/humanoid-actuator-102-thermal-and-control.md) 写进指标，而不是样机出来后再补散热。
- **仿真与台架脱节** 是常见失败模式：电磁算出高转矩，但热模型或驱动限流使 [TN 曲线](../concepts/motor-torque-speed-curve.md) 连续区大幅缩水。
- **控制不是后装件**：弱磁、MTPA、电流采样与极对数对齐直接影响能否跑到设计基速；FOC 参数与电机 \(L_d, L_q, K_t\) 必须同源迭代。

## 流程总览

```mermaid
flowchart TD
  REQ["1 指标定义<br/>峰值/连续转矩、基速、比力矩、散热"]
  TOPO["2 拓扑与槽极<br/>PMSM/IPM、极对数、绕组"]
  EM["3 电磁设计<br/>Maxwell / Motor-CAD / JMAG"]
  TH["4 热与 CFD<br/>连续功率、温升"]
  MECH["5 机械与集成<br/>轴承、机壳、减速器接口"]
  PROTO["6 样机 + 台架<br/>TN / TI / 效率地图"]
  DRV["7 驱动与 FOC<br/>电流环、MTPA、弱磁"]
  SYS["8 关节模组验收<br/>整机 TN + 总线力矩模式"]

  REQ --> TOPO --> EM
  EM --> TH
  EM --> MECH
  TH --> PROTO
  MECH --> PROTO
  PROTO --> DRV --> SYS
  DRV -.->|参数回馈| EM
  TH -.->|温升降额| REQ
```

## 分阶段说明

### 1) 指标定义（需求冻结）

从任务反推电机边界，而非从库存电机反推任务：

| 指标 | 人形语境示例 | 关联页 |
|------|--------------|--------|
| 峰值转矩 / 持续时间 | 蹲起、起跳 1–3 s | [TN 曲线](../concepts/motor-torque-speed-curve.md) |
| 连续转矩 | 行走、站立 | [热学与力矩控制](../overview/humanoid-actuator-102-thermal-and-control.md) |
| 基速 / 最高转速 | 摆腿、跑步关节速度 | TN 曲线恒功率区 |
| 比力矩 (Nm/kg) | 腿部 >15 Nm/kg 量级 | [决策与物种](../overview/humanoid-actuator-102-decision-species.md) |
| 峰值:持续比 | 目标 ≥3:1 | 同上 |

输出物：**目标 TN 包络草图** + 母线电压 / 冷却方式假设。

### 2) 拓扑与槽极

- 选择 **表贴式 / 内置式（IPM）**、槽数极数、绕组形式（集中/分布）。
- IPM 利于 **弱磁扩速**；表贴式 \(L_d \approx L_q\) 控制简单，高基速场景需提前算弱磁电流。
- 槽极组合可先用 [Bavaria Direct 绕组方案计算器](../entities/bavaria-direct-winding-calculator.md) 纸面筛一轮：一次给出绕组系数、齿槽 LCM 与次谐波，筛完再进 FEA。

### 3) 电磁设计（FEA）

工具选型见 [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)。本阶段核心输出：

| 输出 | 用途 |
|------|------|
| 转矩–转速（TN） | 能力边界 |
| 反电势、Ld/Lq | FOC 与弱磁 |
| 齿槽转矩 | 低速振动、力矩纹波 |
| MTPA 轨迹 | 高效区电流分配 |

机器人关节厂常见：**Maxwell（精细电磁）+ Motor-CAD（快速迭代 TN/效率）**。

### 4) 热与流体

- 将 **连续转矩** 与 **冷却方式**（自然风冷、强迫风冷、液冷、机壳导热）耦合。
- CFD 评估风道与搅风损耗；热分析给出绕组/磁钢温升 → 修正连续功率（往往比电磁峰值更能「卡死」设计）。

### 5) 机械与系统集成

- 轴系、轴承寿命、机壳刚度与 **减速器 / 丝杠** 接口（见 [Hardware 101 · 集成执行器](./humanoid-hardware-101-integrated-actuators.md)）。
- 反射惯量、背隙与 **反向驱动** 在模组级一并评估，而非电机单体验收即可。

### 6) 样机与台架测试

台架核心设备是 [电机测功机（Dynamometer）](../concepts/motor-dynamometer.md)（吸收式或电力对拖）；应测全族曲线，而非单点：

| 测试 | 验证什么 |
|------|----------|
| [TN 曲线](../concepts/motor-torque-speed-curve.md) | 峰值/连续/基速是否与仿真一致 |
| [TI 曲线](../concepts/motor-torque-current-curve.md) | \(K_t\)、饱和、驱动限流 |
| 效率地图 | 工作点能耗与热 |
| 温升试验 | 连续转矩是否可维持 |

仿真–实测偏差大时，回到步骤 3–4 改磁路或冷却，而不是仅在驱动器上调参。选型测功机时先核对 **最大扭矩（含堵转）→ 连续机械功率 → 是否需要四象限**。

### 7) 驱动器与 FOC 验证

电机参数冻结后，在逆变器上实现 [磁场定向控制（FOC）](../concepts/field-oriented-control.md)（公式推导见 [FOC 逐步推导](../formalizations/field-oriented-control-derivation.md)）：

| 环节 | 要点 |
|------|------|
| 对齐 | 极对数、电角度零位、相序 |
| 电流环 | 带宽与 PWM 频率（腿足目标常 >20 kHz 电流环量级，见 Actuator 102） |
| MTPA / 弱磁 | 高速区按 \(L_d, L_q\) 分配 \(i_d, i_q\) |
| 力矩模式 | 上层 CAN/EtherCAT 力矩指令 → \(i_q^*\) |

原型阶段可用 [SimpleFOC](../entities/simplefoc.md) 在 MCU 上验证算法；量产关节多用工业伺服栈。

### 8) 关节模组系统验收

- 含减速器、驱动器、传感器的 **整机 TN/TI** 与单电机不同。
- 与 [电机驱动器底软通信协议](../overview/motor-drive-firmware-bus-protocols.md) 联调力矩/阻抗模式，检查延迟与饱和。
- 减速器选型、双编标定、装配一致性与四层测试矩阵见 [自研关节模组开发流程](../concepts/joint-module-self-development-workflow.md)（本页步骤 1–7 的电机子集在此收口为「可交付模组」）。

## 工程实践

- **安装态热边界与连续能力校验**：无框力矩电机装入密封关节后连续能力通常较自由空气样本下降 25%–35%，应按任务 RMS 电流与 1 小时安装态温升实测校准热阻网络，并核对最高速电压裕度与高温峰值退磁（详见 [额定电流过载后果](../concepts/joint-motor-rated-current-overload.md)）。
- **样机–仿真闭环标定**：台架实测反电动势 $K_e$（电钻/对拖反拖）、LCR 相电阻与电感以及测功机 TN/TI 曲线，反哺 FEMM/PYLEECAN 槽满率与端部漏磁修正。
- **CAE / CFD 自动化工具入口**：电磁、结构与热流体分析软件接口可选读 [CAE / CFD 代理工具与技能总览](./cae-cfd-agent-skills-landscape.md)（如 [Ansys AEDT MCP](../entities/laplaceyoung-ansys-aedt-mcp.md)、[Mechanical MCP](../entities/codersag-mechanical-mcp.md)、[OpenFOAM Claude Suite](../entities/swtbkim-openfoam-claude-suite.md)）；自动化结果仍须与台架实测对照。

## 无框力矩电机设计与工具链 / Frameless Torque Motor Design and Toolchain

无框定转子是套件，轴承、壳体、反馈和散热方式决定安装后可交付的能力。先定义安装态工况，再用以下三道闸门决定是否推进样机：

| 闸门 | 输入 → 输出 | 如何验证 |
|---|---|---|
| 电磁与可驱动性 | 负载谱、空间和母线 → 热态转矩常数、反电势、电感、损耗分布与可用速度 | 同时对齐相/线、峰值/RMS、机械/电气角速度单位；在规定安装与冷却下，对照转矩–电流和转矩–转速曲线，分别测断电齿槽与通电纹波。 |
| 热与最小运行气隙 | 电磁损耗、接触热阻、公差与载荷 → 绕组/磁钢温度、最小气隙、连续区与短时过载时间 | 联合核对轴承及支撑变形、偏心、热膨胀，而不是只看冷态名义气隙；将壳温、绕组温升和装配跳动回填模型。 |
| 驱动与反馈 | 电气参数、编码器误差、采样和驱动约束 → 热态电流环和低速平顺性 | 核对真实母线和电流限幅，区分编码器分辨率与装配后精度；按初始温度、峰值持续时间及占空比分别验收。 |

```mermaid
flowchart LR
  DUTY["负载谱、包络和冷却条件"] --> EST["尺寸量级与母线可行域"]
  EST --> EMG["电磁仿真：损耗、反电势、纹波"]
  EMG --> THM["热与公差：温升、受载最小气隙"]
  EMG --> DRV2["驱动与反馈：电压余量、低速精度"]
  THM --> BENCH["安装态台架：曲线、温升、气隙"]
  DRV2 --> BENCH
  BENCH -->|用实测校准| EMG
```

工具链按问题选：表格或脚本先扫可行域；电磁有限元和集总热网络做损耗—温度传递；结构仿真检查热态受载变形；控制仿真核对母线与采样，最终以台架数据确认。短叠厚件要说明二维模型如何处理端部效应，不能把网格尚未收敛的齿槽结果当验收值。

原文给出的尺寸、热阻、绝缘与平衡示例是量级线索，不是模组通用规范。ISO 官方目录标示 ISO 1940-1:2003 已撤销；刚性转子参照 [ISO 21940-11](https://www.iso.org/standard/54074.html) 的适用范围，实际余量和试验条件按产品、工况与适用标准确认。此段是工程流程整理，原文没有配套运行代码，源码运行时序图不适用。

## 关节结构工程师的学习与供应商评审边界

- 需要了解无框电机的定子—壳体接触热阻、气隙同轴度、槽极/齿槽转矩、连续 TN 与质量口径，但不必以独立完成电机量产设计为首个目标。
- 采购评审前统一供应商的测试电压、安装冷却条件、连续/峰值时长以及裸电机/整机质量；用解析初算、FEMM 与台架记录交叉验证。
- 关于**分阶段自学、谐波寿命模型及自研/外购/联合定制的六维判据**，见 [关节模组工程师的电机设计学习边界](../queries/joint-module-engineer-motor-design-literacy.md)；文章归档附标准名称与寿命口径的勘误。

## 常见误区或局限

| 误区 | 实际情况 |
|------|----------|
| 电磁达标即算设计完成 | 连续区往往被热限制；见步骤 4 |
| 台架只测峰值转矩 | 行走看连续转矩与温升 |
| FOC 调通即可上车 | 极对数/相序错误会导致振动与失控 |
| 仿真 TN 直接当关节 TN | 减速器效率、热阻与驱动限流会改写包络 |

## 关联页面

- [力矩电机设计纵深路线（Stage 0–6 学习顺序展开版）](../../roadmap/depth-torque-motor-design.md)
- [电机测功机（Dynamometer）](../concepts/motor-dynamometer.md)
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)
- [开源力矩电机电磁设计完整度对比](../comparisons/open-source-torque-motor-em-design.md)
- [Ironless QDD](../entities/ironless-qdd-actuator.md) · [Caden Axial Flux Part 1](../entities/cadenkraft-ironless-axial-flux-motor.md) · [Halbach Array](../concepts/halbach-array.md) · [FEMM](../entities/femm.md) · [PYLEECAN](../entities/pyleecan.md)
- [电机转矩-转速曲线（TN 曲线）](../concepts/motor-torque-speed-curve.md)
- [电机转矩-电流曲线（TI 曲线）](../concepts/motor-torque-current-curve.md)
- [磁场定向控制（FOC）](../concepts/field-oriented-control.md)
- [Humanoid 执行器 102 技术地图](./humanoid-actuator-102-technology-map.md)
- [自研关节模组开发流程](../concepts/joint-module-self-development-workflow.md) — 传动构型、五件套、制造与验收全链路
- [关节模组工程师是否需要学习电机设计？](../queries/joint-module-engineer-motor-design-literacy.md) — 学习深度、规格评审与自研/外购决策

## 参考来源

- [Zane Hub：无框力矩电机设计与工具链](../../sources/blogs/wechat_zanehub_frameless_torque_motor_toolchain_2026-11-28.md)：原文归档、公式口径及安装指南、软件手册、TI 应用报告和 ISO 标准目录的核验。

- [ansys_motor_cad_electric_machine_design.md](../../sources/sites/ansys_motor_cad_electric_machine_design.md)
- [motor_curves_and_em_simulation_faq.md](../../sources/personal/motor_curves_and_em_simulation_faq.md)
- [simplefoc_documentation.md](../../sources/sites/simplefoc_documentation.md)
- [femm_info.md](../../sources/sites/femm_info.md)
- [motor_dynamometer_primary_refs.md](../../sources/sites/motor_dynamometer_primary_refs.md)
- [开源力矩电机电磁设计策展](../../sources/personal/open_source_torque_motor_em_design_curator.md)
- [Caden Kraft Axial Flux Part 1](../../sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md)
- [Zane Hub：关节模组设计工程师有没有必要自学电机设计？](../../sources/blogs/wechat_zanehub_joint_module_engineer_motor_design_2026-11-22.md)
- [Modern Robotics Ch. 8.9 Actuators](../../sources/papers/modern_robotics_ch8_actuators.md) — 理想 TN 曲线、连续/间歇区与 $G^2$ 反射惯量
- [Halbach 1980](../../sources/papers/halbach_permanent_multipole_magnets_1980.md)


## 推荐继续阅读

- [Ansys Motor-CAD 产品页](https://www.ansys.com/products/electronics/ansys-motor-cad)
- [FEMM 官方文档](https://www.femm.info/doku/doku.php?id=documentation)
- [SimpleFOC — FOC theory](https://docs.simplefoc.com/foc_theory)
- Caden Kraft Axial Flux Part 1：<https://cadenkraft.com/designing-a-coreless-axial-flux-motor-part-1/>
