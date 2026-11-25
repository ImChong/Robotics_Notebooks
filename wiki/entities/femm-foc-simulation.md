---
type: entity
tags: [software, motor, femm, foc, simulation, open-source, education, bldc, femm-foc-simulation]
status: complete
updated: 2026-11-21
original:
  - ../../sources/repos/femm_foc_simulation.md
related:
  - ./femm.md
  - ../comparisons/open-source-torque-motor-em-design.md
  - ../comparisons/motor-em-simulation-software.md
  - ./pyleecan.md
  - ./ironless-qdd-actuator.md
  - ./acmop.md
  - ../concepts/field-oriented-control.md
  - ../formalizations/field-oriented-control-derivation.md
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/femm_foc_simulation.md
  - ../../sources/sites/femm_info.md
  - ../../sources/personal/open_source_torque_motor_em_design_curator.md
summary: "FEMM-FOC-Simulation：开源教学仓，用定/转子 DXF + 已配置 .fem + Lua，按 FOC 电流与转子角扫描径向 BLDC 转矩；适合学 FEMM 工作流，非人形可制造电机。"
---

# FEMM-FOC-Simulation（FEMM 磁场定向控制教学仿真）

## 一句话定义

**FEMM-FOC-Simulation**（[yoga-cycle/FEMM-FOC-Simulation](https://github.com/yoga-cycle/FEMM-FOC-Simulation)）用开源 **[FEMM](./femm.md)** 对小型径向 BLDC 做 **FOC 电流 + 转子机械角扫描**，输出转矩—角度曲线；仓库提供 DXF、`.fem` 与 `FOC_sim.lua`。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FEMM | Finite Element Method Magnetics | 开源 2D 电磁有限元 |
| FOC | Field-Oriented Control | 磁场定向控制 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| DXF | Drawing Exchange Format | 二维 CAD 交换格式，可导入 FEMM |
| Lua | Lua scripting language | FEMM 内置自动化脚本语言 |

## 为什么重要

- 把「打开 FEMM」拆成可跟做的七步：**DXF → 材料 → 绕组方向 → 磁钢磁化 → FOC 电流 → 扫角 → 平均转矩/脉动**。
- 成本接近零，比直接上 Maxwell/Motor-CAD 更适合 [力矩电机纵深 Stage 2](../../roadmap/depth-torque-motor-design.md) 入门。
- 与 [Ironless](./ironless-qdd-actuator.md) 的静态 FEMM 对照互补：本仓强调 **随转子角变化的 FOC 转矩**。

## 核心原理与源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant User as 用户 / FEMM GUI
  participant Lua as resources/FOC_sim.lua
  participant FEMM as FEMM 4.2 求解器

  User->>FEMM: 打开 resources/FOC_sim.fem
  User->>Lua: File -> Open lua script -> FOC_sim.lua
  Lua->>FEMM: mi_saveas("temp.fem") + mi_seteditmode("group")
  Lua->>FEMM: 初始旋转 360/14/2 - 360/12/2 对齐 d 轴与 A 相
  loop n = 0..90 (每步 0.2857° 机械角 = 2° 电角, 共 180° 电角)
    Lua->>Lua: Inverse Park (Id=0, Iq=20) -> Inverse Clarke (ia, ib, ic)
    Lua->>FEMM: mi_modifycircprop("A"/"B"/"C", 1, ia/ib/ic)
    Lua->>FEMM: mi_analyze(1) -> mi_loadsolution()
    Lua->>FEMM: mo_groupselectblock(1) -> mo_blockintegral(22)
    FEMM-->>Lua: 输出转矩并 mi_moverotate(0, 0, 0.2857)
  end
  Lua-->>User: 控制台打印 (rotor_angle, torque) 序列
```

源码核对（`resources/FOC_sim.lua`）：模型为 **14 极 12 槽**电机（1 机械度 = 7 电度），叠长约 **10 mm**，固定设 `I_d = 0`、`I_q = 20 A`，每步调用 `mo_blockintegral(22)`（Weighted Stress Tensor 转矩积分）输出离散角度—转矩表。

## 工程实践

| 步骤 | 操作 |
|------|------|
| 1 | 下载 `resources/` |
| 2 | 安装 FEMM，打开 `FOC_sim.FEM` 观察绕组与磁钢 |
| 3 | 运行 `FOC_sim.lua`，记录转矩曲线 |
| 4 | 改电流幅值/相位，观察平均转矩与脉动 |
| 5 | （进阶）换自己的 DXF，迁移材料与电路定义 |

## 局限与风险

- **教学仿真**：尺寸小，非人形优化；无完整机械、热、绕线制造与成熟样机测试。
- GitHub 无 SPDX license 元数据——二次分发前自行确认。
- 2D 磁静力 + 脚本扫角 ≠ 驱动器实时 FOC；控制理论仍读 [FOC](../concepts/field-oriented-control.md)。

## 关联页面

- [FEMM](./femm.md) — 求解器本体、文档与官方示例入口
- [开源力矩电机电磁设计完整度对比](../comparisons/open-source-torque-motor-em-design.md)
- [电机电磁仿真软件选型](../comparisons/motor-em-simulation-software.md)
- [PYLEECAN](./pyleecan.md) · [Ironless QDD](./ironless-qdd-actuator.md)
- [执行器驱动链选型闭环](../queries/actuator-drive-chain-selection-loop.md)
- [FOC 逐步推导](../formalizations/field-oriented-control-derivation.md)

## 参考来源

- [sources/repos/femm_foc_simulation.md](../../sources/repos/femm_foc_simulation.md)
- [sources/sites/femm_info.md](../../sources/sites/femm_info.md)
- [开源力矩电机电磁设计策展](../../sources/personal/open_source_torque_motor_em_design_curator.md)

## 推荐继续阅读

- 仓库：<https://github.com/yoga-cycle/FEMM-FOC-Simulation>
- FEMM 文档：<https://www.femm.info/doku/doku.php?id=documentation>
- 灵感博文（README）：<https://things-in-motion.blogspot.com/2019/02/how-to-model-bldc-pmsm-motors-kv.html>
