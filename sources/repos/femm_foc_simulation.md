# FEMM-FOC-Simulation

> 来源归档

- **标题：** FEMM simulation of field oriented control for BLDC motors
- **类型：** repo
- **作者：** yoga-cycle
- **链接：** https://github.com/yoga-cycle/FEMM-FOC-Simulation
- **许可：** GitHub API `license: null`
- **星标（截至 2026-07-25）：** ~17
- **入库日期：** 2026-07-25
- **一句话说明：** 用 FEMM + Lua 对径向 BLDC 做 FOC 电流与转子角扫描、输出转矩曲线的教学仿真仓；含定/转子 DXF 与已配置 .fem。
- **开源状态：** **已开源**（公开仓库；无 SPDX license 元数据）
- **沉淀到 wiki：** [femm-foc-simulation](../../wiki/entities/femm-foc-simulation.md)

---

## 仓库资产（核查）

根目录 `resources/` 含：`stator.dxf`、`rotor.dxf`、`FOC_sim.FEM`、`FOC_sim.lua`、示意图与转矩图。README 流程：打开 FEMM → File → Open lua script → 选 `FOC_sim.lua`。

## 对 wiki 的映射

- [FEMM-FOC-Simulation](../../wiki/entities/femm-foc-simulation.md)
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- [电机电磁仿真软件选型](../../wiki/comparisons/motor-em-simulation-software.md)
