# Ironless-QDD-Actuator

> 来源归档

- **标题：** Ironless Rotor Cycloidal Planetary Actuator
- **类型：** repo
- **作者：** CKraft11（Caden Kraft）
- **链接：** https://github.com/CKraft11/Ironless-QDD-Actuator
- **项目页：** https://cadenkraft.com/ironless-cycloidal-planetary-actuator/
- **许可：** README 声明 MIT（GitHub API `license: null`）
- **星标（截至 2026-07-25）：** ~269
- **入库日期：** 2026-07-25
- **一句话说明：** 低成本无铁芯 Halbach 转子 + 采购 10010 定子自绕 36N42P + 摆线—行星减速（~7:1，pygeartrain）+ 集成驱动；仓库含 FEMM/CAD/BOM；报告静态保持约 29.4 N·m。
- **开源状态：** **已开源**
- **项目页归档：** [cadenkraft_ironless_cycloidal_planetary_actuator.md](../sites/cadenkraft_ironless_cycloidal_planetary_actuator.md)
- **博文归档：** [cadenkraft_ironless_cycloidal_planetary_actuator.md](../blogs/cadenkraft_ironless_cycloidal_planetary_actuator.md)
- **齿廓工具：** [pygeartrain.md](./pygeartrain.md)
- **沉淀到 wiki：** [ironless-qdd-actuator](../../wiki/entities/ironless-qdd-actuator.md) · [pygeartrain](../../wiki/entities/pygeartrain.md) · [cadenkraft-ironless-axial-flux-motor](../../wiki/entities/cadenkraft-ironless-axial-flux-motor.md)

---

## 电磁与制造资产（核查）

| 资产 | 位置 / 说明 |
|------|-------------|
| FEMM | `FEMM/`：有/无铁背、Halbach/非 Halbach 多组 `.FEM`、结果图与 `FEMM_Sim_Torque_Data.xlsx` |
| 绕组 | 根目录 `36N42P Winding Scheme.png`；项目页：10010 定子、约 6 匝×6 股 0.4 mm |
| 磁钢 | 主极约 42×(12×5×3 mm) N52 + 辅助小磁钢构成 Halbach |
| CAD | `CAD/actuator.STEP`、`CAD/Print_Files/`（Git LFS） |
| BOM / 驱动 | `BOM.xlsx`、`MKS XDrive Config.json` |

定子硅钢冲片为 **外购 10010**，不是自研模具开源；公开的是槽数、绕线与转子/整机可制造文件。

## 指标读法警告

作者报告的 ~29.4 N·m 是**静态保持力矩**（含减速器），不可直接当作：

- 连续动态输出力矩
- 额定力矩 / 冲击力矩
- 裸电机电磁转矩
- 人形行走可长期使用力矩

## 对 wiki 的映射

- [Ironless QDD Actuator](../../wiki/entities/ironless-qdd-actuator.md)
- [pygeartrain](../../wiki/entities/pygeartrain.md)
- [Caden Kraft Ironless Axial Flux Motor](../../wiki/entities/cadenkraft-ironless-axial-flux-motor.md)
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
