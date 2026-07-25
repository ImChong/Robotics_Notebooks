# pygeartrain

> 来源归档

- **标题：** pygeartrain — gearing calculations and CAD profile export
- **类型：** repo
- **作者：** CKraft11（Caden Kraft）；上游概念致谢 Eelco Hoogendoorn
- **链接：** https://github.com/CKraft11/pygeartrain
- **许可：** MIT
- **星标（截至 2026-07-25）：** ~89
- **入库日期：** 2026-07-25
- **一句话说明：** Python 齿轮系库：行星/复合行星/摆线等传动比符号计算、Matplotlib 运动学可视化、齿廓坐标导出供 SolidWorks；Ironless QDD 用其生成摆线—行星齿形。
- **开源状态：** **已开源**
- **关联项目页：** https://cadenkraft.com/ironless-cycloidal-planetary-actuator/
- **沉淀到 wiki：** [pygeartrain](../../wiki/entities/pygeartrain.md)、[ironless-qdd-actuator](../../wiki/entities/ironless-qdd-actuator.md)

---

## 仓库能力（README 核查）

| 能力 | 说明 |
|------|------|
| 传动比 | 符号计算（如 `Planetary('s','c','r')`） |
| 齿廓 | 摆线、渐开线（WIP）、epi/hypo 混合等 |
| 可视化 | Matplotlib 动画 |
| CAD 导出 | 坐标适合 SolidWorks 导入；`generate_planetary_cad.py` 面向行星 |
| 环境 | Conda `environment.yml` → `conda activate pygeartrain` |

## 对 wiki 的映射

- [pygeartrain](../../wiki/entities/pygeartrain.md)
- [Ironless QDD Actuator](../../wiki/entities/ironless-qdd-actuator.md)
- [Cycloidal Quasi-Direct Drive Actuator](../../wiki/entities/cycloidal-quasi-direct-drive-actuator.md)
