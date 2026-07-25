# 开源机器人力矩电机：电磁设计完整度策展

> 来源归档

- **标题：** 同时公开「定转子几何＋槽极/绕组＋磁钢布置＋电磁仿真＋可制造结构」的开源力矩电机项目筛选
- **类型：** personal（维护者策展 / 选型整理）
- **入库日期：** 2026-07-25
- **一句话说明：** 按电磁设计与可制造性完整度排序开源机器人关节电机相关项目；结论是「完整电磁＋真机关节样机」目前最明确的是 Ironless-QDD-Actuator，其余多为 FEMM 教材、PCB 轴向绕组、或参数化设计工具。
- **沉淀到 wiki：** [wiki/comparisons/open-source-torque-motor-em-design.md](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- **相关策展（关节系统侧）：** [open_source_qdd_actuator_learning_curator.md](./open_source_qdd_actuator_learning_curator.md)

---

## 筛选标准

真正同时公开下列五项的机器人力矩电机项目极少：

1. 定转子几何（或等价可复现几何）
2. 槽极 / 绕组方案
3. 磁钢布置（含磁化方向）
4. 电磁仿真文件（FEMM / FEA 或可复现脚本）
5. 可制造结构（CAD / PCB / BOM 等）

本策展按**完整程度**排序，并区分「固定硬件样机」与「参数化设计工具」。

## 完整度对照表（策展结论）

| 项目 | 电机几何 | 绕组 | 磁钢 | FEM/仿真 | CAD/制造 | 实物验证 | 人形适用性 |
|------|----------|------|------|----------|----------|----------|------------|
| Ironless-QDD | ✅（采购 10010 定子 + 自研转子） | ✅ 36N42P | ✅ Halbach | ✅ FEMM 多方案 | ✅ STEP/打印/BOM | ✅ 保持力矩台架 | 较高（学习向） |
| PCB Motor | ✅ PCB 轴向 | ✅ 多拓扑 | ✅ 可 Halbach | 部分（文献+设计） | ✅ KiCad | 部分 / WIP | 小型关节 |
| FEMM-FOC | ✅ DXF | ✅ | ✅ | ✅ .fem + Lua | DXF | 弱（仿真） | 教学 |
| axfluxmdo | 参数化 | 参数化 | 参数化 | Gmsh/GetDP | 3D 生成 | 无固定样机 | 设计工具 |
| PYLEECAN | 参数化 | ✅ | 参数化 | 自动 FEMM | 可导出 | 非固定样机 | 设计工具 |
| ACMOP | 参数化 | ✅ | 参数化 | FEMM/JMAG | 有限 | 研究案例 | 优化工具 |

## 项目要点（摘录）

### 1. Ironless-QDD-Actuator（最完整样机）

- GitHub：https://github.com/CKraft11/Ironless-QDD-Actuator
- 项目页：https://cadenkraft.com/ironless-cycloidal-planetary-actuator/
- 仓库含：完整执行器 STEP/CAD、FEMM（有/无铁、Halbach/常规对照）、`36N42P Winding Scheme.png`、BOM、MKS XDrive/ODrive 配置、摆线—行星减速结构。
- 电机：采购 **10010** 36 槽定子 + 自绕（约 6 匝×6 股 0.4 mm）+ **42 极** Halbach 无铁芯外转子；低速大直径 QDD 关节。
- 报告 **~29.4 N·m 保持力矩**（含减速器增益）；**≠** 裸电机连续/额定/冲击力矩。
- 局限：个人实验级；缺连续温升曲线、电磁—实测转矩系统对照、退磁/涡流/铁耗/超速/疲劳/人形冲击等工业验证。

### 2. FEMM-FOC-Simulation（径向磁通 FEMM 入门）

- https://github.com/yoga-cycle/FEMM-FOC-Simulation
- 定/转子 DXF、.fem、材料与绕组、Lua 扫转子角 + FOC 电流 → 转矩。
- 教学仿真；小尺寸、无完整机械/热/制造与成熟样机。

### 3. PCB Motor（PCB 轴向磁通绕组）

- https://github.com/ziteh/pcb-motor（WIP，MIT）
- 公开槽极/绕组拓扑对比、KiCad、约 20 极 / 6 层 PCB / 2 mm 厚 / 铜 140 μm / 气隙 1 mm；可考虑 Halbach 转子。
- 适合手指/腕/灵巧手/微型执行器，不适合人形髋膝。

### 4. axfluxmdo（轴向磁通参数化 MDO）

- https://github.com/jman4162/axfluxmdo · https://jman4162.github.io/axfluxmdo/
- 解析 + 2.5D 环带模型、损耗/热/脉动/轴向力、Pareto/贝叶斯优化、Gmsh/GetDP；不替代高保真 3D 瞬态 FEA；无固定加工样机。

### 5. PYLEECAN（径向磁通 PMSM 开源建模）

- https://github.com/Eomys/pyleecan · https://www.pyleecan.org/
- SPMSM/IPMSM 等拓扑 GUI、绕组、材料库、FEMM 非线性、损耗、扫描与多目标优化。
- 适合自设外转子 24/28 或 36/42、集中绕组、48 V、低 KV 人形关节电机方案；本身不是硬件仓。

### 6. ACMOP（交流电机自动优化）

- https://github.com/horychen/ACMOP
- 几何参数化、FEMM/JMAG、BH 曲线、多目标与报告；依赖偏旧 JMAG/Python 环境。
- 适合研究自动扫槽宽/齿宽/磁钢厚度与 Pareto；不建议作为第一入门。

## 策展结论与学习建议

- **「完整电磁设计已公开 + 做成机器人关节样机」** 目前最明确：Ironless-QDD-Actuator。
- 其它项目通常只覆盖子集：仅 FEMM、仅 PCB 绕组、仅参数化工具、或关节开源但电机外购。
- 建议路径：先用 **Ironless** 走通「电磁 → 绕线 → 转子 → FEMM → 减速 → 驱动 → 台架」；再用 **PYLEECAN** 重设更接近人形外转子径向磁通 PMSM。轴向磁通薄、力矩半径大，但公差、轴向磁拉力、散热与多气隙装配通常更难。

## 对 wiki 的映射

- 对比主页：[open-source-torque-motor-em-design](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- 样机实体：[ironless-qdd-actuator](../../wiki/entities/ironless-qdd-actuator.md)
- 工具/教材实体：[pyleecan](../../wiki/entities/pyleecan.md)、[axfluxmdo](../../wiki/entities/axfluxmdo.md)、[pcb-motor](../../wiki/entities/pcb-motor.md)、[femm-foc-simulation](../../wiki/entities/femm-foc-simulation.md)、[acmop](../../wiki/entities/acmop.md)
- 关节系统侧对照：[open-source-qdd-actuator-projects](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
- 仿真软件选型：[motor-em-simulation-software](../../wiki/comparisons/motor-em-simulation-software.md)
- 纵深路线：[depth-torque-motor-design](../../roadmap/depth-torque-motor-design.md)
