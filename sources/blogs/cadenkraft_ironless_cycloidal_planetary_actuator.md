# Ironless Rotor Cycloidal Planetary Actuator（Caden Kraft）

> 来源归档

- **标题：** Ironless Rotor Cycloidal Planetary Actuator
- **类型：** blog
- **作者：** Caden Kraft（CKraft11）
- **链接：** https://cadenkraft.com/ironless-cycloidal-planetary-actuator/
- **入库日期：** 2026-07-25
- **一句话说明：** 低成本全 3D 打印 QDD：Halbach 无铁芯转子 FEMM 四象限对照、采购 10010 自绕 36N42P、摆线齿形行星（~7:1）、MKS XDrive Mini；保持力矩至约 29.4 N·m；CAD/FEMM/BOM 已开源。
- **开源状态：** **已开源**
- **代码：** https://github.com/CKraft11/Ironless-QDD-Actuator
- **齿轮工具：** https://github.com/CKraft11/pygeartrain
- **项目页归档：** [cadenkraft_ironless_cycloidal_planetary_actuator.md](../sites/cadenkraft_ironless_cycloidal_planetary_actuator.md)
- **仓库归档：** [ironless_qdd_actuator.md](../repos/ironless_qdd_actuator.md) · [pygeartrain.md](../repos/pygeartrain.md)
- **沉淀到 wiki：** [ironless-qdd-actuator](../../wiki/entities/ironless-qdd-actuator.md)、[pygeartrain](../../wiki/entities/pygeartrain.md)

---

## 文中要点（归纳，非全文）

### 目标与动机

- 目标：单执行器造价低于 **80 USD**、全定制件可打印、保持力矩不低于 **10 N·m**、易集成（相对 Mini Cheetah 级 COTS QDD 的 500–1000 USD）。
- 前作痛点：SCARA 上为 Nema 23 做高力矩/高精度减速器的挫败；本项目作可复用平台。

### 转子 / FEMM

- 四象限静态转矩对照（作者 FEMM，相对量）：Halbach+铁背 100%；Halbach 无铁 ~91%；常规+铁 ~91%；常规无铁 ~72%。
- Halbach 相对有铁方案接近，且惯量更低（利于换向）。
- 磁钢：主极约 42×(12×5×3 mm) N52 + 辅助小磁钢；气隙约 **0.7 mm**。

### 减速器

- 传统渐开线齿在 FDM 上易圆角与齿根剪切；改用 **摆线齿廓行星**（连续叶瓣、近零背隙叙事）。
- 齿廓由作者 fork/扩展的 [pygeartrain](../repos/pygeartrain.md) 生成并可导出 SolidWorks；本机约 **7:1**；支持螺旋/双螺旋摆线齿。

### 定子 / 驱动 / 整机

- 采购 **10010**（36 槽）；绕组约 6 匝（5 整+1 半）× 6 股 0.4 mm；相电阻公差约 ±0.8%。
- 驱动：MKS XDrive Mini（ODrive 3.6 系 + 集成霍尔磁编）；相对 moteus/新 ODrive 成本可控。
- 材料：PA6-GF；作者事后建议齿轮改用无玻璃纤维尼龙以防磨损。
- 质量约 **728 g**；反驱阻力很小；宣称零背隙。
- \(K_v \approx 79\)（电钻反拖测反电势）。
- 台架：24 V/25 A 电源下保持约 **14 N·m**（电源限流）；Audi e-tron 模组供电下约 **29.4 N·m**（117.5 N @ 250 mm 臂），线圈表面约 78°C，控制器 50 A 限流停；比力矩叙事约 **40 N·m/kg**（保持口径）。
- BOM：执行器约 **40 USD**、含控制器约 **70 USD**（关税前下单价；约 **2.47 USD/N·m** 保持口径）。

## 对 wiki 的映射

- [Ironless QDD Actuator](../../wiki/entities/ironless-qdd-actuator.md)
- [pygeartrain](../../wiki/entities/pygeartrain.md)
- [Caden Kraft Ironless Axial Flux Motor](../../wiki/entities/cadenkraft-ironless-axial-flux-motor.md)（Halbach 前作）
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
