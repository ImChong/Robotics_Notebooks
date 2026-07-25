# Internal-Cycloidal-Actuator

> 来源归档

- **标题：** Internal Cycloidal Actuator
- **类型：** repo
- **作者：** Aaed Musa
- **链接：** https://github.com/aaedmusa/Internal-Cycloidal-Actuator
- **项目页：** https://www.aaedmusa.com/projects/internalcycloidalactuator
- **许可：** GitHub API 显示 `license: null`（使用前自行确认）
- **星标（截至 2026-07-25）：** ~291
- **入库日期：** 2026-07-25
- **一句话说明：** 自制外转子 BLDC + 定子内嵌双摆线减速的一体 QDD 关节；含 CAD、BOM 与装配过程，偏电机本体学习。
- **开源状态：** **已开源**（CAD/BOM；项目页有规格与绕组说明）
- **项目页归档：** [sources/sites/aaedmusa_internal_cycloidal_actuator.md](../sites/aaedmusa_internal_cycloidal_actuator.md)
- **沉淀到 wiki：** [internal-cycloidal-actuator](../../wiki/entities/internal-cycloidal-actuator.md)

---

## 项目页规格（摘录，2024-02）

| 项 | 值 |
|----|-----|
| 槽极 | 36N42P（36 槽定子 / 42 磁极） |
| 尺寸 / 质量 | ⌀125×84 mm / 1023 g |
| 减速 | 8:1 摆线，嵌入定子中心 |
| 力矩 / 转速 | 16.17 N·m / 209 RPM @ 22.2 V |
| 相电阻 / 电感 | 75 mΩ / 41.05 µH |
| 驱动 | ODrive S1 FOC |
| BOM | ~$384 |

## 设计要点

- 外转子大间隙半径 → 高扭矩密度；定子购自成品 10010（36 槽），绕组自绕（6×26AWG、6 匝/槽）。
- 转子 mild steel 1045 机加；N52 磁钢；摆线固定环铝件嵌入定子中心。
- 局限：3D 打印件受线圈热影响易翘曲；缺工业级冲击/寿命验证。

## 对 wiki 的映射

- [Internal Cycloidal Actuator](../../wiki/entities/internal-cycloidal-actuator.md)
- [开源 QDD 执行器项目对比](../../wiki/comparisons/open-source-qdd-actuator-projects.md)
