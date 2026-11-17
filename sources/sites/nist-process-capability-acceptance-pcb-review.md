# NIST：过程能力与抽样验收原文复核

- **类型：** 官方技术资料 / 原文复核
- **核对日期：** 2026-11-16
- **核对范围：** NIST/SEMATECH e-Handbook §6.1.6 What is Process Capability? 与 §6.2.1 What is Acceptance Sampling? 的完整正文、公式、表与图注。未将其参考书目或外链章节视为已读；没有读取微信文章全文或认证某条人形产线。
- **通读：** 已完成

## 原文与版本

- [Process capability](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm)；本次读取材料 SHA-256：`0fdc4d65ce694d1896f03c42b9941e4fe5ea15c89bb50c83be7353804e1b9c4f`。
- [Acceptance sampling](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc21.htm)；本次读取材料 SHA-256：`e51cca4c7f6aa15f746b1bcf0be629e02730eff6dcd99987db62e7c006da5044`。

## 核对结论

- 过程能力比较受控过程分布与规格限，Cp、Cpk 不是对未受控或任意分布过程的万能良率换算。
- Cp=(USL-LSL)/(6σ)；Cpk 为两侧裕量除以 3σ 的较小值。文中示例 Cp=1、均值偏移后 Cpk=0.6667，说明分布宽度相同仍可能因偏心而不合格。
- 能力估计还依赖独立样本量、分布假设和置信区间；中心正态条件下 Cp=1.33 对应约 64 ppm 拒收，不能把任何 Cpk≥1.33 直接写成保证 99.99% 良率。
- 抽样验收的目标是决定接受或拒绝一批，而非估计该批质量，更不能代替持续过程控制。客户接受限值、抽样风险和工程边界须单独约定。

## 对 wiki 的映射

- [humanoid-mass-production-engineering](../../wiki/concepts/humanoid-mass-production-engineering.md)
