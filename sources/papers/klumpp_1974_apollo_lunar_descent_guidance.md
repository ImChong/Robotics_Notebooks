# Apollo Lunar Descent Guidance（Klumpp, Automatica 1974）

> 来源归档（ingest）

- **标题：** Apollo Lunar Descent Guidance
- **作者：** A. R. Klumpp（Charles Stark Draper Laboratory）
- **出版物：** *Automatica*, Vol. 10, No. 2, 1974, pp. 133–146，doi:10.1016/0005-1098(74)90019-3（卷期页码与 DOI 按 [Blackmore 等 2010](blackmore_2010_minimum_landing_error_pdg.md) 参考文献 [8] 的引用核对）
- **类型：** journal paper（另有 1971 年 MIT Draper 实验室报告版本 R-695，本次未能取得）
- **NTRS 记录：** <https://ntrs.nasa.gov/citations/19740044219>（Document ID 19740044219，"Reprint (Version printed in journal)"，出版日期 1974-03-01）
- **入库日期：** 2026-11-06
- **核对范围：** 只读了 NTRS 记录页的摘要与元数据。NASA ALSJ 上的 R-695 PDF 链接返回 404，ibiblio 与存档镜像在本环境无法下载，正文、方程与程序（P63–P66）细节均未核对。
- **通读：** 未完成
- **一句话说明：** Apollo 登月舱动力下降制导的设计总结，是"火箭用推力从高速下降到软着陆"这一制导问题的工程起点。

## 核心摘录（仅摘要内容）

1. **任务跨度。** 摘要写明制导把登月舱从近圆轨道送到触地，"traversing a 17 deg central angle and a 15 km altitude in 11 min"。
2. **机载 + 地面分工。** "A group of interactive programs in an onboard computer guide the descent, controlling altitude and the descent propulsion system throttle. A ground-based program pre-computes guidance targets."
3. **显式 vs 隐式制导。** 摘要结论："the earlier Apollo explicit equation is shown to be an inferior special case of the later implicit equation."
4. **人在回路。** 航天员可改选着陆点（"the two-man crew selects a landing site in favorable terrain"），末段可由航天员控制近乎竖直的下降速率，以安全接触速度在最短时间内着陆。
5. **姿态机动。** "The altitude maneuver routine uses concepts that make gimbal lock inherently impossible."（原文如此，疑为 attitude 的笔误，未能对照正文）

## 后续文献如何引用它

[Blackmore 等 2010](blackmore_2010_minimum_landing_error_pdg.md) 把 Klumpp 归为"忽略约束、求闭式解"的一类方法（引文 [8,17]），并以此说明为什么需要能处理推力上下界、滑翔坡约束的凸优化制导。

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md) — 历史起点
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 1
