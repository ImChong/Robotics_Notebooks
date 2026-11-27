# iMFPO: Improving Mean Flow Policy Optimization with Drift-Aware Velocity Alignment for Embodied Control

- **类型：** 论文（arXiv 预印本）
- **作者：** Lisiyu Pan、Ting Huang、Zeyu Zhang、Hao Tang
- **机构：** 北京大学
- **日期：** 2026-11-25
- **版本：** arXiv:2611.33922v1
- **链接：** [摘要](https://arxiv.org/abs/2611.33922v1) · [全文 PDF](https://arxiv.org/pdf/2611.33922v1) · [全文 HTML](https://arxiv.org/html/2611.33922v1)
- **项目页：** [官方演示](../sites/imfpo.md)
- **代码：** [仓库归档](../repos/imfpo.md)
- **通读：** 已完成
- **核对范围：** v1 PDF 方法、实验、限制与结论；核对图 1–6、表 1–3、式 (6)–(12) 的 JVP、策略目标与单步推理；对照项目页与源码中的评测、训练和导出入口。
- **对 wiki 的映射：** [iMFPO](../../wiki/entities/paper-imfpo.md)

## 核对摘记

1. §3：同一 actor 区分瞬时和平均速度，JVP 修正把两者置于一致的输运监督中；教师只在训练提供动作目标。映射：Wiki 核心机制和流程图。
2. §4 及表 1–3：区分零/随机潜变量与 1/64 NFE，Go2 报告 GPU batch 1 actor p50，不包含环境推进。映射：Wiki 实验与评测。
3. §4.4–6：Go2 的定性真机画面与语言导航集成展示不同于多平台仿真平均回报；作者列出场景和选定检查点的边界。映射：Wiki 结论/局限。
