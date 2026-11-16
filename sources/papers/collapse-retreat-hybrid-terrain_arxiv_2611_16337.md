# 混合地形建模的内存高效坍塌反应式撤退策略学习（2611.16337）

> 论文一手原文归档

- **英文题目：** Memory-Efficient Parallel Policy Learning for Collapse-Reactive Retreat via Hybrid Terrain Modeling
- **arXiv：** https://arxiv.org/abs/2611.16337
- **HTML 全文：** https://arxiv.org/html/2611.16337v1
- **PDF：** https://arxiv.org/pdf/2611.16337v1
- **机构：** 福岛国际研究教育机构（F-REI）、奈良先端科学技术大学院大学
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III 混合地形/SMCM、IV Touch/Sink/Slip、V-A~L 内存/收敛/真机实验、VI 局限；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 稳定区保持 rigid terrain，塌陷区只施加虚拟接触力。State-Multiplexed Contact Model 在环境间共享几何/接触规律，但每环境保留独立塌陷状态，模拟 Touch（接触失支撑）、Sink（下陷）及 Slip（侧滑）；仅用本体感知训练单策略在失支撑后撤退。

## 原文摘录（提炼）
1. **问题与动机：** 真实坍塌区域会随轮足接触改变支撑，若仿真为每个并行环境都复制坍塌刚体则内存高昂。
2. **方法：** 稳定区保持 rigid terrain，塌陷区只施加虚拟接触力。State-Multiplexed Contact Model 在环境间共享几何/接触规律，但每环境保留独立塌陷状态，模拟 Touch（接触失支撑）、Sink（下陷）及 Slip（侧滑）；仅用本体感知训练单策略在失支撑后撤退。
3. **实验：** 各三种坍塌原语完整训练时中位撤退成功率约 90%；省略对应训练原语，相关场景成功率往往低于 40%，去 Touch 时 Touch 成功率低于 10%。论文报告相比刚体复制地形可大幅提高并行规模、缩短收敛时间；Go2W 四类实机坍塌和室外边缘完成撤退验证。
4. **边界与局限：** 虚拟接触不是精细颗粒物理求解；超过 4096 环境时未显著加速收敛。真实数据来自 Go2W 轮足，不能直接推论普通四足/人形。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Collapse-Reactive Retreat](../../wiki/entities/paper-collapse-retreat-hybrid-terrain.md)
