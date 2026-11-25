# 404 Possession Not Found | 参赛技术报告与源码

- **类型：** repo / technical report
- **团队：** 404 Possession Not Found（参赛队）；成员 Rodrigo Fernández Moro、Jorge Arce Nicolás、Alberto Rivas Casal、Gonzalo Fernández González。
- **代码：** [参赛仓库](https://github.com/rodrifdezz/404-Possession-Not-Found) · [固定提交 ead83bb](https://github.com/rodrifdezz/404-Possession-Not-Found/tree/ead83bb0c9d9e1ee844a6aab5fa9ddcfb0937fb5)
- **原始技术文档：** [TECHNICAL.md](https://github.com/rodrifdezz/404-Possession-Not-Found/blob/ead83bb0c9d9e1ee844a6aab5fa9ddcfb0937fb5/docs/TECHNICAL.md) · [v4 修正与验证](https://github.com/rodrifdezz/404-Possession-Not-Found/blob/ead83bb0c9d9e1ee844a6aab5fa9ddcfb0937fb5/docs/BENCH_V4_FIX.md)
- **主办方：** [Metrica Sports / Barça Innovation Hub 官方挑战与评测](https://github.com/metrica-sports/more-than-a-hack-2027) · [数据与规则概览](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md) · [评分规则](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
- **核对范围：** 参赛 README、技术报告（方法、数据、验证、限制与复现）、v4 修正报告、示例说明，主办方定义与评分，及 `make_submission`、`pipeline`、特征、堆叠推断和 HMM 解码入口。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **许可：** 队伍代码/拟合产物可公开查看；固定提交未见明确的 LICENSE，不推定可再分发。官方评测代码 MIT，比赛数据另外授权，整场真值不在公开仓库。
- **对应节点：** [404 Possession Not Found](../../wiki/entities/project-404-possession-not-found.md)

## 原始结果口径

- 经过评测的战术机位帧约 41% 缺球，不等于整段视频的缺球比例。从主办方 ATD 做标签而非从像素训练检测。每帧 89 个基础特征和 18 个触球特征；两级梯度提升再以分半场 Viterbi 解码。
- 10 场外部比赛（3 Metrica、7 IDSSE/DFL）对干净跟踪模拟降质，5 折按整场划分，每折留出 2 场。v3 → v4 的平均 spellF1 为 0.608 → 0.667，macroF1 0.888 → 0.911；同折选择多个候选，报告明确存在乐观偏差。
- 主办方隐藏整场有不同数据分布；[团队成员赛后自述（公开页面转载的原帖内容）](https://ma.linkedin.com/in/saad-el-mouwahid-550620225) 报告 0.553，官方基线 0.2433。0.553 未独立复核正式榜单，不和外部验证 0.667 混写。
- 地面真实标签的来源、输入缺失与规则分开：比赛前 5 分钟公开真值只查接线，不用于调参。本文核对源码，并运行官方仓库的 7 项 HMM 单测；未用授权比赛数据生成整场提交。
