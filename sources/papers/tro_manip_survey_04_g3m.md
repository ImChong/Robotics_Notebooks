# Learning From Videos Through Graph-to-Graphs Generative Modeling for Robotic Manipulation（G3M）

> 来源归档（ingest · T-RO 操作学习 5 篇精选 第 04/5）

- **标题：** Learning From Videos Through Graph-to-Graphs Generative Modeling for Robotic Manipulation
- **简称：** G3M（Graph-to-Graphs Generative Modeling）
- **类型：** paper
- **T-RO 分类：** 03 无标签视频预训练
- **机构：** 北京理工大学
- **出处：** IEEE T-RO 2026 · DOI:10.1109/TRO.2026.3658211
- **姊妹版：** GraphMimic（CVPR 2025, pp. 1756–1768）
- **论文链接：** <https://doi.org/10.1109/TRO.2026.3658211> · CVPR 2025 <https://openaccess.thecvf.com/content/CVPR2025/html/Chen_GraphMimic_Graph-to-Graphs_Generative_Modeling_from_Videos_for_Policy_Learning_CVPR_2025_paper.html>
- **入库日期：** 2026-07-08
- **一句话说明：** 视频帧抽象为 **物体顶点 + 视觉动作顶点** 图结构，图到图生成预训练隐式学操作规律，再条件化下游策略；**20% 标注** 达全量性能。

## 核心摘录（策展，非全文）

- **物体顶点** 捕捉操作对象状态；**视觉动作顶点** 捕捉手-物交互；边编码空间关系。
- 预训练：给定当前帧图，生成未来帧图 → 隐式学习物理/行为逻辑。
- 仿真 +17%+、真机 +23%+；跨本体迁移 +33%+（期刊版 T-RO 文内 +35%+）。
- 图结构捕捉 **可迁移空间关系** 而非本体特定动作细节。

## 对 wiki 的映射

- [paper-tro-manip-04-g3m](../../wiki/entities/paper-tro-manip-04-g3m.md)
- [tro-manip-category-03-video-pretraining](../../wiki/overview/tro-manip-category-03-video-pretraining.md)

## 参考来源（原始）

- T-RO：<https://doi.org/10.1109/TRO.2026.3658211>
- CVPR 2025 GraphMimic：<https://openaccess.thecvf.com/content/CVPR2025/html/Chen_GraphMimic_Graph-to-Graphs_Generative_Modeling_from_Videos_for_Policy_Learning_CVPR_2025_paper.html>
- 微信公众号编译：[wechat_shenlan_tro_manip_5_papers_survey.md](../blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
