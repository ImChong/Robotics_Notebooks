# Legged Load Adapt 官方项目页

> 来源归档（项目页核查）

- **标题：** Beyond Robustness: Learning Unknown Dynamic Load Adaptation for Quadruped Locomotion on Rough Terrain
- **类型：** site / project-page
- **官方入口：** <https://leixinjonaschang.github.io/leggedloadadapt.github.io/>
- **论文：** <https://arxiv.org/abs/2507.07825>
- **机构：** 浙江大学国际联合学院（ZJU-UIUC Institute）
- **入库日期：** 2026-08-02
- **一句话说明：** 展示四足未知动态载荷适应的方法示意、仿真对比（Baseline / NLW / LW / Ours）与真机视频入口。
- **代码：** 项目页标注 **Code (comming soon)**；截至 2026-08-02 **无独立公开仓库 URL**。
- **开源状态（2026-08-02 核查）：** **宣称将开源 / 待发布**；不可复现训练或部署流水线。

## 页面公开信息

- 方法：load characteristics modeling（质量、摩擦、位置、速度）+ RL locomotion；teacher 用特权载荷特征，concurrent estimator 从本体历史预测；teacher–student 蒸馏 latent 以支持盲走部署。
- 仿真对比：6 kg 载荷、μ=0.01 崎岖地形上相对 Baseline / NLW / LW。
- BibTeX 指向 arXiv:2507.07825（非 2109.12343）。

## 对 wiki 的映射

- 论文归档：[legged_load_adapt_arxiv_2507_07825.md](../papers/legged_load_adapt_arxiv_2507_07825.md)
- 实体页：[paper-legged-load-adapt-unknown-dynamic-load.md](../../wiki/entities/paper-legged-load-adapt-unknown-dynamic-load.md)
