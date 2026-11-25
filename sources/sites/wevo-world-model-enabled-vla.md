# WEVO — 官方项目主页核验（2026-11-21）

> 一手资料（academic project website）；详情见 [WEVO 独立论文节点](../../wiki/entities/paper-wevo-world-model-enabled-vla-loco-manipulation.md)。

- **项目名称：** WEVO: World Model-Enabled VLA Optimization for Whole-Body Loco-Manipulation
- **项目主页：** https://qianzhong-chen.github.io/wevo.github.io/
- **论文：** https://arxiv.org/abs/2611.25251v1
- **作者/机构：** Qianzhong Chen 等；亚马逊、斯坦福大学、得克萨斯大学奥斯汀分校
- **公开状态：** 网站头部标注 "Code (coming soon)"，未提供 WEVO 的公开代码、权重或数据集下载链接；有介绍视频及结果演示。论文已在 arXiv 发布，但页面顶部仍写 arXiv (coming soon)，应以已核验的 arXiv 题录为准。
- **入库日期：** 2026-11-21
- **核对范围：** 逐项阅读项目页 Overview、Data、World Model、Results、BibTeX，核对流程图说明、20 条真实轨迹/任务、200 条仿真示范/任务、480 条世界模型轨迹/任务、三任务 Table/各消融及页面头部 Code/Resources 状态，并与 arXiv v1 正文交叉确认。
- **通读：** 已完成

## 官方页面可确认的内容
- LEGS 提供全身合成演示和操作/行走 play；20 条保留真机自主轨迹/任务用以部署域世界模型校准。
- 闭环由 GR00T VLA、SONIC–MuJoCo WBC 和视频世界模型组成，采用阶段感知密集进度奖励。
- 三任务真机平均成功率：BC 23.3%、Sim RL 10.0%、Real RL 48.3%、WEVO DICE-RL 75.0%，均为论文所设实验条件。
- 页面未发布 WEVO 可运行实现，因此不能编写基于真实源码的 train/infer 时序图或安装命令。

## 继续阅读
- [论文原文归档](../papers/wevo_arxiv_2611_25251.md)
- [arXiv HTML 全文](https://arxiv.org/html/2611.25251v1)
