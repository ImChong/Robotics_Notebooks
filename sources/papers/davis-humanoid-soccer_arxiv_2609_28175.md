# DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28175](https://arxiv.org/abs/2609.28175)
- **PDF：** https://arxiv.org/pdf/2609.28175
- **项目页：** https://thusi-lab.github.io/DAVIS/
- **开源结论（2026-09-24）：** **待发布**

## 核心摘录

- **一句话：** 仅头部深度 + 本体历史 + 低维指令，端到端输出 **25-DoF** PD 目标做人形足球射门/带球，无需运行时检测/规划模块。
- **机制：** LightDepthEncoder + HIM 历史；可见性门控辅助几何；GT→prediction annealing + curriculum + AMP 先验；非对称 critic。
- **指标：** 仿真 + Noetix E1 真机 + 消融（文内；射门/带球分任务定义）。

## 对 wiki 的映射

- 实体页：[DAVIS](../../wiki/entities/paper-davis-humanoid-soccer.md)
