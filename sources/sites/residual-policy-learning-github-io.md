# k-r-allen.github.io/residual-policy-learning（RPL 项目页）

- **标题：** Residual Policy Learning
- **类型：** site / project-page
- **URL：** <https://k-r-allen.github.io/residual-policy-learning/>
- **配套论文：** [Residual Policy Learning（arXiv:1812.06298）](https://arxiv.org/abs/1812.06298)
- **代码：** <https://github.com/k-r-allen/residual-policy-learning> — 归档见 [`sources/repos/residual-policy-learning.md`](../repos/residual-policy-learning.md)
- **入库日期：** 2026-07-28

## 一句话摘要

Silver, Allen, Tenenbaum, Kaelbling（MIT CSAIL）的 RPL 官方项目页：用 model-free 深度 RL 改进**不可微**的已有策略（人工控制器或 MPC），在 6 个 MuJoCo 操作任务（部分可观测、传感器噪声、模型失配、控制器失准）上系统验证；页面提供 arXiv、代码与视频入口。

## 公开信息要点（截至入库日）

- **机构：** MIT CSAIL（Tom Silver、Kelsey Allen 共同一作）。
- **页面板块：** 摘要、arXiv 链接、Code 链接（指向 GitHub）、演示视频。
- **核心主张：** 好但不完美的控制器 + 残差学习 ≫ 单独任一方；长视野稀疏奖励任务上纯 RL 失败而 RPL 成功。
- **代码开放度：** **已开源**（环境 + 训练脚本；mujoco-py 150 / TF1 时代技术栈）。

## 为何值得保留

RPL 是 Residual Policy 家族的正式命名文献；项目页三角互证（论文/代码/视频）齐全，是复现与二次开发的入口。

## 对 wiki 的映射

- 实体页：[paper-residual-policy-learning](../../wiki/entities/paper-residual-policy-learning.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
