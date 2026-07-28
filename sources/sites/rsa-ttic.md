# ttic.uchicago.edu/~cbschaff/rsa（Residual Shared Autonomy 项目页）

- **标题：** Residual Policy Learning for Shared Autonomy
- **类型：** site / project-page
- **URL：** <https://ttic.uchicago.edu/~cbschaff/rsa/>
- **配套论文：** [arXiv:2004.05097](https://arxiv.org/abs/2004.05097)，ICRA 2020
- **代码：** <https://github.com/cbschaff/rsa> — 归档见 [`sources/repos/rsa-shared-autonomy.md`](../repos/rsa-shared-autonomy.md)
- **入库日期：** 2026-07-28

## 一句话摘要

Charles Schaff、Matthew R. Walter（TTIC）的 Residual Shared Autonomy 官方页面：把**人当作 base policy**，智能体学习最小干预的加性修正 $a=a_h+a_r$，仅在满足「不坠毁/不出界」等目标无关约束时介入；Lunar Lander / Lunar Reacher / Drone Reacher 三环境验证；页面提供视频与代码。

## 公开信息要点（截至入库日）

- **机构：** Toyota Technological Institute at Chicago（TTIC）。
- **页面内容：** arXiv 链接、GitHub 代码链接、演示视频；作者主页互链。
- **核心设定：** model-free、连续动作空间；不假设已知目标空间/环境动力学/人的策略；残差幅值正则 ⇔ 人的控制权最大化。
- **人测实验：** 16 名参与者（Lunar Lander + Lunar Reacher），copilot 以 BC 模仿代理训练；未对个体微调。

## 为何值得保留

RSA 证明 Residual Policy 的 base **可以是人**：这为遥操作辅助、共享自治与人机共驾提供了与「控制器打底」同构的学习框架，是理解 Residual 家族边界的关键一篇。

## 对 wiki 的映射

- 实体页：[paper-residual-policy-shared-autonomy](../../wiki/entities/paper-residual-policy-shared-autonomy.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
