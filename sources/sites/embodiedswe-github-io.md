# EmbodiedSWE — 项目页

- **标题：** EmbodiedSWE: Coding Agents for Long-Horizon Dexterous Robotics
- **类型：** site
- **链接：** <https://embodiedswe.github.io/>
- **入库日期：** 2026-10-03
- **一句话说明：** 研究 coding agents 如何在物理仿真中迭代编写长时程灵巧操作程序，并将通过验证的解扩展为 VLA 示范数据。

## 项目页核查（2026-10-03）

项目分成四部分：EmbodiedSWE-Bench 基准、coding agent 求解评测、EmbodiedSWE-Gen 示范生成，以及基于已验证结果构造任务并改进 agent。页面指向 arXiv 与官方 GitHub，没有单列模型权重下载入口。

- **代码：** [EmbodiedSWE GitHub 仓库](https://github.com/EmbodiedSWE/EmbodiedSWE)（公开；仓库含 Apache-2.0 许可证）
- **数据 / 仿真资产：** [Hugging Face 数据集](https://huggingface.co/datasets/EmbodiedSWE/robobench-assets)（公开，数据集卡标注 Apache-2.0）
- **论文：** [arXiv:2609.27308](https://arxiv.org/abs/2609.27308)

## 项目页摘要

页面列出 6 个任务套件、28 个场景、17 种具身配置、5 类控制器，任务时长可达约 30 分钟；官方论文对评测子集的表述是 5 种机器人 embodiment。代码仍在积极开发，目录、接口和配置可能变化。

数据生成将一个经验证的 agent 解扩展为多样轨迹，用于 VLA 微调。项目页报告 SmolVLA 在 10 到 400 条示范时平均成功率从 14% 上升至 66%；在相同数据量下，agent 辅助的多样化也改善了 held-out 变体表现。

## 许可证边界

数据集卡称作者制作与烘焙的 USD 场景、碰撞网格、组合资产、纹理降采样、演示视频和 manifest 按 Apache-2.0 发布；第三方来源资产仍受其上游许可约束，部分扫描模型为 CC BY-NC / CC BY-NC-SA，只能按相应限制使用。商业使用前应逐项查阅数据集卡的资产来源与许可表，不能把数据集整体简单理解为全部 Apache-2.0。

## 对 wiki 的映射

- 论文实体页：[EmbodiedSWE](../../wiki/entities/paper-embodiedswe.md)
- 代码归档：[EmbodiedSWE 仓库](../repos/embodiedswe.md)
- 论文归档：[EmbodiedSWE arXiv](../papers/embodiedswe_arxiv_2609_27308.md)

## 参考链接

- 项目页：<https://embodiedswe.github.io/>
- 论文：<https://arxiv.org/abs/2609.27308>
- GitHub：<https://github.com/EmbodiedSWE/EmbodiedSWE>
- 仿真资产：<https://huggingface.co/datasets/EmbodiedSWE/robobench-assets>
