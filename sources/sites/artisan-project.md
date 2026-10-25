# Artisan 项目主页（Physical Vision Group）

> 来源归档

- **标题：** Towards Physically Grounded Articulated 3D Assets: A Multi-Agent Pipeline and Benchmark
- **类型：** site（项目页 / 交互式结果 / 论文与数据入口）
- **链接：** https://physicalvision.github.io/artisan/
- **论文：** [arXiv:2610.13947](https://arxiv.org/abs/2610.13947)
- **代码：** https://github.com/physicalvision/artisan
- **数据集：** https://huggingface.co/datasets/mastercai/Artisan-Benchmark
- **入库日期：** 2026-10-25
- **一句话说明：** 作者团队的项目主页，展示 Artisan 生成的办公室、实验室和厨房等可动关节资产，并提供论文与代码/数据入口。
- **开源核查：** 项目页链接到公开 GitHub 实现；仓库 README 提供 CLI 生成与 benchmark 评估说明。数据集另有 Hugging Face 页面。

## 页面信息摘录

- **目标：** 从文本提示和可选参考图生成带部件、关节与物理属性的 3D 资产。
- **展示：** 可选择生成物体查看运动和物理属性；项目页展示了多种场景中的组合资产。
- **评测说明：** 页面区分空间与物理评测、下游机器人任务，以及人类偏好结果；各项数字应结合论文的任务设置和表格阅读。

## 对 wiki 的映射

- **实体页：** [Artisan](../../wiki/entities/artisan.md)
- **论文：** [arXiv:2610.13947](../papers/artisan_arxiv_2610_13947.md)
- **官方仓库：** [physicalvision/artisan](../repos/physicalvision-artisan.md)

## 备注

- 主页是结果展示与导航入口；方法、指标定义与实验条件以论文为准，运行命令及依赖以仓库 README 为准。
