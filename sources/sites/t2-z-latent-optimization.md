# T2-z 官方项目页

> 来源归档（ingest）

- **标题：** T2-z: Test-Time Optimization of Environment Latents for Robotic World Models
- **类型：** official project page
- **项目页：** <https://t2-z-latent-optimization.github.io/>
- **论文：** <https://arxiv.org/abs/2610.22364>
- **代码：** <https://github.com/T2-z-latent-optimization/T2-z>
- **数据：** <https://huggingface.co/datasets/huagailuowen/T2-z-datasets>
- **入库日期：** 2026-10-17
- **一句话说明：** 展示测试时优化环境潜变量，让动作条件世界模型适应未知物理属性与交互规则的结果和方法。

## 开源核查（2026-10-17）

项目页直接链接公开代码仓库与 Hugging Face 数据集。仓库 README 提供训练和推理入口，但基座视频模型、初始化 checkpoint 与环境依赖需另外下载/配置。

## 发布状态复核（2026-11-26）

项目页的代码、数据与论文链接均核对。作者 [Hugging Face 文件树](https://huggingface.co/datasets/huagailuowen/T2-z-datasets/tree/854d946) 列出 7 个仿真、4 个真机任务目录，以及约 2.56 GB 的 ZIP 和清单；未下载整包或证明所有样本可读。[发布清单](https://huggingface.co/datasets/huagailuowen/T2-z-datasets/raw/854d946/manifest.json) 的 Light Switch 记录 8 个物理环境、80 条 episode；论文表 5 的训练预算为 12×10，故不能把公开数据直接当作论文全量训练集。数据卡的一键预览出现列不匹配（部分分片有 `native_frame_indices`），与文件树/归档存在并不矛盾。未在该数据卡确认许可证，不从代码仓 Apache-2.0 推定数据授权。

## 对 wiki 的映射

- [T2-z 论文来源](../papers/t2_z_arxiv_2610_22364.md)
- [官方仓库归档](../repos/t2-z-latent-optimization.md)
- [T2-z 实体页](../../wiki/entities/paper-t2-z-environment-latents.md)
