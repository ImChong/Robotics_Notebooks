# PrismWAM 项目页

> 来源归档（项目页、论文、代码与模型入口）

- **标题：** PrismWAM: Decoupling State Representations in World Action Models
- **项目页：** https://prism-wam.github.io/
- **论文：** https://arxiv.org/abs/2610.26564
- **代码：** https://github.com/THUSI-Lab/PrismWAM
- **模型：** https://huggingface.co/THUSI-Lab/PrismWAM
- **作者：** Tianyu Xin 等
- **核查日期：** 2026-10-21
- **一句话说明：** 将外观语义目标与深度物理演化分开表示，并根据预测的深度 rollout 生成机器人动作。

## 项目与开放状态
- 项目页为官方展示入口；arXiv 摘要将方法概括为 Semantic Planning → Physical Rollout → Executable Action。
- 官方仓库 README 公布代码、训练脚本、benchmark 适配与 HTTP 部署入口；README 声明代码为 MIT，另含 Apache-2.0 第三方组件。
- README 提供 RoboTwin 2.0、RoboDojo、LIBERO 的预训练策略入口，并报告对应模型权重已发布到 Hugging Face。权重、数据集和仿真器保留各自的许可证。
- 复现条件包括 Linux、NVIDIA GPU、CUDA 12.8；训练需准备相应 benchmark 数据与初始化权重。代码开放不代表数据、仿真环境均随仓库分发。
- 项目页本身在当前资料核查中未能读取；开源结论以官方 GitHub README 的代码、训练与模型发布说明为依据。

## 关联归档
- [论文来源](../papers/prismwam_arxiv_2610_26564.md)
- [官方代码与运行入口](../repos/prismwam.md)
- [wiki 实体](../../wiki/entities/paper-prismwam.md)
