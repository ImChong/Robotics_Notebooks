# RoboAug 官方代码仓库

- **标题：** RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation
- **类型：** repo
- **链接：** https://github.com/Open-X-Humanoid/RoboAug
- **论文：** [arXiv:2602.14032](https://arxiv.org/abs/2602.14032)
- **项目页：** https://x-roboaug.github.io/
- **数据集：** https://huggingface.co/datasets/X-Humanoid/RoboAug-Datasets
- **许可证：** Apache-2.0（代码仓库与数据集页面）
- **开源状态：** 部分开源；Stage 1 区域提取与 Stage 2 语义数据增广已发布，Stage 3 区域对比策略学习尚未开源
- **核查日期：** 2026-10-23
- **一句话说明：** 提供将 HDF5 示教轨迹与单帧 bbox 标注转为增广 HDF5 的数据生成流程，以及 RoboAug-D 标注集和多机器人示教数据的下载说明。

## 仓库入口

- `scripts/demo.sh`：单条轨迹快速运行入口
- `scripts/batch_run.sh`：批量处理入口
- `src/sam2_region_augment.py`：区域提取、mask 传播与语义场景增广主流程
- `src/read_h5.py`：HDF5 轨迹读写
- `dataset/`：数据集格式、下载和加载说明

## 运行链路与依赖

数据流程为：读取多相机 HDF5 RGB/深度轨迹 → 加载该任务单帧 LabelMe bbox → GroundingDINO 提议候选区并以 DINOv2 做 one-shot 匹配 → SAM 2 传播稠密 mask → 预训练文生图模型合成完整背景并保留任务前景 → 将增广图像与 mask 写回 HDF5。

GroundingDINO 与 SAM 2 需要从上游仓库单独安装；模型权重不随仓库分发。README 推荐 Python 3.10 + CUDA。论文中的区域对比策略学习与训练损失尚未包含在公开实现中。

## 数据集

Hugging Face 数据集页链接到 RoboAug-D，包含标注图像，以及 UR-5e、AgileX Cobot Magic 2.0、Tien Kung 2.0 真机数据；代码 README 指向其下载与 HDF5 格式文档。

## 对 wiki 的映射

- [RoboAug 论文实体](../../wiki/entities/paper-roboaug.md) — 统一收录论文、项目页和代码；明确可复现边界
- [论文原始资料](../papers/roboaug_arxiv_2602_14032.md)
- [项目页归档](../sites/x-roboaug-project.md)
