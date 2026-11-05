# Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles

- **类型：** paper
- **机构：** 英伟达（NVIDIA）
- **原文：** <https://arxiv.org/html/2610.30340v1>（v1，2026-10-21）
- **项目页：** <https://research.nvidia.com/labs/sil/projects/foundation4d/>
- **数据集：** <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D>
- **入库日期：** 2026-11-03
- **核对范围：** 已核对摘要、方法与实验部分片段及项目页；全文尚未通读，需继续完成方法、实验、局限与附录的逐节核查。

## 主要内容

- 共享多相机时空表示，分别预测几何、3D 检测、语义占据与 Gaussian 重建；视频级处理拼接窗口并跟踪目标。
- PAI–Foundation4D 包含 1,253 段驾驶片段的稠密深度及语义占据标注；占据体素包括速度。
- 不同基准的结果采用各自的微调与评测方案；nuScenes 检测使用含未来信息的离线窗口，不能直接当作在线性能。

## 对 wiki 的映射

- [Foundation4D](../../wiki/entities/paper-foundation4d.md)
