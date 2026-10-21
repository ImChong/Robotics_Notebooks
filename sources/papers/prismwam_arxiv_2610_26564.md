# PrismWAM 论文来源归档

## 元数据
- **题目：** PrismWAM: Decoupling State Representations in World Action Models
- **作者：** Tianyu Xin、Wanqi Zhong、Bowen Hu、Zhangchi Guo、Lijie Wang、Yiming Wang、Haoran Liao、Heyang Yu、Xiaoyu Tian、Zhiding Yu、Yiming Li
- **预印本：** https://arxiv.org/abs/2610.26564
- **项目页：** https://prism-wam.github.io/
- **代码：** https://github.com/THUSI-Lab/PrismWAM
- **模型：** https://huggingface.co/THUSI-Lab/PrismWAM
- **提交日期：** 2026-10-19
- **入库日期：** 2026-10-21
- **一句话说明：** PrismWAM 将语义目标规划与深度物理状态预测解耦，再以深度 rollout 为动作生成提供条件，研究 WAM 应预测怎样的世界状态。

## 论文要点
论文指出，预训练视频模型的潜表示会混合纹理、颜色等外观信息与占据、姿态等物理状态。PrismWAM 将具身任务拆为“语义规划 → 物理 rollout → 可执行动作”：由初始 RGB 观测学习目标图像，以深度序列预测细粒度物理演化，并从深度 rollout 生成动作块。

论文报告该方法不使用具身动作预训练或额外数据增强，在 RoboTwin 2.0、RoboDojo 与 LIBERO 系列仿真基准取得较强操作结果，并包含真实机器人实验。

## 对 wiki 的映射
- [PrismWAM 实体页](../../wiki/entities/paper-prismwam.md)：表征解耦、两阶段训练、评测读法与复现边界。
- [世界–动作模型](../../wiki/concepts/world-action-models.md)：作为把语义目标和物理未来分开建模的 WAM 实例。

## 开源状态
官方 GitHub 仓库公开训练配方、代码与评测 / 部署入口；README 链接到 RoboTwin 2.0、RoboDojo、LIBERO 模型权重。代码仓库采用 MIT License，第三方组件及模型权重、数据集、仿真器各自适用其许可证。详见[项目页归档](../sites/prism-wam-project.md)与[官方代码归档](../repos/prismwam.md)。
