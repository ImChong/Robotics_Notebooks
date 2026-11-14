# OnlyAct: Extracting Action Representations from Foundation Models for World Models｜从视频基础模型提取动作表征

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14420
- **PDF：** https://arxiv.org/pdf/2611.14420
- **核对范围：** arXiv HTML v1 正文 §1–5、附录 A–D（训练、数据、局限）及 NeurIPS 复现清单；核对官方 README 的表 1–4 脚本；未运行 GPU 训练。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/runjiali-rl/onlyact

## 核心摘录

1. 从无标签视频帧对读取预训练 V-JEPA 2 视频表征。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)
2. 相同运动跨场景的帧对拉近、同场景不同运动的帧对推远，不依赖动作标签。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)
3. 按动作可表达性、场景泄露与动作驱动视频生成评测，不直接等同真机成功率。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-onlyact.md)

## 作者报告的结果与限制

官方项目披露训练 9 个数据集共 168,000 帧对，LoRA 7.1M 可训练参数；13 个数据集的线性探针平均 R²=0.490。

场景解耦依赖样本构造；视频动作转移与动力学控制存在鸿沟。

## 原文核对摘录（2026-11-14）

论文 §3：预训练 V-JEPA 2 ViT-L 上只适配 LoRA，以光流指导的同动作跨场景正例与同场景异动作难负例做对称对比学习，产生 640 维潜动作。9 个数据集 168000 组无动作标签帧对，训练参数约 7.1M，作者报告单 RTX4090 约 1.5 小时。§4 动作 probe 13 个数据集平均 R²=0.490，冻结 V-JEPA2 基线约 0.347；LIBERO 视频迁移动作方向相似度约 0.917，RLBench 约 0.686。各域 PSNR 不一定全部最好；这些是表征和视频世界模型指标，不能称作真机动作控制成功率。附录讨论数据许可、模型基座依赖及跨域外推风险。

## 复现边界

仍未独立运行训练、评测与机器人实验，论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-onlyact.md)。
