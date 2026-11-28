# WACO-pi05-LIBERO：相机轨权重 / LIBERO-Plus Camera-Track Checkpoint

- **类型：** model
- **模型卡：** [Bluexxx/WACO-pi05-LIBERO](https://huggingface.co/Bluexxx/WACO-pi05-LIBERO)
- **论文：** [arXiv:2611.24360v1](https://arxiv.org/abs/2611.24360v1)
- **初始权重：** `lerobot/pi05_libero_finetuned`
- **许可：** Gemma 使用条款（继承上游 π0.5），与 Apache-2.0 代码许可分别看
- **核对日期：** 2026-11-28

公开的是 LeRobot 格式的 LIBERO-Plus 相机轨 π0.5 微调权重，含权重、模型配置、归一化处理器与训练配置。第三人称和腕部图像为 256 × 256，状态为末端位姿与夹爪开度（8 维），输出 7 维末端增量，配置每个动作块执行 10 步。该权重不是论文真机权重的公开入口，论文 93.7% / 92.4% 的发布提示成绩也需要配套 WACO 运行时、标定包与相机轨协议，不是裸权重的成绩。

- **对 wiki 的映射：** [WACO](../../wiki/entities/paper-waco-pose-robust-vla.md)
