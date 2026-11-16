# A Unified Inertial Encoder for Human and Humanoid Motion — 官方项目页

> 来源归档（site；核查日期 2026-11-16）

- **中文标题：** 面向人类与人形机器人运动的统一惯性编码器
- **项目页：** <https://unified-inertial-encoder.github.io/>
- **论文：** [arXiv:2611.10137v1](https://arxiv.org/html/2611.10137v1)，2026-11-06；[论文归档](../papers/unified-inertial-encoder_arxiv_2611_10137.md)
- **代码：** <https://github.com/amazon-far/Unified-Inertial-Encoder>；[源码归档](../repos/unified-inertial-encoder.md)
- **预训练权重：** <https://huggingface.co/Open-IO/Unified-Inertial-Encoder>
- **真机惯性数据：** <https://huggingface.co/datasets/Open-IO/humanoid-g1-odometry>
- **作者与机构：** Yutian Chen 等八位作者；Amazon FAR、UC Berkeley、Stanford University、Carnegie Mellon University
- **对 wiki 的映射：** [统一惯性编码器论文节点](../../wiki/entities/paper-unified-inertial-encoder.md)

## 项目页直接展示的能力

- 人体穿戴/合成 IMU 与 G1 骨盆 IMU + 关节编码器，通过 12 个人体对应部位的惯性信号作为共享输入接口；不输入全局根位姿或速度。
- 连续时间 token 按 125 ms 的**物理时长**切片，避免不同频率/缺样的定采样数 token 歧义。
- 真机 G1 里程计与动作语义检索定性演示；项目页列出五数据集速度误差与轨迹指标，真实 G1 AVE=0.099 m/s、ATE_GT=3.153 m、ATE_Pred=11.50 m。
- TensorRT + NVIDIA L40S 的公开测时为 4 s、12 IMU、200 Hz 窗口 **1.78 ms**；不是已证实的 G1 机载部署时延。

## 开放核查与边界

**已开源：** 官方 Code 链接指向可阅读的训练、预处理、评测代码，Apache-2.0；Hugging Face 有可下载 EMA 模型权重（无优化器状态）及 G1 测试数据。完整多数据集复现仍需各数据集原始文件/许可、硬件与依赖。

**评测边界：** 论文声称零样本真机迁移；官方 [模型卡](https://huggingface.co/Open-IO/Unified-Inertial-Encoder) 说明真机 G1 数据没有参与梯度训练，**但分数参与了已发布 checkpoint 的选择**，并非完全独立盲测模型选择。未验证权重在机载 GPU 上的实时指标，亦未执行本地训练或推理。

## 一手来源

- [项目页](https://unified-inertial-encoder.github.io/)
- [论文全文](https://arxiv.org/html/2611.10137v1)
- [官方 GitHub README](https://github.com/amazon-far/Unified-Inertial-Encoder)
- [Hugging Face 权重说明](https://huggingface.co/Open-IO/Unified-Inertial-Encoder)
