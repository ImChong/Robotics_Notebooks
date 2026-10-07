# RoboReward 基准与资源页

> 来源归档（site）

- **标题：** RoboReward: General-Purpose Vision-Language Reward Models for Robotics
- **项目页：** <https://crfm.stanford.edu/helm/robo-reward-bench/>
- **论文：** <https://arxiv.org/abs/2601.00675>
- **数据集：** <https://huggingface.co/datasets/teetone/RoboReward>
- **8B 模型：** <https://huggingface.co/teetone/RoboReward-8B>
- **作者 / 机构：** Tony Lee 等；Stanford University、UC Berkeley
- **入库日期：** 2026-10-07
- **一句话说明：** 为真实机器人轨迹奖励建模提供数据、评测基准和可用权重。

## 开放状态核查

- **数据与权重：** 已公开链接；HF 模型卡显示用 Qwen3-VL 对视频轨迹输出 1–5 级终局进度分。
- **训练代码：** 项目页/arXiv 指向数据、模型与 evaluation suite，但本次未发现官方训练代码入口；不要把开放权重等同于完整训练管线已开源。
- **项目页用途：** HELM RoboReward Bench 是论文资源和评估信息入口。

## 资源摘要

论文把原始成功轨迹转成校准失败、近失误与中途进展样本，供 reward model 训练和评测；8B HF 模型卡给出了提示词格式和离散分数定义。
