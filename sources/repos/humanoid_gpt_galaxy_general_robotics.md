# Humanoid-GPT（GalaxyGeneralRobotics 官方仓库）

- **标题：** Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking
- **类型：** repo
- **仓库：** <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>
- **论文：** <https://arxiv.org/abs/2606.03985>
- **项目页：** <https://qizekun.github.io/Humanoid-GPT/>
- **收录日期：** 2026-06-04
- **维护方：** Galaxy General Robotics（Galbot Inc. 关联开源组织，以仓库 README 为准）

## 一句话摘要

CVPR 2026 论文 **Humanoid-GPT** 的官方代码入口：实现 **大规模 motion 语料策展、HME 聚类、PPO motion expert 与 Transformer DAgger 蒸馏** 的人形零样本 tracking 管线（具体模块与训练脚本以仓库更新为准）。

## 为何值得保留

- **可复现锚点**：与 arXiv / 项目页并列，便于后续 ingest 工程细节（仿真环境、checkpoint、部署脚本）。
- **生态位**：与 [SONIC](../../wiki/methods/sonic-motion-tracking.md)（NVIDIA，100M MLP）形成 **Transformer + 2B 帧** 对照的开源跟踪基线候选。
- **硬件叙事**：论文与项目页以 **Unitree G1（29-DoF）** 为主平台；仓库预期包含 sim2real 与 TensorRT 部署相关资产（以实际 release 为准）。

## 技术要点（公开材料对齐，细节以代码为准）

1. **三阶段管线**：(a) 多数据集聚合 + G1 retarget + 过滤/增广；(b) HME 聚类 + 簇内 PPO expert；(c) 因果 Transformer + DAgger 蒸馏为单策略。
2. **推理**：项目页强调 **ONNX / TensorRT** 低延迟部署；真机需配合在线 retarget（MoCap→G1）。
3. **评测协议**：仿真 AMASS-test；真机 **训练外** 舞蹈与居家场景零样本片段。

## 对 Wiki 的映射

- [Humanoid-GPT（论文实体页）](../../wiki/entities/paper-humanoid-gpt.md)
- [SONIC（规模化运动跟踪）](../../wiki/methods/sonic-motion-tracking.md) — 同任务不同规模/结构路线的对照阅读

## 参考来源（原始）

- GitHub 仓库首页 — <https://github.com/GalaxyGeneralRobotics/Humanoid-GPT>（2026-06-04 检索：README 标题与 CVPR 2026 标注）
