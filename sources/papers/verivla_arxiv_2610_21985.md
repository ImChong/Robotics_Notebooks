# VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models

> 来源归档（paper）

- **标题：** VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models
- **中文标题：** VeriVLA：利用潜在世界模型对 Flow-Based VLA 进行自适应测试时验证
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.21985>
- **入库日期：** 2026-10-21
- **线索来源：** [公众号综述（33 篇 Manipulation）](../raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（具身运控lab，2026-10-20）
- **一句话说明：** 免训练的两轴触发器识别关键状态，再用 JEPA 式潜在世界模型对候选动作块排序。
- **沉淀到 wiki：** [`wiki/entities/paper-verivla-test-time-verification.md`](../../wiki/entities/paper-verivla-test-time-verification.md)

## 摘要要点（arXiv 摘要页）

- Flow-based VLA 的随机采样会得到质量不一的动作块，测试时通常只取一个样本。
- 用两个初始样本估计块内时间变化与跨样本方差，触发后用潜在世界模型预测动作条件下的未来状态与任务进度来排序。

## 开源状态

- **未核查代码**：arXiv 摘要页未列代码链接。
