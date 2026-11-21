# PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation

> 一手论文归档（paper）

- **中文标题：** PSR：面向接触丰富操作的预测式感觉运动表征学习
- **类型：** paper / predictive-sensorimotor / contact-rich / force-aware-vla
- **arXiv：** https://arxiv.org/abs/2609.21753
- **HTML：** https://arxiv.org/html/2609.21753v1
- **PDF：** https://arxiv.org/pdf/2609.21753
- **项目页：** https://psr-vla.pages.dev/
- **版本：** arXiv v1（2026-09）
- **作者：** Shengbao Li 等（以 arXiv v1 署名为准）
- **核对范围：** arXiv v1 HTML 的预测式预训练、6 层多模态 Transformer、跨层门控注入、实验与消融、接触失败案例、效率和方法局限/附录；与历史项目页归档的代码状态对照。
- **通读：** 已完成
- **入库日期：** 2026-09-21（本次复核 2026-11-21）

## 方法原文摘要

1. 输入当前及历史三类传感信息：RGB/视觉语义、腕部六轴 F/T、关节 proprioception 与 effort；接触和摩擦先验不是从单帧 RGB 直接还原。
2. Stage 1 用六层 predictive sensorimotor Transformer，在预测未来 F/T、关节 effort、终端关节位移等目标下预训练；中间层的 MoE/瓶颈交互提供分层时空表征。
3. Stage 2 将多层 PSR latent 经门控 cross-attention 注入 π0.5 动作专家；训练改为 flow matching 动作目标，预测解码器不再用于部署推理，区分辅助训练与推理路径。
4. 6 类真实机器人接触操作（插入、擦拭、装配等）共 120 次试验：论文报告 PSR 110/120=91.7%，π0.5 74/120=61.7%；ForceVLA 69.2%，适配版 ForceVLA2 72.5%。这些分母与任务组合限定结论。
5. 作者报告约 1210 条示范与 RTX 5080 推理每次额外约 10ms；额外时延不能直接转换为高频力控闭环稳定性保证。

## 限制与开源现状

- 相机/腕力/关节努力历史的传感同步、标定与模型适配成本不能从 headline 成功率推断。
- 受 6 类真实任务及任务数据规模约束，不等于已具备任意接触动力学 OOD 泛化能力。
- 历史 [项目页归档](../sites/psr-vla.md) 截至 2026-09-21 标“待发布”；本次未取得官方可运行代码入口，不编写虚假源码时序图。

## 对 wiki 的映射

- [PSR 唯一论文节点](../../wiki/entities/paper-psr-vla.md)
- [接触丰富操作路线](../../roadmap/depth-contact-manipulation.md)
