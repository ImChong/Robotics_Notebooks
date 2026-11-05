# 人形移动操作前瞻、VLA 梯度路由与 GSBR（待完整核验）

> 状态：待完成。已核对各论文 arXiv 摘要和全文入口；尚未完成逐篇全文、附录、项目页及源码核验。以下仅为待处理资料入口和摘要线索，不代表已完成 ingest；须通读原文后建立或完善实体节点及 Mermaid 流程图，若有可运行官方源码须核对入口并补源码运行时序图。

## Coordinating Humanoid Loco-Manipulation through Action-Conditioned Latent Foresight

- **类型：** paper
- **版本：** [arXiv:2611.04839v1 全文](https://arxiv.org/html/2611.04839v1)
- **中文：** 用动作条件潜空间前瞻协调人形全身操作
- **摘要线索：** 根据视觉、语言与机器人状态生成运动和手部动作候选，利用动作条件终点预测和联合时间先验对候选序列评分，执行首段后重新规划。
- **待核验：** 全文实验条件、局限、项目页和代码开放状态。

## Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs

- **类型：** paper
- **版本：** [arXiv:2611.05249v1 全文](https://arxiv.org/html/2611.05249v1)
- **中文：** 动作专家该不该训练 VLM？混合 Transformer VLA 中的动作 token 监督与梯度路由
- **摘要线索：** 研究动作专家损失到 VLM 的梯度耦合，以及 VLM 的离散动作 token 监督。
- **待核验：** 不同训练设置与评测范围、全文实验和局限、项目页和代码开放状态。

## GSBR: Geometric Support Behavior Regularization for Online RL

- **类型：** paper
- **版本：** [arXiv:2611.05295v1 全文](https://arxiv.org/html/2611.05295v1)
- **中文：** 几何支撑行为正则的在线强化学习
- **摘要线索：** 依据动作与基础策略支撑集的距离进行正则，研究如何在保留先验的同时允许策略改善。
- **待核验：** 支撑集近似方式、任务与吞吐量的评测条件、局限、项目页和代码开放状态。

## 去重与后续

- RoboColosseum、H2RB 和 RAFT 已有实体节点，后续应在原节点上基于全文补充，不再新建重复节点。
- Liber-0 另有草稿 PR [#3366](https://github.com/ImChong/Robotics_Notebooks/pull/3366)，应协调处理避免重复。
- 当前环境未完成仓库检出，尚未运行 `make ci-preflight`。
