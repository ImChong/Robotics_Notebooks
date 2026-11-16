# NSDL: Neuro-Symbolic Division of Labor for Multi-Robot Task Planning｜NSDL：多机器人任务规划的神经符号分工

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.16671 （v1）
- **HTML：** https://arxiv.org/html/2611.16671v1
- **PDF：** https://arxiv.org/pdf/2611.16671
- **代码：** https://github.com/Xincheng-Tan/NSDL
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-nsdl-multirobot-2611-16671.md)

## 方法与评测核对

- **研究问题：** VLM 负责现场语义判断和计划草稿，确定性求解器负责补全协作依赖与可执行调度。
- **方法原理：** 图像与任务文字进入 VLM，输出 GOAL/ASSIGN/SKETCH；detect 工具查询视觉证据，schedule 调用 Φ 求解器插入依赖与前置关系，validator 拦截不可用联合计划；在符号回放基准上训练并评测工具回合。
- **原文结果：** 评测 VIKI-L2 与 LOCUS-Bench 的 ID/OOD 及空行动计划，重点是符号回放有效性与结构错误，不等于真机多机器人成功率。
- **局限与范围：** 依赖任务描述语言、工具输出语义和符号评价器；实际机器人感知/控制/通信约束未在该符号回放中完整验证。
