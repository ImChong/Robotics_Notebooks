# SIM — Self-Improvement Memory 官方项目页

> 官方一手项目页核对归档（2026-11-21）。

- **官方 URL：** https://lucasxinyu.github.io/sim/
- **论文：** [Self-Improvement Memory Enables Continual Recursive Self-Improvement（arXiv:2611.18856v1）](../papers/self_improvement_memory_arxiv_2611_18856.md)
- **论文实体：** [SIM / CRSI](../../wiki/entities/paper-self-improvement-memory-crsi.md)
- **开放状态：** 项目页给论文 PDF、研究说明和可交互示意；页面明确写明「No public implementation is linked here」，没有可运行 SIM 仓库入口。
- **数据开放边界：** 项目页列出与原论文 Table 3 对齐的 10 对 coding 连续优化结果，示意的 SIM 记录是解释性可视化，**不是导出的真实运行 JSON**。

## 项目说明与核对

- 项目页明确区分「evolved harness」与「why we changed it / expected / observed」两种可继承资产； policy weights frozen，memory extensible。
- 作者将记录过程分为 READ（恢复上下文）→ PROPOSE（提出可引用根因的修改）→ SELECT（用目标任务反馈筛选）→ APPEND（保留被选编辑与结果）。
- 十组 Terminus-2 源运行在 TB2.1 上完成演化后，分别启动有/无 SIM 的 TB4 十轮配对延续；9/10 组受益，均值 +2.36；置信区间 [1.05, 3.68]，p=0.003。
- 网页明确指出只验证兼容接口、完整 SIM 读取有成本、1 组反例、跨模型代际 Gemini 4 Argon 下 CRSI 低于 Fresh RSI 0.8 分；不是通用或无限自改进证据。

## 参考来源

- [SIM 官方项目页](https://lucasxinyu.github.io/sim/)
- [arXiv v1 原文](https://arxiv.org/abs/2611.18856v1)
