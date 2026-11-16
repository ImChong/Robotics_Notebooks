# τ₀-VLA 原文核对 / Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation

- **类型：** paper
- **论文：** https://arxiv.org/html/2608.16885v1
- **官方项目：** https://tau0-vla.github.io/
- **代码：** https://github.com/sii-research/tau-0-vla
- **团队：** 上海创智学院、智元 Finch 团队及香港中文大学
- **核对范围：** arXiv HTML I–VII、附录 C（高层策略/世界模型/beam、40D 低层动作、40,115h 数据、评测及路由/异步服务）；对照本仓库已有官方项目和代码归档。
- **通读：** 已完成

## 方法、结果与边界

- 高层用执行记忆选择下一子任务，可在必要时调用 world-model 想象目标终态、value score 和 beam search；低层以统一 40D flow-based VLA 生成具体动作 chunk。
- 低层训练语料为 **40,115 小时**异构真实机器人数据（固定底座、双臂和移动操作），不是所有高层模块单独训练数据规模。
- 高层增强的收益主要体现在长程任务子任务选择/恢复，不能与单次动作推理延迟混淆。论文结果见原文具体评价表，站内保留环境、子任务与闭环评测限定。
- 权重与代码按官方 README **部分开放**（低层与部署示例可用，高层 TTC 的不同模块开放日程需跟踪），不应声称高低层整套 checkpoint 一次 `pip install` 即复现。

## 对 wiki 的映射

- [τ₀-VLA 唯一详情](../../wiki/entities/paper-tau0-vla.md)
