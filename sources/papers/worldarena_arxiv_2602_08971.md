# WorldArena：具身世界模型的感知与功能效用 / A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models

- **作者：** Yu Shang、Zhuohang Li、Yiding Ma、Weikang Su 等。
- **原文：** [arXiv:2602.08971v2](https://arxiv.org/abs/2602.08971v2) · [PDF](https://arxiv.org/pdf/2602.08971v2)
- **项目：** [WorldArena 官网](https://world-arena.ai/) · [官方仓库](https://github.com/tsinghua-fib-lab/WorldArena)
- **核对范围：** v2 全文及补充材料，§3–5、图 4–5、EWMScore 定义、功能任务和分项口径；核对官方 Track 1 入口 `video_quality/evaluate.py`。
- **通读：** 已完成
- **入库日期：** 2026-11-28

1. 1.0 比较 14 个代表性模型，视频侧把 6 类 16 项指标经验归一到 0–100 再取算术平均，EWMScore 不包含下游功能表现。
2. 功能侧测三个不同角色：生成带动作数据增强策略学习、用动作条件模型滚动预测作为评测环境、从预测观察恢复动作并在仿真中执行。人工视频判断是独立参照。
3. 图 5 的 Pearson r 对人工判断为 0.825，对数据引擎为 0.600，对动作规划为 0.360；图 4 的两个策略评测代理都高估所用仿真器成功分数，相关与绝对校准应分别报告。

- **对 wiki 的映射：** [WorldArena](../../wiki/entities/worldarena.md)
