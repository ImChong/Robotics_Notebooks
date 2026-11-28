# CLARE：VLA 持续学习的自主适配器路由与扩展 / Continual Learning via Autonomous Adapter Routing and Expansion

- **类型：** paper
- **作者：** Ralf Römer、Yi Zhang（共同一作）、Yuming Li、Angela P. Schoellig
- **机构：** 慕尼黑工业大学（TUM，Learning Systems and Robotics Lab / MIRMI）、Robotics Institute Germany、Munich Center for Machine Learning
- **论文：** [arXiv:2601.09512v2](https://arxiv.org/abs/2601.09512v2)（2026-05-15）；[HTML](https://arxiv.org/html/2601.09512v2)；[PDF](https://arxiv.org/pdf/2601.09512v2)
- **出版：** IEEE Robotics and Automation Letters 2026，[DOI](https://doi.org/10.1109/LRA.2026.3693992)
- **项目页：** <https://tum-lsy.github.io/clare/>
- **官方代码：** <https://github.com/learnsyslab/clare>
- **核对范围：** v2 全文 11 页，正文 §I–VI、算法 1/2、图 1–9、表 I–VI、附录 VII-A/B/C；结合官方项目页、固定源码快照及 Hugging Face 文件清单核对开放边界，未运行训练、仿真 rollout 或真机。
- **通读：** 已完成
- **入库日期：** 2026-11-28

## 可追溯证据

1. **§III–IV、算法 1/2：** 每个新增任务仍需独立的新演示和明确的训练阶段；部署只使用观测和语言指令，不使用 oracle task ID。新阶段先训练新增函数适配器，再冻结它们训练新判别器。遍历旧判别器时用重构误差分布的 z-score 决定逐层扩容；推理按原始重构误差最小选判别器，而不是按 z-score 选路。多个判别器可映射同一旧适配器。
2. **表 II–IV、图 4/5：** LIBERO-Long、Goal、Spatial 的 AUC 分别为 75.1%、89.3%、87.4%；各高于同表 ER 的 60.5%、76.0%、77.6%。LIBERO-40 的 AUC 83.6%、NBT −1.0%，每任务每阶段只用 10 次 rollout；五项真机 AUC 63.3%、NBT −2.9%，是跨阶段聚合值。AUC 先为每个任务平均其已学习阶段，再在任务间平均；FWT 是初学当阶段成绩的平均。
3. **表 V/VI、图 7/9 与附录：** 全条件路径扩展下，BOWL 入学 100%，第五阶段 90%；负的总体 NBT 并不代表每项从未下降。把取放积木指令改成关抽屉是观测/语言变化引发的动作适配，不是部署中继续更新权重；附录 B 报告该切换在 10 次试验中表现一致。论文报告的路由额外开销小于 3 ms，图 7 的第 6–10 阶段是外推。

- **对 wiki 的映射：** [CLARE](../../wiki/entities/paper-clare-continual-vla.md)
