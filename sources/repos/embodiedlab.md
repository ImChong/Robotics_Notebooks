# EmbodiedLab 官方仓库

> 来源归档（repo）

- **标题：** EmbodiedLab: Benchmarking Embodied Agents That Act to Know
- **类型：** repo
- **链接：** https://github.com/HexaFuture/EmbodiedLab
- **arXiv：** <https://arxiv.org/abs/2610.35385>
- **入库日期：** 2026-10-28
- **一句话说明：** 以物理实验室为评测场，智能体需自主选择实验步骤、操作仪器并从读数得出结论。
- **沉淀到 wiki：** [`wiki/entities/paper-embodiedlab.md`](../../wiki/entities/paper-embodiedlab.md)

## 开源状态

- **已公开**：实验定义、运行框架及评测代码，Apache-2.0；见 [README](https://github.com/HexaFuture/EmbodiedLab/blob/main/README.md)。
- README 的 v0.1.0 支持 T0、T1；论文中的诊断级 T2 技能库不在该发布版。
- `./embodiedlab trial` 调用 `scripts/trial.py`；该脚本导入 `benchmark.evaluation.evaluate` 和 `benchmark.evaluation.process` 执行离线评分。
