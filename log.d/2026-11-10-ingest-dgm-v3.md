## [2026-11-10] ingest | sources/papers/dgm_arxiv_2505_22954v3.md | DGM arXiv v3

- **意图：** 将 Darwin Gödel Machine（Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents）整理为 Robotics_Notebooks 可检索的独立中英双语知识实体，复用已有源码归档。
- **资料：** arXiv:2505.22954v3 正文与附录 A–J；Sakana AI 官方介绍；jennyzzt/dgm 官方 README、DGM_outer.py、self_improve_step.py 及源码入口。
- **主要结论：** 固定基础模型、探索可修改的编码 Agent 代码；用外部编程基准评估与开放式 archive 保留暂时退步的候选。SWE-bench Verified 20.0%→50.0%，完整 Polyglot 14.2%→30.7%；50 题搜索集的 38.0% 与全量复评须分开读。
- **开源结论：** 官方仓库 Apache-2.0，Docker / Python 主入口与示例代码已发布。未实测完整进化，需付费模型调用，并须隔离不可信代码。
- **写回：** wiki/entities/paper-darwin-godel-machine.md；sources/papers/dgm_arxiv_2505_22954v3.md；sources/sites/dgm-sakana-ai.md；更新 sources/repos/darwin-godel-machine.md 与 roadmap/depth-rsi.md。
- **图表：** 论文方法 flowchart、对照源码调用路径的 sequenceDiagram。
- **局限：** 论文目标劫持案例与高算力成本；未实证机器人真机或基础模型权重自我更新。
