# Learn2Design-2026 竞赛站（learn2design2026.com）

> 项目页来源归档（ingest）

- **标题：** Learn2Design Competition — A physics experiment design competition for gravitational-wave detectors
- **类型：** site / competition / gravitational-wave / scientific-instrument-design / neurips-2026
- **URL：** <https://www.learn2design2026.com/>
- **代码：** <https://github.com/artificial-scientist-lab/Learn2Design-2026>（MIT，**已开源**）
- **提交门户：** <https://submit.learn2design2026.com/competitions/4/>（Codabench）
- **论文：** Nature 2026 Review · DOI [10.1038/s41586-026-10898-6](https://doi.org/10.1038/s41586-026-10898-6)
- **入库日期：** 2026-09-12
- **一句话说明：** NeurIPS 2026 官方竞赛页：提交 **优化算法**（非固定设计），在隐藏 UIFO 拓扑上 4h 预算内最大化探测器灵敏度；链到 **Differometor** 仿真器、30k 设计数据与 EUR 25k 奖金。

## 项目页核查（步骤 2.5，截至 2026-09-12）

| 项 | 结论 |
|----|------|
| **GitHub / Starter kit** | 页内与 README 指向 `artificial-scientist-lab/Learn2Design-2026` → **已开源**（MIT） |
| **仿真器** | **Differometor**（PyPI `differometor`，GitHub MIT） |
| **数据集** | ~30,000 设计；`GraviTune-Dataset` 仓 + 竞赛仓 `competition_data/` |
| **评测框架** | `Differometor-Benchmark` / `dfbench` |
| **提交** | Codabench 门户；每月公开榜 + 最终隐藏评测 |
| **奖金** | EUR 25,000（SPRIND）；决赛截止 **2026-10-15**（站面文案） |

## 竞赛格式要点（摘自站面 / README）

- **提交物：** 单个 Python 优化类 + `requirements.txt` + 依赖文件，ZIP 上传。
- **评测：** 每拓扑 **4h** wall-clock（`objective.start_logging()` 起）；记录约束可行下的最优损失；10 隐藏拓扑算术平均。
- **核心问题：** AI 能否发现 **超越人类直觉** 且仍 **物理可行** 的科学仪器设计？

## 对 wiki 的映射

- [`wiki/entities/paper-designing-physics-experiments-with-ai.md`](../../wiki/entities/paper-designing-physics-experiments-with-ai.md)
- [`sources/papers/designing_physics_experiments_with_ai_nature_s41586_026_10898_6.md`](../papers/designing_physics_experiments_with_ai_nature_s41586_026_10898_6.md)
- [`sources/repos/artificial_scientist_lab_learn2design_2026.md`](../repos/artificial_scientist_lab_learn2design_2026.md)

## 当前提炼状态

- [x] 开源状态已写入 wiki 工程实践
