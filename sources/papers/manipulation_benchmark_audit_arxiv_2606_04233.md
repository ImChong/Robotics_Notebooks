# What Are We Actually Benchmarking in Robot Manipulation?（机器人操作基准到底在测什么？）

- **类型：** 原始研究论文 / CoRL 2026（作者 GitHub README 标示）
- **论文：** <https://arxiv.org/abs/2606.04233>
- **HTML 原文：** <https://arxiv.org/html/2606.04233>
- **项目：** <https://ripl.github.io/manipulation_benchmark_audit/>
- **官方代码：** <https://github.com/ripl/ManipulationBenchmarkAudit>
- **核对日期：** 2026-11-12
- **核对范围：** arXiv 全文正文的方法动机、四项诊断、五类操作基准实验、统计显著性分析、局限与附录；对照官方 README、CLAIMS.md、REPRODUCIBILITY.md 与 scripts/recompute_claims.py 确认哪些数值可从公开文件重新计算。未独立运行全部仿真与外部策略评测。
- **通读：** 已完成

## 研究问题（论文原文）

高操作 benchmark 分数不必然意味着语言理解、稳健泛化或长程操作能力；作者审计 **LIBERO、CALVIN、SimplerEnv、RoboCasa、RoboTwin 2.0**，采用四项互补诊断，而非提出又一个得分更高的策略。

1. **Shortcut solvability：** 不需要被声称的能力（如语言）能否获得高分？LIBERO 中低容量 DINO+MLP probe（约 0.09B、没有语言编码器）通过条件设置取得接近榜首成绩，不能将该结果解读为“简单 MLP 拥有全部具身能力”。
2. **Statistical significance：** 只凭公布的 aggregate success 和试次数，能否证明突破超过评测噪声？原论文 LIBERO **19.8%** 是按论文筛选与判定口径可证实显著的历史领先比较比例，并不意味着剩余 80.2% 全部被证伪。
3. **Creeping overfitting：** 测试初始条件、重采样和新配置改变后分数如何变化？CALVIN 在训练分布内调整积木位姿，使受测策略平均任务完成数下降。
4. **Data-source dependence：** 仿真测试附近的示范数据能使较小模型取得高分，是否混淆了训练数据贴近测试条件和策略泛化？SimplerEnv 22M 模型在近测试示范数据下约 94.8%，不同训练设定的 0.9B X-VLA 约 95.8%；这不是严格控制变量的架构实验。

## 数字的版本和证据边界

- **论文原始比例与最新公开包不可强行混用：** 论文/项目摘要以 LIBERO 19.8% 为早期口径；官方仓库 `CLAIMS.md` 记录经 camera-ready 修订的 LIBERO Spatial/Object/Goal **144/789 = 18.3%**，SimplerEnv **24/122 = 19.7%**。官方还明确保留 **88 行** frozen-paper 与 release-current 统计分类差异，不应无版本标注地统一写 19.8%。
- **可重算不等于独立重做：** `scripts/recompute_claims.py` 使用公开 CSV/JSON/YAML 核验 headline 数字；训练和第三方测评仍依赖外部 checkpoint、场景和运行环境。仓库 README/REPRODUCIBILITY.md 区分 CPU claim regeneration、作者训练脚本和第三方策略评测三类证据。
- **试次与受测任务必须同时报告：** 对微小提升还需原始成功/失败计数、共享测试实例/随机种子，以及多次重做的差异。

## 对 wiki 的映射

- [操作基准审计独立节点](../../wiki/entities/paper-manipulation-benchmark-audit.md)
- [DISCO 真机小样本的检验口径](../../wiki/entities/paper-distilled-tree-search-multi-object-rearrangement.md)
- [操作任务](../../wiki/tasks/manipulation.md)

## 一手资料

- [论文](https://arxiv.org/abs/2606.04233)
- [官方项目页](https://ripl.github.io/manipulation_benchmark_audit/)
- [研究组开源仓库 README](https://github.com/ripl/ManipulationBenchmarkAudit/blob/main/README.md)
- [Claim-to-artifact 证据映射](https://github.com/ripl/ManipulationBenchmarkAudit/blob/main/CLAIMS.md)
- [复现边界](https://github.com/ripl/ManipulationBenchmarkAudit/blob/main/REPRODUCIBILITY.md)
