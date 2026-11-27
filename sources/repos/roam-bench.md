# ROAM-Bench：评测与数据源码

- **源码：** <https://github.com/rxjfighting/ROAM-Bench>
- **论文：** [arXiv 原文归档](../papers/roam-bench_arxiv_2611_33143.md)
- **项目页：** [项目主页归档](../sites/roam-bench.md)
- **许可：** MIT 覆盖主仓代码、任务标注与作者资产；第三方场景和机器人资源按各自许可。
- **入库日期：** 2026-11-27
- **核对范围：** 静态检查 README、CLI、harness、轨迹和判分调用；未运行 Isaac Sim 或模型端点。
- **Wiki：** [ROAM-Bench](../../wiki/entities/paper-roam-bench.md)

[`code/scripts/evaluate.py`](https://github.com/rxjfighting/ROAM-Bench/blob/main/code/scripts/evaluate.py) 转给 [`roam_bench.cli.main`](https://github.com/rxjfighting/ROAM-Bench/blob/main/code/roam_bench/cli.py)。它加载任务和初始条件，建立 `EvaluationClient` 与 `ROAMBackend`，由 [`harness.run_model_episode`](https://github.com/rxjfighting/ROAM-Bench/blob/main/code/roam_bench/harness.py) 驱动每轮探索；结束后以 `sim.exploration_evidence()` 和 `enhanced_evidence.judge` 独立核对证据。各题有 `trajectory.json`，汇总记录 A/N/S/T、模型资源用量与配置。Isaac Sim、G1/Inspire 资产和视觉模型端点需要使用者准备。
