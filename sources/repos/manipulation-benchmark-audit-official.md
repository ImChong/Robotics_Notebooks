# ManipulationBenchmarkAudit 官方仓库与源码入口

- **仓库：** <https://github.com/ripl/ManipulationBenchmarkAudit>
- **项目主页：** <https://ripl.github.io/manipulation_benchmark_audit/>
- **论文：** <https://arxiv.org/abs/2606.04233>
- **状态：** 核心审计数据与统计重算脚本公开，部分训练与评价依赖外部模型/数据
- **核对日期：** 2026-11-12
- **核对范围：** README.md、CLAIMS.md、REPRODUCIBILITY.md、scripts/recompute_claims.py 的入口、统计分类与不包含内容；未执行 GPU 训练或第三方 benchmark。
- **通读：** 已完成

## 源码调用依据

- `python scripts/recompute_claims.py`：读取四个诊断下的公开 CSV/JSON/YAML，按项目声称的成功数、样本量、显著性分类、场景扰动等对照再计算，输出包含 status、errors 与 results 的 JSON。
- `python scripts/validate_release.py --skip-exact-inputs`：轻量公开资产一致性检查；精确输入需按 REPRODUCIBILITY.md 另装 NumPy/PyTorch。
- `shortcut_solvability/`：简单策略训练/测试结果和脚本。
- `statistical_significance/`：共享初始场景对比和汇总排行榜显著性分类。
- `creeping_overfitting/`：CALVIN、SimplerEnv、LIBERO 的扰动场景、输出与脚本。
- `data_source_dependency/`：测试数据来源依赖及 WidowX 收集/重放/训练路径。
- `analysis/run_analysis.py paper-inputs` 与 `analysis/run_analysis.py current`：重新生成冻结论文版本与当前公开版本的部分分析产物。
- 不能将公开 `CLAIMS.md` 的 camera-ready LIBERO **18.3%** 与旧摘要 **19.8%** 的比例直接视为同一口径；详见 [论文归档](../papers/manipulation_benchmark_audit_arxiv_2606_04233.md)。

## 与 Wiki 对接

[操作基准审计](../../wiki/entities/paper-manipulation-benchmark-audit.md)
