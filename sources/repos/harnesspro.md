# HarnessPro 官方仓库

- **代码：** <https://github.com/pzs19/HarnessPro>
- **核查版本：** 6fd2d4dbc3b73881b2b48833764fcd803aacb7b8
- **许可：** MIT；第三方组件遵循各自许可。
- **论文：** <https://arxiv.org/abs/2610.26770v2>
- **评估文档：** <https://github.com/pzs19/HarnessPro/blob/6fd2d4dbc3b73881b2b48833764fcd803aacb7b8/docs/evaluation.md>
- **项目页：** <https://pzs19.github.io/HarnessPro/>
- **入库日期：** 2026-11-02

仓库提供三项基准各三种模式的 harness、评估入口、分割清单和部分评估器源码；基准数据、模型服务、凭据及相关容器镜像需另行获取。论文的 GLM-5.2 演化快照未包含在此版本。README 给出 Linux、Python 3.12/3.13、uv、Docker 和 Codex CLI 等条件。

Terminal-Bench 2 入口为 `scripts/run_tb2_harness_eval.py`；它加载数据及运行策略，通过 `src/rho/orchestrators/solve.py` 的 `solve_workspace` 和 `solve_in` 运行 agent，再调用 task 的评分接口。当前仅核查该入口及 solve 模块，其他路径尚待核查。
