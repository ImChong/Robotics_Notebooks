# MARGoS（tombackert/margos）

> 来源归档（repo；核对日期：2026-10-12）

- **代码：** <https://github.com/tombackert/margos>
- **论文：** <https://arxiv.org/abs/2610.13772>
- **许可：** MIT（Copyright 2026 Tom Backert）
- **状态：** README 标注 research MVP；最近提交 `1c6b7cb docs: add springer doi and citation metadata`
- **沉淀到 wiki：** [paper-margos-reproducible-marl-argos.md](../../wiki/entities/paper-margos-reproducible-marl-argos.md)

## 仓库内容（浅克隆核查）

- `margos/`：Python 包——`cli.py`（Typer，入口 `margos` / 别名 `m`）、`config/{loader,schema}.py`（加载、路径解析、`hash_config`、`save_frozen_config`）、`orchestrator/runner.py`（`run_experiment`：加载校验配置 → 建输出目录 → 写冻结配置与哈希 → 采集环境指纹 → 设全部种子（在导入训练脚本之前）→ 日志回调 → 动态导入并执行训练脚本 → `verify_config_integrity`）、`utils/fingerprint.py`、`analysis/{compare,report}.py`（`compare_runs` 比较尾部奖励均值、AUC、配置哈希与完整性，输出 handoff_pass / repro_pass）、`export/{bundle,importer}.py`（ZIP 实验包与导入时指纹对比、路径重写）、`logging/callbacks.py`、`argos_zoo/`（ZeroMQ 客户端与 PettingZoo 环境）。
- `argos_plugins/`：ARGoS C++ 控制器与 loop-functions 插件，需 CMake 编译。
- `experiments/`：`configs/*.yaml`（如 `aggregation_v1.yaml`：experiment.name/seed、scenario.file、training.script/iterations、output.dir）、`scenarios/*.argos`、`training/aggregation.py`、`run_srq3_batch.py`。
- `tests/`：含配置、种子传播、指纹、导出导入、CLI 端到端等 pytest；慢速端到端测试需 ARGoS + RLlib。
- 安装：`pip install -e ".[ml]"`，另需 ARGoS 3 与自行编译插件。本次未实际运行训练。
