# Model-Free Runtime Measurement of Task-Relevant Information — 官方代码归档

- **仓库：** https://github.com/dimipan/ast-telemetry
- **论文：** https://arxiv.org/abs/2611.14111
- **核对范围：** 官方 GitHub 嵌套子项目 README、reproduce.sh 的全部入口与工作流、实验/监控模块表；未获取独立数据包或实际运行。
- **通读：** 已完成
- **开源状态：** 官方公开仓库存在；代码位于嵌套子目录，尚未核实可运行入口。

## 实际入口

- **项目路径：** `ast_telemetry_anonymous_repo/ast-telemetry/`，不是仓库根。
- **运行前：** `export AST_DATA=/path/to/ast-data`，`pip install -e ".[envs,rl,test]"`。
- **运行：** `./reproduce.sh studies`（实验 1–3）、`JOBS=16 ./reproduce.sh rl`（实验 4）、`./reproduce.sh`（全部）；脚本内部最终调用 `experiments/figures.py` 与 `experiments/check.py`。
- **主要模块：** `asttel/monitor.py`、`asttel/bindings/`、`experiments/study1_sar.py` 至 `study4_evaluate.py`。
- **证据：** [官方 reproduce.sh](https://github.com/dimipan/ast-telemetry/blob/main/ast_telemetry_anonymous_repo/ast-telemetry/reproduce.sh)。

## 对 wiki 的映射

- [AST 论文实体](../../wiki/entities/paper-ast-task-information.md)
