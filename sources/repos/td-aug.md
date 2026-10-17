# TD-Aug（td-aug/td-aug）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/td-aug/td-aug>
- **默认分支：** `main`
- **源码状态：** 已公开
- **许可证：** MIT
- **核验日期：** 2026-10-17
- **项目页：** <https://td-aug.github.io/>
- **论文：** <https://arxiv.org/abs/2610.19129>
- **知识节点：** [TD-Aug](../../wiki/entities/paper-td-aug.md)

## 仓库说明

论文代码。目录 `agents/`、`data_gen_scripts/`、`ogbench/`、`utils/`；入口 `main.py`（训练策略）、`main_gen.py`（先训 tdflow 再训带增广目标的策略）、`main_tdflow_gcrl.py`（README 未说明）；`hyperparameters.sh` 列出 GCFBC / MC / TD2 / MCHC / Dyn-Aug 命令。安装 conda `environment.yaml` 或 uv + `requirements.txt`；数据从 Hugging Face `daniellawson9999/tdaug-datasets` 下载到 `data/`；基于 OGBench 与 TRA-OGBench。核验时 1 次提交、无 Release。

## 相关归档

- [论文摘录](../papers/td_aug_arxiv_2610_19129.md)
- [项目页归档](../sites/td-aug-project.md)
