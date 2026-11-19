# World Models as Unit Tests — 官方源码归档

- **仓库：** https://github.com/DongzheZheng/world-models-as-unit-tests
- **许可：** MIT（仓库 LICENSE）
- **项目页：** https://dongzhezheng.github.io/world-models-as-unit-tests/
- **数据集：** https://huggingface.co/datasets/DenzelZheng/world-models-as-unit-tests
- **交互演示：** https://huggingface.co/spaces/DenzelZheng/world-models-as-unit-tests
- **归档日期：** 2026-11-19
- **源码开放结论：** 已发布示例模拟环境与固定控制程序、生成的世界模型、复现脚本、实验记录及小型 harness。**不含重新运行 LLM 创建世界模型的完整历史流程**；用户环境的 harness 尚未由作者通过模型做端到端试运行。
- **核对范围：** README.md 的 Install、Reproduce、Harness、What is not here；reproduce.py 的 TASKS/REGISTRY、run_real 与 world 分支；harness/README.md 的环境 contract、三种角色、config、run/verify 示例；harness/cli.py 的 cmd_run/cmd_verify、harness/verify.py 的评分函数定义；非逐文件代码审计。
- **通读：** 已完成

## 官方文件与真实入口

| 位置 | 作用 |
|---|---|
| `reproduce.py` | 冻结策略注册表与评测入口；`list` 列出 8 个 task/多种 controller；`real <task> <controller>` 在原模拟器以既定 seed 重评；`world <task>` 在已写世界模型中重评程序 |
| `reallab/` | 隔离“真实环境”的 4 类仿真：Doom、drone、pushing、cable/rope |
| `worlds/` | Agent 已写出的显式 Python 模型 |
| `pilots/` | CaP 程序版本 |
| `experiments/<task>/code_wm/` | WM agent 控制器、拟合、仿真评测辅助模块 |
| `runs/` | 官方冻结评测结果、日志、参数 |
| `cawm/` | Trial 预算记账、MPC、PPO 与共用组件 |
| `harness/` | 面向新 Python 环境的受限试验与 held-out 验证器 |

官方 README 给出的复现命令（**只核对代码/文档，本次未运行**）：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
python -m pytest tests reallab -q
python reproduce.py list
python reproduce.py real drone_C cap_in_code_world
python reproduce.py world pusht
python -m harness run --config harness/examples/toy/run.toml
```

## 运行时可核对的关键路径

- `reproduce.py` 的 registry 把 task/controller 对应到 `pilots.*`、`experiments.*` 控制器工厂及 `runs/` 的记录；real 评测加载 `reallab/` 环境，给定 seed 逐 episode 运行，并将回放成功率与记录并列输出。
- `reproduce.py world` 分支对冻结的 CaP 程序在 `worlds/` 对应模型中重算测试成绩；不同 task 有独立的评测分支。
- `harness` 的入口是 `python -m harness run --config ...`；`harness/cli.py` 负责编排 agent / lab 服务与 verify，`harness/verify.py` 分别给 `world.py` 与隔离环境 held-out 打分并输出报告。作者只对无模型的 toy replay 等路径做了测试，切勿将其表述为通用真机部署套件。
- 测试方式与配套依赖见 README；本文没有声称我们成功在本地复现。

## 映射

- [项目页原文](../sites/world-models-as-unit-tests.md)
- [wiki 项目实体](../../wiki/entities/world-models-as-unit-tests.md)
