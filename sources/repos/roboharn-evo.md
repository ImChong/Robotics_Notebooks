# RUCKBReasoning/RoboHarn-Evo（官方代码仓）

> 来源归档（ingest）

- **标题：** RoboHarn-Evo
- **类型：** repo
- **URL：** <https://github.com/RUCKBReasoning/RoboHarn-Evo>
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-16
- **配套论文：** [RoboHarn-Evo（arXiv:2609.37583）](https://arxiv.org/abs/2609.37583) — [论文归档](../papers/roboharn_evo_arxiv_2609_37583.md)
- **项目页：** <https://ruckbreasoning.github.io/RoboHarn-Evo/> — [项目页归档](../sites/roboharn-evo-project.md)

## 一句话摘要

公开的 Python 实现提供共享 HPK agent、RMBench 与 LIBERO-PRO benchmark 应用适配器、服务脚本和配置；README 给出任务列表、dry-run 与运行入口。仿真依赖、模型 checkpoint、SAM3 源码/权重和许可资产需单独安装或配置。

## 仓库核查（2026-10-16）

| 项 | 状态 |
|----|------|
| **公开可见** | 是 |
| **许可证** | Apache-2.0（仓库 LICENSE） |
| **可运行入口** | 有；scripts/run_rmbench.py、scripts/run_libero_pro.py 及模型服务脚本 |
| **主要实现** | roboharn_evo/；benchmark 应用与任务定义位于 benchmarks/ |
| **安装** | Python 3.10+；python -m pip install -e '.[api,expert]' |
| **依赖边界** | 模拟器依赖、模型 checkpoint、SAM3 与权重、外部数据集及受许可资产按 benchmark 单独准备 |
| **无执行副作用的检查** | --list-tasks、--list-suites、RMBench --dry-run |
| **运行输出** | README 指向 eval_result/ |

## 复现读法

1. 先按所选 benchmark 的 README 配置运行时、模型服务与数据/许可资产。
2. 用任务列表和 dry-run 核对配置，再按 README 给出的显式参数运行 benchmark。
3. 读取各 benchmark 自己定义的成功与终止条件；将论文的消融、held-out 和真实机器人数字分开比较。

代码把“可演化知识”落到 agent 配置和运行时流程，复现实验仍需匹配 benchmark、模型服务及资产版本。

## 关联资料

- 论文：[sources/papers/roboharn_evo_arxiv_2609_37583.md](../papers/roboharn_evo_arxiv_2609_37583.md)
- 项目页：[sources/sites/roboharn-evo-project.md](../sites/roboharn-evo-project.md)
- wiki：[wiki/entities/paper-roboharn-evo.md](../../wiki/entities/paper-roboharn-evo.md)
