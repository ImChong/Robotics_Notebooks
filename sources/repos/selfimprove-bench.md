# SelfImprove-Bench｜官方源码

> 来源归档（公开代码仓，静态核对）

- **类型：** repo
- **链接：** <https://github.com/theseus-labs-rsi/SelfImprove-Bench>
- **项目页：** <https://theseus-labs-rsi.github.io/SelfImprove-Bench/>
- **论文：** <https://arxiv.org/abs/2611.02797>
- **入库日期：** 2026-11-09
- **核对范围：** README、入口脚本和文中列出的主要运行模块；没有在本次环境运行完整依赖或评测。
- **通读：** 未完成

## 已核查入口

- **运行入口：** `python -m benchmark configs/main/farm_ace.json`；`--initial` 使用相同保留集测试初始框架。
- **核心模块（静态核对）：** `benchmark/__main__.py` 读配置并调用 `benchmark.runner.run_benchmark`；`runner.py` 的 `AgentController` 将 `benchmark.game_adapters.evaluate_task` 的反馈交给 `rsi_session.refine`，在 holdout 加载冻结状态。
- **结构：** `benches/` 管环境与评分，`rsi_methods/` 提供 ACE/AHE/HyperAgents 适配器，`analyze/` 分析轨迹。
- **许可：** 主仓 MIT；HyperAgents 部分 CC BY-NC-SA 4.0 仅非商业；GameWorld 资源受另外的研究/教学条款限制。

## 工程复现建议

1. 参照主仓 README 使用 Python 3.13.12 建环境；AHE 还需要 Linux bubblewrap，GameWorld 需 Playwright，HyperAgents 需 Docker。
2. 训练 ACE Farm 示例：`python -m benchmark configs/main/farm_ace.json`。
3. 用同一配置运行 `python -m benchmark configs/main/farm_ace.json --initial`，固定 holdout 种子比较初始/改进状态。
4. 使用 `python -m benchmark.hci --trained outputs/main/farm/ace --initial outputs/main/farm/ace_initial --output analysis/farm_ace_hci.json`，并按游戏/难度报告而非仅给全局平均值。

## 局限

这是一套以游戏环境为主的 RSI 评测，不能直接推出机器人真机泛化；HyperAgents 内嵌代码有非商业许可，主仓 MIT 不覆盖全部三方依赖。

## 对 wiki 的映射

- [SelfImprove-Bench 主实体](../../wiki/entities/paper-selfimprove-bench.md)
- [论文归档](../papers/selfimprove-bench_arxiv_2611_02797.md)
- [官方项目页归档](../sites/selfimprove-bench.md)
