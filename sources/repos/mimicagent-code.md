# MimicAgent 官方代码仓库核查

- 核查日期：2026-10-30
- 四足：<https://github.com/Luckykantnayak/mimicagent-datagen>，main 仅含 README 和 .gitignore；README 表示代码即将发布。
- 人形：<https://github.com/Luckykantnayak/mimicagent-humnaoid>，核查版本 [daac033](https://github.com/Luckykantnayak/mimicagent-humnaoid/tree/daac0331fd3cbc32c24e63449ca6ed6c900fbcc7)，Apache-2.0。

## 人形代码入口与边界

- README 的生成入口为 `cd agent && ROBOT_NAME=g1 python run.py --skill "..." --name SKILL_CRAWL`；生成需 Anthropic API key。
- `agent/run.py` 准备技能并调用 `pipeline.build_graph`；`agent/pipeline.py` 编排 goal contract、规划、代码生成、执行、测试与诊断修复。
- 回放入口 `python simulator/run.py --robot g1 --mode SKILL_CRAWL`；运行结果包括代码、报告、轨迹及预览。README 说明 `best_effort` 和 `unverified` 并非通过。
- 这些是运动学参考；通过检查或生成 GIF 不证明控制器能够跟踪。四足训练、部署实现未在四足仓库公开。

## 对 wiki 的映射

- [MimicAgent](../../wiki/entities/paper-mimicagent.md)
