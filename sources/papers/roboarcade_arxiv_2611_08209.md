# RoboARCade: Benchmarking Agentic Robot Control / 智能体式机器人控制评测

- **类型：** 原始论文（arXiv: 2611.08209v1；2026-11-05）
- **论文：** https://arxiv.org/abs/2611.08209
- **HTML 全文：** https://arxiv.org/html/2611.08209v1
- **PDF：** https://arxiv.org/pdf/2611.08209
- **官方项目：** https://roboarcade.org/
- **代码开放：** https://github.com/Satvik1701/RoboARCade
- **核查日期：** 2026-11-09
- **核对范围：** 阅读 arXiv HTML v1 的框架 §3、实验 §4、结论及相关附录；打开官方榜单和 GitHub README、core/run_eval.py、core/run_seed.py、core/mcp_server.py 入口，未实际跑 benchmark。
- **通读：** 已完成

## 核心机制（源自论文正文）
1. RoboARCade 将长时程操作分成可观察、可评分的子任务，含 22 个任务，支持多视角 RGB-D 与 VIA 视觉界面等代理工具接口。
2. 在 MuJoCo 中测试智能体如何选择工具、组织顺序、处理环境状态与执行机器人控制；报告不同本体与接入方式。
3. 支持轨迹记录、分段评分、失败分析与自主改进实验；与纯低层运动策略测试不同。
4. 源码 core/run_eval.py 调用 core/run_seed.py，后者启仿真和隔离的代理，core/mcp_server.py 按接口的 tools.py 分发控制调用，core/grade_episode.py 给出任务结果。

## 原始实验数字及读法
- 官方榜单给出的较强组合最高子任务成功比率约 49.8%；该数是 subtask completion，不是 22 任务的整任务成功率。
- 论文归纳策略选择/目标分解与精细控制是常见失败因素；不同 agent、接口与预算不能简单同列为纯模型能力。

## 局限及复现边界
- 主要基于仿真，受任务集合、资源预算和接口约束；未等同于真实机器人通用可靠性。
- 不同视觉界面带来的 token/延迟/成本变化可能改变结果。

## 对 wiki 的映射
- [RoboARCade 项目节点](../../wiki/entities/paper-roboarcade.md)

## 参考一手资料
- [arXiv HTML v1](https://arxiv.org/html/2611.08209v1)
- [官方项目主页](https://roboarcade.org/)
