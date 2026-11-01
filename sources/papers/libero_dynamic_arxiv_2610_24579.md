# LIBERO-Dynamic（arXiv:2610.24579）

> 来源归档（paper）

- **标题：** LIBERO-Dynamic: Benchmarking Robotic Manipulation in Dynamic Environments
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.24579>
- **PDF：** <https://arxiv.org/pdf/2610.24579>
- **项目页：** <https://renshaojie233.github.io/LIBERO-Dynamic/>
- **代码：** <https://github.com/renshaojie233/LIBERO-Dynamic>
- **入库日期：** 2026-11-01
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** 模型在静态桌面上成功，不代表能抓住移动中的目标。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 模型在思考，目标还在移动。LIBERO-Dynamic把物体运动、实测推理延迟和动作队列放进同一评测：机器人思考时，物理世界继续运行。
- 模型在静态桌面上成功，不代表能抓住移动中的目标。
- 指标：公众号转述：覆盖全部 40 项 LIBERO 任务，每个模型—执行配置 1,800 回合；同一 π₀.₅ 检查点在各自实测最佳动作时域下，异步 / 同步成功率为 48.4% / 43.1%（各自有限扫描中的最佳点）；结论限定于规定轨迹和标定延迟，复杂随机运动、时延抖动与更广泛真机任务仍待验证。

## 对 wiki 的映射

- [paper-libero-dynamic](../../wiki/entities/paper-libero-dynamic.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
