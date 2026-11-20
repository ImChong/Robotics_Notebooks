# Frontier Design Benchmark｜Contra Labs × BenchFlow 官方一手资料

- **主研究：** https://contralabs.com/research/frontier-design-benchmark （2026-11-18）
- **首页：** https://contralabs.com/frontierdesign
- **榜单：** https://contralabs.com/frontierdesign/leaderboard
- **方法：** https://contralabs.com/frontierdesign/methodology
- **Rubric / Slop：** https://contralabs.com/frontierdesign/rubric
- **数据集说明：** https://contralabs.com/frontierdesign/dataset
- **运行方 SDK：** https://github.com/benchflow-ai/benchflow
- **核对日期：** 2026-11-20
- **核对范围：** 通读官方研究报告任务、模型 rollout 与人类 golden、偏好投票方法、结果、Gestalt Preview、结论；并核对官方 methods/rubric/dataset/leaderboard 全页。
- **通读：** 已完成

## 研究与执行协议

- 研究报告与方法页称初始集 33 个任务：27 个由 Contra 真实委托改写，6 个来自创业公司实际设计项目；网页、Web app、移动应用均覆盖。实时榜页面另写“27 Contra jobs”，不能假定各轮全部 33 题均进入同一统计口径。
- BenchFlow 在沙盒中运行 agent、记录网站产物、调用与成本；模型提供方 OpenRouter，尽可能用各实验室自有 harness。
- Human golden 为专业设计师在 moodboard、反馈、迭代后交付的设计，不是未经训练的普通人随手完成。
- 35 位设计师在 5 轮完成 4,311 次两两盲投（报告快照），以 Bradley–Terry 估算 Elo；200 次随机分半后模型名次 Spearman 平均 0.95。
- 主榜基于 expert preference；另有 Design Critique Rubric 与 AI Slop Detector，后者低分表示较少典型 AI 味，不等于“高质量”。
- **官方口径差异：** 研究报告写 34 个 rubric 设计维度，方法页写 35 条高质量 rubric，不能未经核实把它们当同一计数。

## 结果与开放边界

| 日期/来源 | Gestalt (stealth) |
|---|---|
| 2026-11-18 研究报告 | Elo 1928，89%，378/423 |
| 2026-11-20 在线榜 | 第 1 名，Elo 1931，90%，420 票 |

- 研究报告另称在与 human golden 两两比较 28 票中胜出 20 票；不能据此推出“任意设计任务都胜人类”。
- [Dataset](https://contralabs.com/frontierdesign/dataset) 明确 **By request / dataset isn’t public**：任务、golden、rollouts、专家票与 rubrics 不可直接全量公开下载。
- [BenchFlow SDK](https://github.com/benchflow-ai/benchflow) 有开源实现，但不等于此私有基准的全部任务和专家投票数据已开源。
- 是**网站及应用设计**评测，不是 VLA/具身机器人动作成功率测评。

## 对 wiki 的映射

- [Frontier Design Benchmark](../../wiki/entities/frontier-design-benchmark.md)
- [Gestalt (stealth)](../../wiki/entities/gestalt-stealth-design-model.md)
