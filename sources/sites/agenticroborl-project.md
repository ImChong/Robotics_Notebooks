# AgenticRoboRL 官方项目页

> 来源归档（site / 官方项目页）

- **标题：** AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics
- **类型：** site
- **链接：** <https://xinxinliu-ai.github.io/AgenticRoboRL/>
- **论文：** <https://arxiv.org/abs/2610.28589>
- **代码：** <https://github.com/xinxinliu-ai/AgenticRoboRL>
- **模型：** <https://huggingface.co/ciciciciliu/AgenticRoboRL-Qwen3.5-9B>
- **作者与机构：** Xinxin Liu、Bingxin Xu、Wenrui Bao、Ming Li、Wentao Bao、Chen Chen、Yuzhang Shang；University of Central Florida、University of Southern California、Michigan State University
- **许可证：** Apache-2.0（官方仓库）
- **入库日期：** 2026-10-26
- **一句话说明：** 以 SFT + 多轮 RL 后训练机器人 harness 的 planner；同一方法适配 VLA skill 调用与控制代码生成两种执行接口。

## 项目页核查摘要

- **方法：** 对成功教师轨迹做 SFT，再通过 planner 与机器人 harness 多轮交互采集 rollout，以终局成功奖励优化 planner。
- **系统边界：** planner 调用的 VLA、感知、运动控制和 harness 工具保持不变，训练只更新 planner。
- **结果展示：** RPent / LIBERO-PRO 的 Qwen3.5-9B 从 18.1% 提升到 71.0%；LIBERO-Plus 同 checkpoint 报告 72.2%；项目页显示模型 checkpoint 可下载。
- **两种 harness：** RPent 调用冻结 VLA skill；CaP-X 允许 planner 生成并执行 Python 控制程序。
- 项目页图示训练数据流和两类 harness，并提供 benchmark episodes、结果和 BibTeX。

## 开源核查（2026-10-26）

| 项目 | 状态 |
|---|---|
| 代码 | **已开源**：训练、adapter、harness 配置、minimal harness 与评测控制器见官方仓库 |
| 模型 | **已发布**：项目页链接 Qwen3.5-9B SFT / RL checkpoints |
| 训练数据 | 教师轨迹未随仓库发布，完整重训受数据可用性限制 |
| 许可证 | Apache-2.0 |

## 对 wiki 的映射

- [AgenticRoboRL 实体页](../../wiki/entities/paper-agentic-roborl.md)
- [论文来源归档](../papers/agenticroborl_arxiv_2610_28589.md)
- [代码仓库归档](../repos/agenticroborl.md)
