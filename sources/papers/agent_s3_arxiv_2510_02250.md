# Agent S3: Scaling Agents for Computer Use（arXiv:2510.02250）

> 来源归档（paper / 原始论文）

- **标题：** Agent S3: Scaling Agents for Computer Use
- **论文：** <https://arxiv.org/abs/2510.02250>（[HTML v2](https://arxiv.org/html/2510.02250v2)、[PDF](https://arxiv.org/pdf/2510.02250)）
- **机构：** Simular Research（以论文署名为准）
- **项目／代码：** <https://github.com/simular-ai/Agent-S/tree/main/gui_agents/s3>
- **官方技术文章：** <https://www.simular.ai/articles/agent-s3>
- **版本：** arXiv v2（2026-02-03）；论文报告于 TMLR 2026
- **入库日期：** 2026-11-08
- **核对范围：** 原始论文 HTML v2 全文：摘要、引言、方法（Agent S3 / Behavior Judge）、OSWorld 与跨平台实验、消融、局限、附录 A–H；数字以论文表格与官方仓库 README 对照。
- **通读：** 已完成
- **对 wiki 的映射：** [Agent S3 统一实体页](../../wiki/entities/paper-agent-s3.md)

## 原文要点与证据边界

### 1. Agent S3：扁平的截图驱动 Worker

- 不使用 Agent S2 的多层次 manager-worker 规划，减少跨模块调用；单个 Worker 反复消费任务和截图，生成下一 GUI 操作，必要时调用反思模块和可选本地 Code Agent。
- 以屏幕 UI 为主要执行界面；模型生成动作通过 grounding 转换为可执行的鼠标／键盘操作。
- **对 wiki 的映射：** 核心机制、源码时序图；与机器人低层实时控制不是同一类问题。

### 2. Behavior Judge（BJudge）：选择完整轨迹，不是每步投票

- 为同一任务启动可独立重置的多条执行轨迹；Behavior Narrator 结合动作前后截图、操作位置与局部放大信息，生成基于画面的事实叙述。
- Comparative Judge 读取候选轨迹的起始／最终截图和叙述，选择最可能完成任务的一条。
- 多次运行的**平均单轨能力**和多次采样后**择优能力**必须分开报告；该方法要求隔离环境，否则轨迹可能相互污染。
- **对 wiki 的映射：** Mermaid 流程图、评测边界与部署限制。

### 3. 论文报告的 OSWorld 结果（100 步设置）

| 指标 | 结果 | 口径 |
|---|---:|---|
| Agent S3 + GPT-5，单轨平均 | 62.6% | 论文 Table 1，10 次运行平均 |
| Agent S3 + GPT-5，BJudge 10 条 | 69.9% | 多轨迹选择 |
| Agent S3，GPT-5 + Claude Opus 4.5 混合各 5 条，BJudge 10 条 | **72.6%** | 最强组合；不能当作单次成功率 |
| 人类基线 | 72.36% | OSWorld 原始比较基线（约 72%） |
| GTA1 + GPT-5 | 63.4% | 原文列出的既有结果 |

- **注意版本差异：** 项目 README 后续报告单独 Agent S3 的 100-step OSWorld 成绩 **66%**；论文 v2 Table 1 的 62.6% 为所述多次运行平均，不能直接混写成相同评测数据。
- **WindowsAgentArena：** 单轨 50.2% → 选择 3 条 56.6%；**AndroidWorld：** 单轨 68.1% → 选择 3 条 71.6%；二者不等于 OSWorld 上 72.6% 的泛化保证。
- **消融：** BJudge 关键不只是最后截图；动作前后可核对的行为事实叙述有助于轨迹区分。

### 4. 开放与局限

- 论文成绩依赖底座模型、UI grounding、独立环境重置和 BJudge，**不是**“下载安装就免费获得 72.6%”。
- 单桌面不易安全复现 N 条并行轨迹；共享文件、邮箱、云端服务等外部副作用不一定可重置。
- 论文分析包含判断事实错误、图形界面与代码代理交接错误等失败模式；基准上的“超过人类平均”不代表所有真实办公场景超人类。
- **对 wiki 的映射：** 开源状态、复现路径、安全风险。

## 交叉来源

- [Agent-S 官方仓库核查](../repos/simular_agent_s3.md)
- [Agent S3 官方文章](../sites/simular_agent_s3.md)
- [Agent S3 wiki 实体页](../../wiki/entities/paper-agent-s3.md)
