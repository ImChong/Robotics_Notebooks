# SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents｜自我改进智能体的泛化评测

> 原论文归档（截至 2026-11-09：核查过论文摘要、官方项目说明和代码 README；全文逐节核对尚未完成）

- **类型：** paper
- **原文：** <https://arxiv.org/abs/2611.02797>
- **项目：** <https://theseus-labs-rsi.github.io/SelfImprove-Bench/>
- **代码：** <https://github.com/theseus-labs-rsi/SelfImprove-Bench>
- **入库日期：** 2026-11-09
- **核对范围：** 论文摘要与方法/评测选段、官方项目页、主仓 README 及相关运行入口；附录和全文尚未逐页完成核对。
- **通读：** 未完成

## 研究问题与核心事实

- **论文：** <https://arxiv.org/abs/2611.02797>
- **方法：** 将 RSI 改进按 G0–G4 五级泛化测试而非只看训练曲线上升；生成任务难度、跨任务和跨类别保留集。
- **方法比较：** ACE / AHE / HyperAgents 和固定 Agent 基线，训练与 holdout 分离。
- **评估：** 分析收益、HCI、反馈轨迹与修订对象的范围，刻画改进有无真正转移。
- **边界：** 研究以六类游戏为可控环境，不是现实硬件或真实人机交付评测。

## 方法与评测摘录

- **五级泛化阶梯**：G0 同一实例重放；G1 同难度新实例；G2 同任务的新难度；G3 同一能力类别的其他任务；G4 跨能力类别任务。难度随层级增加，但不同游戏分数不可无条件横比。
- 覆盖 **6 个游戏**：Farm、Combat、Sweepmine、Hero's Journey、Minecraft、Astray，组织在探索、规划和推理等能力类别下。
- 对比 **ACE**（经验层）、**AHE**（harness 层）与 **HyperAgents**（元层）等 RSI 方法，也提供 Codex、Claude Code 等固定系统基线。
- 训练阶段采用任务序列和中途反馈；holdout 使用独立任务且**不继续自我修改**，防止将测试学习当作泛化。
- 通过同配置初始框架与改进后框架的对照、HCI 与轨迹归因分析，检查改进是不是稳定和可迁移。

## 复现与保留意见

这是一套以游戏环境为主的 RSI 评测，不能直接推出机器人真机泛化；HyperAgents 内嵌代码有非商业许可，主仓 MIT 不覆盖全部三方依赖。

本归档将已查过的一手材料与尚待全文复核的结论分开；未在本次会话实际运行论文实验，不应把代码入口的存在解释为分数已复现。

## 对 wiki 的映射

- [SelfImprove-Bench 主实体](../../wiki/entities/paper-selfimprove-bench.md)
- [官方站点归档](../sites/selfimprove-bench.md)
- [代码仓归档](../repos/selfimprove-bench.md)
