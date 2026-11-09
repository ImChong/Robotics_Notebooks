# SelfImprove-Bench｜官方项目页

> 来源归档（官方站点）

- **标题：** SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents
- **类型：** site
- **链接：** <https://theseus-labs-rsi.github.io/SelfImprove-Bench/>
- **论文：** <https://arxiv.org/abs/2611.02797>
- **代码：** <https://github.com/theseus-labs-rsi/SelfImprove-Bench>（已公开，范围见代码归档）
- **数据：** <https://huggingface.co/datasets/DreamEternal/SelfImprove-Bench-Cases>
- **入库日期：** 2026-11-09
- **核对范围：** 官方网站的项目说明、论文/代码与数据入口；配合代码主仓 README 验证开放范围。
- **通读：** 未完成

## 关键信息

- **五级泛化阶梯**：G0 同一实例重放；G1 同难度新实例；G2 同任务的新难度；G3 同一能力类别的其他任务；G4 跨能力类别任务。难度随层级增加，但不同游戏分数不可无条件横比。
- 覆盖 **6 个游戏**：Farm、Combat、Sweepmine、Hero's Journey、Minecraft、Astray，组织在探索、规划和推理等能力类别下。
- 对比 **ACE**（经验层）、**AHE**（harness 层）与 **HyperAgents**（元层）等 RSI 方法，也提供 Codex、Claude Code 等固定系统基线。
- 训练阶段采用任务序列和中途反馈；holdout 使用独立任务且**不继续自我修改**，防止将测试学习当作泛化。
- 通过同配置初始框架与改进后框架的对照、HCI 与轨迹归因分析，检查改进是不是稳定和可迁移。

## 开放与复现边界

这是一套以游戏环境为主的 RSI 评测，不能直接推出机器人真机泛化；HyperAgents 内嵌代码有非商业许可，主仓 MIT 不覆盖全部三方依赖。

## 对 wiki 的映射

- [SelfImprove-Bench 主实体](../../wiki/entities/paper-selfimprove-bench.md)
- [论文归档](../papers/selfimprove-bench_arxiv_2611_02797.md)
- [代码仓归档](../repos/selfimprove-bench.md)
