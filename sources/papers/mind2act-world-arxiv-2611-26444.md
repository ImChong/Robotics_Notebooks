# Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation｜Mind2Act World：具身认知、物理动作与组合能力诊断基准

- **类型：** paper / embodied manipulation / diagnostic benchmark
- **论文：** <https://arxiv.org/abs/2611.26444>；[HTML 全文](https://arxiv.org/html/2611.26444)（arXiv:2611.26444v1，2026-11-19）
- **项目：** <https://mind2act-world.com/>
- **排行榜：** <https://mind2act-world.com/leaderboard.html>
- **诊断卡：** <https://mind2act-world.com/diagnosis.html>
- **代码：** <https://github.com/Mind2Act/Mind2Act_World>
- **训练数据：** <https://huggingface.co/datasets/Mind2Act-World/mind2act-train>
- **入库日期：** 2026-11-22
- **核对范围：** 通读 arXiv v1 HTML 正文 §1–§8，核查附录 A–F 的平台、预算、训练设置、完整分项表、诊断规则及 30 个任务卡；交叉阅读官方 GitHub README、docs/source-layout.md、docs/agent-runtime.md、eval/cli.py、eval/runner/__init__.py、eval/batch.py、eval/scoring.py。
- **通读：** 已完成
- **沉淀到 wiki：** [Mind2Act World 论文实体](../../wiki/entities/paper-mind2act-world.md)

## 原文摘录与核验

1. **问题：** 传统基准常把「想对（Mind）」与「做到（Act）」混入同一个最终成功率，或分别测试却缺少对后续物理反馈驱动决策（Mind2Act）的显式验证（§1、§3）。
2. **任务：** 30 个任务 / 12 种能力类别 / 5 类应用域。每个任务设 Easy、Medium、Hard 和 Clean/Random 变体，组成 180 个“任务×等级×场景设置”配置。每个视角各 10 项：Mind 强调历史身份、顺序、位置和规则记忆；Act 强调动态响应、持续接触、轨迹精度；Mind2Act 强调前置条件、状态验证和失败修复（§3、附录 F）。
3. **物理判据：** 使用场景物理量（如电阻读数、接触、释放后稳定、事件顺序、温度）判断任务完成。分项部分进度得分与二元 success 分开；任务 0–100 分；整体 Score 按 0.25×Mind + 0.25×Act + 0.50×Mind2Act 聚合，而整体 SR 是成功 episode 的比例（§4、附录 B）。
4. **环境：** Isaac Sim 5.1 + ARX X5 双臂和双平行夹爪、头部及左右腕 RGB；物理 120 Hz，动作目标 30 Hz。仿真在策略推理时暂停，这**不能证明** agent 的真实世界反应延迟符合物理任务时限（§4、§7、附录 A）。
5. **样本：** 4,500 条专家示范、75.8 小时；9 个系统：3 个工具 Agent、4 个 VLA、2 个 WAM。Agent 每系统 900 个 episode（每配置 5），端到端系统每个 9,000（每配置 50），合计 56,700（§3–§5）。
6. **对比：** 论文中 GPT-6 Astra（Agent）的 Mind / Act / Mind2Act SR 分别为 52.7% / 39.0% / 12.0%；端到端策略的 Mind2Act SR 均低于 3.6%。这体现组合评测中的能力缺口，不等于该模型在开放世界机器人中只具备对应百分比的能力（§5）。
7. **协议差异：** Agent 拥有四倍 physics budget、至多 150 决策和运动规划工具；端到端策略以示教的 90 分位轨迹长度为基准乘 1.2 倍预算，故分别上榜，禁止跨榜直接认定某架构更强（§4、附录 B）。
8. **诊断：** 利用阶段/事件规则识别首个阻断点，辅以 LLM judge 分类 Mind/Act/Composition 并生成模型诊断卡；作者明确该归因是诊断性而非严谨因果实验（§6、§7、附录 E）。
9. **局限：** 单一仿真双臂平台、固定英语指令、有限的模型与任务类别样本，未验证真实机体迁移；成功示教缺少失败后修复轨迹，容易放大组合/修复的训练数据盲区（§6–§7）。

## 对 wiki 的映射

- 独立节点：[Mind2Act World](../../wiki/entities/paper-mind2act-world.md)
- 纵深路线：[具身模型测评 Stage 3](../../roadmap/depth-embodied-eval.md)
- 关联评测：[RoboDojo](../../wiki/entities/robodojo.md) 与 [操作任务](../../wiki/tasks/manipulation.md)
- 代码和发布边界见 [仓库归档](../repos/mind2act-world.md)、[项目页归档](../sites/mind2act-world.md)。

**记录原则：** 论文数字对应 v1 固定实验样本；线上排行榜可更新，不将其视为与论文表格始终一致。