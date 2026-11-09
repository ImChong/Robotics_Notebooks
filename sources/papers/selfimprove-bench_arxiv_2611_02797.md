# SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents｜自我改进智能体的泛化评测

> 原论文归档（2026-11-09，已覆盖 v1 正文与附录；官方仓库静态核对，未独立重跑实验）

- **类型：** paper
- **原文：** <https://arxiv.org/abs/2611.02797>
- **项目：** <https://theseus-labs-rsi.github.io/SelfImprove-Bench/>
- **代码：** <https://github.com/theseus-labs-rsi/SelfImprove-Bench>
- **入库日期：** 2026-11-09
- **核对范围：** arXiv HTML 全文：§1 引言、§2 RSI 相关工作、§3 进化协议/任务分类/G0–G4/语义过程分析、§4 主要实验与消融、§5 结论、References；附录 A（任务、回放、反馈、标注）、B（Farm/Combat/Sweepmine/Hero/Minecraft/Astray 游戏任务与生成器）、C（典型失败和 RSI 修订案例）。
- **通读：** 已完成

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

## 论文全文复核要点（v1）

- **能力与范围（§3.2、附录 B）：** 三项主能力为 Exploration、Reasoning、Planning；六类游戏 Farm、Combat、Sweepmine、Hero's Journey、Minecraft、Astray。前四种支持难度 1–5，后两种使用 GameWorld 固定保留任务。
- **五层泛化（§3.3）：** G0 对相同历史实例的 replay 比较（不触发 refine）；G1 同难度新实例；G2 同游戏更高难度；G3 同能力类别中另一游戏；G4 跨类别迁移。**不要把 G0 称为未见实例泛化**。
- **保持测试独立（§3.1、附录 A.1–A.3）：** 训练候选验证只使用 held-in 任务；held-out 仅测试已保存的改进状态，禁止在其上修订。Replay 由 interval 和 lag 定义，不计入训练任务数；不同种子控制实例生成，去重是 descriptor/hash 层面的而非游戏真实状态严格唯一。
- **主结果（§4.2，表 2）：** 18 个方法×游戏组合中，**8 个 G0 复测低于历史表现**，但 **16 个 G1 HCI 为正**；G1 改善不能保证 G2–G4 改善。Hero→Sweepmine 能改善而反方向退步，说明 transfer 方向敏感。HyperAgents 在跨能力八个方向中五个 HCI 为正。
- **修订归因（§3.4、§4.3）：** 对最终保留的 915 条 refinement opinions 按语义、适用范围、目标功能、修改面标注；ACE、AHE、HyperAgents 中类别通用/通用的比例分别为 14.2%、7.8%、36.5%。主实验保留修改中 87.9% 指向 prompt/memory、12.1% 指向 harness，未观察到对 improver 本身的改写。
- **消融（§4.4）：** 随机中等难度训练在部分 Combat/Hero 的更难 holdout 上优于逐级增难；直接修订 DeepSeek Harness 的 HyperDSH 是探索性实验，和主实验元级修改对象不同。
- **证据限制（附录 A.4）：** 100 条样例的人工辅助复核是带机器标签的事后审阅，Scope 88%、Surface 99% 等一致性**不是盲测分类准确率**。

## 对 wiki 的映射

- [SelfImprove-Bench 主实体](../../wiki/entities/paper-selfimprove-bench.md)
- [官方站点归档](../sites/selfimprove-bench.md)
- [代码仓归档](../repos/selfimprove-bench.md)
