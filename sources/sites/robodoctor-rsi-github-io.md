# RoboDoctor 官方项目页

- **类型**：研究项目页 / 结果与运行记录展示
- **收录日期**：2026-10-22
- **项目页**：<https://robodoctor-rsi.github.io/>
- **官方代码入口**：<https://github.com/THUSI-Lab/RoboDoctor-Preview>
- **论文**：截至收录日，项目页将 Paper / arXiv 标为待发布，未提供论文链接
- **开源状态**：部分公开。项目页公开方法说明、任务结果和运行记录；代码按钮所指的官方 Preview 仓库当前为空（GitHub 仓库大小为 0），尚不能据此复现运行时。

## 一句话

RoboDoctor 是一个训练免费（training-free）的机器人策略自我改进系统：在冻结低层策略的前提下，基于观测诊断任务困难，适时改用 Agent 控制或修正策略动作，并复用经过验证的操作经验。

## 项目页核查摘录（2026-10-22）

官网把问题拆成策略与执行两个层面：先通过 Warmup 判断低层策略是否适合当前任务；若策略仍有用，则采用 Agent 辅助的 Hybrid 控制；若语义动作策略更适合，则切换到 Direct Agent 控制。运行时可检查策略提出的动作、利用视觉与深度几何定位目标，并在动作后重新观测验证结果。经验记录包含诊断、介入和操作结果，供后续回合复用。

## 官网公布的 RoboDojo 结果

官网的 `data/task-results.json` 标注更新日期为 2026-10-21，列出 54 项任务及五个维度。Warmup 平均分为 **49.74/100**，成功率 **40.83%**，有效 episode **162/162**；RSI 平均分为 **57.10/100**，成功率 **49.44%**，有效 episode **301/324**，其中 **46/54** 个任务达到完整 episode 数。按任务汇总的指标不应被读作每个任务都达到相同覆盖度。

官网的对照图同时显示已发表方法的 RoboDojo 分数；但其来源、样本覆盖和评估时间不完全相同，因此不能仅按柱高断言严格同条件排名。以上是项目团队公布的实验数据，尚非独立复现结论。

## 对 wiki 的映射

- 主实体：[RoboDoctor](../../wiki/entities/robodoctor-rsi.md)
- 代码入口归档：[RoboDoctor-Preview](../repos/thusi-lab-robodoctor-preview.md)
- RSI 背景：[递归自我改进](../../wiki/concepts/recursive-self-improvement.md)

## 参考来源（原始）

- 项目主页与方法说明：<https://robodoctor-rsi.github.io/>
- 官网实验数据：<https://github.com/RoboDoctor-RSI/robodoctor-rsi.github.io/blob/main/data/task-results.json>
- 官网数据说明：<https://github.com/RoboDoctor-RSI/robodoctor-rsi.github.io/blob/main/data/README.md>
