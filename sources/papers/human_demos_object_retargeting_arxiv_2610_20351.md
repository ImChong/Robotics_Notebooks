# Human Demonstrations as Post-Training Data for Any Robot Body: Retargeting the Object, Not the Hand（arXiv:2610.20351）

> 来源归档（ingest）

- **标题：** Human Demonstrations as Post-Training Data for Any Robot Body: Retargeting the Object, Not the Hand
- **类型：** paper / human demonstration / data pipeline / cross-embodiment / object-centric retargeting / simulation feasibility
- **arXiv：** <https://arxiv.org/abs/2610.20351>（v1，提交 2026-10-14，cs.RO；HTML：<https://arxiv.org/html/2610.20351v1>；补充材料：<https://arxiv.org/src/2610.20351v1/anc/supplement.pdf>）
- **项目页：** <https://playground.dxtr.co/>（展示部分 episode 与逐机器人结果；提供 URDF 提交入口）
- **代码：** 未列（arXiv 与项目页均未给代码仓库）
- **作者：** Nagasrikanth Kallakuri（单作者）
- **机构：** 未核实（页面未列机构）
- **入库日期：** 2026-10-15
- **一句话说明：** 用手机采集的人类示范记录物体位姿、接触区间与三维物体，而非重定向人手；再对每个 URDF 描述的机器人做仿真可执行性筛选，逐 robot-episode 给出通过、命名失败模式或拒绝。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 无 GitHub、代码或数据下载链接，无 "coming soon" 文字；页面提供的是「发送 URDF 与 mesh，团队跑同一流程并返回通过/拒绝及原因」的服务入口，未给联系方式或提交链接。
- **arXiv 页：** 未给代码或数据链接；补充材料为 PDF。
- **结论：** **未开源**（代码与训练数据均未发布；项目页仅展示子集结果）。

## 摘录 1：问题与主张（摘要）

- **问题：** 人类示范数据通常以人手为中心做重定向，换一个机器人本体就要重做。
- **主张：** 以物体运动为中心记录示范（物体位姿随时间、接触区间、重建的 3D 物体），与人体结构解耦；在每个目标机器人上单独检查可执行性。
- **流程：** 每个 capture 与 URDF 机器人的每个 episode 组合在运动前给出判定：通过 / 命名失败模式 / 拒绝。

## 摘录 2：结果（摘要）

- **样本：** 42 个 capture，8 个机器人，157 个 robot-episode 组合。
- **结果：** 约 31% 通过，52% 失败，14% 被拒绝，2.5% 不可审计。
- **拒绝项校验：** 13 个被拒绝但仍强行执行的组合全部失败，作者以此支持拒绝机制。
- **预测能力：** 仅凭机器人 URDF 描述预测结果，几乎不比随机猜测好。
- **边界：** 论文明确未用这些记录训练任何策略，真机迁移未测试。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-human-demos-object-retargeting.md`](../../wiki/entities/paper-human-demos-object-retargeting.md)：流程总览（采集→物体运动→逐机器人可执行性判定），强调「当前证据是仿真运动学筛选，不是策略效果」。
- 开源状态写为「未开源」；后续若出现训练结果或数据集，需重新核查。
