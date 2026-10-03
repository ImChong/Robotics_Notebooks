# EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation

> 来源归档（ingest）

- **标题：** EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco-Manipulation
- **缩写 / 框架：** **EgoAlign**
- **类型：** paper / humanoid / loco-manipulation / robot-free-demonstration / VLA
- **arXiv：** <https://arxiv.org/abs/2609.38046>（v2，2026-09-30）
- **论文 HTML：** <https://arxiv.org/html/2609.38046v2>
- **项目页：** <https://lambdahumanoid.github.io/EgoAlign/>（归档见 [sources/sites/egoalign-project.md](../sites/egoalign-project.md)）
- **代码仓：** <https://github.com/LambdaHumanoid/EgoAlign>（截至 2026-10-03 仅项目博客与媒体；代码待发布，归档见 [sources/repos/egoalign.md](../repos/egoalign.md)）
- **作者：** Yiming Jiang、Jin Chen、Chongyang Xu、Yilun Chen、Aimin Hao、Yisheng He
- **机构：** 北京航空航天大学（Beihang University）；上海创智学院（Shanghai Innovation Institute）；四川大学（Sichuan University）；阿里巴巴集团（Alibaba Group）
- **提交日期：** 2026-09-29；**更新日期：** 2026-09-30
- **入库日期：** 2026-10-03
- **一句话说明：** 把第一视角人类示范适配为机器人可用的动作与状态监督：用目标机器人模型、SONIC–MuJoCo 闭环修正人体动作，再回放重建控制器所需的机器人状态，最终仅用人类任务数据微调 VLA，并在 G1 上零样本执行长程移动操作。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-03）：** 项目页链接到公开的 [LambdaHumanoid/EgoAlign](https://github.com/LambdaHumanoid/EgoAlign)。该仓 README 说明仓内目前是项目博客和媒体资源，代码将在未来发布；当前未提供可运行的训练、重定向或部署实现，也未找到数据集下载链接。
- **结论：** 项目页和占位仓已公开；**方法代码与数据待发布**。复现结论以官方仓后续发布为准。

## 摘录 1：问题与方法

- 人体示范缺少策略训练需要的目标机器人状态历史；人体与人形机器人的身材比例、控制器响应也会让手部轨迹执行偏移。
- EgoAlign 以目标机器人模型和连续全身控制器为接口，通过动作适配与因果状态重建，将人类观测和机器人可执行动作、状态配对。
- **对 wiki 的映射：** [paper-egoalign 实体页](../../wiki/entities/paper-egoalign.md)；联系 [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)。

## 摘录 2：数据构造管线

- **采集：** PICO 4 Ultra 头显与五个追踪器记录人体动作；两台 GoPro 分别提供向下近场视角与水平远距离视角。SONIC–MuJoCo 实时反馈使采集者能看到机器人实际运动并补大被控制器抑制的动作。
- **适配：** 先做面向 G1 身体比例的上身运动学尺度对齐，再用 SONIC–MuJoCo 回放误差做最多两轮闭环 refinement；保持全局移动和下肢参考，重点校正手部接触几何。
- **重建：** 以最终适配动作因果回放，在每个控制 tick 先记录机器人状态与状态历史，再记录同一时刻的动作 token，避免用当前动作生成的未来状态作为当前监督。
- **对 wiki 的映射：** [paper-egoalign 实体页](../../wiki/entities/paper-egoalign.md)，其流程图呈现采集、适配、重建和训练/部署关系。

## 摘录 3：训练与真机评测

- 在原始人类图像和语言指令上微调 π₀.₅ 架构；动作监督包含 64 维 SONIC 动作 token 和左右手命令。部署时由 SONIC 根据实时本体状态历史解码 token。
- Unitree G1 真机任务包括约 10 米篮筐搬运、多位置导航和脚踩垃圾桶踏板。每个设置 20 次试验；目标位分为训练位置和偏移 0.5 米的未见位置。
- 完整移动操作在直接目标上的成功率为 65%，多位置 seen/unseen 均为 40%；导航在 direct / seen / unseen 为 65% / 70% / 65%，脚踩交互在 seen / unseen 均为 60%。
- **对 wiki 的映射：** [paper-egoalign 实体页](../../wiki/entities/paper-egoalign.md)的「评测与实验证据」与「结论」。

## 当前提炼状态

- [x] 论文正文与 v2 版本核对
- [x] 项目页与代码仓开源状态核查
- [x] 摘录和 wiki 映射填写

