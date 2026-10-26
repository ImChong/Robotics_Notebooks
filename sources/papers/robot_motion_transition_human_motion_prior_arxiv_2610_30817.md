# Robot Motion Transition with a Human Motion Prior（STAMP，arXiv:2610.30817）

> 来源归档（ingest）

- **标题：** Robot Motion Transition with a Human Motion Prior
- **简称：** STAMP（Spatio-Temporal Alignment with a Motion Prior）
- **类型：** paper / humanoid / motion-transition / motion-prior / motion-tracking
- **arXiv：** <https://arxiv.org/abs/2610.30817>（v2；PDF：<https://arxiv.org/pdf/2610.30817>）
- **项目页：** <https://childtoy.github.io/STAMP-project/>
- **作者：** Taehyun Byun、Taemoon Jeong、Yoonbyung Chai、Woojin Wie、Insu Park、Kyungjae Lee、Sungjoon Choi
- **机构：** 高丽大学（Korea University）；乐百机器人（ROBOTIS）
- **时间：** 2026-10-22 提交 v1；2026-10-23 更新 v2
- **一句话说明：** STAMP 先在共享人体动作空间内为两段独立机器人动作选择空间位置、朝向和过渡时长，再用人体动作先验生成中间动作，最后重定向回机器人并训练序列跟踪策略。

## 项目页与源码状态

- **项目页核查（2026-10-26）：** 项目页介绍了 Robot-to-Human-to-Robot 流程、方法、G1/K1 结果与 K1 真机演示。页面显示 Code 按钮，但当前可读取页面未提供可核实的仓库 URL。
- **结论：** 截至入库日，公开源码链接**未核实**；这不等同于断言项目未开源。项目页也未列出独立数据集下载链接。

## 摘录 1：问题与方法（论文摘要、项目页 Overview）

- **问题：** 独立获得的动作片段单独可跟踪，但直接连接时，片段边界的相对位置、朝向和时间安排可能令插值产生滑步或令生成模型补出多余位移与转向。
- **方法：** STAMP 将动作从机器人重定向到共同的人体骨架；学习到的对齐模型依据两段动作的边界上下文，对下一段动作的候选位置、朝向与过渡时长排序；人体动作先验在选定边界条件下补全过渡，完整动作再重定向回机器人。
- **对 wiki 的映射：** 升格到 [STAMP 论文实体页](../../wiki/entities/paper-stamp.md)，并与 [BeyondMimic](../../wiki/methods/beyondmimic.md)、[运动重定向 GMR](../../wiki/methods/motion-retargeting-gmr.md) 互链。

## 摘录 2：评测设置与结果（项目页转引论文 Tables II–III）

| 平台 | STAMP | 插值 | 其他基线 |
|---|---:|---:|---|
| Unitree G1 | 80.69% | 69.59% | MotionBricks 56.91%；Kimodo† 50.69% |
| ROBOTIS K1 | 87.52% | 84.70% | Kimodo† 64.55% |

- **设置：** 18 对动作片段；每个动作对、方法与机器人平台进行 1,024 次 rollout。
- **指标解读：** 各方法针对自身生成的参考序列评估跟踪成功率，衡量的是可跟踪性；不是与一条共同真实过渡动作的相似度。
- **真机演示：** 项目页展示 ROBOTIS K1 的跑步接出拳击，以及较长动作组合。

**对 wiki 的映射：** 实体页「实验与评测」保留平台、基线与指标边界；「结论」强调对齐选择相较直接插值的收益及其评估口径。

## 摘录 3：适用场景与复现边界

- STAMP 适用于将动作库中的独立片段编排成长时序动作；生成前先改变边界条件，能让动作先验有机会产生合理的连接动作。
- 项目页称每个完整组合序列（含生成的过渡）会单独作为参考，训练对应的跟踪策略（使用 BeyondMimic）。
- 复现判断：项目页没有提供可核实的公开代码仓库地址，因此暂不记录代码运行入口；真机证据以 K1 演示为主。

**对 wiki 的映射：** 实体页「工程实践」与「局限与风险」明确序列策略训练需求及源码链接未核实状态。

## 参考链接

- [arXiv 论文页](https://arxiv.org/abs/2610.30817)
- [STAMP 项目页](https://childtoy.github.io/STAMP-project/)
