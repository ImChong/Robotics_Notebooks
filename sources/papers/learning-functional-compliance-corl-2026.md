# Learning Functional Compliance in Whole-body Manipulation｜全身操作中的功能性柔顺学习

- **类型：** paper
- **标题：** Learning Functional Compliance in Whole-body Manipulation
- **作者：** Gio Huh, Yue Zhuo, Zehua Wang, Alan Yu, Yibo Cheng, Yanbing Han, Gabriel B. Margolis, Ge Yang
- **机构：** FortyFive Labs、麻省理工学院、斯坦福大学、加州理工学院（用户提供，待论文首页核对）
- **会议：** CoRL 2026，Poster（作者主页确认录用）
- **原文：** <https://openreview.net/forum?id=wALhOwQWnx>
- **PDF：** <https://openreview.net/pdf/aced2ffb6232ae3ddfe52e5e013b4fbd5e5e4899.pdf>
- **作者主页：** <https://zehuaw1.github.io/>
- **摘要检索线索：** <https://hongsong-wang.github.io/CoRL2026/>
- **入库日期：** 2026-11-25
- **版本：** CoRL 2026 收录记录；PDF 版本待核实。
- **核对范围：** 作者主页的论文记录、会议目录转载的摘要；OpenReview 论坛和 PDF 均触发浏览器验证，未取得正文。
- **通读：** 未完成
- **开源状态：** 未核实；未找到可确认的官方项目页、代码或权重入口。
- **详情：** [功能性柔顺学习（草稿）](../../wiki/entities/paper-learning-functional-compliance.md)

## 摘要线索（非全文核对）

- 问题：全方向统一降低刚度会削弱接触任务需要的控制能力。
- 方法：根据任务方向和执行阶段调整柔顺性；同时学习柔顺位姿奖励塑形的控制策略与外力估计器。
- 评测：摘要报告 Unitree G1 的钥匙插入、书本重定向和开柜门任务；前两项优于对照，开柜门与最强基线相当，动作跟踪精度与先进方法相当。具体数值、试验次数与基线待全文核验。

## 对 wiki 的映射

- [功能性柔顺学习](../../wiki/entities/paper-learning-functional-compliance.md)：保留论文身份与待核对问题。
- [移动操作](../../wiki/tasks/loco-manipulation.md)：接触中方向性与时变柔顺。
- [接触丰富操作路线](../../roadmap/depth-contact-manipulation.md)：Stage 3 的任务级柔顺控制候选阅读。

## 待核对

- 训练奖励、力估计器输入输出、柔顺参数表示与部署接口。
- 实验指标、消融、真实机器人条件、论文自述局限与源码状态。
- 全文读取后再补项目流程图与（若有可运行代码）源码运行时序图。
