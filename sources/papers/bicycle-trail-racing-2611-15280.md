# Autonomous Trail Racing with a Bicycle Robot｜自行车机器人自主林间赛道竞速

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.15280 （v1）
- **HTML：** https://arxiv.org/html/2611.15280v1
- **PDF：** https://arxiv.org/pdf/2611.15280
- **项目：** https://pages.rai-inst.com/autonomous-trail-racing/
- **代码：** https://github.com/rai-opensource/sem_map
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-bicycle-trail-racing-2611-15280.md)

## 方法与评测核对

- **研究问题：** 将可通行林道语义编码进地形并联合训练局部导航与骑行控制。
- **方法原理：** 训练时在程序化地形叠加 trail 语义与难度课程；部署时 SemMap 把 RGB 可通行性与 LiDAR 高程融合成语义地形图，策略直接以局部地图驱动骑行控制与行进路径选择。
- **原文结果：** 官方项目报告 959 米真实山地林道、平均 3.26 m/s；不同模拟竞速与实地道路指标不能混读。
- **局限与范围：** 高草和植被可能制造虚假障碍，岔路选错与轮胎失去抓地力会影响安全；公开 SemMap 感知仓不是整个骑行控制栈。
