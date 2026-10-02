# Flatness-Preserving Residual Learning for Real-Time Tight Quadrotor Formation Flight

> 来源归档（ingest）

- **标题：** Flatness-Preserving Residual Learning for Real-Time Tight Quadrotor Formation Flight
- **类型：** paper / multi-robot / aerial / control / system-identification
- **arXiv abs：** <https://arxiv.org/abs/2607.12275>
- **PDF：** <https://arxiv.org/pdf/2607.12275>
- **视频：** <https://www.youtube.com/watch?v=uF26IkRFQMk>
- **机构：** 宾夕法尼亚大学 GRASP Laboratory
- **发表 / 上传：** 2026-07-14（arXiv）
- **入库日期：** 2026-10-02
- **索引来源：** [IROS 2026 九篇获奖盘点](../blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)

## 开源状态（步骤 2.5，2026-10-02）

- **待发布：** arXiv 与 YouTube 可访问；**截至入库日无 GitHub/项目页代码链接**。

## 摘录 1：问题

- 紧密编队飞行受下洗等气动干扰；未建模会导致碰撞风险。

## 摘录 2：方法

- 在刚体标称动力学上学习 **physics-informed 残差**，并约束残差只依赖编队位置/速度，使联合系统保持 **微分平坦性**。
- **反馈线性化 + 前馈** 补偿气动扰动；先验下洗模型 + 小网络学剩余误差。

## 摘录 3：指标（文内/论文摘要）

- ~28 s 飞行数据训练；**5 ms** 控制周期；平均跟踪误差较标称 **−31%**；算力约为 NMPC **一个数量级** 更低。

**对 wiki 的映射：** [`wiki/entities/paper-flatness-preserving-quadrotor-formation.md`](../../wiki/entities/paper-flatness-preserving-quadrotor-formation.md)

## 当前提炼状态

- [x] 步骤 2.5 开源核查
- [x] 升格 wiki 实体页
