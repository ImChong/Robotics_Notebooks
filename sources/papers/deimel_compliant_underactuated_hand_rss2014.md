# A Novel Type of Compliant, Underactuated Robotic Hand for Dexterous Grasping

> 来源归档

- **标题：** A Novel Type of Compliant, Underactuated Robotic Hand for Dexterous Grasping
- **类型：** paper（hardware design）
- **出处：** RSS 2014 · 原始会议论文；IJRR 2015 扩展版
- **RSS 2014 DOI：** <https://doi.org/10.15607/rss.2014.x.018>
- **IJRR DOI：** <https://doi.org/10.1177/0278364915592961>
- **作者：** Raphael Deimel, Oliver Brock（TU Berlin，Robotics and Biology Lab）
- **入库日期：** 2026-07-20
- **奖项：** RSS 2026 Test of Time Award
- **一句话说明：** 高度顺应性欠驱动气动拟人手；少量气动执行器通过自然机械耦合 + 接触力学自适应抓取；轻量安全廉价；奠定后续软体手（RBO Hand 系列）研究基础；RSS 2026 Test of Time Award 肯定十二年影响。

---

## 核心摘录（策展，非全文）

### 核心设计哲学

- **问题：** 全驱动灵巧手昂贵、重、控制复杂、易损坏 → 实用化困难。
- **洞见：** 通过 **机械顺应性** 让接触力自然调节手指关节构型，极少驱动器即可实现多样抓取。
- **关键原则：** "把接触从干扰变成驱动力"——接触几何决定抓取形态，无需精确规划每个关节。

### 技术要点

| 维度 | RBO Hand 1（本文） |
|------|-------------------|
| 执行器类型 | PneuFlex 气动弹性执行器 |
| 执行器数量 | 极少（约 5 路气压，对应拇指 + 四指分组） |
| DOF 数 | 远多于执行器数（欠驱动） |
| 材料 | 硅橡胶 + 弹性纤维增强 |
| 制造 | 注模；约 $300 原材料成本 |
| 控制 | 气压值控制弯曲幅度；精细构型由接触被动调节 |
| 安全性 | 本质柔软，抗冲击，接触力分散 |

### 抓取能力

- 验证了 Feix 等分类中多种抓取类型（圆柱抓、指尖抓、侧捏、钩状等）
- 成功抓取不同形状、大小的物体（评测使用标准抓取物体集）

### 影响与后续

- 直接催生 RBO Hand 2（IJRR 2015，更完整评测）、RBO Hand 3（Handed Shearing Auxetics，2017+）
- 影响了软体机器人、欠驱动手、柔顺抓取等多个方向十余年的研究路线
- RSS 2026 Test of Time Award（距原始发表 12 年）

### 局限（论文自述）

- 抓持力有限（顺应性 vs 力量 trade-off）
- 气动系统需气泵基础设施
- 难以精确建模，规划依赖接触反馈

### 对 wiki 的映射

- [paper-deimel-compliant-underactuated-robotic-hand](../../wiki/entities/paper-deimel-compliant-underactuated-robotic-hand.md)
- [manipulation](../../wiki/tasks/manipulation.md)
- [midas-hand](../../wiki/entities/midas-hand.md)
- [contact-rich-manipulation](../../wiki/concepts/contact-rich-manipulation.md)

## 参考来源（原始）

- RSS 2014 论文：<https://doi.org/10.15607/rss.2014.x.018>
- IJRR 扩展版（2015）：<https://doi.org/10.1177/0278364915592961>
