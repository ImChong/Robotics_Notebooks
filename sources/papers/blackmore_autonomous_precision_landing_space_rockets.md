# Technical Article: Autonomous Precision Landing of Space Rockets

> 来源归档（ingest）

- **标题：** Autonomous Precision Landing of Space Rockets
- **作者：** Lars Blackmore（本文发表时任 SpaceX 火箭着陆工程师）
- **出版物：** *Frontiers of Engineering: Reports on Leading-Edge Engineering from the 2016 Symposium*，National Academies Press，2017，章节 pp. 33–42；文章亦刊于 *The Bridge* 46(4), 2016, pp. 15–20
- **类型：** technical article / book chapter
- **原文：** <https://nap.nationalacademies.org/read/23659/chapter/10>
- **作者提供的 PDF：** <https://www.larsblackmore.com/nae_bridge_2016.pdf>
- **一句话说明：** SpaceX 着陆工程师从精确着陆的角度概述 Falcon 9 Reusable（F9R）的返回轨迹、误差收敛和末段动力下降控制难点。

## 核心摘录

1. **末段目标是触地时同时消除位置和速度误差。** 对 F9R，文章描述的边界条件是高度到零时速度也到零；若速度过早降为零，火箭会重新上升，而减速过晚则会撞地。
2. **着陆燃烧承担最后的轨迹修正。** 文章指出，助推段返场燃烧和大气层内格栅翼控制先压缩轨迹离散；着陆燃烧仍需具备横向 divert 能力，把火箭引向目标区，并以适合着陆的静止姿态结束。
3. **制导计算必须快速并可重规划。** 车辆需要依据当前位置计算满足硬件约束、到目标点静止的轨迹；计算需在极短时间内完成，硬件故障时还可能需要重新规划。
4. **文中公开的优化实现线索是 CVXGEN。** Blackmore 写道，SpaceX 使用 CVXGEN 生成定制的机载凸优化代码。此处是该文对当时 F9R 技术路线的公开概述，不是完整的 Falcon 9 飞控设计说明或源代码。
5. **这篇文章没有把 Falcon 9 算法称为 G-FOLD。** G-FOLD 是相关的行星动力下降制导研究；不应仅因作者或数学方法相关，就推断它是 Falcon 9 的实际回收算法。

## 对 wiki 的映射

- [控制建模与求解](../../wiki/concepts/modeling-and-solving-for-control.md) — 约束优化和在线制导的背景知识。
- 本次未新增 Falcon 9 控制器实体页：这篇文章是概览性技术资料，公开信息不足以复原完整飞控算法。

## 相关一手资料

- SpaceX 官方视频：[Grasshopper 24-Story Hover Slam 3/7/13](https://www.youtube.com/watch?v=orUjSkc2pG0)
- SpaceX 官方任务时间线：[Twilight Mission](https://www.spacex.com/launches/twilight)
