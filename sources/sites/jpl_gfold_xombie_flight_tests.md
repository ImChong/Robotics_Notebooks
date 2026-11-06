# JPL G-FOLD × Masten Xombie 飞行试验（官方新闻与视频页）

> 来源归档（ingest）

- **JPL 新闻：** <https://www.jpl.nasa.gov/news/jpl-masten-testing-new-precision-landing-software/>（2013-08-12，2014-01-22 更新）
- **NASA Armstrong 转载：** <https://www.nasa.gov/centers-and-facilities/armstrong/mastens-xombie-tests-jpls-g-fold-precision-landing-software>
- **JPL 视频页：** <https://www.jpl.nasa.gov/videos/g-fold-diversion-test/>（2014-01-22）
- **类型：** 官方新闻 / 视频说明
- **入库日期：** 2026-11-06
- **核对范围：** 三个官方页面全文；技术论文（Açıkmeşe 等 AAS 2013、Scharf 等 IEEE Aerospace 2014、Scharf 等 JGCD 2017、Dueri 等 JGCD 2017）未取得全文（AIAA 页面对抓取返回 403）。
- **通读：** 已完成
- **开放状态：** G-FOLD 未见公开源码；无损凸化的算法本身已在论文中公开（见 [Blackmore 等 2010](../papers/blackmore_2010_minimum_landing_error_pdg.md)）。

## 核心摘录

1. **试验对象。** JPL 的 G-FOLD（Fuel Optimal Large Divert Guidance）在 Masten Space Systems 的 XA-0.1B "Xombie" 垂直起降火箭上飞行，参与方含 UT Austin 与 NASA Flight Opportunities 项目。
2. **2013-07-30 飞行。** Xombie 先朝错误着陆点下降，"About 90 feet into the profile, the G-FOLD flight control software was automatically triggered to calculate a new flight profile in real-time"，最终改道到约 2,460 英尺外的正确着陆点。
3. **2013-09-20 飞行（试验收官）。** 火箭起初斜向远离目标，G-FOLD 规划出 "half-mile-long (0.8-kilometer), three-dimensional divert"；视频页称新轨迹 "was first calculated only one second earlier"，落点在目标 9 英寸以内，最高约 1200 英尺。
4. **官方性能对比。** "Compared to the software used to land NASA's Mars Curiosity rover in August 2012, G-FOLD can provide six times more divert range for a lander of that class."
5. **关于 2012 年试验。** 官方页面未提 2012 年飞行；[Malyuta 等教程](../papers/malyuta_2021_convex_trajectory_generation_tutorial.md) 的参考文献列有 2012 年 7 月（750 m 改道）与 8 月（500 m 改道）的 JPL/Masten 视频。

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md)
- [无损凸化（Lossless Convexification）](../../wiki/methods/lossless-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 2 / 5
