# RAI Institute：机构背景与公司路线来源

- **类型：** site / research-organization
- **官网：** <https://rai-inst.com/>
- **机构：** 机器人与人工智能研究所（Robotics and AI Institute，RAI Institute）
- **核查日期：** 2026-10-08
- **Wiki：** [RAI Institute](../../wiki/entities/rai-institute.md)

## 机构与成立时间

- [About](https://rai-inst.com/about/)：Marc Raibert 领导，办公室在美国 Cambridge 与瑞士 Zurich；定位为研究机构。
- [成立公告](https://rai-inst.com/resources/press-release/hyundai-launches-boston-dynamics-ai-institute/)：正文宣布日期为 **2022-08-12**，初始名为 Boston Dynamics AI Institute；网页日期栏为 08-11，路线按正文事件年份 2022 排序。
- [2025 年回顾](https://rai-inst.com/resources/blog/rai-institute-2025-a-year-of-innovation-for-robotics-and-ai/)：2026-02-03 发布，再次确认 founded in 2022。
- 公告称 Hyundai Motor Group 与 Boston Dynamics 初始投资超过 4 亿美元。研究所与 Boston Dynamics 机器人公司的路线分别阅读；Atlas / Spot 上的合作不意味着全部研究归属 Boston Dynamics。

## 当前研究方向

[Research](https://rai-inst.com/research/)列出五个方向：灵巧操作、学习控制、物理交互的数据驱动模型、复杂环境导航、机器人社会伦理。学习控制强调 RL、技能组合和新环境适应；物理交互模型强调数据学习与物理建模结合。公司路线本轮以已入库的控制、动态操作与部署工具为主，不以研究愿景推定已发布通用 VLA / WAM。

## 时间轴与资源核查

| 节点 | 事件时间与官方证据 | 开放范围 / 对应归档 |
| --- | --- | --- |
| ZEST | [arXiv v1](https://arxiv.org/abs/2602.00401)提交于 2026-01-30；2026-08 期刊版另见既有详情 | [既有论文归档](../papers/zest.md)未列可运行官方训练栈；本次不重写其历史开放核查结论 |
| AthenaZero 硬件 | [官方博客](https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/) 2026-04-07；不是 2026-09 期刊版日期 | [既有归档](./rai-athenazero-blog.md)：有效质量分析及部分实验数据公开，完整硬件/真机控制不在同一开放范围 |
| Sumo | [arXiv v1](https://arxiv.org/abs/2604.08508) 2026-04-09；[当前项目页](https://sumo.rai-inst.com/)链接代码 | [rai-opensource/sumo](../repos/rai-opensource-sumo.md)有仿真 GUI / headless MPC 入口；不推定真机驱动或全部数据公开 |
| Robot Juggling 演示 | [官方视频页](https://rai-inst.com/resources/videos/a-new-benchmark-in-robot-juggling/) 2026-05-27 | 页上有视频、无训练代码按钮；页面称 cascade 在不到 10 分钟真实交互中学会，不与后续论文的统计口径混用 |
| SMPC-to-RL | [arXiv v1](https://arxiv.org/abs/2608.12063) 2026-08-12；[项目页](https://pages.rai-inst.com/smpc2rl/) | 2026-10-08 项目页仍无自身 Code 按钮；judo 是被引用的采样 MPC 工具，不能代替完整方法实现 |
| Exploy | [官方博客](https://rai-inst.com/resources/blog/introducing-exploy-simplifying-rl-policy-deployment-in-autonomous-robotics/) 2026-10-07 | [既有归档](./exploy-docs.md)：部署工具代码公开；不等于配套全部机器人策略或驱动公开 |

上表是并行研究与工具的公开事件，不表示一个 checkpoint 依次升级。时间轴复用原有项目详情，不拆分论文、硬件资源与代码节点。
