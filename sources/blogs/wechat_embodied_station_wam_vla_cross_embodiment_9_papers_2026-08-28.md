# 具身智能开源资源集中上新：9篇论文，WAM、VLA、跨本体一次看全

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能开源资源集中上新：9篇论文，WAM、VLA、跨本体一次看全
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ
- **发表日期：** 2026-08-28
- **入库日期：** 2026-08-28
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md`](../raw/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)
- **一句话说明：** 汇总 9 篇近期具身/机器人论文（文内均给项目页或代码链），主线是为策略加入结构化接口：人类视频作任务规格、流式时间记忆、相机坐标动作几何、语言推理与多臂子目标、三维世界令牌、置信度主动学习、韧性里程计与建筑任务词表；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 9**；同一 arXiv **不重复造页**；GaussianDream++ 与既有 Awesome 索引级 [GaussianDream](../../wiki/entities/paper-sa-2605-20752-gaussiandream-a-feed-forward-3d-gaussian-world-m.md) 为不同 arXiv）。

## 核心摘录（归纳，非全文）

文内判断：这批工作共同把具身策略的关键接口显式化——视频可以成为任务说明，时间上下文可以持续流入 VLA，动作可以在本体之外用统一几何表示，语言推理可以分配测试时计算，置信度可以直接指导世界模型补课。开放资源形态分化：部分已提供代码/模型/数据，部分目前以项目页和发布计划为主。阅读时把「论文结果」「项目演示」「可下载资产」分开判断。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Zero-WAM | [2608.26103](https://arxiv.org/abs/2608.26103) | **待发布**：仓已建，代码/模型/数据计划 2026-09-15 前 | [paper-zero-wam](../../wiki/entities/paper-zero-wam.md) |
| 02 | StreamPI | [2608.26067](https://arxiv.org/abs/2608.26067) | **待发布**：项目页写明官方仓计划 2026-08-30 公开 | [paper-streampi](../../wiki/entities/paper-streampi.md) |
| 03 | UCAG-P | [2608.26058](https://arxiv.org/abs/2608.26058) | **待发布**：GitHub Pages 仓 README 写 Code Release Soon | [paper-ucag-p](../../wiki/entities/paper-ucag-p.md) |
| 04 | R³ | [2608.26053](https://arxiv.org/abs/2608.26053) | **待发布**：项目页 Code Coming Soon | [paper-r3-robotic-reasoner](../../wiki/entities/paper-r3-robotic-reasoner.md) |
| 05 | MA-VLA | [2608.25864](https://arxiv.org/abs/2608.25864) | **已开源** 训练/部署 + MACG 基准 | [paper-ma-vla](../../wiki/entities/paper-ma-vla.md) |
| 06 | GaussianDream++ | [2608.25659](https://arxiv.org/abs/2608.25659) | **部分开源**：仓为 GaussianDream v1 实现；++ 入口未在 README 标明 | [paper-gaussiandream-plusplus](../../wiki/entities/paper-gaussiandream-plusplus.md) |
| 07 | ConfAL-WM | [2608.25572](https://arxiv.org/abs/2608.25572) | **已开源** 主动学习管线 + HF 权重/数据 | [paper-confal-wm](../../wiki/entities/paper-confal-wm.md) |
| 08 | SUPER ODOMETRY 2.0 | [2608.25427](https://arxiv.org/abs/2608.25427) | **部分开源**：slim LiDAR-inertial ROS 2 已开；完整四级自适应以论文为准 | [paper-super-odometry-2](../../wiki/entities/paper-super-odometry-2.md) |
| 09 | TARCAT | [2608.25395](https://arxiv.org/abs/2608.25395) | **已开源** 分类体系 JSON + 视频标注（非训练策略） | [paper-tarcat](../../wiki/entities/paper-tarcat.md) |

### 文内要点速记

1. **Zero-WAM** — 人类视频作 in-context 任务规格；HumanGen 7.42 万配对 / 8600 任务；RoboTwin 2.0 七个未见任务 47.0%（+29.5 pp）。
2. **StreamPI** — 无新增参数的流式时间记忆；单元内双向注意力、单元间因果注意力；LIBERO / 真机均优于 π0.5。
3. **UCAG-P** — 相机坐标锚点运动统一手臂/人形/人手；单检查点 LIBERO 98.3%、RoboTwin Easy/Hard 88.7%/89.2%。
4. **R³** — 自由形式自然语言推理作测试时计算；中期训练 + 量表奖励单步 RL；Language Table 与双臂杂货打包。
5. **MA-VLA** — 逐臂原子动作分配 + Arm Shuffle；未见协作模式基准上既有 VLA 大多失败。
6. **GaussianDream++** — 20 个世界令牌；训练期高斯解码、推理期移除；LIBERO 98.6% / LIBERO-Plus 87.8%；真机 29.2%→52.5%。
7. **ConfAL-WM** — 稠密置信度风险图驱动主动后训练；任务/帧/图块三级评分。
8. **SUPER ODOMETRY 2.0** — 四级自适应融合；学习式 IMU；200 km / 800 h 空中、轮式、腿式验证。
9. **TARCAT** — 41 个动作原语 / 12 组 / 3 类；91 项 O\*NET 任务 + 30 段教学视频。

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 9** 个实体；**0 篇复用既有 complete 页**。GaussianDream++（arXiv:2608.25659）与 Awesome 索引级 GaussianDream（arXiv:2605.20752）**不是同一论文**，分别保留节点并交叉链接。
- 阅读坐标：[WAM / VLA / 跨本体 9 篇技术地图](../../wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[里程计与激光雷达融合](../../wiki/methods/lidar-odometry-fusion.md)、[Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（9 新建 / 0 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
