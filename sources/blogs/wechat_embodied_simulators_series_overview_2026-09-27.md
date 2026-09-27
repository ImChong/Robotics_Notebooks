# 具身智能仿真器系列 · 总览篇 | 十大仿真器横评

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能仿真器系列 · 总览篇 | 十大仿真器横评
- **类型：** blog
- **作者：** 微信公众号「具身智能仿真器系列」（入库日 HTML 未解析 nick_name；第三方转载多指向 **Xbotics 具身智能实验室** 系科普，**以原文页脚为准**）
- **原始链接：** https://mp.weixin.qq.com/s/evU4IsliLfmsb9RoYXU65A
- **发表日期：** 2026-09-27（入库日；正文未显式日期）
- **入库日期：** 2026-09-27
- **抓取方式：** Cursor WebFetch 结构化正文；Camoufox / wechat-article-for-ai 因本环境未装 `camoufox fetch` 失败
- **姊妹关系：** 系列 **10 篇分平台正文 + 本总览**；与 [深蓝 TOP 8 十年史](wechat_shenlan_sim_platforms_top8_decade.md) **互补**——本系列偏 **2026 主流十平台横评 + 任务选型 + 多平台管线**，非历史被引量排行
- **一句话说明：** 十大具身仿真平台（MuJoCo→Gazebo/CoppeliaSim）同屏对比 **物理/渲染/并行/可微/生态/上手** 六维，给出 **按任务与资源选型表** 与 **三条多平台研究管线**；**10/10 复用既有 wiki 实体**，升格 **技术地图总览页**（无新 paper 节点）。

## 十平台 → 本库节点（全部复用）

| # | 文内平台 | Wiki 节点 | 开源/许可（文内） |
|---|----------|-----------|-------------------|
| 01 | MuJoCo / dm_control | [mujoco](../../wiki/entities/mujoco.md)、[dm-control](../../wiki/entities/dm-control.md) | Apache 2.0 |
| 02 | Isaac Sim + Lab | [isaac-sim](../../wiki/entities/isaac-sim.md)、[isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md) | NVIDIA 条款 |
| 03 | SAPIEN | [sapien](../../wiki/entities/sapien.md) | 见官方 |
| 04 | Genesis | [genesis-sim](../../wiki/entities/genesis-sim.md) | Apache 2.0 |
| 05 | ManiSkill | [maniskill2](../../wiki/entities/maniskill2.md)、[ManiSkill 论文实体](../../wiki/entities/paper-rcl-ref-18fe7d2bfb0e93f4c45e-maniskill-generalizable-manipulation-skill-bench.md) | Apache 2.0 |
| 06 | Habitat | [habitat-sim](../../wiki/entities/habitat-sim.md) | 见官方 |
| 07 | RoboCasa | [robocasa](../../wiki/entities/robocasa.md) | 见官方 |
| 08 | LIBERO | [libero-benchmark](../../wiki/entities/libero-benchmark.md) | 见官方 |
| 09 | PyBullet | [pybullet](../../wiki/entities/pybullet.md) | MIT/zlib |
| 10 | Gazebo / CoppeliaSim | [gazebo-sim](../../wiki/entities/gazebo-sim.md)、[coppeliasim](../../wiki/entities/coppeliasim.md) | Apache 2.0 / 见官方 |

## 对 wiki 的映射

- **升格主页面：** [embodied-simulators-series-technology-map](../../wiki/overview/embodied-simulators-series-technology-map.md)
- **交叉：** [sim-platforms-decade-technology-map](../../wiki/overview/sim-platforms-decade-technology-map.md)（历史 TOP 8）、[simulator-selection-guide](../../wiki/queries/simulator-selection-guide.md)（locomotion 三选一）、[robot-training-stack-layers-technology-map](../../wiki/overview/robot-training-stack-layers-technology-map.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

## 可信度与使用边界

- 六维 **★ 表** 为公众号相对感受，非 benchmark；星标在 WebFetch 抓取中未保留，读原文或技术地图文字结论。
- 平台版本与维护状态以 **官方仓库 / 文档** 为准（如 Habitat 社区维护、Isaac Gym→Lab 迁移等以本库实体页为准）。
- 原始摘录见 [wechat_embodied_simulators_series_overview_2026-09-27.md](../raw/wechat_embodied_simulators_series_overview_2026-09-27.md)。

## 当前提炼状态

- [x] sources/raw + sources/blogs 归档
- [x] wiki 技术地图（选型 + 管线 + 十平台 hub）
- [ ] 系列 10 篇分平台正文（用户未指定单篇 URL，待后续 ingest）
