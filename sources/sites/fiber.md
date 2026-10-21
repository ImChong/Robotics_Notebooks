# FIBER 官方项目页归档

> 来源归档（ingest）

- **标题：** FIBER — Action-Aligned Latent Planning for Versatile Humanoid Loco-Manipulation
- **类型：** site
- **链接：** <https://fiberaa.github.io/>
- **论文：** <https://arxiv.org/abs/2610.24675>
- **代码：** <https://github.com/Yutang-Lin/FIBER>
- **机构：** Amazon Frontier AI & Robotics（Amazon FAR）；Carnegie Mellon University；Stanford University
- **作者：** Yutang Lin, Guanya Shi, C. Karen Liu
- **入库日期：** 2026-10-21
- **一句话说明：** FIBER 以冻结 BFM 的动作对齐潜变量作为规划接口，在 Unitree G1 上演示全身移动操作。

## 项目页核查与开源状态（2026-10-21）

- **官方仓库已公开，但可运行代码尚未发布。** 项目页的 Code 链接指向 [Yutang-Lin/FIBER](https://github.com/Yutang-Lin/FIBER)。仓库 README 明确写明 “Code coming soon”，并说明当前内容主要是 teaser、视觉概览、链接和 citation；目前不包含可核验的训练、latent projection 或部署步骤。
- **项目页演示：** G1 使用机载深度完成箱子拾取、开门和抓取；页面介绍技能串联、waypoint steering，并提供交互 MuJoCo playground。项目页说明专家、latent projector 和 planner 均在仿真训练。
- **运行频率：** 页面报告 planner 以 5 Hz 重规划，冻结 BFM 以 50 Hz 根据新状态闭环输出全身动作。页面展示 18 次真机 rollout；不把演示次数解读为统计成功率。

## Robotics_Notebooks 映射

- 论文归档：[sources/papers/fiber_arxiv_2610_24675.md](../papers/fiber_arxiv_2610_24675.md)
- 官方仓库归档：[sources/repos/fiber.md](../repos/fiber.md)
- wiki 实体：[wiki/entities/paper-fiber-action-aligned-latent-planning.md](../../wiki/entities/paper-fiber-action-aligned-latent-planning.md)

## 一手来源

- 项目页：<https://fiberaa.github.io/>
- 官方代码仓库：<https://github.com/Yutang-Lin/FIBER>
- 论文：<https://arxiv.org/abs/2610.24675>
