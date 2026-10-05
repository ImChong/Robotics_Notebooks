# SkeleWAM 官方项目页（Peking University）

> 来源归档（site / project-page）

- **标题：** SkeleWAM: Skeleton World-Action Modeling for Efficient Robotic Manipulation
- **类型：** site / project-page
- **URL：** <https://skelewam-project.github.io/>
- **关联论文：** <https://arxiv.org/abs/2610.02120>
- **论文 HTML：** <https://arxiv.org/html/2610.02120v1>
- **机构 / 作者：** Peking University；Juyi Sheng、Hua Wang、Mengyuan Liu
- **演示：** 项目页提供五项任务的 RGB / 骨架交互演示（开抽屉、关抽屉、抽屉内放积木、叠积木、叠碗）
- **代码 / 权重：** 截至 2026-10-05，官方页面未提供公开代码仓库、权重或数据下载入口；页面中的可视化演示不等于可运行训练/部署代码
- **入库日期：** 2026-10-05
- **一句话说明：** 项目将机器人自身关节、物体中心与接触相关交互点组成稀疏 3D 骨架，以未来骨架预测辅助动作生成，并报告 LIBERO-Plus 与 ARX R5 真机结果。

## 页面核心信息

- **方法演示：** 从 RGB-D 观测与机器人状态提取当前骨架；训练时共同学习未来骨架和动作；执行时只生成动作，MAC 选择代表性动作轨迹，然后执行一段并重观测、重规划。
- **LIBERO-Plus（RGB-D）：** 85.9% 整体成功率；摄像头扰动 93.4%；57.1M 参数。七类扰动中布局变化为 66.6%，低于 π₀.₅ 的 84.1%。
- **真机：** ARX R5，五项桌面操作任务，各 20 次试验，SkeleWAM 平均 89%；Cosmos-Policy 87%、Fast-WAM 83%、π₀.₅ 76%。
- **结果口径：** 项目页标注数值来自论文，主结果使用 RGB-D；另有使用特权仿真坐标的 sim-state 诊断设置，不应与实际视觉输入结果混为一谈。

## 对 wiki 的映射

- [paper-skelewam-efficient-manipulation](../../wiki/entities/paper-skelewam-efficient-manipulation.md)
- [论文归档](../papers/skelewam_arxiv_2610_02120.md)
