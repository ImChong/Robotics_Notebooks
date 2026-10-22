# MoCaP（arXiv:2610.20998）

> 来源归档（ingest）

- **标题：** MoCaP: Mobile Code-as-Policies for Zero-Shot Mobile Manipulation from Passive Human Videos
- **类型：** paper
- **作者：** Yiyang Cai、Tatsuya Kamijo、Ninghao Lu、Zijin Hu、Sherry Yang（Cai 与 Kamijo 为共同第一作者）
- **单位：** New York University、Fudan University、NYU Shanghai
- **预印本：** <https://arxiv.org/abs/2610.20998>
- **HTML：** <https://arxiv.org/html/2610.20998v1>
- **版本：** v1，2026-10-14，cs.RO
- **项目页：** <https://mocap-7.github.io/MoCAP-page/>
- **代码：** <https://github.com/MoCAP-7/MoCAP>（公开可访问；README 提供安装与运行入口；仓库未提供 LICENSE 文件）
- **入库日期：** 2026-10-23
- **一句话说明：** 从被动人类视频提取场景空间记忆和接近物体时的站位先验，让移动操作机器人在找到目标后继续搜索可达、无碰撞的抓取底盘位姿。
- **沉淀到 wiki：** [`wiki/entities/paper-mocap-mobile-code-as-policies.md`](../../wiki/entities/paper-mocap-mobile-code-as-policies.md)

## 核心论文摘录

### MoCaP: Mobile Code-as-Policies for Zero-Shot Mobile Manipulation from Passive Human Videos（Cai et al., 2026）

- **链接：** <https://arxiv.org/abs/2610.20998>
- **核心贡献：** 将日常被动人类视频转成可复用的空间记忆与“人准备伸手前站在哪里”的互动线索；代码代理结合实时观察调用导航、抓取候选筛选、逆运动学（IK）和全臂规划原语，闭环完成找物与移动操作。关键观察是“找到目标”还不等于“站到机械臂够得到的位置”。
- **方法线索：** 离线从视频建空间记忆和互动事件；执行时用视频中的人类接近方向指导机器人初始靠近，再按机器人自身的可达性、碰撞和 IK 约束搜索邻近底盘姿态；移动后用新观测重新核验抓取可行性。
- **论文报告：** YOR 真机平台上，纸箱抓取成功 16/20、罐子抓取 8/10（均为 80%）；目标搜索任务中视频空间记忆成功率 10/10，无先验版本 7/10；接近方向研究整体 manipulation-readiness 为 38/45，对照为 18/45。作者指出该汇总差异在当前样本规模下不显著，收益集中于与人类站位先验不一致的起始方向。
- **开源核查：** 项目页提供代码链接；官方仓库公开，包含 `nav_planner/`、`yor_agent/` 和多个导航基线。仓库 README 给出依赖安装、视频记忆提取和 `yor-agent` 任务运行命令；未见许可证文件，复用前需确认许可。
- **对 wiki 的映射：**
  - [MoCaP 论文实体页](../../wiki/entities/paper-mocap-mobile-code-as-policies.md)
  - [Manipulation 任务页](../../wiki/tasks/manipulation.md)
  - [Zero-shot Object Navigation](../../wiki/tasks/zero-shot-object-navigation.md)

## 主要解读

MoCaP 的重点不是单独改进目标检测或抓取生成，而是把人类视频里“人从哪一侧准备操作”转成底盘姿态搜索的先验，并用机器人自身几何检查决定哪些姿态真的可执行。论文的 zero-shot 指不需要为每个任务采机器人示范或微调策略；系统仍依赖机器人标定、在线 RGB-D/代价地图、导航与机械臂规划服务，以及冻结的视觉/代码模型。

## 一手来源

- arXiv 摘要与论文：<https://arxiv.org/abs/2610.20998>、<https://arxiv.org/html/2610.20998v1>
- 项目页：[mocap-7.github.io/MoCAP-page](https://mocap-7.github.io/MoCAP-page/)
- 官方代码：[MoCAP-7/MoCAP](https://github.com/MoCAP-7/MoCAP)
