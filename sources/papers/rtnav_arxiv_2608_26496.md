# RTNav: Towards Real-Time Zero-Shot Object Navigation（arXiv:2608.26496）

> 来源归档（ingest）

- **标题：** RTNav: Towards Real-Time Zero-Shot Object Navigation
- **类型：** paper / objectnav / zero-shot / real-time / asynchronous-navigation / habitat
- **来源：** arXiv abs / PDF / HTML（v1，2026-08-27）
- **原始链接：**
  - <https://arxiv.org/abs/2608.26496>
  - PDF：<https://arxiv.org/pdf/2608.26496>
  - HTML：<https://arxiv.org/html/2608.26496v1>
- **项目页：** <https://generalroboticslab.com/RTNav> — 归档见 [`sources/sites/duke-rtnav-project.md`](../sites/duke-rtnav-project.md)
- **仿真代码：** <https://github.com/generalroboticslab/RTNav> — 归档见 [`sources/repos/duke-rtnav.md`](../repos/duke-rtnav.md)
- **真实机器人代码：** <https://github.com/generalroboticslab/RTNav-RealWorld> — Stretch 3 + Jetson AGX Thor 部署
- **作者 / 机构：** Easop Lee、Lingyu Zhang（共同一作）、Boyuan Chen；杜克大学（Duke University）
- **入库日期：** 2026-10-24
- **一句话说明：** 将导航评测从暂停式同步仿真改为 30 Hz 墙钟实时推进，并让感知、建图、规划和导航异步并行；昂贵 VLM 推理按需触发，减少等待对导航覆盖率和完成时间的影响。

## 开源状态（核查 2026-10-24）

- **已开源：** 官方仿真/评测仓库公开，README 提供 Habitat + ROS 2 + Docker 评测说明，并列出 RTNav 与六种基线。
- **已开源：** 官方真实机器人仓库公开，README 给出 Stretch 3 与 Jetson AGX Thor 的运行步骤；该仓库以子模块固定 RTNav 版本。
- **边界：** 仓库注明论文结果使用 NVIDIA RTX A6000；实时结果受硬件影响，复现实验应在同一目标硬件配置下重跑基线。
- **互指：** 主实体页 [`wiki/entities/paper-rtnav.md`](../../wiki/entities/paper-rtnav.md)

## 摘要级要点

- **评测问题：** 标准 Habitat step-based 接口在 agent 推理期间暂停环境，计算耗时不计入预算；实时接口保持环境按 30 Hz 墙钟推进，旧动作在新动作到达前继续执行。
- **系统架构：** perception、mapping、planning、navigation 在独立线程中读写共享状态；昂贵 VLM 以按需服务运行，模块无需等待完整的串行感知—推理—动作周期。
- **感知：** OWLv2 约 15 Hz 提议目标候选；Qwen3.5-9B 做语义核验；MobileSAM 细化确认框以投影到点云。
- **探索与执行：** 无目标确认时，通过 VLM 从 frontier 候选视图中选目标；机器人由 PointNav policy 持续朝最新 frontier 或目标移动。
- **结果：** 在 500 s 实时预算下，RTNav 相对最强同步基线在 OVON、HM3D-v1、HM3D-v2 的成功率分别高 6.4%、10.5%、9.1%；成功加权完成时间（SCT）分别高 2.45、5.17、4.17。
- **部署：** 独立真实世界仓库展示 Stretch 3 + Jetson AGX Thor 的 ROS 1/ROS 2 桥接及 RTNav 运行入口。

## 对 wiki 的映射

- [RTNav 实体页](../../wiki/entities/paper-rtnav.md) — 方法、评测、源码状态与部署路径
- [零样本目标导航](../../wiki/tasks/zero-shot-object-navigation.md) — 补充实时评测与异步架构案例
- [导航与 SLAM 自主栈](../../wiki/overview/navigation-slam-autonomy-stack.md) — wall-clock 约束下的模块化系统设计