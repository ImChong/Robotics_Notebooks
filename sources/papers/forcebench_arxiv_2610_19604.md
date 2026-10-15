# ForceBench: Benchmarking Force Integration for Robotic Manipulation（arXiv:2610.19604）

> 来源归档（ingest）

- **标题：** ForceBench: Benchmarking Force Integration for Robotic Manipulation
- **类型：** paper / benchmark / manipulation / force-sensing / vla / simulation
- **arXiv：** <https://arxiv.org/abs/2610.19604>（cs.RO；PDF：<https://arxiv.org/pdf/2610.19604>）
- **作者：** Xuan Han, Zinuo Jin, Tianlin Li, Yunfan Lou, Xin Liu, Yuchen Xue, Yuanpeng He, Boxian Yao, Shuhan Zhan, Ruihai Wu, Zhibo Pang
- **机构：** 未核实（arXiv 摘要页与项目页均未列作者单位）
- **项目页：** <https://forcebench-platform.github.io/>
- **开源核查（2026-10-15）：** **待发布**。项目页写明论文、代码、示范与检查点的公开下载链接"暂未提供"，将在发布后更新；项目页与 arXiv 页均未列 GitHub 仓库。
- **文章来源：** [公众号综述摘录](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（31 篇论文汇总）

## 核心主张（摘要要点）

- 将 VLA 策略的六维腕力融合方式放在统一仿真基准中比较：8 个任务（物性推断、阻力交互、接触丰富装配），1600 条同步示范，任务级成功判据。
- 对比无力基线与六种力融合设计，骨干相同（论文中为预训练 VLA 骨干）。
- 仿真：Prompt Fusion 均值 35.31%，基线 23.25%；Predictive 与 Temporal 设计主要在物性推断任务上受益，两种 Temporal 设计在装配任务上低于基线。
- 真机（8 个系统、5 个任务）：Prompt Fusion 与 Dual-Path Fusion 均值 63.00%，基线 40.00%；只有 Prompt Fusion 在每个任务上都优于基线。

## 硬件与平台

- 仿真：Isaac Lab / Isaac Sim，Franka 机械臂 + 平行夹爪，物理 60 Hz、控制 15 Hz；每任务 200 条专家示范，每方法每任务 200 episode，共享随机种子。
- 真机：Franka Research 3 + KWR75B 六轴力传感器 + Robotiq 夹爪 + 外部与腕部相机。

## 链接

- arXiv 摘要：<https://arxiv.org/abs/2610.19604>
- 项目页（无代码链接）：<https://forcebench-platform.github.io/>
- wiki 节点：[paper-forcebench](../../wiki/entities/paper-forcebench.md)
