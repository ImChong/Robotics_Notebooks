# PhySGA: A Physics-Space Grounded Agentic Framework for Generalizable and Interactive 3D Composition（arXiv:2610.20560）

> 来源归档（ingest）

- **标题：** PhySGA: A Physics-Space Grounded Agentic Framework for Generalizable and Interactive 3D Composition
- **缩写 / 框架：** **PhySGA**（Physics-Space Grounded Agentic Framework）
- **类型：** paper / 3d-composition / agentic / physics-simulation / isaac-sim
- **arXiv：** <https://arxiv.org/abs/2610.20560>（Submitted 2026-10-14；PDF：<https://arxiv.org/pdf/2610.20560>）
- **代码（项目页与文章均链接）：** <https://github.com/AgenticEI/PhySGA>（Apache-2.0 LICENSE；仓库 HEAD `558f688 Update README`）
- **核查结果（2026-10-15）：** 浅克隆仓库后，文件仅有 `README.md`、`LICENSE`、`docs/assets/pipeline.png`、`docs/assets/teaser.jpg`，**不含训练、推理或仿真代码**。README 仅含方法概述与 BibTeX。
- **项目页：** <https://agenticei.github.io/PhySGA-page/>（页面外链至作者主页与 arXiv、代码仓库）
- **作者：** Weichen Lyu、Chong Xia、Mingyu Dong、Long Xu、Shengjun Zhang、Yueqi Duan（通讯）
- **机构：** Tsinghua University；Zhejiang University
- **入库日期：** 2026-10-15
- **文章中的转述：** 见 [微信图文归档](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)（第 15 篇，标注「资源、项目页」）
- **一句话说明：** 用 Assembly Agent 做全局规划与粗调整，Critic Agent 基于接触反馈做细修正，经 Action Router 映射到 Isaac Sim 的物理原语，并用自进化技能库沉淀经验，完成刚体、铰接与可变形物体的交互式组合。

## 开源状态（步骤 2.5，核查日期 2026-10-15）

- 项目页与文章均指向 `AgenticEI/PhySGA`，但该仓库目前只有 README 与许可证，**未发布可运行代码**。
- **结论：待发布**（论文与仓库均声明开源方向，代码尚未落库）。

## 摘录 1：方法（摘要 / README）

- **前馈构造：** Assembly Agent 进行几何与运动学推理，给出全局步骤规划与粗调整。
- **反馈修正：** Critic Agent 读取仿真运动与接触反馈，诊断物理失败并做细粒度闭环修正。
- **动作路由：** Geometric、Articulated、Velocity 三类 Router 把语义目标映射为几何变换、关节运动与释放速度。
- **技能进化：** Self-Evolving Skill Library 在任务前检索，任务后把交互记忆与最终物理结果蒸馏为可复用技能。

## 摘录 2：实验（摘要）

- **平台：** Isaac Sim 内的组合任务，涵盖悬挂、重力沉降、隐藏容纳等。
- **摘要结论：** 在 relation fidelity、physical plausibility 与 visual quality 上相对已有方法有显著提升。**具体指标数字本次未读取到，记为未核实。**

**对 wiki 的映射：** 升格 [`wiki/entities/paper-physga.md`](../../wiki/entities/paper-physga.md)；标注代码仓库为占位，不作运行时序图。
