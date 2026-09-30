# PADP：Policy Action Dynamics Projection 四足移动操作（RA-L 2026）

> 来源归档（ingest）

- **英文标题：** From Policy Actions to Whole-Body Dynamics: A Unified Projection-Based Framework for Versatile Legged Loco-Manipulation
- **标题：** 从策略动作到全身动力学：面向 versatile 四足 loco-manipulation 的统一投影框架（PADP）
- **类型：** paper
- **作者：** Chengzhen Yan, Qingchen Liu, Jiahu Qin, Yiming Jiang, Yang Shi
- **机构：** 中国科学技术大学（USTC，前四位）；维多利亚大学（University of Victoria，Yang Shi）
- **刊物：** IEEE Robotics and Automation Letters（RA-L）
- **DOI：** <https://doi.org/10.1109/lra.2026.3734869>
- **IEEE：** <https://ieeexplore.ieee.org/document/11692614/>
- **arXiv：** **无**（截至 2026-09-30 未检索到同题 preprint；勿与 arXiv:2609.36012 混淆）
- **平台（用户 / 摘要级信息）：** Unitree Go2 + Unitree Z1（MuJoCo 仿真）；云深处 DEEP Robotics LYNX M20 + Unitree Z1（真机）
- **关键词：** Legged Robots；Whole-Body Optimization；Mobile Manipulation；Policy Action Dynamics Projection（PADP）
- **开源：** 截至 **2026-09-30**，IEEE 页与公开检索 **未见** 官方 GitHub / 项目页；按 **未开源** 归档，后续 lint 跟进
- **入库日期：** 2026-09-30

## 核心论文摘录

### 1) PADP：策略动作 → 全身动力学投影

- 提出 **Policy Action Dynamics Projection（PADP）**，在统一 **projection-based** 框架下，把上层 **policy actions** 映射为满足 **whole-body dynamics** 与任务约束的可执行全身指令，支撑 **versatile legged loco-manipulation**（具体公式与 QP/优化结构以 RA-L 正文为准）。
- **对 wiki 的映射：** [../../wiki/entities/paper-padp-projection-legged-loco-manipulation.md](../../wiki/entities/paper-padp-projection-legged-loco-manipulation.md)

### 2) 仿真与跨平台真机

- **MuJoCo：** Go2 + Z1 组合验证 loco-manipulation 管线。
- **真机：** LYNX M20 四足底盘 + Z1 机械臂；与仿真栈形成 sim-to-real 对照（数值与任务表以 IEEE PDF 为准）。
- **对 wiki 的映射：** [../../wiki/entities/paper-padp-projection-legged-loco-manipulation.md](../../wiki/entities/paper-padp-projection-legged-loco-manipulation.md)

### 3) 与「纯 RL 全身策略 / 纯 WBC 轨迹优化」的接口定位

- 叙事重点在 **policy 层灵活性与 dynamics 层可行性** 的 **统一投影接口**，便于在移动底盘 + 臂系统上切换 manipulation 任务而不重训底层全身模型（细节见原文 Related Work）。
- **对 wiki 的映射：** [../../wiki/tasks/loco-manipulation.md](../../wiki/tasks/loco-manipulation.md)、[../../wiki/concepts/whole-body-control.md](../../wiki/concepts/whole-body-control.md)

## 当前提炼状态

- [x] Crossref / OpenAlex 元数据核对（DOI、作者、机构）
- [x] wiki 实体页（指标待读 IEEE 原文补全）
- [ ] 官方代码发布后再建 `sources/repos/`
