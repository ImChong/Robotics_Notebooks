# SCPToolbox.jl

> 来源归档（ingest）

- **标题：** SCPToolbox.jl — sequential convex programming trajectory optimization toolbox
- **类型：** repo
- **链接：** <https://github.com/UW-ACL/SCPToolbox.jl>（旧名 `dmalyuta/scp_traj_opt`，访问旧地址会进入本仓库，README 写明 "The repository name has changed to `SCPToolbox.jl`"）
- **语言 / License：** Julia / GPL-3.0
- **入库日期：** 2026-11-06
- **核对范围：** 读了 GitHub 仓库首页 README 与目录结构；未运行代码。
- **通读：** 已完成
- **一句话说明：** [Malyuta 等教程](../papers/malyuta_2021_convex_trajectory_generation_tutorial.md) 的官方实现，`src/solvers/` 下实现 PTR、SCvx、GuSTO、LCvx 四种算法。

## 开源状态（2026-11-06 核查）

| 产物 | 状态 |
|------|------|
| 算法实现（PTR / SCvx / GuSTO / LCvx） | **已开源** |
| 示例（`test/examples/`） | **已开源**：double_integrator、Mars rocket landing（`rocket_landing`）、SpaceX Starship landing "flip" maneuver（`starship_flip`）、oscillator、quadrotor、freeflyer、rendezvous_planar、rendezvous_3d |
| 教程算例代码 | 教程 Fig. 2 指向旧仓库的 `csm` 分支 |

README 给出的引用：Malyuta 等，*IEEE Control Systems* 42(5):40–113, 2022, DOI 10.1109/MCS.2022.3187542。求解器依赖 README 未写明，未核对。

## 对 wiki 的映射

- [序列凸化（Successive Convexification, SCvx）](../../wiki/methods/successive-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 3
