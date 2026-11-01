# Reflex 项目页（Visually Grounded Reactive Humanoid Control）

> 来源归档

- **标题：** Reflex: Visually Grounded Reactive Humanoid Control
- **类型：** site（论文项目页 / 视频演示）
- **URL：** <https://toberj.github.io/reflex/>
- **论文：** <https://toberj.github.io/reflex/static/paper/reflex.pdf?v=1008n>
- **代码：** <https://github.com/ToberJ/reflex>
- **作者 / 机构：** Taoyang Jia 等；University of Washington、Stanford University、University of Southern California、National University of Singapore、University of North Carolina at Chapel Hill、Allen Institute for AI
- **入库日期：** 2026-10-12
- **再核日期：** 2026-10-30
- **代码状态：** 官方仓库已公开仿真训练、评测和 checkpoint；真机部署栈未包含。
- **一句话说明：** 通过三阶段训练，使 Unitree G1 从仿真特权状态接球策略过渡到 RGB-D 视觉历史驱动的实时箱体拦截，并在真机报告 2.5 m / 5 m 成功率。

## 项目页内容核查

- 页面列出论文 PDF、GitHub、作者与机构，以及箱体接球视频。
- 方法简介为三个阶段：状态 PPO 接球、从延迟/不完整观察推断箱体动力学、训练 egocentric RGB-D encoder 恢复动力学 latent，同时冻结 controller。
- 结果表述区分仿真（视觉策略 85.7%，特权观测 87.9%）与真机 G1（2.5 m 65%，5 m 40%）；机器人分别达到人类接球率的约 76% / 59%（人类成功率为 85% / 68.3%）。
- 页面还含多距离投掷演示与交互式 Isaac Sim replay；真机部署代码未随仿真代码公开。

## 对 wiki 的映射

- 主实体：[Reflex 论文实体页](../../wiki/entities/paper-reflex-visually-grounded-reactive-humanoid-control.md)
- 论文归档：[reflex_visually_grounded_reactive_humanoid_control.md](../papers/reflex_visually_grounded_reactive_humanoid_control.md)
- 官方仓归档：[toberj-reflex.md](../repos/toberj-reflex.md)
