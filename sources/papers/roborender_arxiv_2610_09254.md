# RoboRender: Robot-Oriented Video Generation for Visual Sim-to-Real Transfer

> 来源归档（ingest · arXiv 预印本）

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.09254>
- **版本/日期：** v1，2026-10-07
- **作者：** Huang Huang、Wensi Ai、Ziyu Chen、Youhui Wang、Zijian Du、Yang Liu、Jiaolong Yang、Li Fei-Fei、Jiajun Wu
- **项目页：** <https://robo-render.github.io/>
- **核查日期：** 2026-10-10
- **沉淀到 wiki：** [RoboRender](../../wiki/entities/robo-render.md)

## 论文摘要整理

RoboRender 面向模拟到真实迁移中的视觉差异：以模拟器产生的深度视频、语言指令和机器人 RGB mask 视频为条件，生成外观更接近真实世界的 RGB 操作视频，同时保留模拟轨迹中的几何、机器人运动和动作标签。生成视频随后与模拟器状态和动作配对，用于训练策略。

论文报告在 pick-and-place、关节物体操作和移动操作任务中，基于 RoboRender 视频训练的策略在真机任务上的平均成功率为 71%；该结果是论文所测任务与设置，不代表通用机器人能力。作者还报告相对原始模拟渲染和常规视觉域随机化的提升，并展示增加每条模拟轨迹生成的视频数对开门任务的影响。

## 解释边界

生成模型改变视觉呈现，不替代物理仿真器对状态、动作和接触的定义。该论文结果说明生成视频可改善测试设置内的视觉迁移，不代表模拟动力学误差、执行器差异或安全验证问题已被解决。本文当前记录的原始来源是 arXiv 论文和论文所列项目页；未发现论文声明的官方代码仓库，故不补填第三方仓库。

## 一手入口

- [arXiv 摘要、作者和版本历史](https://arxiv.org/abs/2610.09254)
- [论文 PDF](https://arxiv.org/pdf/2610.09254)
- [论文列出的项目页](https://robo-render.github.io/)
