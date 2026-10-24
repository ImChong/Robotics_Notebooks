# ETH SIPLAB：HumanoidPose 项目页

> 来源归档

- **标题：** Vision as Proprioception: Estimating Humanoid Robot Pose from Video
- **类型：** site / project-page
- **URL：** <https://siplab.org/projects/HumanoidPose>
- **论文：** <https://arxiv.org/abs/2610.25288>
- **代码：** <https://github.com/eth-siplab/HumanoidPose>（待发布）
- **作者：** Jiaxi Jiang, Matthias Heyrman, René Zurbrügg, Marco Hutter, Christian Holz
- **机构：** ETH Zürich 计算机科学系；ETH Zürich 机械与过程工程系
- **入库日期：** 2026-10-24
- **配套代码归档：** [eth-siplab/HumanoidPose](../repos/eth-siplab-humanoidpose.md)
- **论文归档：** [arXiv:2610.25288](../papers/humanoidpose_arxiv_2610_25288.md)
- **沉淀到 wiki：** [HumanoidPose](../../wiki/entities/paper-humanoidpose-video-pose-estimation.md)

## 项目页要点

- HumanoidPose 从视频估计全身 3D 人形机器人配置，输入无需机器人板载状态。
- 共享模型以机器人运动学描述为条件，覆盖 16 种 embodiment；训练数据由重定向到机器人的人体动作合成渲染，项目页称不使用真机数据训练姿态模型。
- 真实域适配通过未标注网络视频的伪标签自训练改进 2D 机器人检测器。
- 实机评测平台为 Unitree G1、EngineAI PM01、Fourier N1；项目页报告相对各平台最佳基线，关节角误差降低 18–32%。
- 页面展示真实机器人、互联网视频和 AI 生成视频的恢复结果。

## 源码与资源开放核查（2026-10-24）

项目页链接到 ETH SIPLAB 的官方 GitHub 仓库。仓库目前公开 MIT 许可证、README 和演示素材，但 README 标注 **Code and models: Coming soon**，截至入库日尚无可运行训练或推理入口；因此按**待发布**记录，不将仓库存在等同于实现已开源。
