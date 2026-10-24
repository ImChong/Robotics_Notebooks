# Vision as Proprioception: Estimating Humanoid Robot Pose from Video

> 来源归档（ingest）

- **标题：** Vision as Proprioception: Estimating Humanoid Robot Pose from Video
- **短名：** HumanoidPose
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25288>
- **PDF：** <https://arxiv.org/pdf/2610.25288>
- **项目页：** <https://siplab.org/projects/HumanoidPose>
- **代码：** <https://github.com/eth-siplab/HumanoidPose>（仓库已建，代码与模型待发布）
- **作者：** Jiaxi Jiang, Matthias Heyrman, René Zurbrügg, Marco Hutter, Christian Holz
- **机构：** ETH Zürich 计算机科学系；ETH Zürich 机械与过程工程系
- **入库日期：** 2026-10-24
- **一句话说明：** 将普通视频当作视觉本体感知，估计多种人形机器人的全身关节配置与根部姿态。
- **项目页归档：** [HumanoidPose 项目页](../sites/siplab-humanoidpose.md)
- **代码归档：** [eth-siplab/HumanoidPose](../repos/eth-siplab-humanoidpose.md)
- **沉淀到 wiki：** [HumanoidPose](../../wiki/entities/paper-humanoidpose-video-pose-estimation.md)

## 开源状态（步骤 2.5）

- **待发布（截至 2026-10-24）**：官方项目页链接到 GitHub；仓库目前只有 README、资产与 MIT 许可证，README 明确写着代码与模型即将发布，未见可运行训练或推理入口。因此记录官方仓库，但当前不能据此复现实验。

## 核心摘录（面向 wiki 编译）

### 摘录 1：跨形态视频姿态估计

- 输入视频帧；模型结合人类预训练的视觉/姿态特征与机器人运动学描述，预测对应机器人根部姿态和关节配置。
- 机器人运动学描述作为条件输入，使同一模型可服务多个 embodiment；论文报告以合成数据覆盖 16 种人形机器人。
- 项目页称训练图像由人体运动重定向到机器人后在 Unreal Engine 渲染，并为每帧保留对应机器人配置。

**对 wiki 的映射：** [HumanoidPose](../../wiki/entities/paper-humanoidpose-video-pose-estimation.md)、[GMR 运动重定向](../../wiki/methods/motion-retargeting-gmr.md)

### 摘录 2：真实视频适配与评测

- 对未标注网络视频进行自训练，提升 2D 人形机器人检测器对真实视频的适应能力。
- 在 Unitree G1、EngineAI PM01 和 Fourier N1 三个平台上评测；作者报告相对每个平台表现最佳的基线，关节角误差降低 18–32%。
- 项目页还展示互联网公开视频与 AI 生成视频上的姿态恢复；论文摘要称同时改善关节位置精度、全局定位与未见形态泛化。

**对 wiki 的映射：** [HumanoidPose](../../wiki/entities/paper-humanoidpose-video-pose-estimation.md)

## 当前提炼状态

- [x] 一手论文与官方项目页核对
- [x] 官方代码开放状态核查
- [x] wiki 实体、来源页与 GMR 关联
