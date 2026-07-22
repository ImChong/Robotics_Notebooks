# HumanoidUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation

> 来源归档（ingest）

- **标题：** HumanoidUMI: Bridging Robot-Free Demonstrations and Humanoid Whole-Body Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.27239>
- **项目页：** <https://baai-aether.github.io/HumanoidUMI>
- **机构：** Beijing Academy of Artificial Intelligence（BAAI）
- **入库日期：** 2026-07-03
- **更新日期：** 2026-07-22
- **开源状态：** 待发布；项目页按钮写明 **Code(Coming Soon)**，截至 2026-07-22 未发现官方可运行仓库。项目页 BibTeX 又写作 **BifrostUMI / arXiv:2605.03452**，需后续核对官方命名。
- **一句话说明：** **PICO VR + UMI 式夹爪** 采集骨盆、左右 TCP、左右脚等稀疏关键点、腕部视角与夹爪动作；高层 Diffusion Policy 预测关键点动作块，经 **Spatial Keypoint Retargeting** 与 learned whole-body controller 在 **Unitree G1** 五类真机任务验证，无需机器人遥操作即可采集全身 loco-manip 示范。

## 摘要要点

- 目标：降低人形全身 visuomotor policy 的数据采集门槛，避免每条示范都依赖目标机器人遥操作。
- 采集：PICO 4 VR 系统追踪腰/脚/手持控制器，双 gripper 同步记录 wrist-view fisheye 图像和 gripper aperture。
- 表示：默认五关键点（pelvis、left/right TCP、left/right foot），下肢参与强的任务加入 left/right knee。
- 学习：高层扩散策略从双腕图像和下半身本体预测未来关键点与夹爪命令。
- 执行：SKR 将关键点映射为 G1 root pose + joints，低层 WBC 在真机上跟踪。

## 方法要点

- **Robot-free acquisition：** 采集阶段无需物理 G1；在线 SKR 可视化帮助操作者判断示范是否可转移。
- **High-level policy：** DINOv2 编码左右腕 RGB；融合三帧 15-D lower-body proprioception；输出 H=48 的 receding-horizon action chunk。
- **Action space：** 默认 47-D = 5 个关键点 ×（3D translation + 6D rotation）+ 2 个 gripper widths；七关键点为 65-D。
- **Spatial Keypoint Retargeting：** 只对腿部垂直位移进行 anisotropic adjustment，实验中 leg scale 为 0.75；其余度量关系保留。
- **IK 与控制：** 两阶段加权 IK 先保证足端支撑/朝向，再细化 pelvis/TCP；低层 controller 以 50 Hz 输出 29-D residual joint-position actions。
- **Latency matching：** 对动态投掷任务校准腕部视觉、gripper state、高层推理和机器人执行延迟。

## 实验与数字

- **任务：** cluttered tabletop pick-and-place、bimanual vegetable collection、dynamic ball-shooting、under-table waste disposal、walking coffee delivery。
- **吞吐对比：** 10 分钟内有效示范数，HumanoidUMI 在 bimanual、throw trash、walk + coffee 均高于 TWIST2；novice walk + coffee 为 **61 vs 1**，experienced user 为 **62 vs 5**。
- **消融：** GMR 替换 SKR 会削弱 manipulation 任务；去掉 latency matching 会伤害 dynamic shooting；去掉膝关键点会削弱 under-table / walking delivery。
- **部署：** Unitree G1 上验证单臂、双臂、动态释放、屈膝伸身、行走递送五类全身操作。

## 开源 / 复现状态

- **代码：** 项目页显示 **Code(Coming Soon)**，未发现官方 GitHub 仓库。
- **数据：** 未发现公开数据下载链接。
- **项目页命名：** 页面标题为 HumanoidUMI，但 BibTeX 为 `BifrostUMI`、arXiv `2605.03452`；本 source 仍按用户指定 arXiv `2606.27239` 维护，并在 wiki 中注明风险。
- **复现边界：** 需要 PICO 4 VR、脚/腰 trackers、双 gripper 硬件、DINOv2/DP/SKR/mink/MJLab/Unitree SDK 全链路实现。

## 对 wiki 的映射

- [paper-humanoidumi](../../wiki/entities/paper-humanoidumi.md) — 完整实体页，含流程、机制、实验和开源状态。

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2606.27239>
- 项目页：<https://baai-aether.github.io/HumanoidUMI>
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
