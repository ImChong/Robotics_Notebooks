# EgoHTR: Egocentric 4D Demonstrations of Human Terrain Traversal

> 来源归档（ingest）

- **标题：** EgoHTR: Egocentric 4D Demonstrations of Human Terrain Traversal
- **缩写：** **EgoHTR** / Egocentric Human-Terrain Reconstruction
- **类型：** paper / dataset / 4d-reconstruction / human-motion / perceptive-locomotion / humanoid
- **arXiv：** <https://arxiv.org/abs/2607.13472>（PDF: <https://arxiv.org/pdf/2607.13472.pdf>）
- **HTML：** <https://arxiv.org/html/2607.13472>
- **项目页：** <https://egohtr.github.io>
- **机构：** 苏黎世联邦理工（ETH Zürich）；斯坦福大学（Stanford）；加州大学伯克利分校（UC Berkeley）；慕尼黑工业大学（TU Munich）
- **作者：** Alex Brandes、Haig Conti Georges Sajelian、Manthan Patel、Dominik Hollidt、Chenhao Li、Matthias Heyrman、Oliver Hausdörfer、Manuel Kaufmann、Xi Wang、Jonas Frey、Angela P. Schoellig、Christian Holz、Marc Pollefeys、Marco Hutter 等
- **状态：** arXiv 预印本（约 2026-07-15）；**数据与代码计划开放**（项目页 Dataset / Code 均标 *coming soon*）
- **入库日期：** 2026-07-21
- **一句话说明：** 用 Aria 眼镜 + Rokoko IMU 服 + Leica BLK2GO 扫描，在 rough terrain 上采集 **55** 条场景对齐的 4D 人体运动（约 **1.37 h / 150k** 帧），并以此训练 Unitree G1 感知 locomotion。

## 摘录 1：问题与贡献

- **动机：** 无结构地形上的人形穿越仍难；纯 RL 依赖重奖励工程，而基于人类先验的 mimic 又缺 **场景对齐、粗糙地形** 的多模态数据。现有数据集要么无稠密场景、要么仅室内/平地/受限空间，或依赖外视重建精度不足。
- **贡献（论文三条）：**
  1. **可扩展重建管线**（可穿戴 + 便携扫描），面向社区续采。
  2. **EgoHTR 数据集**：7 场景、8 被试、55 序列、多模态（SMPL-X、ego/exo Aria、场景 mesh、可选固定相机与 mocap GT）。
  3. **应用**：HPS / 4D human-scene 重建基准；厘米级参考运动支撑感知 locomotion（G1 真机）。
- **关键词：** Terrain Traversal、4D Reconstruction、Human Motion Dataset、Mimic。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-egohtr.md`](../../wiki/entities/paper-egohtr.md)；交叉 [人形参考运动数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md)、[AMASS](../../wiki/entities/amass.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)。

## 摘录 2：采集与三阶段重建

- **核心传感器：** Project Aria Gen.1（头戴 ego）、Rokoko Pro II（IMU 全身）、Leica BLK2GO（稠密场景）；可选第二副 Aria（exo）与固定相机。
- **重建流水线：**
  1. **Body model：** MoCap → SMPL-X（手/脸固定 identity）；旋转重定向 + IK 精修（对齐 GMR 思路）。
  2. **Temporal：** 开场拍手同步；弃用硬线 PPS，换移动自由度；跨传感器对齐经验上 < 60 ms；序列上限约 5 min 以控时钟漂移。
  3. **Spatial：** 身体锚定 Aria SLAM 轨迹（头–眼镜静态平移）；VGGT 粗对齐 + ICP 将 Aria 点云配准到扫描场景，得世界系下人–场景耦合。
- **设计取舍：** 便携商用传感器，在「可野外部署」与「厘米级精度」之间折中；相对纯外视优化管线更抗遮挡与高动态。

**对 wiki 的映射：** 实体页「流程总览」；与 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) / [GMR](../../wiki/methods/motion-retargeting-gmr.md) 对照上游数据质量。

## 摘录 3：数据集规模与精度

- **规模：** 7 场景（办公室、实验室跑酷厅、健身房、户外废墟等，约 25–1000 m²）；动作含 parkour、爬、翻、窄道、踏石、楼梯等；55 序列 / 1.37 h / ~150k @ 30 fps；平均序列约 90 s；36 序列含第二视角 Aria（0.88 h）；约 0.7 h 标记 mocap GT 测试子集。
- **局部精度（相对 mocap GT）：** MPJPE **73.2 mm**、PA-MPJPE **54.3 mm**；相对 SLOPER4D 在更难动态/粗糙地形下仍略优。
- **全局：** 报告 W-MPJPE / WA-MPJPE / RTE（自称首个报告全局 HPS 估计的人–场景运动数据集之一）。
- **定性：** 粗糙地形上脚滑、穿地、时序抖动较少；管道/箱体等强遮挡场景仍可用。

**对 wiki 的映射：** 实体页「数据集速查」；更新 [人形参考运动数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md)。

## 摘录 4：下游感知 locomotion 与精度必要性

- **训练：** 按参考 clip 分别训 Unitree G1 专家；PPO；actor 观测本体 + 重定向关节指令 + yaw 对齐地形高度扫描；除 mimic 奖励外加 **时间脚接触奖励**（稀疏踏石上尤为关键）。
- **消融：** 接触奖励在 stepping stones 上提升成功率并加速收敛（文中 Table 3）。
- **参考精度门槛：** 对参考根位置加高斯噪声——约 **σ≤0.05 m** 仍可训，**>0.1 m** 崩溃；论证当前单目人–场景重建全局误差常超该窗，而 EgoHTR 锚定 Aria SLAM + 稠密扫描可满足 foothold-critical 任务。
- **重定向栈：** 场景感知 retarget 基于 OmniRetarget、GMR、CoACD（项目页方法说明）。
- **真机：** beam / box-up 等原子技能在 G1 上部署演示。

**对 wiki 的映射：** [Locomotion](../../wiki/tasks/locomotion.md)、[Terrain Adaptation](../../wiki/concepts/terrain-adaptation.md)、[OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)、[RPL](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)。

## 摘录 5：HMR 基准与局限

- **基准：** 外视（如 JOSH / Human3R）、ego（EgoAllo）等在遮挡、运动模糊、全局漂移上暴露失败模式；数据集可作 fine-tune 资源。
- **局限：** 规模尚不足以大规模预训练；仅静态环境、无关节物体；手部未并入身体模型；无事后联合人–场景优化；无特征环境 / 高加速机动可能失败。
- **开放边界（项目页核查，截至 2026-07-21）：** 页头 **Dataset (coming soon)**、**Code (coming soon)**；GitHub org `egohtr` 仅公开项目站仓 `egohtr/egohtr.github.io`，**无可运行重建/训练仓与数据下载 URL** → 归类 **宣称将开源 / 待发布**。

**对 wiki 的映射：** sites 归档与实体页「开源状态 / 源码运行时序图：不适用」。

## 当前提炼状态

- [x] arXiv HTML / 项目页 / GitHub org 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-egohtr.md` 新建
- [x] 开源边界写入 sites / wiki 局限（无 repos：尚无代码 URL）
