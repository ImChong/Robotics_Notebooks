# MacroHands 官方技术博客原文归档

- **英文标题：** MacroHands: 3D hands and camera pose from egocentric video
- **中文标题：** MacroHands：从第一人称视频联合估计三维手部与相机位姿
- **类型：** 官方技术博客 / 项目发布，不是同行评议论文
- **作者/机构：** Macrodata Labs
- **发布日期：** 2026-11-10
- **原文：** https://macrodata.co/blog/macrohands-3d-hands-and-camera-pose-from-egocentric-video
- **模型：** MacroHands-Mono、MacroHands-Stereo
- **核对日期：** 2026-11-15
- **核对范围：** 官方博客 TL;DR、数据与真实值获取、3D hands 指标/完整采样规则、MacroHands-Mono、Ruler test、Finding the right hands 内部基准、MacroHands-Stereo、Camera pose 的 ATE/RPE/scale 对照、Action MPJPE 1/2 s、Conclusion、Citation/References；核对正文及表格，未独立运行模型。
- **通读：** 已完成
- **沉淀到 wiki：** [MacroHands](../../wiki/entities/macrohands.md)

## 官方摘要与任务

MacroHands 将第一人称头戴相机的单目或双目视频转为相机坐标中的手部 3D 关键点（每只 21 个）及相机到世界的逐帧位姿。它针对机器人学习的目标是恢复真实世界中随时间移动的手部轨迹，而不只是相对手腕对齐的手形。

作者列出 Mono 与 Stereo 两个模型；支持在有 IMU 的头显上使用 IMU 优化相机轨迹，但 **技术博客没有公开可复现的模型内部网络细节、训练配方、权重或源码入口**。这里的「流程」只能归纳输入、输出及评估，不应虚构网络结构。

## 真值与指标的核对

- **手部基准：** SHOW3D（Quest 3、93 scenes、32 人、0.89 h），HOT3D（Quest 3、6 sequences、0.21 h），ARCTIC s05（mono、34 recordings、0.24 h）。
- **相机位姿：** HOT3D（无 IMU）、Monado Index（stereo + IMU、11 recordings、0.99 h）、Aria Gen 2 Pilot（stereo + IMU、12 recordings、1.10 h）。作者称这些相机序列未参与训练。
- **Cam-MPJPE：** 直接在相机系按真值 3D 关节测毫米误差，不做对齐；**PA-MPJPE：** 对每只手做旋转、平移、尺度相似变换后测形状误差。只报后者会掩盖深度失准。
- **Recall：** 左右手身份正确且预测框相对标注框 IoU > 0.5；SHOW3D 使用 60 fps 发布数据的隔帧采样至 30 fps，质量异常帧排除。缺失的预测影响 recall，而非 Cam-MPJPE 的分子。
- **相机 ATE：** 仅一次刚体对齐、不拟合尺度；RPE 测 1 秒增量；还单报运动尺度误差。
- **Action MPJPE：** 各方法使用自己的相机路径，将未来 1 秒或 2 秒手部运动放回第一帧相机坐标，逐关节无对齐比较。SHOW3D 的相机姿态有明显头转问题，因此作者不在 Action MPJPE 表使用 SHOW3D。

## 核心结果（以下全部为作者报告，非独立复测）

| 基准 / 度量 | MacroHands | 对照 | 读法 |
|---|---|---|---|
| SHOW3D Cam-MPJPE（mm） | Mono 32.2；Stereo 13.1 | HaWoR mono 55.9；UmeTrack stereo 46.4 | 双目改进主要来自深度 |
| SHOW3D PA-MPJPE（mm） | Mono 8.4；Stereo 8.5 | — | 形状在对齐后并未因双目明显改善 |
| HOT3D Cam-MPJPE（mm） | Mono 12.2；Stereo 10.6 | UmeTrack stereo 46.2 | 数据集范围不同不可混比 |
| ARCTIC s05 Cam-MPJPE（mm） | Mono 19.4 | ACE-Ego-Hand* 18.7 | 该指标并非 MacroHands 全场第一；* 对照方法训练过 ARCTIC |
| 内部第一人称检测 AP50（%） | 75.8 | HaWoR 43.3 | 内部未公开 benchmark，外界不可复算 |
| Monado Index stereo+IMU ATE（cm） | 5.7 | DROID-SLAM stereo+IMU 23.6 | 完整轨迹漂移 |
| Aria Gen 2 stereo+IMU ATE（cm） | 18.9 | DROID-SLAM stereo+IMU 74.0 | 完整轨迹漂移 |
| HOT3D stereo 1s Action MPJPE（mm） | 13.6 | HaWoR 自带 SLAM + infiller 65.9 | 相机与手部误差组合后的动作质量 |

内部情境测试：其他人手误检占模型输出 0.2%（HaWoR 23.6%）；存在遮挡时正确找到 80% 佩戴者手（基线最高 59%），戴手套时找到 79%（基线最高 48%）。该内部数据不能公开，不能把这些数字当成外部可重复基准。

## 原文结论与机器人含义

手部形状在局部看起来准确，并不保证世界坐标中的动作正确；必须共同控制深度、佩戴者身份、遮挡下检测、相机 1 秒 RPE 和整段 ATE。双目优先用于新采集架构，已采集单目素材仍可用 Mono 测试。对策略训练的实际收益仍需在公开数据及机器人系统上验证，本博客没有报告端到端机器人成功率/良率增益。

## 可用性与局限

- 作者提供提交少量 mono/stereo 视频做免费标注对比的商业接触入口，不等同公开模型/API/推理权重。
- 截至核对日，原文未列公开 MacroHands GitHub、论文 PDF、权重下载或完整推理 SDK；因此无法核查训练架构、复现实测与源码运行时序。
- SHOW3D、HOT3D 样本规模有限；内部真实场景数据无法公开；部分对照在 HOT3D/ARCTIC 的训练集合有重合（作者图表以星号标记）；指标是作者选择的协议与数据集，不是跨域统一结论。

## 官方来源

- [Macrodata Labs：MacroHands 原文（含 benchmark 采样与评分细则）](https://macrodata.co/blog/macrohands-3d-hands-and-camera-pose-from-egocentric-video)
- [SHOW3D 论文](https://arxiv.org/abs/2603.28760)
