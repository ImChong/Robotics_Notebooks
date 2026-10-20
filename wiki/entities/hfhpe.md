---
type: entity
project_id: hfhpe
project: https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation
code: https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation
tags: [perception, head-pose, computer-vision, onnx, open-source]
status: complete
updated: 2026-10-20
related:
  - ./gnm-head.md
  - ./sam-3d-body.md
  - ./paper-face-anything-4d-face-reconstruction.md
sources:
  - ../../sources/repos/hfhpe.md
summary: "HFHPE 是一个针对整头图像 crop 的轻量头部姿态估计管线：用 DINOv3 教师蒸馏轻量学生，在单次姿态图推理中输出全角度 yaw、pitch、roll，并提供 ONNX/TFLite 模型。"
---

# HFHPE：全角度头部姿态估计

**HFHPE（High-Angle Robust Fast Head Pose Estimation）** 是 PINTO0309 开源的头部姿态估计项目：检测器先定位整头 crop，轻量网络随后一次推理输出 yaw、pitch、roll，针对后脑、极端角度与遮挡设计；代码采用 MIT 许可证，模型权重可从发布页获取。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HFHPE | High-Angle Robust Fast Head Pose Estimation | 高角度鲁棒快速头部姿态估计 |
| DINOv3 | Self-DIstillation with NO labels, version 3 | 项目教师模型所用的视觉特征骨干 |
| ONNX | Open Neural Network Exchange | 跨运行时模型交换格式；仓库提供 ONNX 部署图 |
| MAAE | Mean Absolute Angular Error | 平均绝对角度误差，越低越好 |

## 流程总览

```mermaid
flowchart LR
  DET["头部检测器"] --> CROP["整头 crop"]
  CROP --> ROLL["roll 分支"]
  ROLL --> DEROT["图内反旋转"]
  DEROT --> YP["yaw / pitch 主干"]
  ROLL --> OUT["yaw / pitch / roll + κ"]
  YP --> OUT
```

检测器是姿态模型之外的前置环节；整合后的姿态网络把 roll 估计、crop 反旋转与 yaw/pitch 回归封装在一次模型调用中。κ（von Mises 集中参数）为各角度提供置信度线索。

## 核心机制

- **覆盖整圈角度：** yaw、roll 的输出覆盖 0–360°，pitch 覆盖 −120° 至 +120°；后者超过常见正面俯仰范围，面向仰头等 back-tilt 情况。
- **圆周角度表示：** 每个角度使用 biternion 的单位 cos/sin 对表达，并带 von Mises 集中参数 κ，避免把周期角度直接当普通标量回归。
- **解耦 roll：** roll 分支估计图像平面旋转，ONNX 图中对 crop 做反旋转后再估计 yaw/pitch；因此整合图既提供 roll，也让主干在较直立 crop 上工作。
- **蒸馏到小模型：** DINOv3 ViT-L/16（320×320）作教师，轻量学生包含 ViT-T/16、PP-HGNetV2-B0 与 MBConv；64×64 输入有不同精度、速度的取舍。

## 速度与精度读法

官方 README 的测试条件为 i9-10900K、ONNX Runtime 1.22 CPU EP，采用 batch-1 图：

| 整合姿态模型（64×64） | 参数量 | CPU 推理 | yaw / pitch MAAE |
|---|---:|---:|---:|
| ViT-T 学生 | 6.4M | 2.5 ms | 0.76° / 0.67° |
| PP-HGNetV2-B0 学生 | 2.6M | 1.4 ms | 2.34° / 2.02° |
| MBConv 学生 | 1.5M | 1.4 ms | 5.55° / 7.55° |

表中角度误差来自项目自己的合成数据验证集，不能据此与使用不同数据集的论文结果直接比较。README 示例的 DEIMv2 检测步骤约需 24 ms，故姿态网络的 1–2.5 ms 不能代表完整视频处理延迟；检测器可另行替换。

## 工程实践与局限

- 权重发布页提供 ONNX/TFLite 版本；模型消费整头 crop，应用端还需配置检测器、裁剪/缩放与坐标约定。
- README 描述训练数据主要为合成头部姿态图像，并加入作者自拍样本；常见公开头姿 benchmark 未用于该项目报告，部署选型前应在目标相机与目标人群上独立验证。
- 项目测试显示极低照度时，头部纹理线索不足会使 yaw 估计失效；遮挡鲁棒性不能消除曝光不足、模糊或检测框偏移带来的风险。
- 这是头部朝向估计器，不输出头部 3D 网格或机器人颈部关节命令。机器人使用时需自行定义相机坐标、角度符号与颈部控制映射。

## 关联页面

- [GNM Head](./gnm-head.md) — 参数化头脸 3D 网格；与 HFHPE 的朝向估计任务不同，可在感知链路中互补
- [SAM 3D Body](./sam-3d-body.md) — 从图像恢复全身人体网格，任务范围更广
- [FaceAnything（4D 脸重建）](./paper-face-anything-4d-face-reconstruction.md) — 单目视频头脸重建，侧重几何与动态外观

## 参考来源

- [HFHPE 仓库归档](../../sources/repos/hfhpe.md)
- [HFHPE 官方 README](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation#readme)
- [权重发布页](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/tag/weights)

## 推荐继续阅读

- [HFHPE 权重发布页](https://github.com/PINTO0309/High-Angle_Robust_Fast_Head_Pose_Estimation/releases/tag/weights) — 查看 ONNX/TFLite 文件与具体模型变体
- [DINOv3 项目页](https://github.com/facebookresearch/dinov3) — 查看教师模型的视觉骨干来源
