---
type: entity
tags: [repo, motion-retargeting, human-motion, monocular-video, mixamo, blender, glb, gvhmr]
status: complete
updated: 2026-10-13
code: https://github.com/MaxLiu22/Motion2MixamoRig
summary: "Motion2MixamoRig 将单人视频或照片中的动作恢复并重定向到 Mixamo 角色，提供 CLI、Blender 扩展与蒙皮 GLB 导出。"
related:
  - ./gvhmr.md
  - ./mixamo.md
  - ../concepts/motion-retargeting.md
  - ../concepts/motion-retargeting-pipeline.md
sources:
  - ../../sources/repos/motion2mixamorig.md
---

# Motion2MixamoRig

[Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) 是一套开源“视频/照片→Mixamo动画”工具：给定单人全身视频或静态姿态图、Mixamo 角色文件及所需人体模型资产，运行人体动作恢复与重定向，导出蒙皮 .glb、骨架中间结果和预览视频。它把上游人体恢复和目标角色适配封装成可复用工程，不是一个新的动作估计模型。

## 流程与接口

~~~mermaid
flowchart LR
  A[视频 / 照片] --> B[人体动作恢复]
  B --> C[骨架序列]
  C --> D[Mixamo 重定向]
  E[Mixamo 目标 FBX] --> D
  D --> F[GLB / 预览视频]
~~~

仓库提供 m2mr doctor、m2mr run 和 Blender 4.2+ 扩展。CLI 支持视频、静态图片、角色 rig 和 skeleton 复用参数；推理后可保留 skeleton，以便换角色时避免重复提取动作。项目 README 指出需要下载 GVHMR 等模型权重，故不应将整个流水线简化为“仅运行 GVHMR”。

## 复现与许可注意

首次运行需下载约 5 GB 权重；推理速度受硬件、视频长度和是否复用 skeleton 影响。用户需自行获取 Mixamo FBX 与 SMPL-X body model。仓库代码采用 Apache-2.0，但 SMPL-X 默认非商业许可及 Mixamo/上游权重各有独立条款；部署或分发前逐项确认。

## 关联

- [GVHMR](./gvhmr.md)：人体运动恢复上游之一。
- [Motion Retargeting](../concepts/motion-retargeting.md) 与 [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)：所属任务链。
- 来源：[仓库归档](../../sources/repos/motion2mixamorig.md)。
