---
type: entity
tags: [dataset, human-object-interaction, multiview, 4d-reconstruction, nvidia, robot-learning]
status: complete
updated: 2026-10-06
summary: "FORM-HOI：NVIDIA 四相机人–物交互数据集；清洗发布集含 4,135 段、约 29.144 小时、61 个对象和 22 位参与者，CC BY 4.0。"
related:
  - ./paper-cari4d.md
  - ./project-cari4d.md
  - ./paper-hoi-retarget.md
sources:
  - ../../sources/datasets/form_hoi_nvidia.md
  - ../../sources/sites/cari4d-project-page.md
---

# FORM-HOI：多视角人–物交互数据集

**全称：** Foundry for Reconstruction from Multiview Human-Object Interaction  
**发布者：** NVIDIA  
**数据卡：** [Hugging Face nvidia/form-hoi](https://huggingface.co/datasets/nvidia/form-hoi) · **v0.1.0** · **CC BY 4.0**

## 数据规模与组成

数据卡的统计针对清洗后的公开 release，排除 held-out object sequences。

| 指标 | 数值 |
|---|---:|
| Episodes | 4,135 |
| 总时长 | 约 29.144 小时 |
| 独立物体 ID | 61 |
| 独立人员 ID | 22 |
| 下载归档 | 约 4.504 TB |

总时长按 episode 计一次，不把四路相机视频重复计时；数据卡注明归档大小为十进制 TB。

每段序列含四路标定 RGB 与深度流、人/物掩码、相机内外参、人体姿态参数、逐帧刚体物体位姿、物体纹理网格及地面平面。RGB 为 1536×1152 H.264/MP4；深度为 768×576/HDF5；人体姿态提供 SOMA 与 MHR；物体 pose 为逐帧 4×4 刚体变换；对象网格为带度量尺度的 GLB。

## 数据与标注流程

序列由人类操作员录制，再由自动化管线生成姿态与物体标注；人工检查并标记不合格时间段。数据卡中的 failure_segments.json 记录人工质检、Chamfer distance 和 silhouette containment 标记。几何/轮廓阈值命中代表**潜在质量问题**，不等同于已确认标注错误。

\`\`\`mermaid
flowchart LR
  capture["四路同步 RGB-D"] --> calibration["相机标定与深度"]
  calibration --> reconstruction["人体与物体重建"]
  reconstruction --> review["自动指标与人工质检"]
  review --> release["清洗后的序列与轨迹"]
  release --> consumers["CARI4D 训练、策略 grounding、动作重定向"]
\`\`\`

## 适用方向与关联项目

- **HOI 重建：** 数据卡指出可用于训练 CARI4D 等重建模型。
- **机器人策略 grounding：** 可将视频与人体/物体轨迹作为场景和交互参考；不代表数据集含已验证的机器人策略或真机控制标签。
- **动作重定向：** 与 [HOI-Retarget](./paper-hoi-retarget.md) 等流程互补。FORM-HOI 提供人–物状态，HOI-Retarget 负责求解机器人可执行轨迹。

## 可访问性与限制

- Hugging Face 数据卡报告约 4.5 TB；下载前应确认存储、带宽与处理能力。
- failure_segments.json 是质量审查提示；metric-based flags 可能来自深度噪声或分割噪声，不宜自动视作人工标注错误。
- 数据许可为 CC BY 4.0。涉及人体视频和运动轨迹的下游应用仍需自行核验适用法律、授权、隐私和产品需求。
- 数据卡称技术报告仍在撰写中；代码集成于 [NVIDIA Video to Data](https://github.com/nvidia-isaac/video_to_data) 仓库。

## 来源

- [FORM-HOI 数据卡来源归档](../../sources/datasets/form_hoi_nvidia.md)
- [CARI4D 论文](./paper-cari4d.md)
- [CARI4D 项目及代码](./project-cari4d.md)
- [CARI4D / Video to Data 入口](../../sources/sites/cari4d-project-page.md)
