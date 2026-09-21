# Nymeria Dataset（Project Aria 官方页）

> 来源归档（ingest · 数据集产品页）

- **标题：** Nymeria Dataset
- **类型：** site / dataset
- **机构：** Meta · Project Aria
- **URL：** <https://www.projectaria.com/datasets/nymeria/>
- **论文：** [arXiv:2406.09905](../papers/nymeria_arxiv_2406_09905.md)
- **代码：** <https://github.com/facebookresearch/nymeria_dataset>
- **Explorer：** Aria Dataset Explorer（projectaria.com）
- **入库日期：** 2026-09-21
- **一句话说明：** 世界最大野外人类 motion 多模态 egocentric 数据集产品页：300 h 活动、3600 h 视频、层级 motion-language、CC BY-NC 4.0 开放研究使用。

## 数据集速查（官方 Highlights）

| 维度 | 数值 |
|------|------|
| 活动时长 | **300 h** 日常活动 |
| 视频总量 | **3600 h** |
| 序列 | **1200** |
| 参与者 | **264** |
| 地点 | **50** indoor/outdoor |
| 场景脚本 | **20** |
| 语言标注 | **230 h** motion-language · **310.5K** 句 · **8.64M** 词 |
| 轨迹 | 头显 **~400 km** · 腕部 **~1053 km** |
| 人体模型 | Meta **Momentum** 参数化重定向 |

## 设备栈

- **Aria 头显：** 1×RGB + 2×灰度 + 2×眼动 + 2×IMU + 磁力计 + 气压 + 音频；**MPS** 输出 6DoF 轨迹、半稠密点云、眼动深度。
- **miniAria 腕带：** 腕部 egocentric 多模态（未来可穿戴形态）。
- **XSens MVN Link：** 17 IMU 全身 kinematics GT。
- **Observer Aria：** 第三人称跟随视角。

## NymeriaPlus 升级

- 优化 **MHR / SMPL** 人体运动
- 室内物体/结构 **3D+2D bbox**
- **ShapeR** 实例级 3D 物体重建
- 额外模态：basemap、腕带视频、头显音频等

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已发布** | 邮件申请 + JSON 下载链接；`aria_dataset_downloader` CLI |
| **体积** | 每版约 **~80 TB**（1100 序列）；可按 group / sequence 筛选 |
| **许可** | CC BY-NC 4.0 |

## 对 wiki 的映射

- [nymeria-dataset.md](../../wiki/entities/nymeria-dataset.md)
- [paper-nymeria.md](../../wiki/entities/paper-nymeria.md)
