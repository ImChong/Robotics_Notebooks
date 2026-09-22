# 清华大学最新综述！具身AI中多传感器融合感知：背景、方法、挑战

> 来源归档（blog / 微信公众号）

- **标题：** 清华大学最新综述！具身AI中多传感器融合感知：背景、方法、挑战
- **类型：** blog / wechat / survey / perception
- **作者：** 具身智能之心（编辑；原文作者 Shulan Ruan 等）
- **原始链接：** https://mp.weixin.qq.com/s/fLitz6CshVfAAQPQuusq4A
- **发表日期：** 2026-09-22（推断；配套 arXiv:2506.19769 中文导读）
- **入库日期：** 2026-09-22
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`wechat_embodied_heart_msfp_survey_tsinghua_2026-09-22.md`](../raw/wechat_embodied_heart_msfp_survey_tsinghua_2026-09-22.md)
- **配套论文：** [MSFP Survey（arXiv:2506.19769）](../papers/msfp_survey_arxiv_2506_19769.md)
- **一句话说明：** 对清华/USTC/合工大 **MSFP 具身感知综述** 的中文导读——四条技术轴（多模态 / 多智能体 / 时间序列 / MM-LLM）× 融合粒度（点/体素/区域/多级；密集/稀疏/混合 query）；升格 [`paper-msfp-embodied-ai-survey`](../../wiki/entities/paper-msfp-embodied-ai-survey.md)。

## 核心摘录（归纳，非全文）

### 总判断

- **MSFP** 是具身 AI 连接物理世界与数字智能的核心环节；单模态在光照、雨雾、遮挡下各有盲区，融合目标是 **补盲 + 降不确定性**。
- 既有综述多 **绑单一任务**（如 3D 检测 / 自动驾驶）或 **只讲多模态融合**；本文主张 **任务无关** 组织，覆盖 **多智能体协作**、**时间序列** 与 **MM-LLM** 四条轴。

### 四条技术轴（文内 taxonomy）

| 轴 | 子类 / 粒度 | 代表方法（文内点名） | 工程读法 |
|----|-------------|----------------------|----------|
| **多模态融合** | 点 / 体素 / 区域 / 多级 | PointPainting、CenterFusion、TransFusion、MVX-Net、EPNet++ | 对齐粒度决定算力与稀疏场景鲁棒性 |
| **多智能体融合** | V2X / 协作 BEV | CoBEVT、V2VNet、HM-ViT、When2Com / How2Com | 通信带宽 vs 遮挡补盲；中间特征 vs 原始数据 |
| **时间序列融合** | 密集 / 稀疏 / 混合 query | BEVFormer、BEVDet4D、StreamPETR、Sparse4D、UniAD | 实时性：稀疏 query 更适合低延迟闭环 |
| **MM-LLM 融合** | 视觉-语言 / 视觉-LiDAR-语言 | DriveVLM、OmniDrive、LiDAR-LLM、MAPLM | 可解释规划 vs 点云-文本对齐成本 |

### 背景速记

- **传感器：** 相机（语义 rich / 光照敏感）、LiDAR（几何精确 / 稀疏+天气敏感）、毫米波雷达（速度+恶劣天气 / 轮廓稀疏）。
- **任务：** 2D/3D 检测、语义分割、深度估计、占用预测（occupancy）。
- **数据集：** KITTI、nuScenes、Waymo Open、Argoverse、A*3D 等（文内逐条规模与传感器配置）。

### 开源核查（步骤 2.5，2026-09-22）

| 资源 | 结论 |
|------|------|
| 官方代码 / 权重 | **不适用** — 文献综述，arXiv 无项目页 |
| 一手 PDF | [arXiv:2506.19769](https://arxiv.org/abs/2506.19769) |

## 对 wiki 的映射

- **主新建页：** [paper-msfp-embodied-ai-survey](../../wiki/entities/paper-msfp-embodied-ai-survey.md)
- **论文归档：** [msfp_survey_arxiv_2506_19769.md](../papers/msfp_survey_arxiv_2506_19769.md)
- **交叉：** [Sensor Fusion](../../wiki/concepts/sensor-fusion.md)（状态估计侧）、[六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)、[Ultra-Fusion SLAM](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] arXiv:2506.19769 论文 source + 实体页
- [x] 四条 MSFP 轴 taxonomy 与 mermaid 流程图写回 wiki
