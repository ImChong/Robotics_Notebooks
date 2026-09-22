# A Survey of Multi-sensor Fusion Perception for Embodied AI（MSFP Survey，arXiv:2506.19769）

> 来源归档（ingest）— 策展解读编译，非原文镜像

- **标题：** A Survey of Multi-sensor Fusion Perception for Embodied AI: Background, Methods, Challenges and Prospects
- **短名：** MSFP Survey
- **类型：** paper / survey / perception / embodied-ai
- **作者：** Shulan Ruan, Rongwei Wang, Xuchen Shen, Huijie Liu, Baihui Xiao, Jun Shi, Kun Zhang, Zhenya Huang, Yu Liu, Enhong Chen, You He
- **机构：** 清华大学；中国科学技术大学；合肥工业大学
- **年份：** 2025（arXiv v1: 2025-06-24）
- **原文：** https://arxiv.org/abs/2506.19769
- **DOI：** https://doi.org/10.48550/arxiv.2506.19769
- **代码：** 无（综述）
- **项目页：** 无
- **入库日期：** 2026-09-22
- **中文导读：** [wechat_embodied_heart_msfp_survey_tsinghua_2026-09-22.md](../blogs/wechat_embodied_heart_msfp_survey_tsinghua_2026-09-22.md)
- **一句话说明：** 任务无关地梳理具身 AI 中 **多传感器融合感知（MSFP）** 的四条技术轴——多模态、多智能体、时间序列与 MM-LLM——并讨论数据/模型/应用层开放挑战。

## 开源状态（步骤 2.5）

- **结论：** 综述；截至入库日 arXiv 无官方代码仓库或项目页。

## 摘录（编译自摘要与公众号导读，非原文复制）

### 摘录 1 — 动机与缺口

现有 MSFP 综述多面向 **单一任务/领域**（如 3D 检测或自动驾驶），或只从 **多模态融合** 单视角展开，缺少对 **多智能体协作**、**时间序列融合** 与 **MM-LLM** 等多样性的系统覆盖。本文以 **task-agnostic** 视角组织文献。

**对 wiki 的映射：** [`wiki/entities/paper-msfp-embodied-ai-survey.md`](../../wiki/entities/paper-msfp-embodied-ai-survey.md)

### 摘录 2 — 四条技术轴

1. **多模态融合：** 点级 / 体素级 / 区域级 / 多级（如 PointPainting、TransFusion、EPNet++）。
2. **多智能体融合：** 多车/多机协作感知（CoBEVT、V2VNet、HM-ViT；通信效率 When2Com/How2Com）。
3. **时间序列融合：** 密集 query（BEVFormer、BEVDet4D）、稀疏 query（StreamPETR、Sparse4D）、混合 query（UniAD、FusionAD）。
4. **MM-LLM 融合：** 视觉-语言（DriveVLM、OmniDrive）与视觉-LiDAR-语言（LiDAR-LLM、MAPLM）。

**对 wiki 的映射：** [`wiki/entities/paper-msfp-embodied-ai-survey.md`](../../wiki/entities/paper-msfp-embodied-ai-survey.md)

### 摘录 3 — 开放挑战

从 **数据层**（质量、增强、标定）、**模型层**（跨模态对齐、鲁棒性、MM-LLM 集成）与 **应用层**（实时性、可解释性、跨场景迁移）讨论 MSFP 未来方向。

**对 wiki 的映射：** [`wiki/entities/paper-msfp-embodied-ai-survey.md`](../../wiki/entities/paper-msfp-embodied-ai-survey.md)

## 与本库关系

- 升格详情页：[`wiki/entities/paper-msfp-embodied-ai-survey.md`](../../wiki/entities/paper-msfp-embodied-ai-survey.md)
- 状态估计侧融合概念：[`wiki/concepts/sensor-fusion.md`](../../wiki/concepts/sensor-fusion.md)
- 退化感知 SLAM 实例：[`wiki/entities/paper-ultra-fusion-multi-sensor-slam.md`](../../wiki/entities/paper-ultra-fusion-multi-sensor-slam.md)
