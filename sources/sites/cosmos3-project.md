# Cosmos 3 官方项目页（NVIDIA Cosmos Lab）

> 来源归档

- **标题：** Cosmos 3 — Omnimodal World Models for Physical AI
- **类型：** site / 项目页
- **URL：** <https://research.nvidia.com/labs/cosmos-lab/cosmos3/>
- **技术报告 PDF：** <https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf>
- **入库日期：** 2026-06-06
- **一句话说明：** Cosmos 3 产品叙事页：以 **单一全模态世界模型** 贯通理解、生成、仿真与动作，并提供各能力维度的交互式 demo 与榜单摘要。

## 页面要点（ingest 快照）

### 核心 slogan

「Multiple modalities, many applications. One single model.」— 文本、图像、视频、音频、动作在同一模型内流转。

### 能力演示维度（站点交互 Tab）

| 能力 | 简述 |
|------|------|
| **Vision-Language Reasoning** | 在图像/视频中做空间关系、时序事件、2D 轨迹、目标检测与具身任务推理 |
| **Image Generation** | 文本→高保真工业/机器人场景图，强调物理细节与空间布局 |
| **Audio-Visual Generation** | 文本/图像/片段→带同步音效的视频（碰撞、机械声、环境声） |
| **Robot Policy** | 语言+视觉→操纵动作与 rollout（多桌面操作任务 demo） |
| **Forward Dynamics** | 观测+手部位姿/控制→未来视频 rollout |
| **Inverse Dynamics** | 观测状态变化→恢复控制轨迹 |
| **Reasoning + Generation** | 先输出 2D 末端轨迹与 CoT，再生成对应物理交互视频（可含音频） |

### 榜单叙事（页面 footer）

- **Reasoning：** 在 Robotics / Smart Space / Driving 等 benchmark 平均上，开源模型排名第一（页面口径）。
- **Generation：** 开源 **Text-to-Image**、**Image-to-Video** 与 **robot policy** 在 R-Bench、Artificial Analysis、RoboLab、RoboArena 等榜单领先（页面口径）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Cosmos 3](../../wiki/entities/cosmos-3.md) | 主实体页 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 开源全模态生成式世界模型平台 |
| [World Action Models](../../wiki/concepts/world-action-models.md) | policy + forward/inverse dynamics 统一栈 |
| [mimic-video](../../wiki/methods/mimic-video.md) | 前代 Cosmos-Predict2 作为 VAM 骨干的对照 |
| [NVIDIA SO-101 Sim2Real](../../wiki/entities/nvidia-so101-sim2real-lab-workflow.md) | 课程中 Cosmos 数据增广策略的落地语境 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/cosmos3_arxiv_2606_02800.md`](../papers/cosmos3_arxiv_2606_02800.md)
- 代码入口：[`sources/repos/nvidia_cosmos.md`](../repos/nvidia_cosmos.md)
- 实体页：**`wiki/entities/cosmos-3.md`**
