# HumanPlus-1000（项目页）

> 来源归档

- **标题：** HumanPlus-1000 — 1000-Hour Embodied Motion Dataset
- **类型：** site / dataset announcement
- **链接：** <https://humanplus-ai.github.io/HumanPlus1000.github.io/>
- **Hugging Face（预览子集）：** <https://huggingface.co/datasets/humanplus-ai/humanplus-1000>
- **代码 / 可视化：** <https://github.com/humanplus-ai/humanplus-1000>
- **机构：** HumanPlus（与 [HumanPlus: Humanoid Shadowing and Imitation from Humans](../papers/loco_manip_161_survey_012_humanplus.md) 同品牌；论文侧关联 Stanford University）
- **联系：** info@humanplus.xyz（全量数据申请）
- **入库日期：** 2026-09-15
- **一句话说明：** **1000+ 小时** 同步第一人称视觉 + 全身/手部运动 + SLAM/IMU/深度的真实世界人类行为多模态数据集；HF 当前为 **100 session 预览**，全量需邮件申请。

---

## 发布要点（项目页，入库日核查）

| 维度 | 内容 |
|------|------|
| 目标规模 | **1000+** 小时 |
| 采集者 | **200+** 人 |
| 任务 | **500+** 种活动 |
| 场景 | **100+** 地点 |
| 叙事 | 「From the real world to world-human models」— 为模仿学习、World-Human Models 与机器人学习提供可扩展数据底座 |
| 模态 | 第一人称视觉、全身运动、手部运动、人在环境中的移动（SLAM/相机轨迹）、多模态时间同步、结构化标注与质控 |
| 代表活动（页内 demo） | 洗碗、日常家务与长程真实活动（页内视频卡片） |

## 开源状态（步骤 2.5，2026-09-15）

| 产物 | 状态 | 入口 |
|------|------|------|
| 预览数据 | **已开源**（100 session 子集，ungated） | [humanplus-ai/humanplus-1000](https://huggingface.co/datasets/humanplus-ai/humanplus-1000) |
| 全量 1000h | **部分 / 申请制** — README 写联系 info@humanplus.xyz | 邮件申请 |
| 读取 / 可视化代码 | **已开源**（MIT） | [humanplus-ai/humanplus-1000](https://github.com/humanplus-ai/humanplus-1000) |
| 数据许可 | **CC BY-NC 4.0** | HF card |
| 代码许可 | **MIT** | GitHub `LICENSE` |

## 对 wiki 的映射

- **wiki/entities/humanplus-1000-dataset.md** — 数据集实体页（主升格）
- **sources/datasets/humanplus-1000.md** — HF 数据卡归档
- **sources/repos/humanplus-1000.md** — 官方 viewer / loader 仓
- **wiki/entities/paper-loco-manip-161-012-humanplus.md** — 同品牌 HumanPlus 人形 shadowing 论文对照
- **wiki/overview/ego-category-01-data-collection.md** — Ego 大规模人类采集旁路
- **wiki/queries/humanoid-training-data-pipeline.md** — 人体视频 → 重定向管线候选来源
