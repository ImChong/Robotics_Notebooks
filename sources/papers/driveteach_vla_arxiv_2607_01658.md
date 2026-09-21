# DriveTeach-VLA: Teaching Vision-Language-Action Models What to See and Where to Look（arXiv:2607.01658）

> 来源归档（ingest）

- **标题：** Teaching Vision-Language-Action Models What to See and Where to Look
- **类型：** paper / vla / autonomous-driving / end-to-end-planning / navsim
- **arXiv：** <https://arxiv.org/abs/2607.01658>
- **代码：** <https://github.com/ShivaTeam/DriveTeach-VLA>
- **机构：** 北京航空航天大学（BUAA）、清华大学智能产业研究院（AIR）、滴滴、中国传媒大学
- **出处：** **ECCV 2026**（README 称 2026-06-24 接收）
- **入库日期：** 2026-09-21
- **一句话说明：** 用 Driving-aware Vision Distillation（DVD）教 VLA「看什么」、2D Trajectory-Guided Prompts（2D-TGP）教「看哪里」，再经 TGP-guided SFT + GRPO 对齐 BEV 轨迹；NAVSIM 与 nuScenes SOTA；Apache-2.0 已开源。

## 开源状态（步骤 2.5，2026-09-21）

| 资源 | 状态 |
|------|------|
| arXiv PDF | **已发布** |
| GitHub 仓库 | **已开源** Apache-2.0 — `data_engine/`、`dvd/`、`sft/`（LLaMA-Factory 补丁） |
| 标注与数据集 | **已发布** — [Google Drive](https://drive.google.com/drive/folders/1oOz6EVfsrxGvOXvYkwxNhcgqjYP-LqSa?usp=drive_link) |
| GRPO 训练代码 | **部分外置** — 见 [Curious-VLA](https://github.com/Mashiroln/curious_vla) |
| 结论 | **已开源**（主仓训练/数据引擎 + 外站数据；RL 阶段需 Curious-VLA） |

## 核心论文摘录

### 1) 问题与动机（Abstract）

- **缺口：** 现有驾驶 VLA 过度依赖 **文本中心 VQA / CoT** 预训练，强调语言推理而非 **action-grounded planning**；表征有语义但缺 **空间依赖**，轨迹预测不稳。
- **主张：** 显式教 VLA **what to see** 与 **where to look**，再 **how to act**。
- **对 wiki 的映射：**
  - [DriveTeach-VLA 论文实体](../../wiki/entities/paper-driveteach-vla.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 2) 三阶段 vision-guided pipeline

| 阶段 | 模块 | 作用 |
|------|------|------|
| **What to see** | **DVD**（Driving-aware Vision Distillation） | bbox 增强图像 → 自蒸馏，向视觉编码器注入 **交通场景先验** |
| **Where to look** | **2D-TGP** + TGP-guided **SFT** | 相机投影可行轨迹关键点，提供与驾驶路径对齐的 **空间 grounding** |
| **How to act** | TGP-guided **GRPO** | 强化学习进一步对齐轨迹预测与驾驶偏好 |

- **推理：** 双模型 — **TGP-Prompter** 从前视图像预测 2D-TGP → **TGP-Planner** 条件于 2D-TGP 生成 **BEV 轨迹**。
- **对 wiki 的映射：**
  - [GRPO](../../wiki/methods/grpo.md)
  - [S²-VLA](../../wiki/entities/paper-s-squared-vla.md)（同为 NAVSIM 驾驶 VLA 对照）

### 3) 实验（Abstract / README）

- **基准：** **NAVSIM**、**nuScenes** — 报告 **SOTA 级**（具体分项见原文 Table）。
- **数据：** 基于 **NAVSIM v2.0.0**；`data_engine/` 可配置 Extract → Enrich → Render 流水线（SampleIR、2D-TGP 投影、LLaMA-Factory 格式）。
- **骨干：** **Qwen2.5-VL** + LLaMA-Factory SFT；DVD 预训练在 `dvd/`。
- **对 wiki 的映射：**
  - [Autonomous Driving Core Algorithms](../../wiki/overview/autonomous-driving-core-algorithms-series.md)

### 4) 工程与复现（README）

- **Data Engine 三 pipeline：** `poutine_label`（VLM 伪标签）、`prompter`（DVD / 2D-TGP）、`planner`（CoT-SFT）。
- **训练入口：** `python data_engine/main.py --config ...`；DVD/SFT 见 `dvd/README.md`。
- **依赖：** Qwen2.5-VL、NAVSIM、Transformers、LLaMA-Factory；Prompt 设计参考 ReCogDrive、Poutine。
- **对 wiki 的映射：**
  - [sources/repos/driveteach-vla.md](../repos/driveteach-vla.md)

### 5) 局限

- **RL 代码外置：** GRPO 阶段不在主仓，需 Curious-VLA。
- **双模型推理：** Prompter + Planner 两趟前向，部署需算力与延迟预算。
- **域绑定：** 数据引擎与评测强依赖 NAVSIM / 驾驶前视设定，迁移到其他机器人操作域需重设计 TGP。
