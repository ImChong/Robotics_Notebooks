# Diffusion-Based World Models: A Survey（Preprints.org, 2026）

> 来源归档（ingest）

- **标题：** Diffusion-Based World Models: A Survey
- **类型：** paper / Review / survey / world-models / diffusion / generative-ai
- **出处：** Preprints.org manuscript **202609.1022**（v1）
- **DOI：** <https://doi.org/10.20944/preprints202609.1022.v1>
- **预印本页：** <https://www.preprints.org/manuscript/202609.1022>
- **代码 / 项目页：** <https://github.com/energy588/Diffusion-based-World-Models>（Living Survey 仓库；见 [`diffusion_based_world_models_survey_github.md`](../repos/diffusion_based_world_models_survey_github.md)）
- **作者：** Gang Wang、Zhen Liu、Mingliang Zhou、Ziying Song、Yugui Zhang、Lei Yang、Yuanyan Tang、Zheng Zhu、Lin Gu、Guang Yang（bibtex 见 GitHub README）
- **机构：** Crossref 元数据 **未附 affiliation**；以预印本 PDF / 作者页为准（截至 **2026-09-25** 未写入 `schema/institutions.json` tag）
- **代码与数据：** **已开源（策展型）** — GitHub 提供 **360+** 论文清单、`papers/` 三分域 markdown、数据集/ benchmark 索引与视觉资产；**非**单一可训练世界模型代码库
- **入库日期：** 2026-09-25
- **一句话说明：** 首篇 **专聚焦 diffusion-based world models** 的结构化综述：连接扩散过程与世界建模原理，按 **自动驾驶 / 具身智能 / 通用世界** 三分域 + 统一 taxonomy，评估优势（多模态、条件可控、高保真）与局限（算力、长程一致、误差累积、评测瓶颈），并给出 open problems 路线图；配套 **Living Survey** GitHub。

## 相关资料

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.20944/preprints202609.1022.v1](https://doi.org/10.20944/preprints202609.1022.v1) | Preprints.org v1 |
| 手稿页 | [202609.1022](https://www.preprints.org/manuscript/202609.1022) | 全文入口（入库环境 HTML 极短，以 Crossref 摘要 + GitHub 为准） |
| GitHub | [energy588/Diffusion-based-World-Models](https://github.com/energy588/Diffusion-based-World-Models) | 论文表、taxonomy 图、数据集 hub |
| 本库机器人 WM 坐标 | [robot-world-models-training-loop-taxonomy.md](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | arXiv:2605.00080 训练闭环三线 |
| 生成式 WM 方法页 | [generative-world-models.md](../../wiki/methods/generative-world-models.md) | 像素/Token rollout 工程折中 |

## 摘要级要点（Crossref JATS）

- **动机：** 扩散模型已成为 **表征型世界模型** 的重要路径，但社区缺少 **专门、系统** 的 diffusion-WM 综述来归纳优势、局限与待解科学问题。
- **内容结构：** 扩散 ↔ 世界建模原理 → 主流方法框架与代表系统 → 进展 / 内在特性 / 常用数据集 → **统一 taxonomy**。
- **世界模仿评估：** 优势含 **多模态假设覆盖、条件可控生成、高保真**；局限含 **计算低效、长程一致、误差累积、评测瓶颈**。
- **展望：** 识别关键 open problems 与未来方向；提供 consolidated reference + research roadmap。

## GitHub Living Survey 要点（README / survey-notes）

| 维度 | 内容 |
|------|------|
| 规模 | **360+** 参考文献（2016–2026） |
| 三分域 | **Autonomous Driving**；**Embodied Intelligence**；**General-purpose Worlds** |
| 能力主题 | 长程演化、多模态融合、交互性、时空一致、环境多样化 |
| Open problems | 效率、可控性、因果推理、物理 grounding、评测与验证 |
| 数据集 hub | 驾驶：nuScenes / Waymo / nuPlan 等；具身：Open X-Embodiment / CALVIN / LIBERO 等；通用：UCF101 / WebVid / Minecraft 等 |

## 核心摘录（面向 wiki 编译）

### 1) 与「泛 WM 综述」的分工

- 既有 survey 覆盖 world models 全谱（理解 vs 预测、机器人、驾驶等），本文 **切片在 diffusion 生成范式**。
- **对 wiki 的映射：** 与 [World Model for Robot Learning（2605.00080）](../../sources/papers/wm_robot_survey_arxiv_2605_00080.md) **正交** — 本文为 **生成机制 / 扩散子族** 索引；后者为 **机器人学习闭环** 坐标。

### 2) 三分域 taxonomy（GitHub）

- **自动驾驶：** 场景生成、轨迹条件仿真、occupancy/BEV 预测、闭环规划、长尾合成。
- **具身智能：** 动作条件预测、操作、VLA 推理、策略学习、sim2real、交互想象。
- **通用世界：** 长视频、数字孪生、游戏式仿真、3D/4D 世界、规模化 foundation WM。
- **对 wiki 的映射：** Mermaid 主干图 + 实体页「与其他工作对比」表。

### 3) 开源与复现边界

- **已开源：** GitHub 策展仓库（markdown 论文表、`assets/` 图、`CONTRIBUTING.md` PR 流程）。
- **非开源：** 综述 PDF 训练代码、统一 benchmark 跑分脚本（各 cited 论文各自仓库）。
- **对 wiki 的映射：** 源码运行时序图 = **维护者更新 Living Survey** 路径。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-diffusion-based-world-models-survey.md`](../../wiki/entities/paper-diffusion-based-world-models-survey.md)**
- 代码归档：**[`sources/repos/diffusion_based_world_models_survey_github.md`](../repos/diffusion_based_world_models_survey_github.md)**
- 交叉：**[`robot-world-models-training-loop-taxonomy.md`](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)**、**[`generative-world-models.md`](../../wiki/methods/generative-world-models.md)**
