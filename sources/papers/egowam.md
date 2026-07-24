# EgoWAM（项目页 · arXiv 待公布）

> 来源归档（ingest）

- **标题：** EgoWAM: World Action Models Beyond Pixels with In-the-Wild Egocentric Human Data
- **缩写：** **EgoWAM**
- **类型：** paper / world-action-model / egocentric-human-data / human-robot-co-training
- **项目页：** <https://gatech-rl2.github.io/egowam.github.io/>
- **arXiv：** 截至 ingest **尚未在 arXiv 检索到**（项目页提供 Paper / arXiv 入口，待官方公布 ID 后补链）
- **代码：** 项目页标注 **Coming soon**
- **数据集：** [EgoVerse](https://egoverse.ai/)（野外 egocentric 人示教；见本库 [egoverse_arxiv_2604_07607.md](./egoverse_arxiv_2604_07607.md) / [wiki/entities/paper-egoverse.md](../../wiki/entities/paper-egoverse.md)）
- **作者：** Baoyu Li\*、Xinchen Yin\*、Mengying Lin、Yixin Zhang、Danfei Xu（\*Equal contribution）
- **机构：** 佐治亚理工学院（Georgia Tech）· Robot Learning and Reasoning Lab（RL²）
- **资助：** Toyota Research Institute（TRI University 3.0）
- **入库日期：** 2026-07-09
- **一句话说明：** 在 **固定 HPT 骨干、动作头与数据混合** 的受控 **人–机协同训练** 框架下，仅替换 **世界预测目标**（Pixel / DINO / 3D motion flow），系统研究 **野外 egocentric 人数据** 能否通过 WAM **状态预测分支** 弥合 **具身差距**；相对朴素 **BC 协同训练**，WAM 更能随多样化人数据扩展，且 **DINO** 显著改善 **OOD 物体/场景泛化**（最高约 **4×**），**3D flow** 带来 **20–30%** 域内增益，并在 **未对齐人数据** 下仍保持鲁棒（BC 可跌至 **robot-only 以下**）。

## 核心论文摘录（MVP）

### 1) 问题：人数据丰富，但 BC 协同训练未必有益（Abstract / Takeaways）

- **链接：** <https://gatech-rl2.github.io/egowam.github.io/>
- **核心贡献：** **Egocentric 人数据** 为机器人操作提供可扩展监督，但 **行为克隆（BC）** 会把 **可迁移内容**（物体、场景、任务语义）与 **不可迁移因素**（人体形态、头部运动、行为风格）**纠缠** 在一起；因 **具身差距（embodiment gap）**，朴素 **人–机 BC 协同训练** 有时 **损害** 而非提升策略。论文追问：**World Action Models（WAM）** 是否提供更好训练信号——要求策略不仅预测动作，还要预测 **场景如何演化**；并进一步问：**何种世界表征** 最利于 **人→机迁移**。
- **假设（有效世界目标的三条）：** 抽象外观、捕捉 **跨具身一致** 的物理效应、把 **相机运动** 与 **环境变化** 分离。
- **对 wiki 的映射：**
  - [EgoWAM 论文实体](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)（BC 与人数据缩放语境）

### 2) 受控框架：仅替换世界预测头（Method / Architecture）

- **链接：** 项目页 § Method
- **核心贡献：** **EgoWAM** 在 **Heterogeneous Pretrained Transformer（HPT）** 上构建：**具身专用 stem**（ego vision / proprio / wrist vision）→ **共享 Transformer trunk**；**动作头**（共享、**conditional flow matching**）读 **action tokens**；**可替换世界模型头** 读 **future tokens**，在选定目标空间重建 **未来观测**。实验 **固定** 骨干、动作头与 **三源数据混合**，**只改变世界预测目标**：**Pixel（VAE）**、**DINO（RAE）**、**3D motion flow**。推理时 **关闭世界头**，仅走动作路径。
- **对 wiki 的映射：**
  - [EgoWAM 论文实体](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) — 流程总览与 Mermaid
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint 族 + 可替换世界监督轴

### 3) 三源协同训练与野外人数据缩放（Data Gallery / Performance）

- **链接：** 项目页 § Data Gallery、§ Performance
- **核心贡献：** 每项 **双臂真机任务** 混合三类数据：**机器人遥操作**、**域内人示教**（同场景/物体，视点与行为不匹配）、**野外人示教**（EgoVerse：多样场景、物体与演示者）。在 **三项真实双臂任务** 上，**WAM 协同训练** 随 **野外 egocentric 人数据** 扩展 **优于 BC**。**Pixel** 世界预测迁移弱；**DINO** 在 **OOD 物体与场景** 上相对 BC 最高约 **4×** 泛化增益；**3D flow** 在 **域内** 带来约 **20–30%** 性能提升。训练方案对比：**Robot Only**、**+ In-Domain Human**、**+ EgoVerse（野外）**。
- **对 wiki 的映射：**
  - [EgoWAM 论文实体](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)（双臂操作与人数据缩放）
  - [EgoScale](../../wiki/methods/egoscale.md)（对照：VLA 路线的人视频缩放）

### 4) 对齐消融：未对齐人数据下 BC 崩溃、3D Flow 仍鲁棒（Alignment Ablation）

- **链接：** 项目页 § Alignment Ablation
- **核心贡献：** 当协同训练的人数据与机器人 **故意不对齐**（不同抓取策略、更大头部运动）时：**BC** 在 **Bag Grocery** 任务跌至 **20%**，**低于 40% robot-only 基线**；**3D-Flow WAM** 仍保持 **75%**。在 **对齐** 人数据上（Cup on Saucer），各世界目标均优于仅域内设定，**3D Flow** 最高且最稳（**85% → 85%**）。说明 **动力学监督** 比纯动作模仿更能 **抵抗 misalignment**，并把对齐人数据当作 **可扩展飞轮**。
- **对 wiki 的映射：**
  - [EgoWAM 论文实体](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 5) 表征对齐：UMAP 显示 WAM 拉近人–机嵌入（Representation）

- **链接：** 项目页 § Representation (UMAP)
- **核心贡献：** 对 trunk 表征做 **UMAP**：**BC（HPT action features）** 下机器人与人数据 **分簇**；**WAM（3D-Flow 协同训练，concat features）** 把人机嵌入 **拉入共享空间**——与「通过 **任务相关动力学监督** 对齐跨具身表征」的叙事一致。
- **对 wiki 的映射：**
  - [EgoWAM 论文实体](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md)
  - [World Action Models](../../wiki/concepts/world-action-models.md)

## 推荐继续阅读（外部）

- [EgoWAM 项目页](https://gatech-rl2.github.io/egowam.github.io/)
- [EgoVerse 数据集与平台](https://egoverse.ai/)
- [EgoVerse 论文（arXiv:2604.07607）](https://arxiv.org/abs/2604.07607)
- [EgoVerse 本库实体页](../../wiki/entities/paper-egoverse.md)
- [World Action Models 综述（arXiv:2605.12090）](https://arxiv.org/abs/2605.12090)
