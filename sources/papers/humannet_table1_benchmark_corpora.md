# HumanNet 论文 Table 1：代表性人类视频 / 行为语料与官方入口

> 来源归档（ingest）。表格维度与「Embodied Use」等定性标签 **转录自** Deng et al., *HumanNet: Scaling Human-centric Video Learning to One Million Hours*（[arXiv:2605.06747](https://arxiv.org/abs/2605.06747)）正文中的 **Table 1**；**规模数字以论文表格为准**，若与各数据集官方数据卡不一致，以后者为准。

## 总览

| 分组 | 数据集 | 论文表格中的规模（摘要） | 视点 | 活动范围（论文用语） | Embodied Use（论文用语） |
|------|--------|-------------------------|------|----------------------|-------------------------|
| Ego | EPIC-KITCHENS-100 | ~100 h | 第一人称 | 厨房动作 | Limited |
| Ego | Ego4D | ~3,670 h | 第一人称 | 日常活动 | Indirect |
| Ego | HOI4D | 2.4M RGB-D 帧 / >4k 序列 | 第一人称 | 类别级 HOI | Direct |
| Ego | EgoDex | 829 h | 第一人称 | 灵巧操作 | Direct |
| Ego | OpenEgo | 1,107 h | 第一人称 | 灵巧操作 | Direct |
| Ego | EgoScale | 20,854 h | 第一人称 | 灵巧操作 | Direct |
| Ego | EgoVerse | 1,362 h / 80k episodes | 第一人称 | 人类示教 | Direct |
| Exo | ActivityNet | >648 h | 第三人称 | 未裁剪人类活动 | Indirect |
| Exo | Kinetics | 最多约 650k clips | 第三人称 | 人类动作 | Indirect |
| Exo | Charades | 9,848 videos / 68.8 h | 第三人称 | 室内日常 | Indirect |
| Exo | AVA | 430 clips / 107.5 h | 第三人称 | 原子视觉动作 | Indirect |
| Exo | Something-Something V2 | 220,847 videos | 第三人称 | 细粒度交互 | Indirect |
| Exo | HACS | 1.5M clips / 139k segments | 第三人称 | 人类动作片段 | Indirect |
| Exo | FineGym | Gym99 / 288 / 530 等标注规模 | 第三人称 | 细粒度体操 | Indirect |
| Exo | HowTo100M | 136M clips / 1.22M videos | 多为第三人称 | 教学流程 | Indirect |
| Exo | Ego-Exo4D | 1,286 h | 第一 + 第三人称 | 技能型活动 | Indirect |
| Exo | Human2Robot (H&R) | 2,600 episodes | 第三人称 | 人示教→机器人动作学习 | Direct |
| Ours | HumanNet | 1,000,000 h | 第一 + 第三人称 | 细粒度人类活动 | Direct |

## 官方入口（链接索引）

下列为入库时可核对的 **项目页 / 论文 / 常用下载入口**（截至 2026-05-17；若 404 请检索数据集名称 + `official`）。

### Ego-centric 行

- **EPIC-KITCHENS-100**  
  - 项目：<https://epic-kitchens.github.io/>  
  - 数据与基准：<https://github.com/epic-kitchens/epic-kitchens-100-annotations>（常用辅助；主站见上）

- **Ego4D**  
  - 官网：<https://ego4d-data.org/>  
  - 论文入口（项目引用）：<https://arxiv.org/abs/2110.07058>

- **HOI4D**  
  - 项目页：<https://hoi4d.github.io/>  
  - 说明与工具：<https://github.com/leolyliu/HOI4D-Instructions>

- **EgoDex**  
  - 论文：<https://arxiv.org/abs/2505.11709>  
  - 数据与代码入口：<https://github.com/apple/ml-egodex>

- **OpenEgo**  
  - 项目页：<https://www.openegocentric.com/>  
  - 论文：<https://arxiv.org/abs/2509.05513>  
  - 统一格式与工具（常用 GitHub 入口）：<https://github.com/physicalinc/openego>

- **EgoScale**  
  - 论文：<https://arxiv.org/abs/2602.16710>  
  - NVIDIA GEAR 官方页：<https://research.nvidia.com/labs/gear/egoscale/>（仓库内另见 [sources/sites/nvidia-research-egoscale.md](../sites/nvidia-research-egoscale.md)）

- **EgoVerse**  
  - 项目页：<https://egoverse.ai/>  
  - 论文：<https://arxiv.org/abs/2604.07607>  
  - 代码：<https://github.com/gatech-rl2/egoverse>

### Exo-centric 行

- **ActivityNet**  
  - 官网：<http://activity-net.org/>（含 Challenge 与下载说明）

- **Kinetics**（系列；论文表格为 clips 量级）  
  - 论文（Kinetics-700）：<https://arxiv.org/abs/2010.06465>  
  - 常用托管与下载脚本：<https://github.com/cvdfoundation/kinetics-dataset>

- **Charades**  
  - AllenAI / PRIOR：<https://prior.allenai.org/projects/charades>

- **AVA**  
  - Google Research 项目页：<https://research.google.com/ava/>  
  - 论文：<https://arxiv.org/abs/1705.08421>

- **Something-Something V2**  
  - Qualcomm 开发者数据集页：<https://developer.qualcomm.com/software/20bn-something-something-dataset-v2>

- **HACS**  
  - 项目 / 数据说明（常用入口）：<https://github.com/hmehrian/HACS>  
  - 论文：<https://arxiv.org/abs/2012.09943>

- **FineGym**  
  - 项目页：<https://sdolivia.github.io/FineGym/>

- **HowTo100M**  
  - INRIA / Willow 项目页：<https://www.di.ens.fr/willow/research/howto100m/>  
  - 论文：<https://arxiv.org/abs/1906.03327>

- **Ego-Exo4D**  
  - 官网：<https://ego-exo4d-data.org/>  
  - 论文：<https://arxiv.org/abs/2311.18259>

- **Human2Robot (H&R)**  
  - 论文：<https://arxiv.org/abs/2502.16587>  
  - Hugging Face 数据集：<https://huggingface.co/datasets/dannyXSC/HumanAndRobot>

### HumanNet（Table 1 中的「Ours」）

- 论文：<https://arxiv.org/abs/2605.06747>  
- 项目页：<https://dagroup-pku.github.io/HumanNet/>  
- 代码：<https://github.com/DAGroup-PKU/HumanNet/>  
- 仓库内归档：[humannet.md](humannet.md)、[../repos/humannet.md](../repos/humannet.md)

## 对 wiki 的映射

- [HumanNet 相关人类视频语料（Table 1 对照）](../../wiki/comparisons/humannet-table1-human-video-corpora.md)：把上表提炼为 **视点 × 活动语义 × 具身向可用性** 的选型阅读框架，并回链本索引页与各实体/方法页。
