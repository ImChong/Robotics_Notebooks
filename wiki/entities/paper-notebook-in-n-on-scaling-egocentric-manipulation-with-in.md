---
type: entity
tags: [paper, humanoid-paper-notebooks, humanoid, manipulation, egocentric, human-video, vla, co-training, ucsd, unitree, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-09-28
arxiv: "2511.15704"
code: https://github.com/XiongyiCai/Human0
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../methods/π0-policy.md
  - ./paper-hrl-stack-34-gr00t_n1.md
  - ./paper-notebook-humanoid-policy-human-policy.md
  - ../concepts/data-flywheel.md
  - ./paper-egowam-egocentric-human-wam-co-training.md
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/humanoid_pnb_in-n-on.md
  - ../../sources/papers/sun_awesome_ego_2511_12643_in-n-on-scaling-egocentric-manipulation.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
summary: "第一视角（egocentric）视频是学操作策略的宝贵可扩展数据源，但数据异质性大，多数方法只把人类数据用于简单预训练，没释放全部潜力。本文先给出一套可扩展配方：把人类数据分成两类——野外（in-the-wild）与任务对齐（on-task），并系统分析如何使用。作者整理出数据集 PHSD，含 1000+ 小时多样野外第一视角数据与 20+ 小时直接对齐目标任务的任务数据。据此训练一个大型语言条件流匹配策略 Human0；配合域适应技术，Human0 缩小人到人形的差距。实证表明，规模化人类数据带来若干新性质：仅凭人类数据就能听从语言指令、少样本学习、以及用任务数据提升的鲁棒性。"
---

# In-N-On

**In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

第一视角（egocentric）视频是学操作策略的宝贵可扩展数据源，但数据异质性大，多数方法只把人类数据用于简单预训练，没释放全部潜力。本文先给出一套可扩展配方：把人类数据分成两类——野外（in-the-wild）与任务对齐（on-task），并系统分析如何使用。作者整理出数据集 PHSD，含 1000+ 小时多样野外第一视角数据与 20+ 小时直接对齐目标任务的任务数据。据此训练一个大型语言条件流匹配策略 Human0；配合域适应技术，Human0 缩小人到人形的差距。实证表明，规模化人类数据带来若干新性质：仅凭人类数据就能听从语言指令、少样本学习、以及用任务数据提升的鲁棒性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Egocentric | 第一视角（头戴视角） |
| In-the-wild / On-task | 野外 / 任务对齐数据 |
| PHSD | 本文数据集（1000h 野外 + 20h 任务） |
| Human0 | 语言条件流匹配策略 |
| Flow Matching | 流匹配生成策略 |
| Domain Adaptation | 域适应，缩小人↔人形差距 |

## 为什么重要

- **"野外 + 任务对齐"二分**是用好海量人类数据的关键洞见：规模来自野外、对齐来自少量任务数据；
- **语言条件 + 流匹配**让策略可指令驱动且高效；
- **域适应**是人到人形落地的必备环节；
- 与 Dexterity from Smart Lenses、EgoDex 等共同推进第一视角数据规模化。

## 解决什么问题

第一视角人类数据潜力大但用不好： - **异质性大**，多数只做**简单预训练**； - 缺**如何分类与使用**数据的系统配方； - 人到人形有**域差距**。

In-N-On 要：一套**可扩展配方**（野外 + 任务对齐）+ 数据集 + 策略，释放第一视角数据潜力。

## 核心机制

1. **可扩展第一视角数据配方**：野外 + 任务对齐两类 + 使用分析；
2. **PHSD 数据集**：1000h 野外 + 20h 任务对齐；
3. **Human0 语言条件流匹配 + 域适应**：缩小人↔人形差距；
4. **涌现新性质**：仅人类数据听指令、少样本、任务数据增鲁棒。

方法拆解（深读笔记小节）：数据分类：野外 + 任务对齐；PHSD 数据集；Human0：语言条件流匹配 + 域适应；涌现新性质；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html> |
| arXiv | <https://arxiv.org/abs/2511.15704> |
| 源码 | **已发布**：仓库 [XiongyiCai/Human0](https://github.com/XiongyiCai/Human0)、数据 [PHSD](https://huggingface.co/datasets/XiongyiC/PHSD)、模型 [Human0](https://huggingface.co/XiongyiC/Human0)；截至 2026-09-28 仓库 README 只有流程简介，未给出训练 / 部署命令 |
| 作者 | Xiongyi Cai、Ri-Zhao Qiu、Geng Chen、Lai Wei、Tianshu Huang、Xuxin Cheng、Xiaolong Wang（UC San Diego） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**设置**：PHSD 数据集 = 1000+ 小时野外第一视角人类 / 人形数据 + 20+ 小时与目标任务对齐的「on-task」数据。Human0 以 π₀ 权重初始化，8×H200 训练 10 万步得到基座；后训练在单张 H100 上 3 万步。平台为 Unitree G1（主要）与 H1，均配 Inspire 五指手。

真机结果（论文 Table 1，I.D. / O.O.D.）：

| 方法 | 单物体抓取 | 多物体抓取 | 汉堡组装 | 倾倒（1 条机器人演示） |
|------|------|------|------|------|
| π₀ | 19/20 · 19/20 | 25/30 · 16/30 | 5/12 · 3/12 | 0/20 |
| GR00T N1 | 18/20 · 13/20 | 6/30 · 8/30 | 4/12 · 3/12 | 0/20 |
| HAT（含人类数据） | 17/20 · 15/20 | — | — | 2/20 |
| Human0 去掉人类数据 | 18/20 · 18/20 | 23/30 · 15/30 | 7/12 · 2/12 | 2/20 |
| **Human0** | **20/20 · 19/20** | **29/30 · 30/30** | **8/12 · 7/12** | **5/20** |

- **零样本语言跟随**：只在人类数据里出现过的物体 / 食材（如马苏里拉奶酪），Human0 也能按指令抓取；π₀ 在多物体 O.O.D. 中近似随机二选一。
- **1-shot 学新行为**：双臂倾倒只给 1 条机器人演示，Human0 达 5/20；但仅靠人类数据学全新行为在当前规模下仍做不到（论文原话 "the answer is no"）。
- **域判别器消融**（梯度反转）：分阶段倾倒成功率 15% → 25%；线性探针显示加判别器后中间特征无法区分具身（概率约 50%）。
- 人类数据在机器人数据少时的正则化作用最明显（单物体抓取随机器人数据比例变化的曲线）。

## 与其他工作对比

| 工作 | 数据配方 | 与 In-N-On 的差异 |
|------|------|------|
| [HAT](./paper-notebook-humanoid-policy-human-policy.md) | 仅任务对齐的 PH2D | 专才策略、不能做语言条件预训练；In-N-On 再加 1000+ 小时野外数据做基座 |
| [π₀](../methods/π0-policy.md) / [GR00T N1](./paper-hrl-stack-34-gr00t_n1.md) | 机器人数据为主 | 多物体 O.O.D. 与汉堡组装明显落后，未见指令跟随弱 |
| [EgoWAM](./paper-egowam-egocentric-human-wam-co-training.md) | 野外第一视角人数据 + 世界–动作模型 | 同样利用野外人数据，但用世界模型而非人体中心动作表示 |
| [EgoDex](./paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md) | 大规模第一视角灵巧操作数据 | 提供数据；In-N-On 回答的是「野外 vs 任务对齐」如何组合使用 |

## 结论

**In-N-On 真正的主张是「人类第一视角数据该怎么切」——把野外数据与任务对齐数据分开、各司其职，而不是又训一个更大的预训练模型。**

- 规模与对齐是两件事：PHSD 用 **1000+ 小时野外** 数据买规模、**20+ 小时任务对齐** 数据买对齐；后者体量很小但负责鲁棒性，这正是「只拿人类数据做简单预训练」漏掉的部分。
- 承载载体是 **Human0**（大型语言条件流匹配策略）加域适应；论文把域适应放在必备位置，而非可选加分项——人↔人形的差距不会靠数据量自动消失。
- 最值得注意的是涌现性质：仅凭人类数据就能听语言指令、少样本学习，说明当前瓶颈更多在 **数据配方**，而不只是机器人本体数据不够。
- 边界：域差距靠梯度反转判别器压缩（倾倒 15%→25%），但仅凭人类数据仍学不会全新行为，至少需要少量机器人演示。
- 谱系上与 Dexterity from Smart Lenses、EgoDex 同属「第一视角数据规模化」路线，差别在本文补的是 **使用配方** 而非又一个采集设备。

## 局限与风险

- **只在人形上验证**：G1 / H1 + 灵巧手，其他机器人形态留作后续（论文结论）。
- **不能仅靠人类数据学新行为**：需要至少少量机器人演示；1-shot 倾倒只有 5/20。
- **on-task 数据仍需专门采集**：20+ 小时任务对齐数据是鲁棒性的来源，换任务就要重录。
- **训练成本**：基座训练 8×H200；后训练可单卡，但依赖基座权重。
- **开源边界**：仓库、数据、模型均已发布，但 README 未给运行命令；源码运行时序图 **不适用**（无可辨识的训练 / 部署入口）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 初始化权重与对比基线 π₀：[π0-policy](../methods/π0-policy.md)
- 对比基线 GR00T N1：[paper-hrl-stack-34-gr00t_n1](./paper-hrl-stack-34-gr00t_n1.md)
- 同组前作 HAT / PH2D：[paper-notebook-humanoid-policy-human-policy](./paper-notebook-humanoid-policy-human-policy.md)
- 人类数据作为可扩展数据源：[data-flywheel](../concepts/data-flywheel.md)
- 野外第一视角人数据协同训练的另一路线：[paper-egowam-egocentric-human-wam-co-training](./paper-egowam-egocentric-human-wam-co-training.md)

## 参考来源

- [humanoid_pnb_in-n-on.md](../../sources/papers/humanoid_pnb_in-n-on.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html>
- 论文：<https://arxiv.org/abs/2511.15704>
- 论文正文（Table 1–2、消融）：<https://arxiv.org/html/2511.15704>
- 仓库：<https://github.com/XiongyiCai/Human0>
- [`sources/papers/sun_awesome_ego_2511_12643_in-n-on-scaling-egocentric-manipulation.md`](../../sources/papers/sun_awesome_ego_2511_12643_in-n-on-scaling-egocentric-manipulation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)

## 推荐继续阅读

- [机器人论文阅读笔记：In-N-On](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html)
- 项目页：<https://xiongyicai.github.io/In-N-On/>
- PHSD 数据集：<https://huggingface.co/datasets/XiongyiC/PHSD>
