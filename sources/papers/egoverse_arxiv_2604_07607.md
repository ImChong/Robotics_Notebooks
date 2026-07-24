# EgoVerse：面向机器人学习的全球 Egocentric 人类数据集与联盟级迁移研究

> 来源归档（ingest）

- **标题：** EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World
- **类型：** paper / dataset / consortium study / human-to-robot transfer
- **机构：** Georgia Tech；Stanford；UC San Diego；ETH Zürich；MIT CSAIL；Meta Reality Labs Research；Mecka AI；Scale AI（联合）
- **作者（项目主导）：** Ryan Punamiya\*、Simar Kareer\* 等（\*Equal）；学术 PI：Marc Pollefeys、Robert Katzschmann、Xiaolong Wang、Shuran Song、Judy Hoffman、Danfei Xu
- **原始链接：**
  - <https://arxiv.org/abs/2604.07607>
  - 项目页：<https://egoverse.ai/>
  - 代码：<https://github.com/GaTech-RL2/EgoVerse>
  - 数据浏览器：<https://partners.mecka.ai/egoverse>
- **入库日期：** 2026-07-24
- **一句话说明：** 提出 **活的** egocentric 人类示教生态（EgoVerse-A 标准化学术采集 + EgoVerse-I 产业野外规模）与 **EgoDB** 统一接入；当前释放约 **1,362 h / 80k episodes / 1,965 tasks / 240 scenes / 2,087 demonstrators**；并在多实验室、三具身上复现协议，证明 **人–机协同训练有效**，但 **缩放收益依赖域对齐人数据锚定**，且 **场景多样性** 在有限预算下主导泛化。

## 核心论文摘录（MVP）

### 1) 问题：机器人数据贵，静态人数据集难持续扩展

- **链接：** <https://arxiv.org/abs/2604.07607>
- **核心贡献：** 机器人模仿学习依赖规模与多样性，但真机采集贵；egocentric 人数据可提供日常操作行为，但既有人数据集多为 **一次性静态发布**，且跨机构碎片化。EgoVerse 目标是 **持续生长的联盟式人类数据生态** + **可复现的人→机迁移科学**。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [具身规模法则](../../wiki/concepts/embodied-scaling-laws.md)

### 2) 数据集双轨：EgoVerse-A（受控可复现）+ EgoVerse-I（规模与开放任务）

- **链接：** 论文 §III
- **核心贡献：**
  - **EgoVerse-A：** Project Aria（学术标准设备）+ 共享 **dataset unit** 协议；六大 **Flagship Tasks**（object-in-container、cup-on-saucer、bag-grocery、fold-clothes、scoop-granular、sort-utensils）；沿 **任务 / 场景 / 演示者** 三轴控多样性；含受控多样性子集用于消融。
  - **EgoVerse-I：** 产业伙伴定制头戴/双目鱼眼等；近 **1,400 h**、约 **2,000** 开放任务、稠密 **1–2 s** 语言标注，面向 VLA / 通才策略。
  - **统一信号：** ego RGB + **双手 21 关键点** + **6-DoF 头/相机位姿**；A 侧轻量 episode 元数据，I 侧更密语言与上下文标签。
  - **手机采集：** iPhone 头带 ultrawide 1080p@30fps → 云端恢复头姿与手关键点，降低入门门槛。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
  - [HumanNet Table 1](../../wiki/comparisons/humannet-table1-human-video-corpora.md)
  - [EgoScale](../../wiki/methods/egoscale.md) — 同为 Direct 档 egocentric 人数据，但叙事侧重 VLA 预训练缩放

### 3) EgoDB：持续摄入的数据管理与训练接入

- **链接：** 论文 §III-C；项目页 Dataset browser
- **核心贡献：** 云端入库（S3 系存储）→ 统一训练就绪格式 → 夜间预处理/校验/索引；SQL 元数据支持按任务/具身/场景/来源查询；Web 浏览；本地按配置同步过滤子集。相对静态数据集，强调 **living dataset**。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
  - [仓库归档](../repos/egoverse.md)

### 4) 联盟级研究：三具身协议复现 + 协同训练架构

- **链接：** 论文 §IV
- **核心贡献：**
  - **Robot A/B：** 双 ARX5 + Aria / 腕相机（安装与动作表示不同）；**Robot C：** Unitree G1 + Inspire 灵巧手 + ZED 2。
  - **对齐：** 人动作将未来手位姿投到当前 device 帧；人机特征做 **分位数归一化**（1%–99% → [-1,1]）；训练期随机裁剪与颜色抖动。
  - **策略：** 具身专用 stem + 共享 Transformer（HPT 风格）+ **flow matching** 动作解码；BC 协同训练损失 = 人 CFM + 机 CFM。
  - **评测：** 四项旗舰任务，各方法 **20 ID + 20 OOD** rollout，子任务分项聚合为归一化分数。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
  - [EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) — 同实验室后续 WAM 共训路线，野外数据来自 EgoVerse
  - [Manipulation](../../wiki/tasks/manipulation.md)

### 5) 关键发现：共训有效、对齐锚定缩放、场景多样性主导有限预算泛化

- **链接：** 论文 §IV-E / §IV-F / Key Findings
- **核心贡献：**
  1. **人–机共训**在标准化跨实验室设定下 **一致提升** ID/OOD（最高约 **+30%**）；个别具身/任务（如 Robot B 的 bag-grocery）可因策略偏离人示教而下降。
  2. **仅多样人数据或仅域对齐人数据**都不足以驱动显著缩放；**少量域对齐人数据锚定**后，多样 EgoVerse-A 数据才出现正向缩放。
  3. **演示者多样性**提升对未见演示者的鲁棒性；**场景多样性**在有限数据预算下对未见场景泛化更关键，数据密度增加收益递减。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)
  - [具身规模法则](../../wiki/concepts/embodied-scaling-laws.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 6) 局限（论文自述）

- **链接：** 论文 §V
- **核心贡献：** 主线聚焦 **人–机协同训练**，未系统覆盖 pretrain–finetune 等更广算法；场景/演示者多样性消融主要依赖 **离线 Avg-MSE**，需更多真机 rollout 验证是否转化为操作泛化。
- **对 wiki 的映射：**
  - [EgoVerse 论文实体](../../wiki/entities/paper-egoverse.md)

## 当前提炼状态

- [x] Abstract / 贡献三段 / 数据集双轨 / EgoDB / 三具身研究 / 关键发现 / Limitations 已摘录到可维护粒度
- [x] 项目页与 GitHub 开源边界已交叉核查（见 `sources/sites/`、`sources/repos/`）

## BibTeX

```bibtex
@misc{punamiya2026egoverse,
      title={EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World},
      author={Ryan Punamiya and Simar Kareer and Zeyi Liu and Josh Citron and Ri-Zhao Qiu and Xiongyi Cai and others},
      year={2026},
      eprint={2604.07607},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.07607},
}
```

## 交叉链接（sources 互指）

- 项目页：[egoverse-ai.md](../sites/egoverse-ai.md)
- 仓库：[egoverse.md](../repos/egoverse.md)
