# Vision-Based Tactile Intelligence for Robotics: Sensing, Learning, and Embodied Manipulation

> 来源归档（ingest）

- **标题：** Vision-Based Tactile Intelligence for Robotics: Sensing, Learning, and Embodied Manipulation
- **类型：** paper（survey）/ vbts / tactile-foundation-model / sim-to-real / tactile-datasets
- **arXiv abs：** <https://arxiv.org/abs/2608.15490>
- **arXiv HTML：** <https://arxiv.org/html/2608.15490>
- **PDF：** <https://arxiv.org/pdf/2608.15490>
- **机构：** 大湾区大学（Great Bay University）；香港大学（The University of Hong Kong）；清华大学（Tsinghua University）；南洋理工大学（Nanyang Technological University）；香港理工大学（The Hong Kong Polytechnic University）；华南理工大学（South China University of Technology）；KTH 皇家理工学院（KTH Royal Institute of Technology）；伦敦国王学院（King's College London）
- **入库日期：** 2026-09-23
- **复核日期：** 2026-11-18
- **原文版本：** arXiv:2608.15490v1（2026-08-16）
- **核对范围：** arXiv HTML 原文 §I–VII，Fig. 1–5、Table I–IV 及相关叙述；覆盖传感原理、硬件分类、感知学习、仿真与数据集、Sim2Real 和未来挑战。
- **通读：** 已完成
- **作者：** Peng Zhou, Jun Hu, Sihan Chen, Zeqing Zhang, Haofei Ma, Zhenyu Lu, Sichao Liu, Xueqian Wang, Pai Zheng, Xiang Li, Shan Luo, Jia Pan, David Navarro-Alarcon, Chenguang Yang, Michael Yu Wang
- **一句话说明：** 将 VBTS **硬件 taxonomy**（弹性体 / 尺寸形状 / 光学系统）、**分层触觉智能学习**（低层信号 → 任务策略 → foundation models）与 **仿真–数据集–sim2real**  scaling 层作为 **一体化 sensing-and-learning 系统** 综述，并讨论 open challenges。

## 开源核查（2026-11-18 复核）

| 项 | 状态 |
|----|------|
| 综述配套代码仓 | **未列**（复核 arXiv v1 HTML 原文，未见该综述官方 GitHub 或可运行入口；引用的第三方项目不视为本综述源码） |
| 引用数据集/模型 | 指向 Sparsh、GelSight、DIGIT、TacBench 等第三方开源资源（见正文 Table / 引用） |
| 结论 | **N/A（综述）** — 无独立「论文代码」；以文献索引与 taxonomy 为主 |

## 原文表格补充核对（v1）

- **Table I：光学读出** — marker tracking 适合解释剪切/滑移，受 marker 密度与跟踪质量约束；photometric stereo 适合细纹理和几何恢复，依赖照明与标定；stereo vision 通过视差求 3D 几何，但硬件复杂；shading-based 重建允许非理想光学，存在推断歧义。
- **Table II：硬件三轴** — 弹性体（多模态表皮、标记/图案、弹性体本体）、尺寸形状（大面积、操作器集成、微型探头）、光学（照明、紧凑光路、多通道、事件相机）；不能把表中器件排序理解为统一性能排行榜。
- **Table III：学习模型** — Sparsh/T3/AnyTouch 等偏通用表征；OmniVTLA、T-Rex、UniTacVLA 等进一步将触觉接入动作策略；表中实验数据与结果分别来自其所引用的论文，并非本综述训练或复现。
- **Table IV：仿真/数据/评测** — TACTO、Taxim、DiffTactile、Taccel、SimTac 是不同路线的仿真平台；TacBench/ManiFeel 是评测资源；ObjectFolder、Touch100K、TVL 等是数据资源。
- **§VI 七类挑战** — 灵巧手尺寸匹配、全身表皮、触觉基础模型与 VTLA、Sim2Real、多模态对齐、示教触觉反馈、第一视角触觉数据采集。

## 摘要级要点

- **动机：** 传统电子触觉 spatial resolution / 信号 richness 有限；VBTS 将接触变形转为图像，天然对接 CV / multimodal / robot learning。
- **贡献 1 — 硬件 taxonomy：** 相对 GelSight baseline，沿 **deformable elastomer**、**sensor size & shape**、**optical system** 三维度组织代表性 VBTS（Fig. 3 分类图）。
- **贡献 2 — 学习层次：** 低层信号理解 → 物理推理 / 几何位姿 → 识别与 multimodal fusion → manipulation → **tactile foundation models**。
- **贡献 3 — Scaling 层：** 仿真平台、触觉数据集、sim-to-real、**cross-sensor adaptation** 作为训练/评测/部署基础设施。
- **信息层级：** VBTS 信号分 **直接几何**、**间接力相关**、**时序动态**（滑移、接触转移）三层，推理难度递增。
- **光学机制：** Table I 对比 marker tracking、photometric stereo、stereo vision、shading-based reconstruction。
- **与既有 survey 差异：** 强调 hardware + learning + simulation + datasets **耦合** 为 embodied intelligence 组件，而非孤立子话题。

## 核心论文摘录（MVP）

### 1) VBTS 流水线：从接触到光学读数

- **链接：** <https://arxiv.org/html/2608.15490#S2>
- **摘录要点：** 四组件（弹性体、照明、成像光学、图像传感器）；四阶段（接触变形 →  optical response → 图像采集 → 物理量推断）。
- **对 wiki 的映射：**
  - [触觉传感](../../wiki/concepts/tactile-sensing.md) — VBTS 原理与信息层级
  - [GelSlim](../../wiki/entities/gel-slim.md) — 硬件族谱实例

### 2) 硬件设计 taxonomy

- **链接：** <https://arxiv.org/html/2608.15490#S3> Figure 3
- **摘录要点：** 按 elastomer interface / form factor / optics 分类 DM-Tac、TacTip、DIGIT、GelStereo、TacShade 等；solid vs dashed border 表主/次贡献维度。
- **对 wiki 的映射：**
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)

### 3) 学习、仿真与 tactile foundation models

- **链接：** <https://arxiv.org/html/2608.15490#S4>–§V
- **摘录要点：** 触觉表征、力/滑移推理、pose、multimodal fusion、manipulation policy、SSL foundation（Sparsh 等）；仿真与数据集作 scaling；sim2real / cross-sensor。
- **对 wiki 的映射：**
  - [Sparsh 归档](./sparsh_arxiv_2410_24090.md)
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)
  - [SimTac（Awesome Touch）](../../wiki/entities/paper-sa-2511-11456-simtac-a-physics-based-simulator-for-vision-base.md)
  - [VLA](../../wiki/methods/vla.md) — VTLA / tactile-VLA 小节语境

### 4) Open challenges（§VI）

- **链接：** <https://arxiv.org/html/2608.15490#S6>
- **摘录要点：** 标准化 benchmark、跨传感器泛化、实时部署、数据规模、与 LLM/VLA 融合等未来方向（详见正文）。
- **对 wiki 的映射：**
  - [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)

## 对 wiki 的映射（汇总）

- 概念枢纽：[触觉传感](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)
- 方法/任务：[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)
- 本批 ingest 交叉：[Sparsh](./sparsh_arxiv_2410_24090.md)、[Tactile-VLA](./tactile_vla_arxiv_2507_09160.md)、[OmniVTLA](./omnivtla_arxiv_2508_08706.md)、[TaF-VLA](./taf_vla_arxiv_2601_20321.md)
- **唯一详情节点：** [Vision-Based Tactile Intelligence](../../wiki/entities/paper-vision-based-tactile-intelligence.md)（已存在，避免同一 arXiv ID 重复建页）

## 当前提炼状态

- [x] 三大贡献轴、光学机制表、学习/仿真层次、open challenges 框架已摘录
- [x] v1 正文 §I–VII 与 Table I–IV 的结论、局限已复核；未逐项转录表格中全部传感器/论文条目

## BibTeX

```bibtex
@article{zhou2026visionbasedtactile,
  title={Vision-Based Tactile Intelligence for Robotics: Sensing, Learning, and Embodied Manipulation},
  author={Zhou, Peng and Hu, Jun and Chen, Sihan and Zhang, Zeqing and Ma, Haofei and Lu, Zhenyu and Liu, Sichao and Wang, Xueqian and Zheng, Pai and Li, Xiang and Luo, Shan and Pan, Jia and Navarro-Alarcon, David and Yang, Chenguang and Wang, Michael Yu},
  journal={arXiv preprint arXiv:2608.15490},
  year={2026},
  url={https://arxiv.org/abs/2608.15490},
}
```
