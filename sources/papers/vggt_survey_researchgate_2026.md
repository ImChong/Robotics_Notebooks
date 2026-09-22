# VGGT for 3D Reconstruction and Beyond（ResearchGate preprint, 2026）

> 来源归档（ingest）

- **标题：** VGGT for 3D Reconstruction and Beyond: A Survey of Geometric State Strengthening and Its Applications
- **类型：** paper / Survey / 3D reconstruction / VGGT / geometric state
- **年份：** 2026（preprint）
- **DOI：** <https://doi.org/10.13140/RG.2.2.14069.33767>
- **项目页：** <https://richardchen225.github.io/vggt_survey/>
- **GitHub companion：** <https://github.com/richardchen225/Awesome-VGGT>
- **作者：** Ruiyang Chen、Feiran Li、Ruiyang Cheng、Jiashuo Yang、Chu Zhou、Heng Guo*、Boxin Shi、Zhanyu Ma（* corresponding）
- **机构：** 北京邮电大学（BUPT）；北京大学（PKU）；国立情报学研究所（NII，东京）；Independent Researcher
- **入库日期：** 2026-09-22
- **一句话说明：** 以 **几何状态**（多视图 latent **Z<sub>geo</sub>** + 结构化输出 **R<sub>geo</sub>**）统一 VGGT 系 **142 篇** 后续工作：五向 **state strengthening** 与五向 **state reuse**，并系统整理 **71** 个数据集 / 评测任务与三类开放挑战。

## 核心论文摘录

### 1) 几何状态定义（Overview）

- **VGGT** 一次前馈联合预测 **cameras、point maps、depth maps、tracks**。
- **Z<sub>geo</sub>**：跨帧 / 跨视图聚合的多视图几何 **latent features**。
- **R<sub>geo</sub>**：由 latent 解码的 **结构化几何输出**（相机、点图、深度、跟踪）。
- 综述主张：后续文献可归为 **如何加强 Z/R 的形成与持久性** vs **如何在下游任务复用 Z/R**。

### 2) 十类 taxonomy（catalog 2026-09-17）

**State strengthening（63 篇）**

| ID | 方向 | 篇数 | 要点 |
|----|------|------|------|
| S1 | Diverse-Input 3D Reconstruction | 11 | 可选相机/深度、几何感知多传感器融合、鱼眼/全景/事件等成像系统 |
| S2 | Efficient 3D Reconstruction | 22 | 量化压缩 + token merging / sparse attention / 架构加速（FastVGGT、VGGT-Ω 等） |
| S3 | Robust 3D Reconstruction | 5 | 噪声、遮挡、域移下的可靠几何状态 |
| S4 | Streaming and Long-Sequence | 17 | 流式 / 长序列 / 有界内存下的状态保持（LingBot-Map、VGG-T³ 等） |
| S5 | Dynamic 3D Reconstruction | 8 | 动态 / 4D 场景扩展（D4RT 等） |

**State reuse（79 篇）**

| ID | 方向 | 篇数 | 要点 |
|----|------|------|------|
| R1 | Novel View Synthesis | 15 | 冻结或微调几何状态生成新视角 |
| R2 | SLAM | 20 | 前馈几何作 SLAM 前端 / 后端初始化（UniSim-SLAM、SLAMFormer-∞ 等） |
| R3 | Semantic 3D Scene Understanding | 19 | 语义 / 开放词汇 3D 理解 |
| R4 | Geometry-Aware World Models | 10 | 自回归几何世界模型（VGGT-World 等） |
| R5 | Embodied Action and Planning | 15 | 具身动作、规划、手重建等下游 |

### 3) 数据集与评测（Datasets & benchmarks）

- 综述梳理 **71** 个数据集，按 **3D reconstruction / NVS / other downstream** 分组。
- 矩阵化标注各数据集对 **camera pose、depth、3D reconstruction、long stream、dynamic 4D、NVS、SLAM、semantic、world models、embodied** 等评测任务的支持比例。
- 强调：**应用增益** 需区分来自 **几何状态本身改进** 还是 **下游复用方式**——benchmark 设计应控制这一变量。

### 4) 开放挑战（Open challenges）

1. **Reliable and persistent geometric state** — 测量噪声、序列长度、场景运动变化下状态仍可靠。
2. **Adaptive geometric representations for downstream applications** — 单一状态如何暴露不同应用所需信息而无需为每任务重建表示。
3. **Dataset design for benchmarking geometric state** — 可控对比以分离 strengthening vs reuse 的贡献。

## 开源核查（步骤 2.5，2026-09-22）

| 状态 | 说明 |
|------|------|
| **Companion 已开源** | [`richardchen225/Awesome-VGGT`](https://github.com/richardchen225/Awesome-VGGT) + 交互项目页；**非** VGGT 官方训练代码。被引论文各自仓库开放程度见 catalog 内 Code 链接。 |

## 对 wiki 的映射

- 综述视角总览 → [`wiki/overview/vggt-geometric-state-survey.md`](../../wiki/overview/vggt-geometric-state-survey.md)
- 状态估计知识链挂接 → [`wiki/overview/hub-state-estimation.md`](../../wiki/overview/hub-state-estimation.md)
- 站内已有实体（survey catalog 交叉）：[VGG-T³](../../wiki/entities/paper-vgg-ttt.md)、[LingBot-Map](../../wiki/entities/paper-lingbot-map.md)、[SLAMFormer-∞](../../wiki/entities/paper-slamformer-infinity.md)、[Track4World](../../wiki/entities/paper-track4world.md)、[D4RT](../../wiki/entities/paper-d4rt.md)、[Glob3R](../../wiki/entities/paper-glob3r.md)、[Wid3R](../../wiki/entities/paper-wid3r.md)、[UniSim-SLAM](../../wiki/entities/paper-unisim-slam.md)、[VGGT-World](../../wiki/entities/paper-sa-2603-12655-vggt-world-transforming-vggt-into-an-autoregress.md)
