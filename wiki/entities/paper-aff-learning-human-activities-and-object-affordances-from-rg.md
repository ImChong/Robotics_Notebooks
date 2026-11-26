---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-11-26
project_id: aff-learning-human-activities-and-object-affordances-from-rg
url: "https://dl.acm.org/doi/abs/10.1177/0278364913478446"
venue: "IJRR 2013"
summary: "Koppula 等（IJRR 2013）提出 CAD-120 数据集与时空马尔可夫随机场（MRF）+ 结构化 SVM 框架，融合多时序分割假设联合标注人类子活动与物体动态 affordance，并在 PR2 机器人上完成辅助与受约束搬运演示。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_learning-human-activities-and-object-affordances-from-rg.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Learning Human Activities and Object Affordances from RGB-D Videos

**Learning Human Activities and Object Affordances from RGB-D Videos**（IJRR 2013，arXiv:1210.1207v2）是人–物交互（HOI）与动态 affordance 建模的早期里程碑工作，贡献了 **CAD-120** RGB-D 长程活动数据集与基于时空马尔可夫随机场（MRF）的联合推理框架。

## 一句话定义

本文通过构建跨越多个时序片段的物体–子活动无向图（MRF），并用结构化支持向量机（SSVM）联合推理人类子活动序列、物体随时间变化的 affordance 标签及高层活动类别。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MRF | Markov Random Field | 马尔可夫随机场，本文用于建模物体与子活动时空关系的概率图 |
| SSVM | Structural Support Vector Machine | 结构化支持向量机，通过割平面法学习 MRF 节点与边势能权重 |
| CAD-120 | Cornell Activity Dataset 120 | 本文采集的 120 段 RGB-D 长程日常活动视频数据集（61,585 帧） |
| MIP | Mixed-Integer Program | 混合整数规划，图割松弛后用于带互斥约束的精确推理求解 |
| VFH | Viewpoint Feature Histogram | 视点特征直方图，用于 3D 点云物体几何分类与跟踪 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |

## 为什么重要

- 首次系统论证了「物体如何使用（affordance）往往比物体类别本身更有助于长程活动理解」，且同一物体在序列中的 affordance 会随子活动动态演化（如水壶依次呈现 `reachable → movable → pourable`）。
- 提出了带有物体–物体、物体–子活动及跨时间边势能的时空图模型，并构造了机器人辅助基准 **CAD-120**，成为后续 ATCRF、Structural-RNN、ST-AOG 等工作的共同基线。
- 顺着 [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md) 与 [技术地图](../overview/affordance-awesome-technology-map.md) 可对照其从图模型向深度时空网络的演进。

## 核心信息

| 字段 | 内容 |
|------|------|
| 作者 | Hema Swetha Koppula, Rudhir Gupta, Ashutosh Saxena（Cornell University） |
| 出处 | IJRR 2013（arXiv:1210.1207v2） |
| 关键产出 | CAD-120 数据集（120 段 RGB-D 视频，4 名被试，10 类高层活动，12 类 affordance） |
| 分组 | Perception · Scope: Scene；Perception Subcategory: Interaction-driven |
| 论文链接 | <https://dl.acm.org/doi/abs/10.1177/0278364913478446> / <https://arxiv.org/abs/1210.1207> |
| 数据与代码（ROS/PCL） | <http://pr.cs.cornell.edu/humanactivities/> |

## 核心机制

1. **RGB-D 骨架提取与检测–跟踪融合**：
   - 使用 OpenNI 骨架跟踪器提取人体上身 8 个关节（相对头部坐标），生成 1,030 维子活动节点特征 `φ_a(i)`。
   - 在 RGB-D 上融合颜色直方图 + HoG 的 One-vs-All SVM 与基于 VFH 的 kNN 检测器，配合 PCL 粒子滤波跟踪，通过路径得分图在「合并 / 孤立 / 忽略检测」间选取最优物体轨迹，提取 180 维物体节点特征 `φ_o(i)`。
2. **多时序过分割假设（Multiple Temporal Segmentations）**：
   - 长程活动边界模糊，单一分割易出错。作者结合「固定长度均匀分割」与「基于关节欧氏距离及距离变化率的 Felzenszwalb-Huttenlocher 图分割」，刻意偏向过分割以生成多组候选分段集合 `H`。
3. **时空 MRF 建模与 SSVM 学习**：
   - 将每个时空片段构建为含物体节点 `V_o` 与子活动节点 `V_a` 的无向图 `G = (V, E)`，能量函数由 6 项对数线性势能组成：`E_w(x, y) = E_o + E_a + E_oo + E_oa + E^t_oo + E^t_aa`，分别刻画节点自身特征、片段内物体–物体（200 维）与物体–骨架（400 维）几何关系，以及相邻片段间的物体（40 维）与子活动（160 维）时序演化。
   - 训练采用割平面结构化 SVM（图割求解损失增广推理）；测试推理在图割线性松弛后接入通用 MIP 求解器以满足单标签互斥约束，推理速度 **74.9 fps**（端到端含特征提取 **16.0 fps**）。
4. **潜变量多分割融合与高层活动分类**：
   - 将各分割假设下的标注 `y^{h_n}` 视为潜变量，通过两步迭代凸优化融合多分割置信度权重 `θ`；随后统计预测子活动/affordance 直方图及物体遮挡率，经多类 SVM 判定 10 类高层活动。

### 方法流程图

```mermaid
flowchart LR
    RGBD["RGB-D Video Stream"] --> SKEL["OpenNI Upper-Body Skeleton Tracker"]
    RGBD --> TRK["HoG-SVM + VFH-kNN + PCL Particle Filter"]
    SKEL --> SEG["Multiple Temporal Segmentations (Uniform + Graph-Based)"]
    TRK --> FEAT["Spatio-Temporal Node and Edge Features"]
    SEG --> FEAT
    FEAT --> MRF["Spatio-Temporal MRF (Objects + Sub-Activities)"]
    MRF -->|"SSVM Weights + MIP Inference"| FUSE["Multi-Segmentation Latent Fusion"]
    FUSE --> OUT1["Dynamic Object Affordances & Sub-Activities"]
    FUSE -->|"Label Histograms + Occlusion Features"| SVM["Multi-Class SVM"]
    SVM --> OUT2["High-Level Activity & PR2 OpenRAVE Response"]
```

## 评测与指标

### 1. CAD-120 四折留一被试交叉验证（Table V，micro P/R %）

| 设置与变体 | Object Affordance | Sub-activity | High-level Activity |
|------------|-------------------|--------------|---------------------|
| 基准：MEMM（Sung et al. 2012，给定 GT 分割） | – | – | 26.4 ± 2.0 |
| 消融：仅单节点 SVM multiclass（无边势能） | 74.2 ± 0.7 | 56.2 ± 0.4 | 30.6 ± 3.5 |
| 消融：无物体上下文（`sub-activity only`） | – | 71.9 ± 0.8 | 27.4 ± 5.2 |
| 消融：无子活动上下文（`object only`） | 86.9 ± 1.0 | – | 59.7 ± 1.8 |
| 消融：无时序边（`no temporal interactions`） | 87.0 ± 0.8 | 76.0 ± 0.6 | 81.4 ± 1.3 |
| 消融：无物体间交互边（`no object interactions`） | 88.4 ± 0.9 | 85.3 ± 1.0 | 80.6 ± 2.6 |
| **完整模型（GT 时序分割 + GT 跟踪）** | **91.8 ± 0.4** | **86.0 ± 0.9** | **84.7 ± 2.4** |
| 完整模型（GT 时序分割 + 自动跟踪） | 88.2 ± 0.6 | 82.5 ± 1.4 | 79.0 ± 4.7 |
| **完整模型（多分割融合 + GT 跟踪，无 GT 分割）** | **83.9 ± 1.5** | **68.2 ± 0.3** | **80.6 ± 1.1** |
| **完整模型（多分割融合 + 自动跟踪，全自动）** | **79.4 ± 0.8** | **63.4 ± 1.6** | **75.0 ± 4.5** |

### 2. CAD-60 与物体跟踪及 PR2 实机验证（Tables III, IV, VI）

- **CAD-60（"New Person" 12 类活动，Table IV）**：在仅使用子活动节点与 20 帧均匀切分时，平均 Precision **80.8%** / Recall **71.4%**，显著高于 Sung et al. (2012) 的 `67.9%` / `55.5%`。
- **物体跟踪（Table III）**：融合检测与跟踪在 **77.8%** 的帧上达到 `>=10%` 边界框重叠率（`>=40%` 重叠占 `53.5%`）。
- **PR2 机器人（Kodiak + Kinect + OpenRAVE，Table VI）**：
  - 辅助人类任务：识别 *Having Meal*（收餐具）、*Taking Medicine*（递水）、*Making Cereal*（将牛奶放回冰箱）后执行预编程响应。
  - 基于 affordance 的桌面清理：可移动物体搬运准确率 **100%**（19 个实例）；对含液体的 `drinkable / pourable / pour-to` 物体执行不倾斜约束搬运，单次观测准确率 **80%**，聚合多次活动观测后达 **100%**（15 个实例）。

## 与其他工作对比

- **对比 Sung et al. (2012) MEMM**：MEMM 仅依赖人体姿态且限制子活动数，在 CAD-120 高层活动分类上仅达 26.4%；加入物体交互上下文后，本方法在同等条件下提升至 84.7%（全自动无 GT 分割下亦达 75.0%）。
- **对比无结构化图边基线（SVM multiclass）**：完整 MRF 将 affordance 准确率从 74.2% 提升至 91.8%，高层活动从 30.6% 提升至 84.7%，证明物体–物体与跨时间边势能的核心贡献。

## 结论

**Koppula 等（2013）确立了「人–物交互上下文 + 物体间关系 + 时序演化」联合建模在动态 affordance 识别中的基础范式，证明即使不依赖深度神经网络，显式时空图推理也能支撑真机辅助任务。**

1. **物体上下文对动作识别收益巨大**：加入物体节点使子活动 micro P/R 提升 14.1 个百分点（71.9% → 86.0%），高层活动准确率提升逾三倍（27.4% → 84.7%）。
2. **间接交互依赖物体–物体边**：未直接被手触碰的物体（如被倒水的杯子 `pour-to`）依赖物体–物体关系边推断，移除该边会使 affordance macro recall 下降 14.9 个百分点。
3. **多分割潜变量融合有效缓解边界噪声**：相比单次最佳切分，多分割融合将 affordance 与子活动的 macro precision 分别提高 5.8% 与 9.1%。
4. **前置感知误差是主要掉点来源**：从 GT 跟踪切到自动跟踪会导致 affordance / 子活动掉点约 3.5%，高层活动掉点约 5.7%。

## 常见误区或局限

1. **强依赖桌面初始化与 OpenNI 骨架质量**：物体检测需在首帧给定桌面物体初始框，且人体遮挡会导致骨架跟踪失真——接触类 affordance 正确样本的手–物平均距离仅 3.8 cm，而错误样本均值高出 43.3%、方差高出 53.8%。
2. **高耗时在于特征计算而非图推理**：端到端速率 16.0 fps（520 帧视频耗时 32.5 s），其中 MIP 推理仅占 6.94 s，其余开销集中在 SIFT/几何特征构建。
3. **真机动作属于预编程原语**：PR2 仅消费推理出的离散 affordance 标签触发 OpenRAVE 预设运动约束（如禁倾倒），未实现端到端连续策略学习；且每个片段假设唯一主导标签，难以表达单片段多并发标签。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_learning-human-activities-and-object-affordances-from-rg.md`](../../sources/papers/affordance_awesome_learning-human-activities-and-object-affordances-from-rg.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://dl.acm.org/doi/abs/10.1177/0278364913478446>（预印本 <https://arxiv.org/abs/1210.1207>）

## 推荐继续阅读

- [Cornell Human Activities 数据与代码主页](http://pr.cs.cornell.edu/humanactivities/)
- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://dl.acm.org/doi/abs/10.1177/0278364913478446)
