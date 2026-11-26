---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-11-26
venue: "ECCV 2022"
summary: "arXiv:2207.10883v1 针对第一视角视频中的关键步骤发现与排序，构建 62 小时 / 130 人 / 16 任务的 EgoProceL 数据集，并提出纯视觉自监督框架 Correspond and Cut (CnC)：用 TC3I 损失跨视频学习步骤嵌入，再以基于图割的 ProCut 模块定位关键步骤并按视频独立推断执行顺序。"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_noarxiv_129_my-view-is-the-best-view-procedure-learn.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# My View is the Best View

**My View is the Best View: Procedure Learning from Egocentric Videos**（arXiv:2207.10883v1，IIIT Hyderabad & IIT Delhi，ECCV 2022）收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 129/249** 篇，分组 **292 Procedural Learning**。本页基于论文全文完成结构化精读与核实。

## 一句话定义

**My View is the Best View** 提出首个专门面向第一视角无标签长视频的程序学习（Procedure Learning）数据集 **EgoProceL**（62 小时、16 项任务）与纯视觉自监督框架 **CnC（Correspond and Cut）**，通过跨视频时间循环一致性嵌入与图割聚类实现免旁白、可变顺序的关键步骤定位与排序。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CnC | Correspond and Cut | 本文三阶段自监督程序学习框架：跨视频帧对应学习 + 图割关键步骤分割 |
| TC3I | Temporal Cycle Consistency + Contrastive-IDM | 本文提出的复合表征学习损失，兼顾跨视频步骤语义对齐与单视频局部时序连贯性 |
| PCM | ProCut Module | 本文提出的多标签图割（`t-links` + `n-links`，`alpha-Expansion` 求解）关键步骤定位模块 |
| TCC / C-IDM | Temporal Cycle Consistency / Contrastive-Inverse Difference Moment | 分别用于跨序列软最近邻循环匹配与单序列窗口内时序平滑正则 |
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |

## 为什么重要

- **量化验证第一视角相对于第三视角在程序学习中的优势**：在同时拥有第一/第三视角同步录制的 CMU-MMAC 数据集上，同一 `CnC` 框架在第一视角下取得比四种第三视角（Top/Back/LHS/RHS）平均高 **+2.7%** 的 F1-Score（22.7 vs. 均值 20.1），原因在于第一视角遮挡更少、手–物交互区域更清晰且相机位姿更一致。
- **摆脱“固定步骤顺序”与“依赖语音旁白（ASR）”的双重限制**：现实任务（如制作三明治、装机）允许调换部分步骤顺序且常存在大量重复/背景帧；`CnC` 仅用纯视觉模态与自监督损失即可在每段视频中独立解码步骤顺序并统计高频合法流程。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 129/249（Awesome Egocentric Vision） |
| 分组 | 292 Procedural Learning |
| 出处 | ECCV 2022（arXiv:2207.10883v1） |
| 作者与机构 | Siddhant Bansal¹, Chetan Arora², C.V. Jawahar¹（¹IIIT Hyderabad CVIT，²IIT Delhi） |
| 数据规模（EgoProceL） | **62 小时**第一视角视频、**130** 名受试者、**16** 项任务（平均每任务 **8.7** 个关键步骤，平均视频长度 **769.2 s** ≈ 13 分钟）；融合 CMU-MMAC、EGTEA Gaze+、MECCANO、EPIC-Tents 标注并新采集 PC Assembly / PC Disassembly |
| 开源状态 | 已开源（数据集与代码）：<https://sid2697.github.io/egoprocel/> · 镜像 <http://cvit.iiit.ac.in/research/projects/cvit-projects/egoprocel> |

## 核心机制

1. **EgoProceL 多难度第一视角程序学习基准**
   - 用 ELAN 软件对 16 项任务精细标注关键步骤起止时间（Table 7 显示任务特性跨度极大：前景动作占比 `Foreground Ratio` 从 Bacon & Eggs 的 0.15 到 PC Assembly 的 0.79；可缺失率高达 Salad 的 0.30；步骤重复率在 Pasta Salad 达 0.86）。
   - **修正评估指标偏置（§5.1）**：针对未裁剪长视频中背景帧占主导导致传统全局 Hungarian 匹配虚高的问题，提出 **先对每个关键步骤单独计算帧级 Precision / Recall / F1 / IoU，再对所有 $K$ 个步骤取平均**。
2. **三阶段自监督 `CnC`（Correspond and Cut）框架**
   - **阶段一：基于 `TC3I` 损失的跨视频嵌入学习**
     - 以在 ImageNet 预训练的 `ResNet-50`（提取 `Conv4c` 特征，取步长 15 的 $c=2$ 上下文帧堆叠）经两层 3D 卷积、3D 全局最大池化与两层 FC 映射为 **128 维**帧级嵌入（默认 $K=7$，每次随机采样 32 帧，窗口 $\sigma=300$，边界 $\zeta=2.0$，权重 $\xi=1.0$）。
     - **TC3I = TCC + C-IDM**：利用 **TCC**（软最近邻双向循环一致，高斯方差正则）拉近同一任务不同视频中相同关键步骤的嵌入；同时引入 **C-IDM**（Contrastive-Inverse Difference Moment）在单视频内以窗口 $\sigma$ 约束时序相邻帧接近、远距帧互斥，避免重复动作导致 TCC 误匹配到远距离循环帧。
   - **阶段二：基于 `ProCut Module (PCM)` 的图割关键步骤定位**
     - 将聚为 $K$ 个步骤的问题转化为有向图 $G=\langle V, E \rangle$ 的多标签最小割问题：节点包含 $K$ 个终端节点（关键步骤）与 $M$ 个非终端节点（视频帧）。
     - **`t-links` 代价**：由 Fuzzy C-Means 软聚类计算每帧属于各簇的概率 $p$，令分配代价为 $1-p$；**`n-links` 代价**：依据帧间时间距离赋予时序平滑惩罚；使用 **`alpha-Expansion`** 算法求得全局低代价分割。
   - **阶段三：逐视频独立推断关键步骤顺序**
     - 按帧在视频中的归一化时间 $T(v_i^n) = n/p$ 计算每个步骤簇的平均出现时刻并升序排列，从而允许不同视频呈现不同的合法执行路径，最后统计任务中各路径出现频次形成排序榜单。

### 方法流程图

```mermaid
flowchart LR
    A["同任务多段未裁剪第一视角视频 (V_1 ... V_n)"] --> B["ResNet-50 Conv4c + 3D Conv 嵌入器 (128 维帧特征)"]
    B --> C["TC3I 自监督损失: 跨视频 TCC 循环一致 + 单视频 C-IDM 时序平滑"]
    C --> D["ProCut Module (PCM) 构图"]
    D --> E["Fuzzy C-Means 计算 t-links 代价 + 时间距离赋值 n-links"]
    E --> F["alpha-Expansion 多标签图割 -> 帧到 K 个关键步骤的分配"]
    F --> G["按各步骤平均归一化时间排序 -> 逐视频顺序与高频流程榜单"]
```

## 评测与指标

### 1. EgoProceL 第一视角关键步骤定位对比（原文 Table 3，$K=7$，按步骤平均的 F1 / IoU %）

| 方法 | CMU-MMAC (F1 / IoU) | EGTEA G. (F1 / IoU) | MECCANO (F1 / IoU) | EPIC-Tents (F1 / IoU) | PC Assembly (F1 / IoU) | PC Disas. (F1 / IoU) |
|------|---------------------|---------------------|--------------------|-----------------------|------------------------|----------------------|
| `Random` | 15.7 / 5.9 | 15.3 / 4.6 | 13.4 / 5.3 | 14.1 / 6.5 | 15.1 / 7.2 | 15.3 / 7.1 |
| `TC3I + HC`（K-Means 替换 PCM） | 19.2 / 9.0 | 20.8 / 7.9 | 16.6 / 8.0 | 15.4 / 7.8 | 21.7 / 11.0 | 24.9 / 14.1 |
| `TC3I + SS`（子集选择替换 PCM） | 19.7 / 8.9 | 20.4 / 7.9 | 16.3 / 7.8 | 15.9 / 7.8 | 24.8 / 11.9 | 23.6 / 14.0 |
| **`CnC`（TC3I + PCM，本文）** | **22.7 / 11.1** | **21.7 / 9.5** | **18.1 / 7.8** | **17.2 / 8.3** | **25.1 / 12.8** | **27.0 / 14.8** |

### 2. 第三视角基准与视角/模块消融（原文 Table 2, 4, 5, 6）

- **第三视角基准（Table 2，旧标准协议）**：仅使用纯视频输入，`CnC` 在 **ProceL** 达到 **F1 = 21.6**（P 20.7 / R 22.6，超越 Kukleva 等 16.4 达 **+5.2%**，且高于使用“视频+旁白”的 Shen 等 21.1）；在 **CrossTask** 达到 **F1 = 22.6**（P 22.8 / R 22.5，超越 Elhamifar 等 16.3 达 **+6.3%**，同样超越多模态基线 21.0）。
- **第一视角 vs. 第三视角（Table 4，CMU-MMAC 多机位对比）**：`Egocentric` 取得 **F1 = 22.7 / IoU = 11.1**，全面领先第三视角 `Top`（17.9 / 8.1）、`Back`（20.0 / 8.5）、`LHS`（21.8 / 9.7）与 `RHS`（20.6 / 8.7），相对第三视角平均提升 **+2.7% F1**。
- **损失函数消融（Table 5，CMU-MMAC / EGTEA Gaze+）**：`TC3I + PCM`（F1 **22.7 / 21.7**）高于 `TCC + PCM`（19.7 / 19.7）、`LAV + TCC + PCM`（19.7 / 18.6）与 `LAV + PCM`（21.1 / 19.1）。
- **聚类数 $K$ 鲁棒性（Table 6）**：$K=7$ 时表现最佳；当 $K$ 增至 15 时，`CnC` 仍稳定在 20.1 / 19.9，而 K-Means 与子集选择基线掉点明显。

## 与其他工作对比

- **对比弱监督与旁白驱动方法（Alayrac et al.、Fried et al.、Shen et al.）**：不依赖易出错的 ASR 转录文本或预先给出的有序/无序步骤清单，纯视觉自监督即可在第三视角基准上追平甚至反超多模态方法。
- **对比时间轴对齐方法（Kukleva et al.、LAV、TCC）**：摒弃“所有视频关键步骤按相同时间比例和严格顺序出现”的假设，更适配第一视角下步骤跳过、重复或乱序的真实操作流程；后续工作 [STEPs](paper-sa-ego-132-steps-self-supervised-key-step-extraction-and-lo.md) 在此基础上进一步去掉了训练期跨视频配对依赖并引入多模态传感器特征，在 EgoProceL 上取得显著提升。更多横向对照见 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)。

## 常见误区或局限

1. **训练依赖同一任务的多段配对视频（§4.1 & §4.4）**：`TC3I` 损失基于成对视频间的循环一致性构造监督信号，无法像单视频自监督方法那样仅靠 1 条专家演示录像完成训练。
2. **预设固定超参数 $K=7$（§4.4 & Table 7）**：尽管 `CnC` 对 $K$ 变化较稳健，但在实际中仍需手动设定簇数 $K=7$，而 EgoProceL 16 项任务的真实关键步骤数从 4（Greek Salad/Sandwich）跨度到 17（Toy Bike Assembly）。
3. **绝对定位分数仍处于早期可用水平（Table 3）**：在按步骤平均的严格评测协议下，即便最强变体 `CnC` 的 F1 也仅在 **17.2%–27.0%**（IoU 7.8%–14.8%），第一视角剧烈头部运动产生的无关帧和极短步骤（如固定硬盘）仍易丢失。
4. **端到端采样长度受限（Table 11）**：由于训练期需微调 ResNet-50 嵌入网络，每段长视频（平均 13 分钟）每次仅随机采样 32 帧，难以充分利用超长时程全局上下文。

## 结论

**My View is the Best View 奠定了第一视角程序学习的基准与评测标准（EgoProceL），并证明“跨视频循环对应 + 图割步骤分割（CnC）”能够在无文本、可变步骤顺序下超越传统第三视角与强顺序假设方法。**

1. **采集程序教学或机器人示范数据时优先选第一视角** — 头戴第一视角可减少身体遮挡与机位不确定性，在同算法下直接带来约 +2.7% F1 增益。
2. **用“每步骤单独算 F1 再求平均”替代全局帧匹配** — 未裁剪长视频包含大量背景帧，旧指标容易掩盖“把全部帧都分进背景簇”的退化解。
3. **采用 PCM 图割替代 K-Means 可明显改善非固定顺序与超参数敏感性** — `alpha-Expansion` 兼顾软聚类置信度与时间邻域代价，在步骤数 $K$ 失配时退化幅度远小于硬聚类。
4. **若训练期只有极少甚至单条专家视频，应转向无需跨视频对齐的方法** — 如结合冻结预训练特征与多传感器信号的单视频时序对比学习（见 [STEPs](paper-sa-ego-132-steps-self-supervised-key-step-extraction-and-lo.md)）。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 同组对比：[STEPs](paper-sa-ego-132-steps-self-supervised-key-step-extraction-and-lo.md)
- 方法/任务：[vla.md](../methods/vla.md)、[imitation-learning.md](../methods/imitation-learning.md)、[manipulation.md](../tasks/manipulation.md)、[teleoperation.md](../tasks/teleoperation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_noarxiv_129_my-view-is-the-best-view-procedure-learn.md`](../../sources/papers/sun_awesome_ego_noarxiv_129_my-view-is-the-best-view-procedure-learn.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://github.com/ShyamNarasimhan/Procedure-Learning-from-Egocentric-Videos>

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [EgoProceL 官方项目主页](https://sid2697.github.io/egoprocel/)
- [arXiv:2207.10883 原文 PDF](https://arxiv.org/pdf/2207.10883)
