# HRDexDB: A Paired Human-Robot Dataset for Cross-Embodiment Dexterous Grasping（arXiv:2604.14944）

> 来源归档（ingest · 全文消化）

- **标题：** HRDexDB: A Paired Human-Robot Dataset for Cross-Embodiment Dexterous Grasping
- **类型：** paper / dataset / dexterous manipulation / cross-embodiment HOI
- **arXiv abs：** <https://arxiv.org/abs/2604.14944>
- **arXiv HTML：** <https://arxiv.org/html/2604.14944v1>
- **项目页：** <https://snuvclab.github.io/HRDexDB/>（归档见 [`sources/sites/snuvclab-hrdexdb-github-io.md`](../sites/snuvclab-hrdexdb-github-io.md)）
- **作者：** Jongbin Lim*, Taeyun Ha*, Mingi Choi, Jisoo Kim, Byungjun Kim, Subin Jeon, Kanghyeon Cho, Seongho Cha, Hanbyul Joo（* equal）
- **机构：** Seoul National University；RLWRLD（Hanbyul Joo 第二单位）
- **入库日期：** 2026-07-01
- **一句话说明：** 首个在**相同 100 种物体**上 **markerless 配对采集** 人类与多种灵巧机器人手抓取序列的大规模多模态数据集：23 路同步相机 + 统一世界坐标系下的 3D 手/机/物轨迹 + Inspire 系列触觉 + 成败标注，服务跨 embodiment 灵巧操作学习。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://snuvclab.github.io/HRDexDB/> | 数据集可视化、触觉/接触演示、BibTeX |
| 近邻对照 | RealDex [17] | 单 embodiment 人–机同步灵巧抓取；缺多手形态与触觉 |
| 近邻对照 | DexWild [26] | 便携手套 + 单相机；任务级对齐、缺稠密 episode 对齐与 3D GT |
| 近邻对照 | RH20T [10] / H&R [33] | 人–机配对但以平行夹爪为主，缺灵巧手与 3D 手重建 |
| 硬件 | Allegro Hand、Inspire RH56DFTP / RH56F1 | 三套机器人末端；xArm6 臂 + XSens + MANUS 遥操 |

## 摘要级要点

- **痛点：** 既有 HOI 集偏人类侧、机器人集偏单 embodiment 或缺 3D/触觉；人–机**同物体配对**且覆盖**多种灵巧手形态**的公开数据极少。
- **规模（论文投稿版）：** **1.4K** 抓取 trial（含成败）· **100** 物体 · **4** embodiment 类型（人 + Allegro + 2 款 Inspire）· **12.8M** 帧 · **2048×1536** RGB。
- **规模（项目页 2026-07）：** 项目页更新为 **2.1K** sequences · **5** embodiments（含持续扩充叙事）。
- **传感：** **21** 第三人称 RGB + **2** 第一人称立体 RGB；机器人关节状态；物体 **6D pose**；人类 **MANO** 参数；Inspire 指尖**触觉力**；成功/失败标签。
- **采集协议：** **机器人先行、人类复现**（robot-driven mimicry），在统一世界坐标系下保持语义配对，允许速度/微动力学差异。
- **系统：** 23 相机 ChArUco 外参 + 手眼标定；机器人遥操用 **XSens 惯性服 + MANUS 手套**（相对视觉遥操更抗遮挡、度量稳定）。

## 核心摘录（面向 wiki 编译）

### 1) 单条机器人 trial 数据结构

$$\mathcal{T}^{\mathrm{robot}}=\left\{\{\mathbf{I}^{c_{i}}_{t}\}_{c_{i}=1}^{21},\,\mathbf{I}^{\mathrm{ego}}_{t},\,\bm{q}^{\mathrm{robot}}_{t},\,\bm{T}^{\mathrm{object}}_{t},\,\bm{F}^{\mathrm{tactile}}_{t},\,y\right\}_{t=1}^{T_{r}}$$

人类 trial 含 21 路 RGB、ego、MANO $\bm{\theta}^{\mathrm{human}}_{t}\in\mathbb{R}^{51}$、物体 6D 与标签 $y$，无机器人触觉项。

### 2) 与代表性数据集对比（Table 1 维度摘要）

| 数据集 | 类型 | 灵巧机器人手 | 触觉 | Markerless | 3D Hand | Obj 6D | 配对人–机 |
|--------|------|-------------|------|------------|---------|--------|-----------|
| DexYCB / ARCTIC / GigaHands | HOI | ✗ | ✗ | 部分 | ✓ | ✓ | ✗ |
| RealDex | ROI | ✓（单） | ✗ | ✓ | ✓ | ✓ | 部分 |
| DexWild | HROI | ✓ | ✗ | ✗ | ✗ | ✗ | 任务级 |
| **HRDexDB** | HROI | ✓（多） | ✓ | ✓ | ✓ | ✓ | **同物体 episode 配对** |

### 3) 机器人平台

- **臂：** 6-DoF **xArm6**
- **手：** **Allegro Hand**；**Inspire RH56DFTP**；**Inspire RH56F1**（尺寸/指形/材料各异）
- **遥操：** XSens 全身 IMU + MANUS 手套 → 臂腕与指关节映射
- **触觉：** Inspire 系列指尖传感器与视觉帧同步

## 对 Wiki 的映射

- **wiki/entities/hrdexdb-dataset.md**：数据集实体页（归纳级）
- **wiki/tasks/manipulation.md**：灵巧操作数据与跨 embodiment 学习入口
- **wiki/queries/dexterous-data-collection-guide.md**：多模态采集系统与配对协议对照
- **wiki/entities/allegro-hand.md**：Allegro 作为采集 embodiment 之一
- **wiki/overview/topic-grasp.md**：抓取专题下的配对 HOI 数据锚点
