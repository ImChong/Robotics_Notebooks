# TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware（arXiv:2607.29567）

> 来源归档（ingest）

- **标题：** TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware
- **类型：** paper / transparent object manipulation / grasp pose estimation / depth completion / laboratory automation
- **arXiv abs：** <https://arxiv.org/abs/2607.29567>
- **arXiv HTML：** <https://arxiv.org/html/2607.29567v1>
- **PDF：** <https://arxiv.org/pdf/2607.29567>
- **作者：** Hailing Hu、Mingyi Zhu、Yiquan An、Yifei Tian、Tianyou Zuo、Jian S. Dai、Lifeng Zhou*（* corresponding；`lifengzhou@pku.edu.cn`）
- **机构：** 北京大学（Peking University；含先进制造与机器人学院）；上海交通大学（Shanghai Jiao Tong University，Mingyi Zhu）；南方科技大学（Southern University of Science and Technology，Jian S. Dai）
- **发表日期：** 2026-07-31（arXiv v1）
- **入库日期：** 2026-08-05
- **硬件：** AUBO i5 六轴臂 + CTAG2F90C 自适应平行夹爪 + 眼在手上 Intel RealSense D435i
- **训练算力：** PyTorch / Ubuntu 20.04 / NVIDIA RTX 4090
- **开源状态（步骤 2.5，截至 2026-08-05）：** **确认未开源** — 论文正文与 arXiv HTML **无项目页 / GitHub / 数据集下载链接**；GitHub 检索无 `TransGraspNet` / `RobotSci-Glass` 官方仓；RobotSci-Glass 数据集仅文内描述
- **一句话说明：** 针对含液透明实验器皿的安全抓取，用 **边界一致（E-CBAM + Edge Branch）→ 表面一致（EGAG 深度补全）→ 物理一致（质心/主轴/力旋量重打分）** 打通感知到执行；真机 clutter **86%** 闭环成功率，**0.5 m/s** 运液 **零洒出**。

## 摘要级要点

- **痛点：** 透明器皿折射/镜面导致 RGB 纹理泄漏、深度空洞；级联管线各自优化 → 边界错 → depth bleeding → 法向崩 → 倾斜/偏心抓取 → 运液洒出。
- **主张：** 核心不只是「各模块更强」，而是 **跨阶段几何–物理一致性**。
- **三原则：**
  1. **Boundary consistency** — TransGraspNet-Det：Mask R-CNN + Enhanced CBAM + Edge Branch，输出可靠轮廓先验。
  2. **Surface consistency** — TransGraspNet-Depth：TDCNet + Edge-Guided Attention Gate (EGAG) + geometry-preserving / MGR 损失，抑制跨边界深度扩散。
  3. **Physics consistency** — GraspNet-1Billion 出 6D 候选后，用径向/角度/质心对齐 + antipodal + wrench-space $Q$ 重打分（线性回归标定权重，无反传）。
- **数据：** 公有 Trans10K / ClearGrasp 预训练 + 自建 **RobotSci-Glass**（20 类，15 类透明；感知子集 5,000+ RGB-D；深度金标 200 场景，不透明涂层采 GT）。
- **真机：** Simple **96%** / Clutter **86%**（各 50 次，含感知→抓取→运输→放置全闭环）；锥形瓶/玻璃瓶半满液体 **0.5 m/s、1.0 m/s²** 水平运移 **零洒出**。

## 核心摘录（面向 wiki 编译）

### 1) 管线（§III）

| 阶段 | 模块 | 关键机制 |
|------|------|----------|
| 检测/分割 | TransGraspNet-Det | E-CBAM 抑背景泄漏；Edge Branch 监督轮廓 |
| 深度补全 | TransGraspNet-Depth | 边界门控融合 RGB 引导；保留法向保真 |
| 抓取精炼 | Geometry–Physics scorer | PCA 主轴/质心 + antipodal + wrench 凸包半径 → $S_{\mathrm{final}}$ |
| 执行 | approach–grasp–lift–place | 基座系变换后标准轨迹 |

### 2) 关键评测数字

| 设置 | 指标 | 结果 |
|------|------|------|
| RobotSci-Glass 分割消融（Table I） | APmask / Boundary F | Full **78.5% / 65.3%**（基线 72.5 / 48.2） |
| ClearGrasp 分割（Table IV） | Boundary F | **65.1%**（优于 TransLab 58.6） |
| ClearGrasp Test-Real 深度（Table V） | RMSE / δ&lt;1.25 | **0.043 m / 91.5%** |
| Top-1 抓取几何质量（Table III） | Succ / 角度误差 / 偏心 | Geo+Phy **98% / 3.8° / 8.5 mm** vs 视觉置信 94% / 22.5° / 35.2 mm |
| 真机闭环（Table VI） | Simple / Hard | **96.0% / 86.0%**（平均 91%） |
| 动态运液 | 洒出 | **0**（0.5 m/s） |

### 3) 与近邻工作对照

| 工作 | 关系 |
|------|------|
| ClearGrasp | 透明深度多阶段优化；本文强调实时可抓取闭环与边界门控补全 |
| TransLab / Trans10K | 透明语义/实例分割；本文强化边界 F 而非仅 AP |
| GraspNet-1Billion | 提供原始 6D 候选；本文做 **后处理物理重排序**（非重训 GraspNet） |
| AnyGrasp / GSNet | 同属检测式抓取家族；本文特化 **透明含液器皿** 与 upright 约束 |

## 对 Wiki 的映射

| 主题 | 关系 |
|------|------|
| [TransGraspNet（论文实体）](../../wiki/entities/paper-transgraspnet.md) | **主沉淀页** |
| [Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md) | 透明场景下的检测式抓取 + 后处理精炼 |
| [Query：抓取策略选型](../../wiki/queries/grasp-policy-selection.md) | 「透明/反光」失败模式的专用补丁 |
| [AnyGrasp vs GraspNet](../../wiki/comparisons/anygrasp-vs-graspnet.md) | GraspNet 候选 + 任务级重打分对照 |
| [Manipulation](../../wiki/tasks/manipulation.md) / [抓取枢纽](../../wiki/overview/hub-grasp.md) | 实验室自动化抓取案例 |

## BibTeX（arXiv）

```bibtex
@misc{hu2026transgraspnet,
  title={TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware},
  author={Hu, Hailing and Zhu, Mingyi and An, Yiquan and Tian, Yifei and Zuo, Tianyou and Dai, Jian S. and Zhou, Lifeng},
  year={2026},
  eprint={2607.29567},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
