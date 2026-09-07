# Robot-Powered Data Flywheels（arXiv:2511.19647）

> 来源归档（ingest）

- **标题：** Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation
- **类型：** paper / data-flywheel / vlm / mobile-manipulation / in-the-wild-deployment
- **来源：** arXiv abs / PDF / HTML（v1，2025-11-24）
- **原始链接：**
  - <https://arxiv.org/abs/2511.19647>
  - PDF：<https://arxiv.org/pdf/2511.19647>
  - 项目页：<https://scanford-robot.github.io/>
- **作者：** Jennifer Grannen、Michelle Pan、Kenneth Llontop、Cherie Ho、Mark Zolotas、Jeannette Bohg、Dorsa Sadigh
- **机构：** 斯坦福大学（Stanford University）；丰田研究所（Toyota Research Institute）
- **入库日期：** 2026-09-07
- **一句话说明：** Robot-Powered Data Flywheel（RPDF）：野外部署机器人边做有用任务边采集域代表数据，经自动标注/策展微调 VLM；Scanford 在东亚图书馆两周扫 2103 架书架，书识别 32%→71.8%，困难 OCR 英/中 +21.8/+7.2 pp；截至入库日无官方代码/数据集链接。

## 开源状态（核查 2026-09-07）

- **项目页 / arXiv：** 无 GitHub、Hugging Face 或数据集下载外链。
- **结论：** **确认未开源**（方法论文 + 演示站）；不可按官方入口复现训练/部署栈。
- **互指：** [`sources/sites/scanford.md`](../sites/scanford.md)；[`wiki/entities/paper-scanford-robot-powered-data-flywheel.md`](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)；[`wiki/entities/scanford.md`](../../wiki/entities/scanford.md)。

## 摘要级要点

- **框架：** RPDF 闭环 — `RobotDeploy → Curate → Aggregate → Fine-tune → 下一轮部署`；机器人从 FM **消费者** 变为 **数据生成器**。
- **Scanford 实例：** Franka FR3 + TidyBot++ 移动底座；腕部 RealSense D435；底座 Unitree L2 LiDAR；东亚图书馆 **中/日/韩** 书架盘点。
- **自动标注：** VLM（Qwen2.5-VL-7B）+ 图书馆目录 **RAG** 候选书；`Curate` 用字符串相似度 + 局部排序校验 → 无需人工标注。
- **数据规模：** 6 h 部署 raw 8232 图 → 策展 **5019** 对；两周 **2103** 书架；馆员估计节省 **18.7 h**；日均人工干预 **2.6** 次。
- **微调：** Qwen2.5-VL-7B，5 epoch，单 H200，1–7 h；书识别 held-out **71.8%**（预训练 32.4%）；困难英/中 OCR **46.6% / 38.0%**。
- **增益饱和：** 约 **1.5 h（≈1352 图）** 后 domain-specific 与 OCR 增益趋平。

## 核心论文摘录（MVP）

### 1) RPDF 形式化与飞轮闭环

- **链接：** <https://arxiv.org/abs/2511.19647> §III、Algorithm 1
- **摘录要点：** 迭代 $t$：机器人用 $\text{FM}_{t-1}$ 采集 $D^{\text{raw}}_t$，经 `Curate` 得 $D_t$，累加 $\mathcal{D}_t$，在 $\text{FM}_0$ 上微调得 $\text{FM}_t$。强调域代表数据同时提升 **域内** 与 **域邻接** 能力。
- **对 wiki 的映射：**
  - [paper-scanford-robot-powered-data-flywheel](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)
  - [Data Flywheel](../../wiki/concepts/data-flywheel.md)

### 2) Scanford 硬件、导航与 VLM 标注

- **链接：** §IV-A、Fig. 2
- **摘录要点：** 预定义货架高度扫描；LiDAR 点云拟合两侧书架平面做漂移校正；RAG 将 sectional catalog 候选书注入 VLM prompt。
- **对 wiki 的映射：**
  - [Scanford](../../wiki/entities/scanford.md)
  - [TidyBot++ 相关文献] — 移动操作底座

### 3) 自动策展与微调配方

- **链接：** §IV-B–C、§V-A
- **摘录要点：** 字符串匹配 + 排序一致性过滤错误 VLM 标签；AdamW $2\times10^{-7}$，bf16，有效 batch 16。
- **对 wiki 的映射：**
  - [paper-scanford-robot-powered-data-flywheel](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)

### 4) 野外部署效用与干预统计

- **链接：** §V-B、Fig. 4
- **摘录要点：** 10 天 × 4 h/天；26 次干预，每次 <5 min 重居中；任务选择与馆员利益对齐（ZPD 类比）。
- **对 wiki 的映射：**
  - [Scanford](../../wiki/entities/scanford.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-scanford-robot-powered-data-flywheel.md`](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)
- [`wiki/entities/scanford.md`](../../wiki/entities/scanford.md)
- [`wiki/concepts/data-flywheel.md`](../../wiki/concepts/data-flywheel.md)
- [`wiki/queries/humanoid-robot-data-collection-landscape.md`](../../wiki/queries/humanoid-robot-data-collection-landscape.md)
