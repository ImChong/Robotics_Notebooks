# HUMEMBR（arXiv:2606.30404）

> 来源归档（ingest）

- **标题：** HUMEMBR: Learning Human Routines for Predictive Embodied Navigation
- **缩写：** **HUMEMBR**（Human-Centered Memory for Embodied Robots）
- **类型：** paper / embodied-qa / long-horizon-memory / person-reid / semantic-navigation / routine-conditioned-navigation
- **arXiv：** <https://arxiv.org/abs/2606.30404>
- **会议：** IROS 2026（项目页声明）
- **项目页：** <https://samirahuber.github.io/humembr/>
- **代码：** <https://github.com/samirahuber/humembr>（截至 2026-08-06：**已开源**可运行入口；COBD 数据集 README 标明 **private**）
- **作者：** Samira Huber, Klaas Pelzer, Duc M. Nguyen, Xuesu Xiao, Sören Pirk
- **机构：** 基尔大学（Kiel University）；乔治梅森大学（George Mason University）
- **入库日期：** 2026-08-06
- **一句话说明：** 面向办公等真人日常环境的 **人中心长时程记忆** 系统：并行构建身份感知记忆（人脸 DBSCAN + KPR ReID、Qwen 字幕 + 向量检索）与 LLM **工具调用检索**，支持 PersonEQA 与 Spot 上的例行条件导航；相对全上下文基线用更少 token 提升长程推理。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2606.30404>；项目页 <https://samirahuber.github.io/humembr/>
- **核心贡献：** 传统度量地图 / 场景图偏静态空间；既有 EQA / 检索记忆（如 ReMEmbR、Mind Palace）偏物体或短时程，缺 **多日身份感知例行模式**。HUMEMBR 贡献三点：
  1. **PersonEQA** 基准：评测跨日身份聚类与例行推理（六类问答）；
  2. **HUMEMBR** 检索增强 LLM agent：结构化记忆 + 函数调用；
  3. **Spot 真机** 两环境定性/定量部署（预载 COBD 记忆 + GraphNav）。
- **对 wiki 的映射：**
  - [HUMEMBR 论文实体](../../wiki/entities/paper-humembr.md)
  - [视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)（EQA / 语义导航边界）
  - [Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md)（统一 EQA agent 对照）

### 2) 并行记忆构建与查询（§III）

- **链接：** Methodology；Fig. 2–3
- **核心贡献：**
  - **Memory Building：** 2 Hz RGB → ResNet50 去冗余 → GraphNav 路点 + 时间戳；Qwen3-VL 字幕 → mxbai-embed 向量；YOLO 人体 → InsightFace 脸嵌入（DBSCAN 锚）+ KPR 全身 ReID（无脸匹配）。
  - **Querying：** LLM 迭代调用至多五类检索函数（语义观测 / 路点观测 / 人物观测 / 当日人物集 / 人物日摘要）+ `Navigate to waypoint`；相关度 \(s_i=\alpha d_i+\beta(1-\exp(-\lambda\Delta t_i/3600))\)。
  - 与「先建库再离线问」不同：构建与查询 **并发实时**。
- **对 wiki 的映射：**
  - [HUMEMBR 论文实体](../../wiki/entities/paper-humembr.md) — 流程总览 / 源码时序
  - [Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md) — LLM/agent 导航原语对照

### 3) COBD 与 PersonEQA（§IV）

- **链接：** Data
- **核心贡献：** **COBD**（Collaborative Office Behavior Dataset）：Spot 办公室 20 天、31 小时、136 路点、>13k 图；无脚本日常行为。**PersonEQA**：200 题、六类（Spatial / Temporal-Time / Temporal-Duration / Binary / Descriptive / Person ID）；上下文跨小时到多日。
- **对 wiki 的映射：**
  - [HUMEMBR 论文实体](../../wiki/entities/paper-humembr.md) — 评测节
  - [导航纵深 Stage 3/4](../../roadmap/depth-navigation.md)

### 4) PersonEQA 与真机（§V–VII）

- **链接：** Experiments / Results / Real-World Deployment
- **核心贡献：**
  - Gemini 3 Flash 版总准确率 **75.41%** vs 全字幕上下文 **67.33%**，且约 **17%** token（106k vs 632k/题）；开源 Qwen3-VL-235B **66.01%**、约 **2%** token。
  - 全上下文在 **Spatial** 上仅 **34.62%**（上下文稀释），HUMEMBR Gemini **92.31%**。
  - 函数调用上限约 **10** 次较稳；ReID 消融伤 Person/Spatial；字幕偏好 **Interaction-centered**。
  - 真机六类任务（例行搜人 / 多步 / 取件递送 / 遮挡稳健性）：多数高成功率；失败主因遮挡/逆光/幻觉缺席人物。
  - 伦理：同意采集、假名 ID、不支持非自愿监控。
- **对 wiki 的映射：**
  - [导航·SLAM 栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)（GraphNav / Spot 低层对照）
  - [iCrowdNav](../../wiki/entities/paper-icrowdnav.md)（人群导航 vs 人例行记忆）

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-humembr.md`](../../wiki/entities/paper-humembr.md)
- 仓库归档：[`sources/repos/humembr.md`](../repos/humembr.md)
- 项目页：[`sources/sites/samirahuber-humembr-github-io.md`](../sites/samirahuber-humembr-github-io.md)
- 互链参考：[VLN](../../wiki/tasks/vision-language-navigation.md)、[Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md)、[Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md)、[导航栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)、[iCrowdNav](../../wiki/entities/paper-icrowdnav.md)、[导航纵深](../../roadmap/depth-navigation.md)

## BibTeX（项目页）

```bibtex
@inproceedings{huber2026humembr,
  title     = {HUMEMBR: Learning Human Routines for Predictive Embodied Navigation},
  author    = {Huber, Samira and Pelzer, Klaas and Nguyen, Duc M. and Xiao, Xuesu and Pirk, S{\"o}ren},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
}
```
