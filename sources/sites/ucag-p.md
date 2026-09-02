# UCAG-P 项目页（Public-BOTs）

> 来源归档（ingest）

- **标题：** UCAG-P: One Policy, Many Embodiments
- **类型：** site（官方项目页）
- **发布方：** Xiaomi Embodied Intelligence Team；University of Macau
- **原始链接：** <https://public-bots.github.io/UCAG-P>
- **论文：** <https://arxiv.org/abs/2608.26058>
- **代码入口：** 页头写 *Project Page & Code*；实际 GitHub [Public-BOTs/UCAG-P](https://github.com/Public-BOTs/UCAG-P) 截至 **2026-09-02** **无训练/推理代码**
- **复核日期：** 2026-09-02
- **入库日期：** 2026-08-28
- **一句话说明：** 官方对外页：用相机可观测锚点把异构操作演示接到共享几何接口，再经几何条件翻译器出各本体命令；展示 11 数据集 / 6,300+ 小时 / 9 embodiment / 单 checkpoint 数字与真机 Piper 视频。

## 开源状态（项目页核查，2026-09-02）

| 项 | 状态 |
|----|------|
| Paper / arXiv | 已挂链（2608.26058） |
| Code | **coming soon** — 徽章与 README 均写 Release Soon；仓内仅 `assets/`、`web-page/`、README |
| 权重 / 数据下载 | **未列** |
| 复现范围 | **宣称将开源**；当前可看视频、架构图与 BibTeX，不可跑训练 |

## 页面结构（策展）

- **Abstract / 数字条：** 11 datasets、6,300+ hours、9 embodiments、1 unified checkpoint
- **System overview：** 单臂 / 双臂 / 灵巧手 / 人手 → \(p_0/p_1\) 相机系轨迹 → 翻译器（外参、Jacobian、本体状态）
- **三阶段：** Camera-centric specialization → Geometry-conditioned translation → Joint robot-human training
- **Human-to-robot：** 面包抓取、叠碗、擦黑板等锚点示意
- **仿真榜：** LIBERO 98.3%、LIBERO-Plus 82.0%、RoboTwin Easy/Hard 88.7%/89.2%、RoboCasa GR-1 62.0%
- **真机 Piper：** 笔筒、开抽屉、叠碗、人→机面包
- **Limitations：** 标定/深度/运动学/手关键点误差会传导；接触丰富与铰接跨本体仍难
- **Citation：** `@article{xu2026ucag-p, ... arXiv:2608.26058}`

## 对 wiki 的映射

- 论文归档：[`sources/papers/ucag_p_arxiv_2608_26058.md`](../papers/ucag_p_arxiv_2608_26058.md)
- 仓库归档：[`sources/repos/ucag-p.md`](../repos/ucag-p.md)
- 沉淀 **[`wiki/entities/paper-ucag-p.md`](../../wiki/entities/paper-ucag-p.md)**
