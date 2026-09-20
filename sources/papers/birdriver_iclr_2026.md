# BIRDriver: Bird's-Eye-View Informed Reasoning Driver（ICLR 2026）

> 来源归档（ingest）

- **标题：** Bird's-Eye-View Informed Reasoning Driver（BIRDriver）
- **类型：** paper / autonomous-driving / motion-planning / vlm / bev / long-tail
- **会议：** ICLR 2026 Poster
- **OpenReview：** <https://openreview.net/forum?id=TuU95FWkyH>
- **ICLR Proceedings：** <https://proceedings.iclr.cc/paper_files/paper/2026/hash/acb18f946cde3cc29864e6df7df54d11-Abstract-Conference.html>
- **PDF（Proceedings）：** <https://proceedings.iclr.cc/paper_files/paper/2026/file/acb18f946cde3cc29864e6df7df54d11-Paper-Conference.pdf>
- **PDF（OpenReview）：** <https://openreview.net/pdf?id=TuU95FWkyH>
- **ICLR Virtual：** <https://iclr.cc/virtual/2026/poster/10009274>
- **作者：** Yinuo Wang†、Mining Tan†、Yuanxin Zhong†∗、Zhitao Wang、Siyuan Cheng∗
- **机构：** 清华大学（Tsinghua）；华为（Huawei）；中国科学院大学（UCAS）；中国科学院自动化研究所 MAIS（CASIA）
- **关键词：** Autonomous driving, Key Intent Points
- **入库日期：** 2026-09-18
- **一句话说明：** 分层 VLM–规划器：单帧 BEV 图 + 系统/用户 prompt → VLM 输出 ≤3 个相对坐标 key points → PLUTO 运动规划器解码轨迹；三类辅助 SFT 数据 + token 加权损失提升数值精度；nuPlan Test14 多数超 PLUTO 基座，InterPlan 长尾 SOTA。

## 开源状态（核查，2026-09-20）

- **BIRDriver 官方实现：确认未开源。** ICLR Proceedings、OpenReview forum/PDF 与 ICLR Virtual **未列** GitHub / Hugging Face / 项目页；二次检索仍无作者公开训练/闭环仿真入口。
- **可复现边界：** 方法细节（BEV 渲染五类元素、RDP key point、838,824 样本三任务 10:1:2、LoRA on Qwen2.5VL-3B、PLUTO PointEncoder 微调 + VLM 预测噪声增广）已写在正文 §4–5 与 Appendix B；依赖 **nuPlan devkit** 与 **[PLUTO](https://github.com/jchengai/pluto)** 基座可部分对照，但 **VLM 权重与 BIRDriver 联合推理脚本未发布**。
- **源码运行时序图：** wiki 实体页标 **不适用**。

## 摘要级要点

- **问题：** 规则/模仿学习规划在常见场景可用，但在 **长尾**（施工区、借道超车 stalled vehicle 等）泛化差；现有 VLM 接入方式（meta-action / hidden state / 长 waypoint 序列）各有粒度、可解释性或预训练利用不足。
- **核心设计：** **≤3 个相对自车 key points** 表达高层意图（RDP 自适应提取，末点必保留）；VLM **仅看单帧 BEV**（无多相机、无场景文本冗余），最大化 internet-scale 预训练可用性。
- **VLM 微调：** 复合数据集 — **Key Point**（主）、**Spatial Localization**（BEV 像素↔物理距离）、**Driving Scene Stepwise**（先场景类型再 key points）；**Weighted SFT** 对数字/符号 token 分层加权（α=5，两位小数）。
- **规划器：** 基于 **PLUTO**（PointEncoder 融合 key points + 结构化场景特征）；独立微调，训练时对 GT key points 加 **N(0, VLM MAE)** 噪声；推理时用 **上一时刻末规划点** 作额外 key point 保时序一致。
- **基座 VLM：** **Qwen2.5VL-3B**（效率与 Table 4 精度权衡）；对比 InternVL2.5-2B/4B、Qwen2.5VL-7B。
- **评测：** nuPlan **CLS**（0–100）；**Test14-random**（261）、**Test14-hard**（272）、**InterPlan** 长尾。

## 核心论文摘录（MVP）

### 1) 单帧 BEV + ≤3 key points 分层架构

- **链接：** §1、§4.1–4.2、Fig. 1–2
- **摘录要点：** BEV 含 map/agent/红绿灯/route/obstacle 五类符号；VLM 输出文本坐标 key points → KeyPoint Encoder → PLUTO decoder 出多模态轨迹概率。
- **对 wiki 的映射：**
  - [BIRDriver 实体页](../../wiki/entities/paper-birdriver.md)
  - [DriveVLM](../../wiki/entities/paper-drivevlm.md)（waypoint VLM 对照）
  - [Senna](../../wiki/entities/paper-senna.md)（meta-action 对照）

### 2) 三任务 SFT + Weighted SFT Loss

- **链接：** §4.3、Table 2、式 (4)
- **摘录要点：** 838,824 样本，10:1:2；Spatial Localization 对 x/y/ϕ 误差降幅最大；加权 SFT 再降 8–11%。
- **对 wiki 的映射：**
  - [BIRDriver 实体页](../../wiki/entities/paper-birdriver.md)
  - [VLA 方法页](../../wiki/methods/vla.md)

### 3) nuPlan / InterPlan 闭环结果

- **链接：** §5.2、Table 1、Fig. 3
- **摘录要点：** BIRDriver(PLUTO) Test14-random CLS-NR **91.46** / CLS-R **91.26***；Test14-hard **80.56*** / **80.33***；InterPlan CLS-R **55.29***（*超 PLUTO 基线）。InterPlan 相对 PLUTO **+13.0%**、相对 Diffusion Planner **+38.8%**。
- **对 wiki 的映射：**
  - [BIRDriver 实体页](../../wiki/entities/paper-birdriver.md)
  - [E2E 自动驾驶十大算法地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)

## Table 1 摘录（BIRDriver vs 代表基线，CLS）

| 方法 | Test14-random CLS-NR | Test14-random CLS-R | Test14-hard CLS-NR | Test14-hard CLS-R | InterPlan CLS-R |
|------|----------------------|---------------------|--------------------|-------------------|-----------------|
| PLUTO | 91.87 | 90.03 | 80.03 | 76.92 | 48.92 |
| Diffusion Planner | 93.85 | 91.73 | 78.82 | 81.42 | 39.85 |
| PlanAgent (VLM) | 70.31 | 66.96 | 57.37 | 52.95 | — |
| **BIRDriver (PLUTO)** | **91.46** | **91.26*** | **80.56*** | **80.33*** | **55.29*** |

## BibTeX

```bibtex
@inproceedings{wang2026birdriver,
  title     = {Bird's-Eye-View Informed Reasoning Driver},
  author    = {Yinuo Wang and Mining Tan and Yuanxin Zhong and Zhitao Wang and Siyuan Cheng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=TuU95FWkyH}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-birdriver.md`](../../wiki/entities/paper-birdriver.md)
- 会议页归档：[`sources/sites/birdriver-iclr-2026-proceedings.md`](../sites/birdriver-iclr-2026-proceedings.md)
- 互链：[DriveVLM](../../wiki/entities/paper-drivevlm.md)、[Senna](../../wiki/entities/paper-senna.md)、[VLA](../../wiki/methods/vla.md)、[E2E AD 技术地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
