---
type: entity
project_id: relational-spatial-alignment-vla
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-11-01
arxiv: "2610.24705"
project: https://qm-ipalab.github.io/RIGID
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/relational_spatial_alignment_vla_arxiv_2610_24705.md
  - ../../sources/sites/relational-spatial-alignment-vla.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
summary: "Relational Spatial Alignment VLA（arXiv:2610.24705）：RIGID对齐特征之间的空间关系，而非逐点强迫匹配三维教师；仿真、空间推理与真机测试支持保留表达能力，收益仍受教师质量和微调数据限制。"
---

# Relational Spatial Alignment VLA（arXiv:2610.24705）

**Relational Spatial Alignment VLA**（*Relational Spatial Alignment for Vision-Language-Action Models*，[arXiv:2610.24705](https://arxiv.org/abs/2610.24705)，[项目页](https://qm-ipalab.github.io/RIGID)）来自 [具身智能小站 2026-11-01 盘点](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)，属「VLA 与策略学习」组。

## 一句话定义

**RIGID对齐特征之间的空间关系，而非逐点强迫匹配三维教师；仿真、空间推理与真机测试支持保留表达能力，收益仍受教师质量和微调数据限制。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Relational Spatial Alignment VLA | Relational Spatial Alignment for Vision-Language-Action Models | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 给VLA补空间知识，避免表示坍缩。
- 开源结论：**已开源**（步骤 2.5，2026-11-01）。公众号未给代码链接；项目页链接的 QM-IPAlab/RIGID 经 git ls-remote 可达。
- 同期同组工作见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A["多视角 RGB 观测 {I(v)}"] --> B["学生 VLA: 视觉 token Z(v)<br/>(SigLIP + DINOv2 融合视觉编码, LoRA 训练)"]
    A --> C["冻结 3D 基础模型 VGGT 作教师"]
    B --> D["可学习 query/key 投影 gq, gk<br/>对每个视图对 (a,b) 计算注意力图 Avla"]
    C --> E["直接取教师的单视图 (a=b) 与跨视图 (a≠b) 注意力图 A3D"]
    D --> F["把注意力每行视为图像块上的空间分布<br/>以块坐标欧氏距离为传输代价 C"]
    E --> F
    F --> G["Sinkhorn 熵正则最优传输 (Wasserstein) 距离<br/>得到 L_RIGID, 只对齐成对关系, 不做逐点特征相等"]
    B --> H["VLA 动作学习目标 L_action"]
    G --> I["L_total = L_action + β · L_RIGID"]
    H --> I
```

图依据：论文 §3.1-3.2 Method / 式(1)-(7) / Fig.1

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.24705](https://arxiv.org/abs/2610.24705) |
| **开源** | **已开源** |
| **要点** | RIGID对齐特征之间的空间关系，而非逐点强迫匹配三维教师；仿真、空间推理与真机测试支持保留表达能力，收益仍受教师质量和微调数据限制。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（已开源）** — 公众号未给代码链接；项目页链接的 QM-IPAlab/RIGID 经 git ls-remote 可达。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**给VLA补空间知识，避免表示坍缩：RIGID对齐特征之间的空间关系，而非逐点强迫匹配三维教师；仿真、空间推理与真机测试支持保留表达能力，收益仍受教师质量和微调数据限制。**

1. 开源边界：**已开源** — 公众号未给代码链接；项目页链接的 QM-IPAlab/RIGID 经 git ls-remote 可达。
2. 核心机制：RIGID对齐特征之间的空间关系，而非逐点强迫匹配三维教师；仿真、空间推理与真机测试支持保留表达能力，收益仍受教师质量和微调数据限制。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-11-01-56-papers-technology-map](../overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 参考来源

- [relational_spatial_alignment_vla_arxiv_2610_24705.md](../../sources/papers/relational_spatial_alignment_vla_arxiv_2610_24705.md)
- [wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md](../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)
- [arXiv:2610.24705](https://arxiv.org/abs/2610.24705)
- [arXiv PDF](https://arxiv.org/pdf/2610.24705)：已通读方法部分（实验与附录以略读为准）：§3 方法（关系对齐动机、RIGID 注意力蒸馏与 Sinkhorn 最优传输、总损失），§4.1 实验设置与实现细节

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.24705)
- [项目页](https://qm-ipalab.github.io/RIGID)
