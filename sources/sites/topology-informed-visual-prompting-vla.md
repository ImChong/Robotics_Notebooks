# Topology-VLA 项目页

> 来源归档（site）

- **标题：** Topology-Informed Visual Prompting For Vision Language Action Policies
- **类型：** site
- **链接：** <https://topology-vla.github.io/>
- **arXiv：** <https://arxiv.org/abs/2609.23944>
- **机构：** 密歇根大学（University of Michigan）
- **入库日期：** 2026-10-18
- **核对范围：** 通读 `topology_vla.html` 全文（Title、Links、Method 视频/框架图说明、Simulation Results 表、Hardware Results 表、以及论文主体之外补充的 Ablation Results 表）
- **通读：** 已完成
- **一句话说明：** 展示 Topology-VLA 在 Rope Pulling、Cable Threading、Box Pickup 仿真与真机货架取箱实验上的视频对比及 `π0.5` 三项消融数字；未放出任何代码或权重仓库链接。

## 开源状态

- **未开源（项目页未列 GitHub）**（2026-11-14 核查）：页面仅提供 Paper PDF 与演示视频，无 Code / Dataset / Weights 链接。

## 网站独有消融补充（Ablation Results on `π0.5`）

| 变体 | Rope Pulling | Cable Threading | Box Pickup |
|------|--------------|-----------------|------------|
| Topology-Informed Visual Prompting（完整） | **64%** | **77%** | **82%** |
| No Visual Prompting（去视觉箭头，仍保留全部数据） | 52% | 69% | 75% |
| Task-Continuation Data Only（去修正轨 `D_corr`，仅 `D_nominal ∪ D_cont`） | N/A | 24% | 54% |

## 对 wiki 的映射

- [paper-topology-informed-visual-prompting-vla](../../wiki/entities/paper-topology-informed-visual-prompting-vla.md)
- [论文摘录](../papers/topology_informed_visual_prompting_vla_arxiv_2609_23944.md)
