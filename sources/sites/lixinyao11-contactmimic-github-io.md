# lixinyao11.github.io/contactmimic-page（ContactMimic 项目页）

- **标题：** ContactMimic — Humanoid Object Interaction via Contact Control
- **类型：** site / project-page
- **URL：** <https://lixinyao11.github.io/contactmimic-page/>
- **入库日期：** 2026-07-12
- **配套论文：** [ContactMimic（arXiv:2607.08742）](https://arxiv.org/abs/2607.08742) — 归档见 [`sources/papers/contactmimic_arxiv_2607_08742.md`](../papers/contactmimic_arxiv_2607_08742.md)

## 一句话摘要

UIUC（Saurabh Gupta 组）**ContactMimic** 官方站点：提供 **方法概述与真机 contact controllability 补充视频**、contact-aware 奖励与三种轨迹增广说明、仿真四组轨迹 contact ✔/✘ 可视化、BeyondMimic 对照与增广消融图，以及真机 5 条动作成功率表（每条展示 5 次 trial）。

## 公开信息要点（截至入库日）

- **机构：** University of Illinois Urbana-Champaign；Xinyao Li、Xialin He 共同一作。
- **核心主张：** keypoint tracking **不足以**表达擦板、坐椅、推家具等 **接触定义任务**；策略在 **同一 keypoint** 下可通过 **per-body contact 标签** 开启或抑制物理接触。
- **方法板块：** contact-conditioned tracker $\pi_\theta(a_t \mid p_t, \bar{k}_t, \bar{c}_t)$；contact-label matching + contact distance 奖励；增广 ① label 翻转 ② 去物体 ③ 膨胀几何。
- **真机板块：** Unitree G1；5 条 HOI 动作；near/far keypoint × contact ✔/✘ 网格（靠椅背任务）；每条条件展示 **全部 5 次真机 trial**。
- **仿真板块：** 10 条 HUMOTO 动作 contact controllability；搬箱 **无任务专用奖励**；与 BeyondMimic MPJPE 相当但接触指标更高。
- **分析板块：** 去掉数据增广损害 controllability；线性探针显示本体感知已编码 runtime contact。
- **资源：** Paper（arXiv）/ Overview Video / BibTeX；截至入库日 **未标注 Code/Data 开源**。

## 为何值得保留

- **非 PDF 证据：** 概述视频与真机 trial 网格比摘要更直观呈现 **contact ✔/✘ 切换**（擦板留痕 vs 悬停、坐椅承重 vs 悬空蹲姿等）。
- **与 arXiv 三角互证：** 增广三项、真机成功率表、BeyondMimic 对照与论文 §3–§4 一致，便于维护者核对表述。
- **同系工作锚点：** 与 BeyondMimic、SceneBot、OmniRetarget、InterPrior 同属 **人形 HOI / contact-aware tracking** 线；ContactMimic 突出 **运行时 contact 旋钮** 与 **训练数据解耦工程**。

## 关联资料

- 论文归档：[`sources/papers/contactmimic_arxiv_2607_08742.md`](../papers/contactmimic_arxiv_2607_08742.md)
- wiki 实体：[`wiki/entities/paper-contactmimic.md`](../../wiki/entities/paper-contactmimic.md)
- 重定向：[`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)
- 对照 tracker：[`wiki/methods/beyondmimic.md`](../../wiki/methods/beyondmimic.md)
