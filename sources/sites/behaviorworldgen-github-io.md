# BehaviorWorldGen（项目页）

> 来源归档（ingest 关联资料）

- **标题：** BehaviorWorldGen — 交互式驾驶世界模型
- **类型：** site / project-page / demo-videos
- **项目页：** <https://behaviorworldgen.github.io/>
- **论文：** <https://arxiv.org/abs/2608.22187> — 见 [`sources/papers/behaviorworldgen_arxiv_2608_22187.md`](../papers/behaviorworldgen_arxiv_2608_22187.md)
- **团队：** AFARI World Model Team / 千里科技世界模型团队
- **机构：** AFARI, Qianli Technology；MEGVII
- **入库日期：** 2026-09-18
- **一句话说明：** 官方演示站：BehaviorFlow 交互 rollout、双路径 world simulator（动作条件 AWM + 3DGS 外推）、NAVSIM 策略微调表与 Film Preview 视频。

## 开源核查（步骤 2.5，2026-09-18）

项目页 Hero CTA 仅有 **arXiv** 与 **Film Mode Preview**；Navigation / Footer / 全站 HTML **无 GitHub、Hugging Face、Zenodo 或 Code 链接**。

| 链接 | 用途 |
|------|------|
| arXiv:2608.22187 | 论文 |
| `video/film-video/`、`video/behavior-flow/` 等 | 本地托管演示 MP4 |
| GitHub / 权重 / 数据 | **未列出** |

**结论：** **未开源**。论文与视频可引用；复现训练/推理需等待官方发布代码。

## 项目页结构快照

| 区块 | 内容 |
|------|------|
| Framework Overview | 总框架示意图 |
| World Simulators | Tab：Action Condition WM / 3D Gaussian Splatting |
| BehaviorFlow | 直道 cut-in、路口让行等可控行为视频 |
| Quantitative Results | NAVSIM PDMS 表（VLA + E2E）；低分场景分桶 |
| Authors | AFARI + MEGVII 联合作者列表 |

## 关联 wiki

- [`wiki/entities/paper-behaviorworldgen.md`](../../wiki/entities/paper-behaviorworldgen.md)
