# BooST: Bridging Semantics and Motions for Efficient Skill Transfer（arXiv:2608.10600）

> 来源归档（ingest）

- **标题：** BooST: Bridging Semantics and Motions for Efficient Skill Transfer
- **缩写：** **BooST**
- **类型：** paper / skill-transfer / vq-vae / policy-distillation / libero
- **arXiv：** <https://arxiv.org/abs/2608.10600>
- **期刊：** IEEE RA-L 2026（项目页）
- **项目页：** <https://boost-robots.github.io/>（归档见 [`sources/sites/boost-robots.md`](../sites/boost-robots.md)）
- **作者：** Jusuk Lee、Daesol Cho、Jonghun Shin、Seungyeon Yoo、Jonghae Park、Taekbeom Lee、H. Jin Kim
- **机构：** 首尔大学（SNU）；佐治亚理工学院（Georgia Tech）
- **入库日期：** 2026-08-18
- **一句话说明：** 跨模态 VQ-VAE 同时编码「做什么」与「怎么动」，再蒸馏成约 60 Hz 轻量策略；DROID 预训练后少样本迁到 LIBERO / UR3。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-18）：** 有方法、LIBERO 表、真机跨本体视频；GitHub 组织 [boost-robots](https://github.com/boost-robots) **仅** `boost-robots.github.io`，无训练仓。
- **结论：** **项目页已发，实现未开源**；源码运行时序图不适用。

## 摘录 1：两阶段

1. **统一技能预训练：** visuo-linguistic 通路（CLIP ViT × 指令）与 action 通路交替写同一 codebook；监督只有 **动作重建**，不重建像素。
2. **下游适应：** 冻结编码器当教师，轻量因果 skill prior + 低层 BC；执行只看过去观测。

## 摘录 2：数字（项目页）

LIBERO-90 成功率（50 / 20 / 10 demo）：BooST **0.91 / 0.82 / 0.70**，相对次优 +41% / +59% / +140%。干扰物预训练后四套均分 **0.90**（LAPA 0.79、UniVLA 0.70）。真机：Franka 技能 → UR3，每任务 5 条示范。

**对 wiki 的映射：** [`wiki/entities/paper-boost-skill-transfer.md`](../../wiki/entities/paper-boost-skill-transfer.md)；交叉 [LIBERO](../../wiki/entities/libero-benchmark.md)、[模仿学习](../../wiki/methods/imitation-learning.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（项目页无训练仓）
