# MM-Future（arXiv:2609.20377）

> 来源归档（ingest）

- **标题：** MM-Future: Multi-Mode Joint World–Action Modeling for Autonomous Driving
- **类型：** paper / world-action-models / autonomous-driving / multi-mode-planning
- **arXiv：** <https://arxiv.org/abs/2609.20377>（Submitted 2026-09-17；HTML：<https://arxiv.org/html/2609.20377v1>；PDF：<https://arxiv.org/pdf/2609.20377>）
- **作者：** Shuai Liu, Hechangle Gong, Hao Jiang, Runlin He, Junxiang Zhan, Kai Huang, Sheng Yang, Shaoqing Ren（通讯）
- **机构：** 蔚来（NIO）；中国科学技术大学 AGI 研究院；中山大学计算机学院；北京航空航天大学
- **入库日期：** 2026-09-27
- **一句话说明：** 驾驶 **多模态联合 WAM**：并行生成多组 **场景–轨迹假设对**，经 modality-aware diffusion Transformer **双向共演化**；**MM-Tokens** 压缩多视角视频；**future-conditioned scorer** 在 NAVSIM 达 **94.0 PDMS / 91.5 EPDMS**，零样本 HUGSIM **32.3 HD-Score**。

## 开源状态（步骤 2.5，2026-09-27）

| 资源 | 状态 | 说明 |
|------|------|------|
| 项目页 | **无** | arXiv HTML/PDF 与公开检索均未发现独立 `*.github.io` 项目页 |
| 代码 | **待发布** | 正文与 arXiv 页未列 GitHub / Hugging Face 训练或推理入口 |
| 评测数据 | **第三方** | 训练 **NAVSIM navtrain**；开环 **NAVSIM v1/v2 navtest**；闭环 **HUGSIM**（436 场景）。NAVSIM 数据与指标见 [`sources/repos/navsim.md`](../repos/navsim.md) |
| 相关地图数据文档 | **外部** | 用户指定 [Argoverse User Guide](https://argoverse.github.io/user-guide/) 作驾驶数据集/地图生态参考 — 归档见 [`sources/sites/argoverse-user-guide.md`](../sites/argoverse-user-guide.md)（**非**本文训练集直接来源） |

**结论：截至入库日无可运行官方代码；复现需自备 NAVSIM 资产并等待作者发布实现。**

## 摘录 1：问题与范式（§1）

- **痛点：** 级联 WAM 有多模态 rollout 但 **单向** 条件；联合 WAM 有双向交互但通常 **单条** 场景–动作对。驾驶在路口等场景需要 **多种可交互结局** 同时被建模。
- **主张：** **Multi-mode joint world–action modeling** — 每组假设从 **结构化 action prior（GMN）** 与 **独立 scene noise** 初始化，在共享 flow 里 **双向共演化**；推理时用 **配对未来** 而不只看历史来选轨迹。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-mm-future.md`](../../wiki/entities/paper-mm-future.md)；与 [World Action Models](../../wiki/concepts/world-action-models.md)、[generative-world-models](../../wiki/methods/generative-world-models.md) 互链。

## 摘录 2：MM-Tokens 与联合 flow（§3）

- **MM-Encoder：** DINOv2-S + rank-32 LoRA → 每相机 register tokens；按 **2 帧 chunk** 用 learnable queries 聚合成 **64×256-D MM-Tokens**（相对 dense patch 大幅降 token 数）。
- **训练目标：** 独立 flow 时间 \( \rho^a, \rho^x \)；modality-aware Transformer（历史 clean prefix + 双向 scene–action）；**Best-of-Many (BoM)** 用 L1 轨迹误差选 winning mode，只对 winner 回传 velocity loss。
- **推理：** 短 Euler 积分；默认 **64 对** 假设（主结果表 latency **233 ms** @ H800 bf16）。

**对 wiki 的映射：** 实体页「流程总览」Mermaid + 「核心原理」小节；与 [Latent-WAM 清单页](../../wiki/entities/paper-sa-2603-24581-latent-wam-latent-world-action-modeling-for-end.md) 对照 **显式多模态 joint** vs latent 单对。

## 摘录 3：Future-conditioned scorer 与实验（§3–§4）

- **Scorer：** 轨迹 query 先与 **历史 MM-Tokens** cross-attend，再 **块对角** 地只看 **自己的配对未来**；预测 PDMS 分量 logits，BCE 监督（generator 侧 stop-gradient）。
- **NAVSIM-v1 navtest（trainval）：** **94.0 PDMS**（EP **91.6**）；train-only **93.4**。
- **NAVSIM-v2 navtest：** **91.5 EPDMS**（TTC **98.6**）。
- **HUGSIM 零样本闭环：** 平均 **HD-Score 32.3**（Medium **40.0**），较 Latent-WAM **+3.4** 平均 HD-Score。
- **Ablation：** 32 假设 paired **92.9→93.3 PDMS**（加 paired-future scorer）；multi-mode 较 single-mode **更快收敛**（0.80 PDMS @ 3.8k vs 17.5k steps）。

**对 wiki 的映射：** 实体页「实验与评测」「结论」；[e2e 算法 Top10 地图](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md) 可引用 PDMS headline。

## 参考外链

- [NAVSIM 官方仓库](https://github.com/autonomousvision/navsim) — [`sources/repos/navsim.md`](../repos/navsim.md)
- [Argoverse User Guide](https://argoverse.github.io/user-guide/) — [`sources/sites/argoverse-user-guide.md`](../sites/argoverse-user-guide.md)
