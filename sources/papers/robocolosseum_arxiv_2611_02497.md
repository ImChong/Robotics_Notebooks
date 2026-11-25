# RoboColosseum：Evaluating Physical Intelligence at Scale In-The-Wild（arXiv:2611.02497）

> 来源归档（paper）

- **标题：** RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild
- **中文：** RoboColosseum：在真实分布式场景中规模化评测物理智能
- **类型：** paper
- **版本：** arXiv:2611.02497v1，2026-11-02，预印本
- **论文 HTML：** https://arxiv.org/html/2611.02497
- **PDF：** https://arxiv.org/pdf/2611.02497
- **项目页：** https://robocolosseum.ai/
- **代码汇总：** https://github.com/RoboColosseum/RoboColosseum
- **数据/rollout：** https://huggingface.co/RoboColosseum
- **首次入库：** 2026-11-04
- **本次核对：** 2026-11-21
- **核对范围：** 已阅读 arXiv v1 HTML 正文的 Introduction、Related Work、Benchmark（硬件/站点、两个 track 与统计评分）、Experiments RQ1–RQ5（Table 2 与图 3–11）、Limitations & Conclusion、Future，以及附录中的 Fine-tune 训练协议/任务规格/进度判据和 Open Track 的 LLM agent 接口与安全边界；对照官方项目页与 LeRobot 集成文档。本文归档以 2026-11-02 v1 论文为准，和官网当前的动态规模数据分开记录。
- **通读：** 已完成
- **沉淀到 wiki：** [paper-robocolosseum](../../wiki/entities/paper-robocolosseum.md)

## 核心研究问题

仅靠单实验室、小规模 rollout 或仿真评测，无法可靠解释通用机器人策略在不同任务、场景、本体下的真实表现。论文通过多个线下运营站点、共用策略服务接口与可对比的试验协议，把真实硬件的异构性纳入量化评测。

## 研究设计

- **对象：** SO-101 单臂、DROID/Franka、Bimanual-YAM 双臂、Unitree G1 人形，共四种本体；5 个 VLA（GR00T N1.7、π₀.₅、MolmoAct2、G0.5、LingBot-VLA v2），Open Track 另有 3 个 API 代理。
- **Open Track：** 不做任务特定微调；现场创建自然语言任务，在同一机器人与场景条件下依次运行匿名 A/B，记录偏好、成功/进度、反馈和 1–10 的人工难度。按不确定性分配下一组比较（active matchmaking）；用 Difficulty-Calibrated Rasch（DCR）将模型潜在能力 θ 和任务潜在难度 β 联合估计，报告 Elo 样式的线性换算分，不是传统逐轮 Elo 更新。
- **Fine-tune Track：** 4 个本体各 5 项预定义任务，共 20 项；每模型每任务采用同一 100 条示范，再进行 25 次评测。比较 full success rate（SR）与任务进度（PS，已完成评测条件数/条件总数的 trial 均值）。不同模型保留官方各自优化/微调 recipe，不能将差异直接归因为网络架构。
- **安全：** 风险停止的试验计为失败/零进度；视频和失败 rollout 也是应保留的证据。

## 主要实验结果（论文定稿快照：2026-09-17 至 10-28）

| 证据 | 结果 | 正确解读 |
|---|---|---|
| Open Track 规模 | 124 个站点、5,699 次匿名配对比较 | 不是全部为成功执行，也不是 5,699 个 Fine-tune 任务 |
| 代理 vs VLA（有直接比较的子集） | 901 对：代理进度更好 394，VLA 更好 265，平局 242；代理平均 +5.3 个百分点 | 代理优势不代表所有平台/单模型都领先 |
| 极难任务 | 人工难度 9–10：完整成功率 0.4%，进度 3.5%；难度 1–4：成功率 30.7%，进度 56.6% | 难度组有任务/本体构成差异，不能解释为单一因果效应 |
| Fine-tune Bimanual-YAM | π₀.₅ 平均 SR 93.6% | 只适用该平台的 5 个任务和论文协议 |
| Fine-tune DROID/Franka | π₀.₅ 平均 SR 84.0% | 并不代表所有真机平台的泛化排名 |
| Fine-tune G1 | 5 个任务全部模型 SR 0%；模型平均 PS 为 13.3%–23.2% | 零成功率不等于零进度 |
| 跨形态排名 | Kendall W = 0.24 | 本体与任务同时变化，不能把差异全部归因于形态 |
| 主动配对效率 | 达到约 90% 的全量排序一致性需约 400–500 次主动比较，随机配对需约 1,000 次 | 这是基于已有数据子采样的回放分析，不是独立新增真机试验 |
| 站点分半稳定性 | DROID / YAM 约 85% 的模型对顺序一致；SO-101 约 59% | 可复现性随本体、样本和站点组成变化 |

## 附录与限制

- Fine-tune 的 100 条示范起点一致，但不同模型官方训练 recipe、优化器、参数冻结策略不完全一致；以「实际适应能力」比较，不是严格架构消融。
- 排名只覆盖参与评测模型/兼容本体以及每本体 5 项固定 Fine-tune 任务。现场任务难度、操作者主观评价与任务分布可能影响 Open Track 结论。
- 论文中的 124 站点与官网当前的 140+ 场景不是同一个统计时点，不能混写成论文实验数据。
- 论文相对较早的二手公众号摘要曾记“项目页未列 GitHub”，经 2026-11-21 官方源码核对已失效；代码仅公开 client、policy server 与各本体 Fine-tune 评测工具，托管路由/现场运营全栈并非据此完全开源。

## 一手代码与复现入口

- [公开仓库索引](https://github.com/RoboColosseum/RoboColosseum)
- [Robot Client](https://github.com/RoboColosseum/robocolosseum-client)
- [Policy Server](https://github.com/RoboColosseum/robocolosseum-policy-server)
- [Fine-tune G1](https://github.com/RoboColosseum/robocolosseum-finetune-g1)
- [LeRobot 官方接入文档](https://huggingface.co/docs/lerobot/main/en/robocolosseum)
- [源码核对归档](../repos/robocolosseum-client-policy-server.md)

## 对 wiki 的映射

- [RoboColosseum 真机评测论文实体](../../wiki/entities/paper-robocolosseum.md)
- [具身模型测评纵深](../../roadmap/depth-embodied-eval.md)