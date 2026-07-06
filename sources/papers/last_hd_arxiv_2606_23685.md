# LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation（arXiv:2606.23685）

> 来源归档（ingest）

- **标题：** LaST-HD: Learning Latent Physical Reasoning from Scalable Human Data for Robot Manipulation
- **缩写：** **LaST-HD**
- **类型：** paper / vision-language-action / human-to-robot / manipulation / world-model
- **arXiv：** <https://arxiv.org/abs/2606.23685>（PDF：<https://arxiv.org/pdf/2606.23685>）
- **项目页：** <https://siriyep.github.io/last-hd-project-page/>
- **作者：** Jiaming Liu, Yinxi Wang, Chenyang Gu, Siyuan Qian, Xiangju Mi, Hao Chen, Jiawei Chen, Qingpo Wuwu, Xiaoqi Li, Nuowei Han, Yiming Zhang, Xuheng Zhang, Yang Yue, Yeqing Yang, Lei Wang, Peng Jia, Hao Tang, Shanghang Zhang
- **机构：** 北京大学多媒体信息处理国家重点实验室、香港中文大学、Simplexity Robotics、Aether Tech（贡献与通讯作者以 PDF 为准）
- **入库日期：** 2026-07-06
- **一句话说明：** 在 **reasoning-before-acting MoT VLA** 上，用 **动作条件世界模型** 把 **非配对** 的人手与机器人轨迹对齐到 **共享潜式物理推理空间**，以 **前向动力学特征** 监督推理专家；配套 **OOL Glove** 低成本人手采集与 **mixed-to-human**（混合共训 + 人手在线纠偏）训练配方，在 **6 项真机任务 / 3 种本体** 上报告 **仅用人类数据泛化** 与 **20 分钟纠偏达 >90%** 的适应叙事。

## 摘录 1：问题与动机（摘要 / 引言）

- **痛点：** VLA 依赖大规模 **机器人遥操** 数据，采集 **慢、贵**；人手演示 **物理交互先验丰富、可扩展**，但现有 **运动学重定向 / 形态翻译 / 动作级共训** 多停留在 **几何或表征对齐**，对 **人手与机器人操作的动力学一致性** 关注不足。
- **核心问法：** 能否把 VLA 的 **物理推理** 当作 **跨具身中间接口**，让人手数据 **更高效** 进入机器人动作学习，而不只是模仿人手运动学？
- **与相邻路线：** 相对 EgoMimic / DexWild 的 **显式运动学对齐**、H-RDT / EgoScale 的 **大规模跨具身预/中训**，LaST-HD 强调 **潜式物理推理对齐** 而非纯动作或未来帧视觉表征共训。

**对 wiki 的映射：**
- [LaST-HD](../../wiki/entities/paper-last-hd-latent-physical-reasoning.md) — 问题定义、与 EgoMimic / π₀.₅ / LaST0 对照
- [VLA](../../wiki/methods/vla.md) — 「人数据缩放 + 潜式推理」路线补充
- [Imitation Learning](../../wiki/methods/imitation-learning.md) — 人手→机器人 IL 管线

## 摘录 2：架构与人–机潜式对齐（§3.1–3.2）

- **骨干：** 基于 **Janus-Pro** 的 **Mixture-of-Transformers (MoT) VLA**（**DeepSeek-LLM 1.5B**）：**推理专家** 自回归预测 **\(N_{\text{lat}}\)** 个潜式 token；**动作专家** 经 **flow matching** 输出动作块；二者 **共享注意力** 传递推理知识（**reasoning-before-acting**）。
- **视觉：** **SigLIP-Large** 编码 **384×384** 观测；语言指令 + 多视角图像条件化策略。
- **对齐桥：** 在 **混合人手 + 机器人** 轨迹上微调 **动作条件世界模型**（无需严格配对）；以 **动作块 cross-attention** 注入各层，在 **最终去噪步** 抽取 **最深 U-Net 层** 的 **前向动力学特征**，经 MLP **对齐到 \(d_l\)** 维并池化为潜式监督目标。
- **设计要点：** 用 **世界模型潜特征** 监督推理专家，而非直接动作预测或 **未来 SigLIP 帧特征**；**动作标签** 作弱锚点，使不同形态在 **相同物理后果**（如推苹果）下对齐。

**对 wiki 的映射：**
- [LaST-HD](../../wiki/entities/paper-last-hd-latent-physical-reasoning.md) — MoT 双专家、世界模型桥与 Mermaid 流程
- [World Action Models 概念](../../wiki/concepts/world-action-models.md) — 动作条件前向动力学作为跨具身接口

## 摘录 3：OOL Glove 与 mixed-to-human 训练（§3.3–3.4）

- **OOL Glove：** 超轻量（**<100 g/只**）**IMU** 手套，**6 模块** 跟踪 **21 手关键点 + 1 腕点**；**>200 Hz**、**<10 ms** 延迟、**亚毫米级** 平均 RMS；采集 **双腕 + 头/胸** 三视角同步轨迹（语言 + 观测 + 手–腕状态）。
- **统一动作基底：** 原生人手轨迹映射到 **手中心坐标**；平行夹爪由 **指尖距离** 派生，灵巧手经 **IK 重定向** 关节角。
- **Stage 1 混合共训：** 世界模型在预训练混合（含 OOL + Tianji 双臂数据）上训练一次即可为下游提供对齐潜目标；LaST-HD 优化 **\(\mathcal{L}=\mathcal{L}_{\text{act}}+\lambda\mathcal{L}_{\text{latent}}\)**（潜式 **cosine** 损失），对人手与机器人轨迹 **同时** 训练。
- **Stage 2 人手在线纠偏：** 实机 rollout 定位失败态，用 OOL 采集 **人手纠偏**（替代额外机器人遥操）；冻结世界模型，**1–2 epoch** 平衡回放（**DAgger 缓冲** 与历史数据 **1:1** 采样）。

**对 wiki 的映射：**
- [LaST-HD](../../wiki/entities/paper-last-hd-latent-physical-reasoning.md) — 数据采集硬件与两阶段训练配方
- [跨具身迁移专题](../../wiki/overview/topic-cross-embodiment.md) — 人→机器人数据缩放案例

## 摘录 4：实验设置与主要结论（§4）

- **平台与任务：** **6 项真机任务**、**3 种本体** — Galaxea R1 Lite（拧瓶盖、整理盒）、Tianji Marvin 双臂（分拣水果、装袋拉链）、Marvin + **WUJI 灵巧手**（倒水、夹具抓取）；每任务 **100 机器人遥操 + 50 OOL** 域内数据；泛化场景（**未见位置 / 物体 / 背景**）每场景仅 **60 OOL** 人手轨迹。
- **Baselines：** **LaST0**（潜式 CoT VLA）、**\(\pi_{0.5}\)**、**Cosmos-Policy**（世界–动作模型）；均官方全量微调。
- **域内：** LaST-HD（100 机器人）**6 任务平均 73%** SR，优于 LaST0（63%）、\(\pi_{0.5}\)（62%）、Cosmos-Policy（52%）；**Mix-HD**（50 机器人 + 50 OOL）在多数任务与纯机器人版相当。
- **泛化（+ 未见场景人手数据）：** LaST-HD **全局平均 56%** vs LaST0 **46%**；未见物体 **58%**、未见背景 **68%**；**在线纠偏** 仅 **20 分钟 / 60 条 OOL** 可在 Sort Fruits 上将未见背景推至 **100%** SR。
- **消融：** 去掉潜式推理 **73%→60%**；**SigLIP 未来帧** 与 **无动作条件 WM** 均弱于 **动作条件 WM 潜监督**；OOL 数据（73%）优于裸手视觉（63%）与同采集时长机器人数据 Real-12（60%）。

**对 wiki 的映射：**
- [LaST-HD](../../wiki/entities/paper-last-hd-latent-physical-reasoning.md) — 实验表与 baseline 定位
- [EgoMimic](../../wiki/entities/paper-ego-03-egomimic.md) — 人→机器人 IL 对照锚点
