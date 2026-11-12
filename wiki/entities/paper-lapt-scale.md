---
type: entity
project_id: lapt-scale
tags:
  - paper
  - vla
  - latent-action
  - human-video
  - pretraining
  - corl-2026
  - jd
  - tsinghua
  - hitsz
  - hkust-gz
status: complete
updated: 2026-11-12
arxiv: "2610.13732"
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../concepts/embodied-scaling-laws.md
  - ../concepts/robot-data-supervision-signal-types.md
  - ../entities/paper-joyai-ra-05.md
  - ../entities/paper-rcl-2601-04061-clap-contrastive-latent-action-pretraining-for-l.md
  - ../entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md
  - ../entities/paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md
  - ../entities/robotwin.md
  - ../entities/robocasa.md
sources:
  - ../../sources/papers/lapt-scale_arxiv_2610_13732.md
original:
  - ../../sources/papers/lapt-scale_arxiv_2610_13732.md
summary: "LAPT-Scale（arXiv:2610.13732，CoRL 2026）：23.3K 小时第一人称视频上训练连续潜在动作模型（LAM，DINOv2 与光流对齐），标注 5.1K 小时子集做 VLA 的 Stage-1 预训练，再经 250 小时机器人数据对齐与任务后训练；仅增加 Stage-1 人视频（1K 到 5K 小时），真机三任务均值 63.3% 升至 85.0%（每任务 20 次，一个平台）。"
---

# LAPT-Scale：用人类视频潜在动作预训练扩展具身策略（arXiv:2610.13732）

## 一句话定义

**LAPT-Scale** 是一套「人视频 → 连续潜在动作 → VLA 预训练」流程：LAPT-Scale-LAM 在第一人称人类视频上学习连续的帧间潜在动作，LAPT-Scale-Policy 把这些潜在动作当作 Stage-1 预测目标，再用目标机器人数据做具身对齐与任务后训练，推理时输出可执行的机器人动作（论文 §3.2–3.3）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LAM | Latent Action Model | 从相邻帧推断隐式「动作」变量的模型，本文为连续高斯潜变量（d=32） |
| LAPT | Latent Action Pre-Training | 以潜在动作作监督的预训练范式 |
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VQ | Vector Quantization | 离散码本量化，本文作为被对比的离散 LAM 路线 |
| VAE | Variational Auto-Encoder | 变分自编码器，LAM 的结构基础 |
| KL | Kullback-Leibler divergence | 约束后验贴近先验的正则项 |
| LPIPS | Learned Perceptual Image Patch Similarity | 感知相似度，本文用于感知对齐 |
| DINOv2 | self-DIstillation with NO labels v2 | 冻结自监督视觉特征，用于语义对齐 |
| RAFT | Recurrent All-pairs Field Transforms | 冻结光流估计器，用于运动对齐 |
| CFM | Conditional Flow Matching | 动作专家的训练目标（式 5） |
| LARY | Latent Action Representation Yielding benchmark | 第一人称动作识别线性探针基准，用来评价潜在动作语义 |
| PSNR / SSIM | Peak Signal-to-Noise Ratio / Structural Similarity | 下一帧重建质量指标 |

## 为什么重要

- 机器人动作数据昂贵，第一人称人视频规模大但没有机器人动作标签；既有做法（如 VITRA）把视频转成显式伪动作，论文认为估计误差会带来噪声标注，单纯加视频未必提升策略（§1）。
- 论文的核心证据是受控缩放实验：机器人数据、架构、评测固定，只改 Stage-1 人视频量与标注形式（§4.3、表 14）。LAPT-Scale-LAM 在 1K/3K/5K 小时为 63.3/80.0/85.0，而手部运动标注在 5K 小时回落到 55.0，与 0 小时对照相同。
- 它把「标注质量」与「视频规模」分开检验：DreamDojo 的预训练 LAM 在 5K 小时为 76.7，低于 LAPT-Scale-LAM 的 85.0（§4.3，作者推断原因是纯像素重建不足以刻画语义与运动结构，这是作者的解释，论文未做更细的因果隔离）。

## 核心原理与流程总览

```mermaid
flowchart LR
  A["23.3K 小时第一人称视频"] --> B["训练 LAM 连续潜在动作 d=32"]
  B --> C["冻结编码器标注 5.1K 小时清洗子集"]
  C --> D["Stage 1 VLA 预测潜在动作块"]
  D --> E["Stage 2 切到 250 小时机器人动作块对齐"]
  E --> F["Stage 3 任务后训练 仿真与真机"]
  F --> G["推理输出机器人动作"]
```

**LAM（§3.2）。** 时空 VAE：编码器读 (x_t, x_t+1)，由前置的可学习 action query 输出对角高斯后验，采样得到 z_t；解码器把 W_z z_t 加到当前帧补丁嵌入上，预测 x̂_t+1。低维瓶颈加上解码器单独接收当前帧上下文，用来迫使 z_t 记录变化而非静态外观。损失为 MSE+LPIPS（感知）、冻结 DINOv2 的补丁/CLS 特征距离（语义）、冻结 RAFT 光流 L1（运动，目标侧停梯度）与 KL 项（式 1–4）。

**Policy（§3.3）。** VLM 加 Perceiver 式动作专家，架构沿用 JoyAI-RA，三阶段共用同一动作专家，目标统一补零到最多 32 维并在损失里掩码（附录 B.1）。Stage 1 目标是 LAM 后验均值的潜在动作块，Stage 2 换成机器人动作块，Stage 3 在任务数据上继续训练。

**数据（§3.1、附录 A）。** LAM 用 23.3K 小时（含 Egocentric-10K 10.0K 与自采 8.5K）；策略预训练用按「可见手物交互、有意义时序变化」过滤出的 5.1K 小时，其中自采占 2.9K（表 8）。

## 工程实践

- **算力（附录 B.2、表 9–10）：** LAM 训练 32 张 H200、154.3 小时（4,938 GPU 小时）；标注 5.1K 小时约 1,700 GPU 小时；Stage 1 约 43h、Stage 2 约 50h（均 100K 步、批 1024）；真机后训练每任务 30K 步、约 7-8h（表 11）。标注缓存后，缩放实验不需要重跑 LAM 编码器。
- **LAM 配置（附录 B.1）：** patch 16、维 1024、编解码各 24 层、16 头、320×240、帧间随机间隔 1-4；KL 权重 1e-6、LPIPS 0.01、DINO 0.003、光流 3.3e-5，权重量级相差很大，复现时需注意。
- **数据处理（附录 A.2）：** 去损坏与重复片段；策略子集再去掉强模糊、极端相机运动、无手、长静止、严重遮挡的片段；缩放子集尽量嵌套并对大来源限量。
- **评测设置：** 仿真 RoboTwin 2.0 Easy/Hard、RoboCasa-GR1 Tabletop（24 任务×50 次）；真机 AgiBot G1，三个原任务各 20 次，四个更难任务每任务每域 20 次（附录 D）。
- **代码与权重：** 论文正文未给出仓库或项目页，本页因此不填 `code:`。

## 与清单说法的核对

清单（Glen Berseth 的 CoRL 2026 阅读清单，据论文摘要由 LLM 起草，排序为作者个人兴趣模型打分，不是论文质量评价）对该论文的一句话摘要为：「Continuous latent action model on egocentric human video as pretraining supervision for a VLA, then embodiment alignment and post-training.」

| 清单措辞 | 论文核对 | 结论 |
|----------|----------|------|
| Continuous latent action model | 连续高斯潜变量 d=32，与 VQ 离散 LAM 对比（§3.2、表 7） | 成立 |
| on egocentric human video | 23.3K 小时第一人称视频，策略预训练用其中 5.1K 小时清洗子集（§3.1） | 成立；清单未提 LAM 与策略用的数据量不同 |
| as pretraining supervision for a VLA | Stage 1 让 VLA 预测 LAM 后验均值潜在动作块（§3.3） | 成立；推理时不用潜在动作 |
| then embodiment alignment | Stage 2 用 250 小时目标机器人数据（162,767 条 episode，附录 D.1） | 成立；清单没说明这一步数据量不小，对齐数据为自采 |
| and post-training | Stage 3 在 RoboTwin、RoboCasa-GR1 与真机任务上微调 | 成立；仿真与真机都有 |

- 清单摘要未含任何数字、仿真或真机的限定词，也没提评测规模。论文自己的数字：真机三任务均值 63.3% 升至 85.0% 对应 Stage-1 人视频 1K 到 5K 小时，每任务 20 次，同一机器人平台 AgiBot G1（§4.3、§5）。
- 清单的一句话是方法描述，没有夸大；需要读者补上的限定是：效果依赖 250 小时对齐数据，且论文自己说对齐仍是必需的（§5）。
- 清单把该论文列在「Learning from Human Video」组，与论文主题相符。作者署名在清单里只写「Xiang, Liu, Li et al.」，与论文作者列表前三位（Xiang、Liu、Li）一致。

## 局限

**论文自述（§5）：**

- 视觉潜在动作不保证与机器人可独立控制的自由度对齐，所以目标机器人对齐必需。
- 视频操作信号弱、具身差距大、遮挡重或相机运动主导时可能失败。
- 真机只有 7 个任务、1 个平台、每条件 20 次，成功率精度有限，需要更广验证后才能谈部署鲁棒性。

**本页的观察（未经论文验证）：**

- 单次试验量小：每格 20 次时，一次成功相当于 5 个百分点；表 3 里 LAPT-Scale 85.0 对 JoyAI-RA 80.0 的差距不大。论文也未报告多随机种子或置信区间（推测：在已读范围内未见）。
- 手部运动基线 5K 小时回落到 55.0，逐任务波动很大（表 15：Cup 从 100.0 到 40.0），论文把原因归于手部估计噪声，但未单独验证该解释；该基线受限于只含有手部姿态标注的来源，且排除了 Egocentric-10K。
- 缩放曲线只有 3 个点（1K/3K/5K），5K 与全量 5.1K 的主结果混用；论文「持续增益」的结论不能外推到更大规模，作者也把更系统的缩放律列为未来工作。
- LAM 训练集（23.3K 小时）中自采占 8.5K，Stage-1 子集中自采占 2.9K，自采数据未见公开说明，外部复现受限（推测）。
- 表 7 中 VQ-LAM 的三个指标与表 17 中 LAPA 完全相同（18.3416 / 0.6741 / 0.3443），论文未说明二者是否为同一模型，读表时应注意 VQ-LAM 对照可能就是 LAPA 的离散潜在动作。
- Stage-1 受益的对照以同一 JoyAI-RA 架构为基础，且作者团队与 JoyAI-RA 同属京东 Joy Future Academy；JoyAI-RA 行对应的是 0.1 版本（论文参考文献 [44]），与更新的 JoyAI-RA 0.5（[20]）没有在表中直接对比。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [模仿学习](../methods/imitation-learning.md)
- [流匹配具身策略](../concepts/flow-matching-embodied-policy.md)
- [具身缩放律](../concepts/embodied-scaling-laws.md)
- [机器人数据监督信号类型](../concepts/robot-data-supervision-signal-types.md)
- [JoyAI-RA 0.5](paper-joyai-ra-05.md)：同属京东 Joy Future Academy、同样用人视频 latent action 的相邻工作
- [CLAP：对比式潜在动作预训练](paper-rcl-2601-04061-clap-contrastive-latent-action-pretraining-for-l.md)
- [Motus：统一潜在动作世界模型](paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)：本文表 1 的基线之一
- [EgoScale：多样第一人称人类数据扩展灵巧操作](paper-sa-2602-16710-egoscale-scaling-dexterous-manipulation-with-div.md)
- [RoboTwin](robotwin.md)
- [RoboCasa](robocasa.md)

## 参考来源

- [lapt-scale_arxiv_2610_13732.md](../../sources/papers/lapt-scale_arxiv_2610_13732.md)
- [arXiv:2610.13732](https://arxiv.org/abs/2610.13732)

## 继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.13732v1)
- [JoyAI-RA 0.5](paper-joyai-ra-05.md)
- [具身缩放律](../concepts/embodied-scaling-laws.md)
