---
type: entity
project_id: being-h05
tags:
  - paper
  - vla
  - cross-embodiment
  - egocentric
  - human-video
  - dexterous-manipulation
  - beingbeyond
status: complete
updated: 2026-11-12
arxiv: "2601.12993"
code: https://github.com/BeingBeyond/Being-H
original: ../../sources/papers/being-h05_arxiv_2601_12993.md
related:
  - ../methods/vla.md
  - ../methods/being-h07.md
  - ./paper-notebook-being-h0-vision-language-action-pretraining-from.md
  - ./beingbeyond.md
  - ./cn-os-being-h.md
  - ./paper-x-vla.md
  - ./paper-pi05-open-world-vla.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../concepts/mixture-of-experts.md
  - ../methods/action-chunking.md
  - ./paper-training-time-real-time-chunking.md
  - ./libero-benchmark.md
  - ./robocasa.md
sources:
  - ../../sources/papers/being-h05_arxiv_2601_12993.md
  - ../../sources/sites/research-beingbeyond-com.md
summary: "Being-H0.5（arXiv:2601.12993，BeingBeyond）：把人类第一视角手部动作与 30 种机器人本体映射到统一状态-动作空间，用约 35K 小时（16K 人类视频 + 14K 机器人 + 5K 等效小时视觉语言）预训练 MoT 结构 VLA，配 MoF、MPG、UAC；LIBERO 98.9%、RoboCasa 53.9%，五个真机本体上优于 π0.5（作者自报，真机样本量小）。"
---

# Being-H0.5（arXiv:2601.12993）

**Being-H0.5**（*Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization*，BeingBeyond Team，[arXiv:2601.12993](https://arxiv.org/abs/2601.12993)，[项目页](https://research.beingbeyond.com/being-h05)）是 [Being-H0](./paper-notebook-being-h0-vision-language-action-pretraining-from.md) 的扩展版。本页依据 arXiv v1 全文，章节号指论文章节；「作者自述」与「本页判断」分开标注。

## 一句话定义

以人类第一视角手部动作为跨本体的共同「语言」，把人手与多种机器人映射到同一个物理语义对齐的状态-动作空间，用人类视频、机器人轨迹与视觉语言数据混合预训练一个 Mixture-of-Transformers 的 VLA，再用 MoF、MPG、UAC 适配并部署到不同本体（摘要、§1）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MoT | Mixture-of-Transformers | 理解专家与动作专家共享注意力的结构（§5.1，参照 BAGEL） |
| MoF | Mixture of Flow | 动作专家分「共享基础层 + Top-K 路由的专用专家」（§5.1.2） |
| MPG | Manifold-Preserving Gating | 用切片 Wasserstein 距离估计上下文可靠性，门控特征残差（§5.3.2） |
| UAC | Universal Async Chunking | 按本体设定延迟的异步动作块，扩展 Training-Time RTC（§5.3.3、§6.2） |
| ESA | Embodiment-Specific Adaptation | 按动作槽位的适配器库，后训练只更新活跃槽位（§5.3.1） |
| MANO | Hand Model with Articulated and Non-rigid defOrmations | 参数化人手模型，腕部位姿对齐末端执行器、手指对齐精细操作槽（§5.1.1） |
| UniHand-2.0 | — | 本文预训练语料（§3） |
| RTC | Real-Time Chunking | 推理延迟下保持动作块连续的做法，UAC 的前身（§5.3.3） |
| MWDS | Mean Wrist Displacement Similarity | 预测与真值腕部位移向量余弦相似度的均值，Table 8 指标（§7.3.2） |

## 为什么重要

- **作者自述的问题：** 现有 VLA 多是「单语种」，换本体就失效，且机器人数据稀缺、灵巧手数据不到现有语料的 5%（§1、§3）。
- **作者自述的思路：** 用大规模人类第一视角视频作为稠密物理先验，并用统一动作空间避免每个本体一个动作头（§1、§5.1.1）。
- **本页判断：** 这是「人手 + 多本体机器人 + VL 数据」在同一序列建模里联合训练的一个较完整的工程样本，同时给出了真机盲评协议与异步部署细节，对做灵巧手 / 双臂 VLA 的人有参考价值；但关键因果（统一动作空间、人类视频各自贡献多少）没有被单独拆开（见局限）。

## 核心原理与流程总览

```mermaid
flowchart LR
  A[人类第一视角视频 16K小时] --> D[统一状态动作空间]
  B[机器人轨迹 14K小时 30种本体] --> D
  C[视觉语言数据 5K等效小时] --> E[统一序列建模 QA格式]
  D --> E
  E --> F[MoT 预训练 理解专家加动作专家]
  F --> G[后训练 ESA 槽位适配器]
  G --> H[MPG 门控与 UAC 异步分块]
  H --> I[五个真机本体与仿真基准]
```

**数据（§3）。** 人类视频来自 Ego4D、EPIC-KITCHENS、Egocentric-10K 等，用 HaWoR 估计 MANO 手部位姿与相机外参，用 Gemini-2.5 生成每秒指令与 10 秒意图描述，并做运动质量过滤与左右镜像去偏；机器人数据聚合 OXE、AgiBot-World、RoboMIND、RoboCOIN 等并下采样到 30% 帧，仿真数据占 26%；视觉语言数据含通用 VQA、2D 空间定位与可供性、任务规划三类（§3.1 到 §3.3）。自建采集系统 UniCraftor 用头戴 D435、AprilTag 外参标定与脚踏事件，采集 43 个桌面任务、200+ 小时（§4）。

**模型（§5）。** 骨干由 InternVL-3.5 初始化；文本用 next-token，离散人手动作用掩码 token 预测，连续动作用 Rectified Flow；同一动作块同时受连续流匹配与掩码离散监督，二者通过注意力掩码互不可见（§5.2.3）。统一空间里动作为相对位移、旋转为轴角、关节为绝对弧度，且不做归一化（§5.1.1）。

**后训练与部署（§5.3、§6）。** ESA 只更新活跃槽位的适配器；MPG 把可靠性门控放在投影之前，并保留一个不受门控的先验偏置作回退；推理用 K 小于 10 步的 Euler 去噪，MPG 做 1 到 3 轮精炼，UAC 在去噪时锁定已承诺前缀，双线程环形缓冲（缓冲至少 2 倍块长）。作者称可在 NVIDIA Orin-NX 边缘端以及 10 Hz 桌面臂到 50 Hz 人形上部署（§6.3）。

## 工程实践

- **算力：** LIBERO 专家模型 45k 步，4×A800，块长 8，有效 batch 128（§7.2.1）。预训练算力：引言称将发布「1,000 GPU 小时预训练配方」（§1），正文未给预训练训练步数与总算力，**本页不引用未出现的数字**。
- **真机：** 五个本体为 PND Adam-U、Unitree G1 + LinkerBot O6、FR3 + Inspire Hand、BeingBeyond D1、LeRobot SO-101（Table 2）；10 个任务，每任务 30 到 60 分钟示教（§7.1.1）。
- **评测协议：** 统一黑盒推理服务、随机抽布局与策略、操作员不知道在评哪个模型，默认每策略每布局 K=20 trial（§7.1.1）。
- **开源：** 引言承诺发布权重、训练管线、仿真脚本，之后还有真机部署基础设施和预训练配方（§1）。仓库实际内容见 [cn-os-being-h](./cn-os-being-h.md)，那里的核对只覆盖 LIBERO 训练与评测入口，MPG 实现与 RoboCasa 评测未读。

## 评测证据与清单说法的对比核对

清单（CoRL 2026 Glen Berseth 阅读清单，P1，Oral，VLAs & Generalist Policies）对该论文的一句话摘要（作者说由 LLM 据摘要起草）：「Human-centric cross-embodiment VLA pretrained on 35,000 hours from humans and 30 robots, with a unified action space.」排序是作者的兴趣模型打分，不是论文质量评价。

| 清单说法 | 论文实际 | 判定 |
|----------|----------|------|
| 35,000 小时 | 摘要与 §3：35,000+ 小时，但由 16K 人类 + 14K 机器人 + 5K「等效小时」视觉语言数据相加（Figure 3、4）。约 4 亿样本、约 120B token | 数字对；「来自人类与机器人」略去了 5K 等效小时的 VL 数据与「等效」换算 |
| 来自人类与 30 种机器人 | §3：30 种本体，来自 OXE、AgiBot-World 等已有数据集加自采；仿真数据占 26%（Figure 4）。「30」按 >10 小时的本体计（§3 脚注式说明） | 基本对；部分机器人数据是仿真，摘要未提 |
| 统一动作空间 | §5.1.1：物理语义对齐的统一状态-动作空间，人手 MANO 作为一种本体 | 对 |
| 人类中心 | §1、§3.1：16K 小时第一视角人类视频、人手作为「通用末端模板」 | 对 |
| 跨本体 VLA | §7.1：5 个真机本体、单个 checkpoint；仅 Adam-U 上有零样本任务本体组合，成功率非零但低（§7.1.2，正文无具体数字） | 对，但「跨本体泛化」在真机上是 10 个任务级别的验证，零样本只有定性描述 |
| （清单未提）LIBERO 98.9%、RoboCasa 53.9% | §7.2，Table 4、5；摘要即写 | 清单未提，论文的两个主要仿真数字 |

另有一处论文自身的不一致：Table 8 里去掉 L_mask 后 MWDS 反而更高（Lab 0.35 对 0.33，Wild 0.28 对 0.20），与正文「去掉后明显下降」相反，见局限。

## 结论

Being-H0.5 为人类视频与多机器人统一训练提供了系统方案，但跨本体零样本的可靠性仍需更充分验证。

- 统计训练规模时拆开人类、机器人、仿真与视觉语言等效小时，避免把 35K 小时全当机器人遥操。
- 分别测试分布内和未见任务本体组合，仿真均分不能代替目标真机的泛化评测。
- 复现时拆分统一动作空间、预训练、MPG 与 UAC 的贡献，并保留论文表文不一致的掩码实验疑点。

## 局限

**论文自己写出的：**
- 零样本迁移成功率「低于分布内」且动作精度不稳（§7.1.2）。
- 骨干选择影响大，作者表示尚未系统比较其他骨干（§5.1）。
- 仿真数据占比刻意封顶 26%，因其会加剧 sim-to-real 差距（§3.2）。
- 论文没有独立的局限章节，结论（§8）只讲优点。

**本页看到的、论文里未验证的：**
- 真机成功率由柱状图读数（Figure 10），均为 5 的倍数，对应样本量很小，没有置信区间或方差，4 点以内的差距（如双臂通才 45 对 π0.5 的 40）难以判断显著性。
- 真机对比只有 π0.5 一个外部基线，且 π0.5 在「同一每本体数据」上微调；统一动作空间、MoF、MPG、UAC、人类视频各自的贡献没有被单独拆开（真机上只有「有无预训练」与「有无 MPG+UAC」两类消融）。
- Table 8 与正文矛盾：表中去掉 L_mask 数值更高并被加粗，文字却说下降。无法判断是表还是文字有误，因此掩码 token 预测的收益这一论断本页不采信。
- 「首个」「最大」类说法（首次统一人手与多种机器人控制、最大具身预训练配方）是作者自述，以对比数据集的口径为准（§3 称统计本体数时只算 >10 小时的数据），本页未独立核实。
- 预训练的总算力、训练步数、各数据源的混合比例对结果的敏感性未给出，复现难度只能等仓库与配方发布后评估。
- LIBERO 上 EO1 在 L-Spatial 与 L-Object 更高（Table 4），Being-H0.5 的优势主要在平均与 L-Long。LIBERO / RoboCasa 是仿真基准，不等于真机泛化。

## 关联页面

- [VLA 方法页](../methods/vla.md)
- [Being-H0（前作）](./paper-notebook-being-h0-vision-language-action-pretraining-from.md)
- [Being-H0.7（后续潜空间世界动作模型）](../methods/being-h07.md)
- [智在无界（BeingBeyond）](./beingbeyond.md)
- [Being-H 开源仓库源码笔记](./cn-os-being-h.md)
- [X-VLA（跨本体软提示 VLA）](./paper-x-vla.md)
- [π0.5（真机基线）](./paper-pi05-open-world-vla.md)
- [流匹配具身策略](../concepts/flow-matching-embodied-policy.md)
- [Mixture of Experts](../concepts/mixture-of-experts.md)
- [动作分块](../methods/action-chunking.md)
- [Training-Time Real-Time Chunking](./paper-training-time-real-time-chunking.md)
- [LIBERO 基准](./libero-benchmark.md)
- [RoboCasa](./robocasa.md)

## 参考来源

- [being-h05_arxiv_2601_12993.md](../../sources/papers/being-h05_arxiv_2601_12993.md)（论文归档与核对范围）
- [research-beingbeyond-com.md](../../sources/sites/research-beingbeyond-com.md)（BeingBeyond 站点研究列表）
- [arXiv:2601.12993](https://arxiv.org/abs/2601.12993)

## 继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2601.12993)
- [项目页](https://research.beingbeyond.com/being-h05)
- [Being-H 仓库](https://github.com/BeingBeyond/Being-H)
