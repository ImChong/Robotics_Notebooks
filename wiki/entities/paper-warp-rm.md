---
type: entity
project_id: warp-rm
tags:
  - paper
  - berkeley
  - stanford
  - reward-modeling
  - progress-reward
  - data-curation
  - behavior-cloning
  - deformable-manipulation
status: complete
updated: 2026-11-12
arxiv: "2606.28320"
related:
  - ../entities/paper-sarm.md
  - ../entities/paper-robometer.md
  - ../entities/paper-topreward.md
  - ../entities/paper-pi0.md
  - ../entities/paper-progress-reward-modeling-survey.md
  - ../methods/behavior-cloning.md
  - ../methods/imitation-learning.md
  - ../methods/dagger.md
  - ../concepts/reward-design.md
  - ../concepts/flow-matching-embodied-policy.md
original:
  - ../../sources/papers/warp-rm_arxiv_2606_28320.md
sources:
  - ../../sources/papers/warp-rm_arxiv_2606_28320.md
summary: "WARP-RM（arXiv:2606.28320）：对成功示教做变速加倒放的时间扭曲，自监督训练帧级相对进度速度模型，再按动作块终端帧速度过滤并加权 π0 行为克隆；真机双臂折 T 恤与瓶子入桶验证，规模为单本体、每配置 20 次。"
---

# WARP-RM：时间扭曲增强的相对进度奖励模型（arXiv:2606.28320）

**WARP-RM**（*A Warp-Augmented Relative Progress Reward Model for Data Curation*，[arXiv:2606.28320](https://arxiv.org/abs/2606.28320)，UC Berkeley / Stanford / XDOF，v1 2026-06-26）。本页依据 arXiv v1 全文整理，是 Glen Berseth 的 CoRL 2026 阅读清单中 P1「Rewards & Progress Models」一档的条目。

## 一句话定义

**一个只用成功示教、无需人工标注的自监督模型：它预测一段视频内两帧之间的归一化时间位移，聚合成逐帧「进度速度」（约 1 为专家平均节奏、约 0 为停滞、负为倒退），再用这个速度对行为克隆的动作块做过滤与加权（WARP-BC）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WARP | Warp-Augmented Relative Progress | 本文的自监督训练算法，用时间扭曲产生进度标签 |
| WARP-RM | WARP Reward Model | 用 WARP 训练出的相对进度模型 |
| WARP-BC | WARP Behavior Cloning | 用 WARP-RM 速度给动作块过滤并加权的行为克隆 |
| RA-BC | Reward-Aligned Behavior Cloning | 按奖励 / 进度对动作块加权的行为克隆，SARM 提出的做法 |
| AR(1) | First-order AutoRegressive process | 一阶自回归过程，本文用来生成平滑相关的变速曲线 |
| TTC | Time-To-Completion | 成功试验的平均完成时间 |
| BC | Behavior Cloning | 行为克隆 |
| DINOv3 | Self-distillation with no labels v3 | 冻结的视觉骨干（ViT-B/16） |

## 为什么重要

- 遥操作数据难免含犹豫与失误；整条丢弃会连同其中有价值的恢复片段一起丢掉，不剔除又会让策略学到停顿（论文 §1）。WARP 给出帧级、可直接对动作块加权的信号。
- 作者的核心主张（作者自述）：以「相对进度速度」代替「绝对归一化进度」，避免跨示教的时间对齐噪声，同时不需要 SARM 那样的人工子任务标注（§1、§2）。
- 在 §4 的设定里，数据越宽松（混入更多低效示教），vanilla BC 退化越快，而 WARP-BC 退化慢得多，说明在「长度作为低效代理」的数据上，过滤加权很重要。

## 核心原理与流程总览

1. **时间扭曲采样器（§3.2）：** 取 N=32 帧；N-1 个对数速度来自平稳 AR(1) 过程（α=0.5，σ∞=ln 2），指数化后缩放到总路径长 ℓ（取自 [L/3, 5L/3]，与附录表 7 的每间隔 0.5 至 2.5 秒一致）；以 Poisson(λ_rev=1) 个反转点翻转后续速度符号，再以 0.5 概率整窗倒放。
2. **标签（式 3）：** y_j = (i_j - i_0)/C_norm，C_norm = (N-1)·S·f = 1395 帧（S=1.5 s，f=30 Hz，附录 C.2）。
3. **模型（§3.4）：** 冻结 DINOv3 ViT-B/16 取每帧 768 维特征，token = [特征, 与前一帧的特征差]，经 12 层双向 Transformer，每个位置输出 30 个 bin 的分布（two-hot 交叉熵）。
4. **推理与聚合（§3.5，附录 C.4）：** 以每步一帧的滑窗（窗内帧间距 1.5 s）预测累计进度，窗内差分得速度 v_j = (N-1)(ŷ_j - ŷ_{j-1})，对覆盖同一帧的所有窗口取平均得 v̂_t，每个数据集只离线算一次。
5. **WARP-BC（式 4–5）：** 动作块（1.0 秒）权重 w = v̂_end · 1[v̂_end > τ]，τ=1.0；w=0 的块在训练前剔除，其余按 w 加权 flow-matching 损失；策略骨干为 π0（附录 C.3）。

```mermaid
flowchart LR
  A[成功示教 参考子集 最短1950条] --> B[时间扭曲采样器 AR1变速加反转加整窗倒放]
  B --> C[窗口累计位移标签 y]
  C --> D[冻结DINOv3加双向Transformer 30 bin two-hot]
  D --> E[滑窗推理加窗口平均 得逐帧速度]
  E --> F[动作块终端帧速度 大于tau才保留]
  F --> G[按速度加权的pi0 flow-matching行为克隆]
```

## 工程实践

- **参考子集决定「1.0」的含义：** WARP-RM 只在最短的 1950 条示教（≤59.8 s，28.7 小时）上训练一次，并冻结后用于 D1 至 D3 全部分层（§4.2、附录 A）。也就是说「专家节奏」是由长度挑出来的，这一步依赖人对「短即高效」的先验（见局限）。
- **阈值与保留率：** τ=1.0 时 WARP-BC 在 D1 / D2 / D3 保留 35.7% / 34.4% / 22.5% 的动作块（表 1）；τ=0 几乎全保留（97.0%）且只有 3/20（表 3）。
- **选择性很关键：** 表 2 里保留 66% 至 78% 块的 SARM 与 SCIZOR 在 D5 上均 2/20，保留 33.7% 的 DemInf 为 18/20；论文据此推测「过滤不够选择性」是原因，这是作者的解释而非单独验证过的因果。
- **成本：** 速度轨迹逐集离线算好并缓存，不增加策略训练时的开销，且被剔除的块不占算力（附录 C.3、C.4）。
- **超参迁移：** 瓶子入桶任务沿用同一套 WARP 超参，仅换训练数据（附录 B）。
- **评估协议：** 折叠 240 秒超时（最长训练示教的 2 倍）；吞吐按每次失败计 4 分钟，人工复位时间不计；折叠中前 10 秒抓错物体、瓶子任务中桶被撞翻则重做不计分（§4、附录 D）。

## 与清单说法的核对

清单原文（作者称由 LLM 据摘要起草）：「Self-supervised relative-progress reward model from time-warped demonstrations, used to curate and weight mixed-quality teleop data.」

| 清单措辞 | 论文核对 |
|----------|----------|
| Self-supervised | 成立。训练不需要人工标注（摘要、§3.1）。但训练集只取「最短 1950 条」成功示教（§4.2），选择标准是长度，属于一个人为设定的先验。 |
| relative-progress | 成立。预测的是窗口内的归一化时间位移与其差分得到的进度速度，而非绝对完成度（§3.1、§3.5）。 |
| from time-warped demonstrations | 成立。变速加反转加整窗倒放（§3.2）。 |
| reward model | 部分成立。论文称 WARP-RM 为 reward model，但实际用法是速度作为「优势代理」做动作块门控与加权，并非 RL 奖励；论文自己写明它是 proxy 而不是带价值基线估计的 RL 优势（§3.5）。 |
| curate and weight | 成立。先阈值过滤再按 v̂_end 连续加权（式 4–5）；连续加权优于二值掩码（19/20 对 16/20，表 3，单次差异未做显著性检验）。 |
| mixed-quality teleop data | 成立，但「混合质量」的操作化定义是示教时长分层（D1 ≤60 s、D2 ≤90 s、D3 ≤120 s），以长度作为低效代理（§4.2）。 |
| 清单未提及的限定 | 全部结果是真机，不是仿真；只有一个双臂本体（I2RT YAM）、两个任务（折 T 恤与瓶子入桶）、每配置 20 次试验（§4、§5）。摘要里的「19/20 对 2/20、约 18 倍」特指 D2 一档；D3 上 WARP-BC 为 14/20、vanilla 为 0/20（表 1）。 |

## 局限

**论文自己承认（§5、脚注 1）：**
- 策略仍受限于训练集中已有行为，后续才考虑 DAgger、离线 / 在线 RL 与自我改进。
- 只有两个真机任务与单一双臂本体。
- 负进度监督完全来自倒放，该近似继承自 ReWiND，物理上可能不真实；是否对过滤加权有益「不能保证」。
- 衣物二维图像面积可能是进度的隐含强代理，缺少此类视觉线索的任务尚未评估。
- n=20 时 19/20 与 20/20 之差处于二项抽样噪声内，只有大差距才有意义。

**读者视角的未验证之处（本页判断，非论文结论）：**
- 「低效」的真值只由时长代理，没有独立的人工质量标注来检验 v̂ 是否真对应犹豫与失误；图 1 与附录 F 是定性示例。
- 参考集取最短示教，若最优策略并非最快（例如对衣物更谨慎的慢动作），该先验会偏向速度。
- 对比方法的复现细节（SARM 需额外标注集 D_A、DemInf / SCIZOR 的设置）在正文给出有限，基线差异可能部分来自调参。
- 单次真机评估、无多随机种子的方差报告；表 2 的 D4 / D5 因加入 D_A 与表 1 不是同一数据。

## 关联页面

- [SARM（阶段感知奖励模型）](../entities/paper-sarm.md)：论文最接近的对照，RA-BC 的来源，需人工子任务标注
- [Robometer](../entities/paper-robometer.md)：论文列为 VLM 通用奖励模型一类（§2）
- [TOPReward](../entities/paper-topreward.md)：论文引用的 token 概率式零样本奖励（§2）
- [进度奖励建模综述](../entities/paper-progress-reward-modeling-survey.md)
- [π0](../entities/paper-pi0.md)：WARP-BC 使用的策略骨干
- [行为克隆](../methods/behavior-cloning.md)
- [模仿学习](../methods/imitation-learning.md)
- [DAgger](../methods/dagger.md)：论文提到次优片段中的恢复行为类似 DAgger 数据
- [奖励设计](../concepts/reward-design.md)
- [Flow Matching 具身策略](../concepts/flow-matching-embodied-policy.md)

## 参考来源

- [warp-rm_arxiv_2606_28320.md](../../sources/papers/warp-rm_arxiv_2606_28320.md)
- [arXiv:2606.28320](https://arxiv.org/abs/2606.28320)
- [项目页](https://uynitsuj.github.io/warp-rm/)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2606.28320v1)
- [SARM 页](../entities/paper-sarm.md)
- [行为克隆方法页](../methods/behavior-cloning.md)
