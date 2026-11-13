---
type: entity
tags:
- paper
- pi
- physical-intelligence
- vla
- reinforcement-learning
status: complete
updated: 2026-11-12
arxiv: '2604.23073'
venue: '2026'
summary: RL Token（Physical Intelligence，arXiv:2604.23073）：把 π0.6 的最终层嵌入压成一个 RL token，冻结 VLA 后只在线训练小 actor/critic 去修正 VLA 的动作块，四项真机精密操作任务上以分钟到数小时的数据提升关键阶段速度与成功率。
original:
- ../../sources/papers/rl-token-bootstrapping-online-rl_arxiv_2604_23073.md
related:
- paper-rcl-wam-robot-learning-control-survey.md
- ../overview/rcl-awesome-wam-technology-map.md
- ../methods/generative-world-models.md
- ../methods/vla.md
- ../tasks/manipulation.md
- ../tasks/locomotion.md
- ./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md
- ../methods/reinforcement-learning.md
- ../methods/td3.md
- ../methods/action-chunking.md
- ../methods/residual-policy-learning.md
- ../methods/dagger.md
- ./paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md
sources:
- ../../sources/papers/rl-token-bootstrapping-online-rl_arxiv_2604_23073.md
- ../../sources/papers/rcl_awesome_wam_2604_23073_rl-token-bootstrapping-online-rl-with-vi.md
- ../../sources/papers/rcl_awesome_wam_catalog.md
- ../../sources/repos/awesome-world-action-models-rcl.md
- ../../sources/sites/pi-memory-rlt-fast.md
---

# RL Token：用 VLA 表示启动在线强化学习

## 一句话定义

RL Token（RLT）在预训练 VLA（π0.6）的最终层嵌入上训练一个 encoder–decoder，得到一个紧凑的「RL token」；之后冻结 VLA 与该表示，只在线训练小 actor 与 critic，在 VLA 给出的参考动作块附近做局部修正（论文摘要、§IV）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
| --- | --- | --- |
| VLA | Vision-Language-Action | 视觉和语言条件下生成动作块的大模型，这里指 π0.6 |
| RL | Reinforcement Learning | 用交互反馈优化策略 |
| RLT | RL Token | 论文方法名，也指其压缩表示 z_rl（图 2 标为 1×2048） |
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | 双 critic 取最小值的离策略算法，论文 critic 沿用（§IV-B） |
| BC | Behavior Cloning | 此处指 actor 损失里靠近 VLA 参考动作的正则项（β 项，式 5） |
| UTD | Update-to-Data ratio | 更新数据比，论文取 5（§V） |
| HIL-SERL | Human-in-the-Loop Sample-Efficient RL | 论文的基线之一，小网络加 ResNet 编码器（§VI-B） |
| PLD | Probe-Learn-Distill | 基线之一，单步残差策略（§VI-B） |
| DSRL | Diffusion Steering via RL | 基线之一，在扩散噪声空间做 RL（§VI-B） |

## 为什么重要

- 作者要解决的矛盾（§I）：大规模 RL 训练基础模型需要大量数据，而数据高效的真机 RL 通常只训练小模型，牺牲 VLA 的泛化。RLT 想同时用 VLA 的表示与先验和轻量在线 RL 的样本效率。
- 作者自述的结果（摘要）：四项真机任务（螺丝安装、扎带紧固、充电器插入、以太网插入）上，最难阶段速度最多提升约 3 倍，成功率在几分钟到几小时练习内显著提高，部分任务超过人类遥操作速度。
- 我的判断：它给「冻结大模型加轻量在线头」这条路线提供了一个具体、含消融的真机配方，但见下文，其评测口径比标题容易让人以为的更窄。

## 核心原理与流程总览

```mermaid
flowchart LR
  D[少量任务示范 1到10小时] --> A[微调 VLA 并训练 RL token 编码解码器]
  A --> F[冻结 VLA 与 RL token]
  F --> W[基座 VLA warmup 预填 replay buffer]
  W --> R[人工选择关键阶段交给 RL 策略]
  R --> P[actor 输入 z_rl 本体状态与 VLA 参考动作块]
  P --> E[执行动作块 人可接管并给成功失败标签]
  E --> B[replay buffer]
  B --> U[离策略更新 critic TD 与 actor 加 BC 正则]
  U --> P
```

1. **适配（§IV-A）**：给 VLA 最终层嵌入序列附加可学习的 `<rl>` 嵌入，轻量 encoder 取该位置输出为 z_rl；decoder 自回归重建原嵌入（对原嵌入 stop-gradient，MSE），所以 z_rl 是信息瓶颈。可与 VLA 监督微调联合训练（算法 1 中 α>0）。因各任务指令固定，此步丢弃了语言嵌入（§IV-A 脚注）。
2. **critic（§IV-B 式 3）**：输入 (z_rl, 本体状态) 与动作块，对块级转移做 C 步 TD 备份，按 TD3 用目标网络，两个 Q 函数取最小（附录 B）。
3. **actor（§IV-B 式 4、5）**：高斯策略，输入含 VLA 采样的参考动作块 ã，损失为 −Q + β‖a−ã‖²，等于在 VLA 动作附近做局部编辑而非无约束搜索。作者称参考块还保留了 VLA 的多模态信息。
4. **reference action dropout（§IV-B、附录 B）**：训练中 50% 的参考块置零，防止 actor 只会复制参考；推理时始终提供。
5. **系统细节（§V）**：块长 C=10（VLA 的 H=50），步长 2 子采样，异步训练，每次 actor 更新对应 2 次 critic 更新，UTD=5。replay buffer 汇集 VLA warmup、在线 rollout 与人工接管数据，接管时人的动作取代 VLA 参考。
6. **关键阶段策略（§V）**：只对每个任务的「关键阶段」用 RL，其余由基座 VLA 执行；训练时由人选择交接时机，测试时可再微调 VLA 去预测何时交接以自动切换。

## 工程实践

- 基座与规模（§III、§VI-A/B、附录 B）：π0.6，50 Hz，14 维动作，RL 侧动作为 10×14=140 维；每任务 1–10 小时遥操作示范；RL 训练 400–1000 episodes，去掉复位与开销后实际机器人数据约 15 分钟到 5 小时；RL token 与 VLA 微调用 2000–10000 梯度步。
- 网络（附录 B）：actor 与 critic 从零初始化，除螺丝任务外为 2 层 MLP（隐层 256），螺丝任务为 3 层（隐层 512）；输入为 RL token（来自两个腕部相机与一个基座相机）加本体状态（螺丝用关节位置，其他任务用末端位姿）。
- 奖励与监督（§II、§V、附录 B）：仅稀疏二值奖励，由人在 episode 结束时标成功或失败；人还负责接管纠正和阶段切换。
- 复现信息：论文未给代码仓库；本页未访问项目页，不确认官方实现是否发布。因此没有源码运行时序图。
- 读结果时：同时看成功率与吞吐（每 10 分钟成功次数），对已高成功率的任务（以太网、充电器）收益主要体现在速度（§VI-C Q1）。

## 评测与指标

- 两种口径（§VI-A）：**关键阶段评测**从部分完成状态重置（如扎带任务一开始已握住扎带两端），每任务 50 episodes，四项任务都做；**整任务评测**从 home 位置开始，只做螺丝与扎带。
- 论文明写的数字：关键阶段最多约 3 倍加速（摘要、§I）；螺丝成功率「例如从 20% 到 65%」（§I，是否限于关键阶段原文未写明）；整任务评测中螺丝成功率提升 40%、扎带 60%（§VI-C Q1，措辞为 improves by，是相对值还是百分点原文未写）；以太网任务终策略平均步数相对基座减少 2 倍（Q2），图 9 中位数（图轴为 episode length）遥操作 146、基座 228、RLT 66（单位按图轴读，论文正文未写物理单位）。
- 对比与消融只在以太网任务上做（§VI-C Q2、Q3）：HIL-SERL 与 PLD 学不起来，DAgger 与 DSRL 成功率与 RLT 相当但速度提升小；去掉 RL token（换 ResNet-10）吞吐降 50%，去掉 BC 正则降幅最大，去掉 chunk 追不上基座，去掉参考直通学得慢但最终追平（图 6–8）。
- 涌现行为（Q4，图 9）：以太网任务中基座 VLA 常在接触点「试探」（靠近、后退、再调整），RLT 策略一次流畅插入，失败时会加压并轻微晃动利用柔顺性；作者称该行为不在示范数据里，来自在线探索。
- 图 4–9 的柱高与曲线数值本次未逐一读数，其余具体比率以原文图为准。

## 与清单说法的对比核对

清单（Glen Berseth 的 CoRL 2026 阅读清单第 003 条，P0，Oral；作者说明条目摘要由 LLM 据摘要起草，排序是其个人兴趣模型打分而非论文质量评价）的一句话：*Exposes a compact "RL token" from a VLA and trains a small actor-critic on it, anchored to the VLA, for hours-scale real-world RL.*

| 清单用语 | 论文情况 | 判断 |
| --- | --- | --- |
| 从 VLA 暴露紧凑的 RL token | 属实：encoder–decoder 瓶颈压缩，图 2 标 1×2048（§IV-A）。「暴露」需要先微调适配 VLA 并训练该模块，不是现成读出 | 基本相符 |
| 在其上训练小 actor-critic | 属实：2 层或 3 层 MLP（附录 B）。actor 同时输入本体状态与 VLA 参考动作块，不只靠 RL token | 相符，略省略输入 |
| 锚定到 VLA | 属实：参考动作条件加 BC 正则（式 4、5）；去掉正则降幅最大（Q3） | 相符 |
| 小时级真机 RL | 实验确为真机（§VI）。数据量为约 15 分钟到 5 小时的有效机器人数据（§VI-B），整任务结果报告约 5 小时数据后（附录 B）；「小时级」大体准确，但不含复位 | 相符，但「有效数据时间」口径要点明 |
| （未提及的限定） | RL 只作用于人工指定的关键阶段，其余由基座 VLA 执行；奖励、接管与阶段切换均有人参与（§V、§VII）；基线与消融只在一个任务（以太网）上（Q2、Q3） | 清单摘要未体现，读者易高估自主程度与证据广度 |

## 结论

RL Token 提供了冻结 VLA 后以小网络优化关键阶段的路径，但仍需要人工定义训练与切换边界。

- 先核对 token 压缩、动作块修正和基座切换协议，再开展在线训练。
- 区分关键阶段与完整任务的速度和成功率，不能把局部改善等同整流程收益。
- 对齐奖励、接管及基线控制设置，并在其他基座与硬件上复验。

## 局限

作者自述（§VII）：训练需要额外人工介入，包括奖励信号、接管纠正，以及关键阶段与基座策略之间的切换；全自动流水线是未来工作。

我看到的未验证之处（判断）：

- 基线对比和消融只在以太网这一个相对简单的任务上（Q2、Q3 均写在该任务上），螺丝、扎带等难任务没有对应的基线与消融证据。
- 整任务评测只覆盖螺丝与扎带，充电器与以太网只有关键阶段评测。
- 所有结果来自同一实验室、同一基座 π0.6 与同一类硬件，论文没有跨基座或跨实验室验证；论文也未报告多随机种子的方差（正文与图注未见）。
- 附录 C 自述 HIL-SERL 失败与 50 Hz 控制频率及缺少动作空间包围盒有关，说明该基线是否被充分调优有限定。
- 论文未给代码仓库，可复现性取决于后续发布（本次未核实）。

## 关联页面

- [π0.6 / π*0.6：从经验中学习的 VLA](./paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md)：论文引为 RECAP，对全模型离线 RL 的路线
- [SERL](./paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md)：样本高效真机 RL 的前序工作
- [TD3](../methods/td3.md)
- [动作分块](../methods/action-chunking.md)
- [残差策略学习](../methods/residual-policy-learning.md)
- [DAgger](../methods/dagger.md)
- [强化学习](../methods/reinforcement-learning.md)
- [VLA](../methods/vla.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [论文通读归档（本页依据）](../../sources/papers/rl-token-bootstrapping-online-rl_arxiv_2604_23073.md)
- [RCL Awesome WAM 清单条目](../../sources/papers/rcl_awesome_wam_2604_23073_rl-token-bootstrapping-online-rl-with-vi.md)
- [PI 一手资料补核](../../sources/sites/pi-memory-rlt-fast.md)

## 继续阅读

- [RL Token 论文（arXiv）](https://arxiv.org/abs/2604.23073)
- [项目页](https://pi.website/research/rlt)
- [Physical Intelligence 层级概览](./pi-physical-intelligence-layer.md)
