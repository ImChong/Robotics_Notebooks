# Flow Policy Gradients for Robot Control（arXiv:2602.02481）

> 来源归档（paper）

- **标题：** Flow Policy Gradients for Robot Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2602.02481>
- **PDF：** <https://arxiv.org/pdf/2602.02481>（本次读 v1，2026-02-02）
- **项目页：** <https://hongsukchoi.github.io/fpo-control>
- **代码：** https://github.com/amazon-far/fpo-control（项目页给出的链接；论文正文写「We will release code」，本次未核对仓库内容）
- **作者 / 机构：** Brent Yi、Hongsuk Choi 等（Amazon FAR、UC Berkeley、Stanford、HKU、CMU），Pieter Abbeel、Guanya Shi、Karen Liu、Angjoo Kanazawa 等
- **入库日期：** 2026-11-12
- **一句话说明：** 在 FPO（Flow Policy Optimization，无似然的流匹配策略梯度）基础上提出 FPO++（逐样本比值 + 非对称信任域 ASPO），在 IsaacLab 足式运动、人形运动跟踪（含 T1 / G1 真机）与 BC 预训练操作策略微调上验证。
- **会场：** 经 CoRL 2026 阅读清单标 Oral（清单口径，本次未在 CoRL 官方页核对）

## 核查范围与通读

- **核对范围：** arXiv v1 PDF 全文逐段读完：正文 I–V 节（引言、背景、FPO++ 方法、四组实验、局限与结论）、参考文献，以及附录 A–E（相关工作、梯度方差、实验细节与超参表、补充实验 D.1–D.5、流场可视化）。图表看了图注与坐标轴文字；曲线的具体数值只能从文字与表中取，未逐点读图；Fig. 3 真机照片与 Fig. 9 步态图只读图注。未读：项目页视频内容、代码仓库、被引文献（FPO 原文 arXiv:2507.21053 等仅凭本文转述）。
- **通读：** 已完成

## 核心摘录

1. **问题（§I）：** 基于似然的策略梯度（PPO 等）要求动作似然可微，限制策略为高斯等简单分布；流策略的似然需对体积变化积分，代价高。FPO 用条件流匹配（CFM）损失差构造 PPO 比值的代理，绕开似然。
2. **作者自述的失败（§IV-A）：** 原版 FPO 此前只在较简单的合成环境验证，在 IsaacLab 足式环境中易陷局部极小与灾难性失败，调超参未能解决（Fig. 2）。
3. **FPO++（§III-B/C）：** (1) 逐样本比值（式 10），对每个 (tau_i, eps_i) 单独裁剪；(2) 非对称信任域 ASPO（式 11–13），正优势用 PPO 裁剪，负优势用 SPO（Simple Policy Optimization）目标。
4. **零采样 zero-sampling（§III-D）：** 训练时初始噪声取自 N(0, I)，测试与评估时从 eps = 0 起做 Euler 积分。
5. **足式基准（§IV-A，附录 C.1）：** IsaacLab 速度跟踪，Go2、Spot、H1、G1；actor 3 层 MLP 256 单元，critic 768；4096 并行环境；64 Euler 步；Fig. 2 中 FPO++ 为 5 个种子平均，FPO 为 3x3x3 超参网格。
6. **人形 sim-to-real（§IV-B）：** Booster T1 速度条件行走（基于 HumanoidVerse）与 Unitree G1 全身动作跟踪（基于 BeyondMimic 代码库，LAFAN 的 6 段参考动作，每段平均约 2.5 分钟，每段动作一个策略）；G1 训练用 50 步流积分，部署用零采样 + 5 步。作者称这是「据我们所知」首个无需专家蒸馏的流策略、也是首个无显式似然的策略梯度方法的人形 sim-to-real（§IV-B）。真机结果以 Fig. 3 的照片展示（稳定步态、长序列跟踪、外力扰动），文中没有真机定量表。
7. **操作微调（§IV-C，附录 C.3 / D.3 / D.4）：** RoboMimic Can、Square 与 DexMimicGen Box Cleanup、Tray Lift、Threading，共 5 个任务，均在仿真；图像输入 ViT + 3 层 MLP，动作块长 16，10 步流采样；基线为 FPO 与两种 DPPO 变体。Fig. 4：FPO++ 收敛快、最终成功率高；Can 任务基础策略随机采样成功率约 10%，零采样可到 70% 以上（§IV-C、§IV-E），DPPO 在该任务上失败。附录 D.3 另用 DPPO / ReinFlow 原作者代码在 Can、Square 上对比，FPO++ 最终成功率最高（Fig. A.7）。
8. **消融（§IV-D，Fig. 5–6）：** 去掉逐样本比值或 ASPO，4 种机器人上平均回报下降且方差增大；ASPO 保持熵（Fig. 7）。但 ASPO 在操作微调上有时有害（附录 D.5），操作实验因此关闭 ASPO。
9. **与高斯 PPO 对比（§IV-F，Fig. 8）：** FPO++ 在各并行环境数下回报「几乎总是」更高、种子间方差更小；Spot 上产生 trot 步态而高斯 PPO 倾向 pronk（Fig. 9），动作维度间出现相关性（Fig. 10）。
10. **作者承认的局限（§IV-F）：** 墙钟时间比高斯 PPO 长（G1 行走达到回报 25：PPO 19 分钟，FPO++ 23 分钟）；动作跟踪最多慢约 3 倍，回报略低于调好的高斯 PPO（附录 D.2，作者推测缺熵正则与 KL 自适应学习率；补上后高斯 PPO 峰值回报仍更高）。

## 对 wiki 的映射

- [paper-flow-policy-gradients-robot-control](../../wiki/entities/paper-flow-policy-gradients-robot-control.md)
