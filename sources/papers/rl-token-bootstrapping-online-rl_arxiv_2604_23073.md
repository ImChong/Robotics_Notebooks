# RL Token: Bootstrapping Online RL with Vision-Language-Action Models（arXiv:2604.23073）

> 来源归档（paper）

- **标题：** RL Token: Bootstrapping Online RL with Vision-Language-Action Models
- **作者：** Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke（Physical Intelligence）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2604.23073>（v1，2026-04-24，cs.LG）
- **PDF：** <https://arxiv.org/pdf/2604.23073v1>
- **项目页：** <https://pi.website/research/rlt>（本次未打开，见 [PI 一手资料补核](../sites/pi-memory-rlt-fast.md)）
- **代码：** 论文未给出代码仓库
- **入库日期：** 2026-11-12
- **核对范围：** 下载 arXiv v1 PDF 并用 pdftotext 通读全文：摘要、§I 引言、§II 相关工作、§III 预备知识、§IV 方法（A 适配 VLA 与 RL token、B 在线 RL）、§V 完整系统（含算法 1）、§VI 实验（任务、基线与消融、Q1–Q4 结果）、§VII 结论、附录 A（贡献）、B（实验细节）、C（基线细节），以及图 1–9 的图注。图 4、5、6、7、8、9 的柱高与曲线数值无法从文本层读出，本页只引用正文与图注中明写的数字。未读参考文献所引的论文、PI 官网文章与视频，也未运行任何代码。
- **通读：** 已完成

## 核心摘录

1. **问题与思路（§I、§IV）：** 大 VLA 在精密操作「最后一毫米」慢且易失败；在线更新整个 VLA 太贵。做法是先在少量任务示范上微调 VLA 并训练一个 encoder–decoder 把最终层 token 嵌入压缩成一个 RL token（图 2 标 1×2048），之后冻结 VLA 与 RL token，只在线训练小 actor 与 critic。
2. **RL token（§IV-A，式 1、2）：** 给 VLA 最终层嵌入序列附加一个可学习的 `<rl>` 嵌入，经轻量 encoder transformer，取该位置输出为 z_rl；decoder transformer 自回归重建原嵌入（对原嵌入 stop-gradient，MSE 损失），所以 z_rl 是瓶颈。该步在小规模示范集上训练，VLA 对重建损失视为冻结，可与 VLA 监督微调联合（算法 1 的权重 α，α>0 才微调 VLA）。脚注：各任务指令固定，故此步丢弃语言嵌入。
3. **critic（§IV-B，式 3）：** 输入 x=(z_rl, 本体状态 s_p) 与动作块，对 chunk 级转移做 C 步 TD 备份，按 TD3 用目标网络；附录 B 写两个 Q 函数的集成，目标取最小值。
4. **actor（§IV-B，式 4、5）：** 高斯策略（附录 B：小的固定标准差），以 x 与 VLA 采样的参考动作块 ã 为输入，损失为 −Q + β‖a−ã‖²；另有 reference action dropout（附录 B：训练中 50% 置零，推理时始终提供）。
5. **系统（§V）：** 先用基座 VLA 做 N_warm 步 warmup 预填 replay buffer；rollout 时人可接管，接管数据覆盖 reference 存入 buffer；步长 2 的块内子采样；异步训练，每次 actor 更新对应 2 次 critic 更新，更新数据比 5；人工在关键阶段切换 base VLA 与 RL 策略。可在末尾让 VLA 额外预测「何时交接」以便测试时自动切换。
6. **设置（§III、§VI-A、附录 B）：** 基座 π0.6，H=50（1 s）；VLA 实际只执行块的前缀（如前 20 步）；RL 块长 C=10，50 Hz，14 维动作，RL 动作 140 维；RL token 由两个腕部相机和一个基座相机图像产生；稀疏 +1 奖励由操作员给出；actor/critic 为 2 层 MLP（隐层 256），螺丝任务为 3 层（隐层 512）。
7. **任务（§VI-A，图 3）：** 螺丝安装（M3）、扎带紧固、以太网插入、充电器插入；整任务 30–120 s，关键阶段 5–20 s。每任务收集 1–10 小时遥操作示范；RL 训练 400–1000 episodes，剔除复位与开销后约 15 分钟到 5 小时的实际机器人数据（§VI-B）。
8. **评测口径（§VI-A）：** 关键阶段评测从部分完成状态重置开始，每任务 50 episodes；整任务评测只做螺丝与扎带，附录 B 称先在关键阶段训练再切到整任务，报告约 5 小时数据后的性能。指标为成功率（人工二值判定）与吞吐（每 10 分钟成功次数）。
9. **结果（§I、§VI-C）：** 关键阶段最多约 3 倍加速；螺丝任务成功率「例如从 20% 到 65%」（§I，未明写是否关键阶段）；整任务评测中螺丝成功率提升 40%、扎带 60%（§VI-C Q1，措辞为 improves by，未写绝对值还是百分点）；以太网任务终策略平均步数比基座少 2 倍；该任务上最终策略比遥操作示范更快（图 9 中位数：遥操作 146，基座 228，RLT 66，单位见图轴 episode length）；图 7 图注称仅用约 5 分钟关键阶段数据（实验总时长约 40 分钟）就超过基座。
10. **基线（§VI-B、Q2、图 6）：** HIL-SERL、PLD、DSRL、DAgger，仅在以太网任务上比较，数据量对齐（附录 C）。HIL-SERL 与 PLD 在该设置学不起来；DAgger 与 DSRL 成功率与 RLT 相当但速度提升小。附录 C 自述 HIL-SERL 失败与 50 Hz 控制频率及缺少动作空间包围盒有关。
11. **消融（Q3、图 7、8）：** 仅在以太网任务上做。去掉 RL token（换冻结 ImageNet 预训练 ResNet-10）吞吐下降 50%；去掉 BC 正则（β=0）是单项最大下降；去掉 chunk（C=1）无法稳定追上基座，且只能配 ResNet 编码；去掉 reference 直通学得更慢、早期漂移、训练中失败更多，但最终能追平。
12. **局限（§VII）：** 训练仍需人提供奖励信号、接管纠正，以及在关键阶段与基座策略之间切换；作者认为这些部分原则上可以用奖励模型和进度预测自动化，全自动流水线留作未来工作。

## 与本仓库既有页面核对的补记

- 旧页写「官网项目页返回 403」：本次未访问项目页，不重复该判断。
- 论文没有给出代码仓库链接；本页与实体页都不写 `code:`。

## 对 wiki 的映射

- [paper-rcl-2604-23073-rl-token-bootstrapping-online-rl-with-vision-lan](../../wiki/entities/paper-rcl-2604-23073-rl-token-bootstrapping-online-rl-with-vision-lan.md)
