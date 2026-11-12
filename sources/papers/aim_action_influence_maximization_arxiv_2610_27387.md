# Action Influence Maximization 论文来源归档

## 元数据

- **题目：** Training Controllable World Models via Action Influence Maximization（方法简称 AIM）
- **作者：** Samuel M. Bateman, Tenny Yin, Chongyi Zheng, Lei Huang, Brian Wang, Benjamin Eysenbach, Jaime Fernández Fisac, Dhruv Shah
- **单位：** Princeton University（Lei Huang 另署 Columbia University）
- **预印本：** https://arxiv.org/abs/2610.27387（cs.RO，v1，2026-10-19）
- **PDF：** https://arxiv.org/pdf/2610.27387v1
- **项目页：** princeton-prism.github.io/action-influence-maximization（论文首页所列；本次未打开）
- **代码：** https://github.com/princeton-prism/action-influence-maximization（论文首页所列；本次环境无法访问 GitHub，未核实仓库是否已公开及内容）
- **入库日期：** 2026-10-22；2026-11-12 据全文重写
- **核对范围：** arXiv v1 PDF 全文：摘要、§1 引言、§2 相关工作、§3.1-3.4 方法与理论、§4.1-4.3 实验与消融、§5 结论与局限、附录 A（定理 1/2 与引理 1 证明）、B（被动数据缩放）、C（算法 1/2、策略与奖励细节、表 2-4）、D（世界模型结构与表 5）、E（实验设置、基线、表 6）、F（Four Rooms 扩展）、G（Push-T 信任域）、H（Push-T 其他指标）。图（Figure 1-13）只读到图注和正文对图的描述，未逐图核对曲线像素，数值以表 1 和正文为准。没有读：参考文献逐条内容、项目页、代码仓库。
- **通读：** 已完成

## 论文要解决的问题

动作条件的视频世界模型在成功、目标导向的示范数据上训练后，倾向忽略输入动作，不管被问什么动作都生成合理的专家式未来（摘要、§1）。作者把这一失败归因于训练数据缺少足够的「动作影响」（action influence），并证明数据的这一指标是任何在其上训练的模型所能表达的动作影响的上界（Theorem 1，§3.2）。

## 方法要点（作者自述）

- 可控性度量：解释性动作信息 Î（explained action information），即「不知道动作时的预测误差（KL）」减去「知道动作时的预测误差」，式 (1)，§3.1。
- Theorem 1（§3.2，证明在附录 A.1）：Î(ρ,π) ≤ I(ρ,π)，后者是数据的条件互信息 I(A; S′ | S)；作者指出这等价于单步 empowerment 目标（引用 [67,68]）。
- AIM 奖励，式 (4)（§3.3）：r = log q(a|s,s′) − log β(a|s) + log ρ_eval(s) − log d(s)。前两项为动作信息（MI）项，后两项为参考状态覆盖项。Theorem 2：该奖励的玻尔兹曼倾斜策略 π* ∝ β·exp(E[r]) 与重放缓冲上的情境 bandit 问题同解，且若 β≠π* 则 I(ρ,π*) > I(ρ,β)（条件见附录 A.2-A.3）。
- 实用化（§3.4、附录 C，算法 1/2）：批量在线循环。每轮用流匹配模型加 FFJORD-Hutchinson 估计器（训练时自蒸馏摊销散度估计）拟合 q、β、d 三个密度，用 held-out 似然选检查点，对两项分别 z-score 标准化（式 C.2，实际使用的奖励）；再用 IQL 评论家加 AWR 提取到扩散策略，AWR 权重截断到 [0,100]；采集 B_k 条轨迹并入重放。ρ_eval 取目标状态空间上的均匀分布（§3.4）。

## 实验设置（§4、附录 D-E）

- **Four Rooms**：离散动作、完全可观测、确定性网格环境，可穷举所有状态动作对；世界模型为 97.8M 参数的 DiT 流匹配模型（表 5），从零训练；评测为把预测观测按最大似然匹配回仿真状态得到的 Sim Fidelity，n=5 个随机布局（附录 E）。
- **Push-T**：6 维连续动作、像素观测，ManiSkill 仿真；世界模型为 DreamerV4 风格的潜变量 Diffusion Forcing 模型（44.5M 参数，表 5），从零训练。可控性指标为带仿真特权掩码的末端执行器（EE）像素召回（H=10；消融用 256×256、H=33）。评测在三种「学习者」策略的 rollout 上：弱行为克隆（BC）扩散策略、反向任务「Push Away」PPO 策略、两者交织的 BC-Failure。
- 基础数据：Push-T 的「专家」是 ManiSkill 状态型 PPO 策略（不是人类遥操作），先验数据集 3005 条轨迹约 1.1×10^5 步（附录 E）。每轮收集 100 条，Four Rooms 4 轮、Push-T 5 轮（§4.2）。
- 基线：均匀随机、Fixed-Start 随机、随机游走、合成 RaC 式恢复数据，以及基于 SAC 的 DIAYN、DADS、ICM、METRA（附录 E、表 6）。这些内在动机基线的奖励使用仿真特权状态（31 维状态，EE 与 Tee 位姿），作者称它们在纯像素观测下无法有效探索（附录 H）。
- 统计：每个数据集、规模、分辨率只训练一个世界模型；每个模型 3 个推理种子、每个评测分布 100 条 rollout，对轨迹和推理种子做 10000 次自助重采样得到置信区间（附录 E）。

## 报告的结果（论文自报）

- 数据与保真度相关性（§4.1、图 3）：Four Rooms 中，跨数据规模，解释性动作信息与仿真保真度 r=0.979，95% CI [0.96, 0.99]。共训练 68 个被动采集世界模型（Push-T 8 个规模、Four Rooms 9 个规模、每个规模 4 种被动策略）。
- 主要结果（§4.2、图 5）：共 73 个世界模型。Four Rooms 中均匀随机最强，AIM 在更多数据下接近；Push-T 中随机、恢复和内在动机基线在 EE 可控性上持平或下降，AIM 逐轮提升。3k 专家加 500 条 AIM 数据在 BC rollout 的 EE 可控性上持平 300k 专家基线（摘要与结论称数据量缩减 85 倍），在 Anti-Push-T（Push Away）rollout 上只是接近 19k 专家基线。
- 表 1（§4.3，H=33，256×256，BC 评测集）：EE 召回 PPO-3K 0.267，PPO-3K+AIM500 0.485，PPO-300K 0.275；联合召回 0.203 到 0.344（作者称比 PPO-3K 高 69%，比 300K 基线高 54%）；人类偏好 Elo 732 到 1190（1545 次比较、20 名标注者）。
- 表 1 的其他评测集：BC-Failure 的 EE 召回 0.517 到 0.564（300K 为 0.554）；Push Away 的 EE 召回 0.297 到 0.375（300K 为 0.332），但 Tee 召回 0.190 到 0.201，仍低于 300K 的 0.282。Recovery-3K 作基底再加同一份 AIM500，BC 联合召回 +36%（§4.3）。
- 消融（§4.3、表 1）：不重新估计奖励（只用第一版奖励采 500 条）BC 联合召回反而比迭代 AIM 高 10%；去掉新颖性项（只剩 MI）在 BC 与 BC-Failure 人类评测上略高（+16%/+8%）；去掉 MI 项（只剩新颖性）在 Push Away 联合召回高 16%。
- 附录 F-H：Four Rooms 中低数据（150 条）模型几乎完全忽略动作，300k 条时几乎完美（图 8）；Tee 像素召回在所有方法下都偏低，作者归因于 Tee 位置覆盖不足与缺少大规模预训练（附录 H）；未经 AWR 信任域约束的 AIM 奖励会给工作空间外区域高奖励（图 11，附录 G）。

## 作者自述的局限（§5）

- 需要重训多个学习密度奖励模型，实现复杂、算力高；
- 序贯 RL 近似不保证提升情境 bandit 目标；
- 未讨论后训练对数据效率的影响；
- 工作是「仿真中的概念验证」，迁移到持续改进的通用模型尚待未来工作。

## 核对时的观察（读者判断，非论文原话）

- 摘要中的「85 倍」对应 Push-T BC rollout 上的 EE 可控性这一特定指标与设置（§4.2），在 Push Away 评测集上不成立。
- 每个配置只训练一个世界模型，置信区间只反映评测抽样，未包含训练种子方差（附录 E）。
- 基线与 AIM 的观测输入不同（基线用仿真特权状态，AIM 用像素），比较并非同一输入条件（附录 E、H）。
- 全部实验在仿真中完成，没有真机数据。

## Robotics_Notebooks 映射

- 实体页：[Action Influence Maximization](../../wiki/entities/paper-aim-action-influence-maximization.md)

## 一手来源

- arXiv v1 PDF 全文（本次通读）：https://arxiv.org/pdf/2610.27387v1
