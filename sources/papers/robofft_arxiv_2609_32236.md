# RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process（arXiv:2609.32236）

> 来源归档（ingest）

- **标题：** RoboFFT: Finetuning Generative Robot Policy via Online Reinforcement Learning with Forward Process
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2609.32236>（2026-09-26，v1，cs.RO）
  - PDF：<https://arxiv.org/pdf/2609.32236>
  - 项目页：<https://student-of-holmes.github.io/RoboFFT/>
  - 代码：<https://github.com/shshsunny/RoboFFT>
  - Hugging Face：<https://huggingface.co/shshsunny/RoboFFT>
  - ModelScope：<https://modelscope.cn/models/shshsunny/RoboFFT/tree/master>
- **作者：** Yu Li、Shenghe Hu、Yuhan Wang、Yaoxiang Pu、Haotong Zhang、Yuanpei Chen、Yaodong Yang（Li 与 Hu 共同一作；Chen 与 Yang 通讯）
- **机构：** 北京大学人工智能研究院、中关村学院、PKU-PsiBot 联合实验室、南京大学（论文首页脚注）
- **会议：** CoRL 2026（项目页标注录用；论文 PDF 内未见会议标注，未核实）
- **入库日期：** 2026-10-16
- **一句话说明：** 面向扩散与 flow 生成式机器人策略的在线 RL 微调方法：对 rollout 动作做前向加噪，以加权 score / flow matching 损失之差构造 PPO surrogate ratio，不再把逆向去噪链当作多步 MDP。
- **核对范围：** 下载 arXiv v1 PDF（首页标题与编号核对一致），通读摘要、正文第 1-7 节、图 1-10 与图注、局限与结论，以及附录 A-H（方法细节与 ELBO 解释、Robomimic 实现与超参、额外种子与效率与代理比率协议、附加消融、可视化协议、VLA 实现、真机设置与算法 2、补充相关工作）；图中曲线只看趋势，未逐点读数，部分图内数字标注（如图 4 各任务倍率）因文字抽取不全未逐项核对；未读参考文献所引的外部论文，未读项目页视频与代码仓库源码本身（仓库信息沿用既有归档，见 [官方代码归档](../repos/robofft.md)）。
- **通读：** 已完成

## 论文要点（带章节）

### 1) 问题与动机（摘要、第 1 节）

- 作者自述：扩散与 flow 策略缺乏可计算的精确似然，使 PPO 难以直接使用；现有方法把反向去噪过程建模成嵌入环境 MDP 的多步 MDP（DPPO、ReinFlow 等），带来计算与显存开销，且长去噪链可能梯度不稳（第 1 节，图 1(a)）。这是动机陈述，并非本文单独验证的命题。
- RoboFFT 沿前向加噪过程，用与预训练同形的加权 matching loss 近似对数似然。灵感来自图像生成与简单运动控制领域的前向过程工作（DiffusionNFT、AWM、FPO、FPO++，第 1、2.2 节）；作者明确说 FPO 系工作主要面向 locomotion，RoboFFT 面向操作并覆盖扩散、flow、VLA 与真机（第 2.2 节）。

### 2) 方法（第 3、4 节，附录 A）

- 预训练损失：式 (3)，权重 w(λ_k) 乘以 -dλ_k/dk，扩散取噪声目标，flow 取速度目标 v = ε − a⁰（rectified flow，α_k = 1−k，σ_k = k）；预训练通常令乘积恒为 1（第 3.2 节）。
- SDE 采样（第 4.1 节）：flow 策略 rollout 时注入 Euler-Maruyama 噪声，σ_sam(k) 实践中取常数 η；该采样器只用于收集动作，不参与似然计算。
- surrogate ratio（第 4.2 节，式 5-6）：ρ̂ = exp(L_old − L_new)，用于标准 PPO 裁剪目标，优势用 GAE（式 13）。动机是加权去噪目标与 ELBO 的联系（Kingma 与 Gao），附录 A.1 给出推导：ELBO 比等于真实似然比乘以变分间隙项的指数，故 surrogate 带一个额外修正项，即它不是精确似然比。
- 实现技巧（第 4.3 节）：Monte Carlo 取 N_mc 个 (k, ε)，新旧策略复用同一批 (k, ε) 以稳定差分；ratio temperature α_ratio 重新标定损失尺度（式 8）。
- RoboFFT-D（附录 B.6）：基于 DPPO 代码库；噪声预测损失按离散 Δlog-SNR（式 16-17，σ² 设下限）加权；N_mc = 16；困难任务用 best-evaluation anchor 正则（式 21-23）；负优势样本折扣系数 β_neg（式 24）。RoboFFT-F 基于 ReinFlow 代码库，权重恒为 1，N_mc 通常 1 或 2（表 4、表 6）。

### 3) 仿真实验（第 5.1、5.3 节，附录 B-D）

- 任务：Robomimic Lift、Can、Square、Transport，state 与 pixel 两种输入，稀疏奖励（表 1）；pixel 为 96×96 RGB（Transport 为两视角）；动作维度 7（Transport 为 14）；预训练 BC 检查点与 DPPO、ReinFlow 的 Robomimic 数据一致，不额外引入示范（附录 B.2）。基线：DPPO、ReinFlow（反向过程），DRWR、FPO（前向过程，FPO 为作者自行改入同一代码库，附录 B.4）。
- 结果（图 3，3 个随机种子均值与一个标准差）：Lift、Can 上最终成功率与反向过程基线相当，早期收敛有时较慢（附录 C.1）；Square、Transport（尤其 pixel）上作者称 RoboFFT（特别是 RoboFFT-F）更稳定、更强；DRWR 与 FPO 通常较弱。图 3 为曲线，正文未给汇总表。附录 C.2：Square-state 用 10 个种子、Transport-state 用 5 个种子对 RoboFFT-F、FPO、ReinFlow 复核，Square-state 上 RoboFFT-F 与 ReinFlow 相近（作者称 FPO 微调中退化），Transport-state 上 RoboFFT-F 平均成功率高于两个 flow 基线。
- 效率（图 4，第 5.1 节，附录 C.3）：计时只含策略与价值网络更新，不含 rollout 采样；取每个配置前 10 次训练迭代的均值与标准差，误差棒反映迭代间波动而非跨种子；RoboFFT-F 相对 ReinFlow 平均 1.53×；四个 state 任务上 RoboFFT-D 相对 DPPO 平均 38.0×（N_mc = 16）。倍率为各任务倍率的算术平均。论文自己提醒：速度比与网络结构、更新设置、内部采样步数有关，不等于整轮迭代加速。
- 消融（第 5.3 节，附录 D）：Square-state 上 SDE 噪声过小导致策略坍塌，过大也退化（图 6a）；N_mc 由 1 增至 2 收敛加快，再增收益有限（图 6b）；α_ratio = 2 最佳；去掉优势加权或去掉 PPO 裁剪会不稳（Transport-state 上两者均坍塌）；RoboFFT-D 上去掉负优势折扣（β_neg = 1）明显恶化，N_mc = 1 明显弱于 4 或 16，Δλ 加权优于朴素加权，anchor 与 ratio temperature 影响较小。
- 代理比率诊断（第 5.3 节，图 8，附录 C.4）：在 RoboFFT-F 的 Square-state 微调中，每个迭代抽 2048 个动作块，对比 ReinFlow 式闭式参考比率。Spearman 秩相关约 0.2，相对误差训练末期 90% 以上样本低于 5%；论文自己说明这只刻画数值差异，并不证明两种似然形式严格对应。
- 可视化（图 7，图 15，附录 E）：用辅助 Q 网络 Q_viz（仅作诊断，不参与更新）度量，微调策略在抓取、搬运、对齐等关键阶段 ΔQ 上升而样本多样性下降；论文自注 t-SNE 密度不能直接解读为似然或模态数，也不单独证明是否发生模式坍塌。

### 4) VLA 微调（第 5.2 节，图 5，附录 F）

- 模型为 π0（3.3B，VLM 骨干加 flow 动作头），在 LIBERO-Object 与 LIBERO-Spatial 上微调，基线为 πRL；基于 RLinf 实现，只把对数概率计算替换为 surrogate matching loss，采集与 PPO 更新不变（附录 F）。
- 表 7 的配置差异：KL 系数 β 为 0.5（πRL 为 0.0），PPO 裁剪范围 0.05（πRL 为 0.20），N_mc = 8。
- 结果只以曲线呈现（图 5，横轴样本数至约 4×10⁶）：RoboFFT 相对预训练 π0 提升，与 πRL 相比作者称有竞争力。正文与图注都没有给出具体成功率数字，本归档不编数。第 5.2 节同时说明，动作块维度高时直接对 flow matching 损失求和会使损失幅值大、更新不稳，是扩展到 VLA 的难点。

### 5) 真机实验（第 5.4 节，图 9、10，附录 G）

- 并非纯在线：采用受 RL100 启发的离线到在线流程。每任务 50 条 SpaceMouse 示范，预训练 rectified-flow 策略，IQL 预训练 critic；离线阶段用 IQL 估计优势并做 RoboFFT 更新；部署后收集 N_roll = 1 轮、每轮 d = 30 条轨迹，与示范合并做模仿学习；再在线 RoboFFT K_on = 5 次迭代；奖励为人工判定的 0/1（附录 G.2，算法 2）。
- 平台：Pick Eggplant 与 Cabinet Retrieval 用 Franka Research 3，Peg Insertion 与 Pour 用 Franka Emika Panda；两个腕部与一个第三视角 RealSense 相机（附录 G.1）。
- 结果（图 9，每任务 20 次试验）：Pretrain / Offline / Online 成功次数为 Pick Eggplant 12/18/20，Peg Insertion 10/19/20，Pour 9/15/18，Cabinet Retrieval 5/14/17。由此汇总 36/80 = 45.0%、66/80 = 82.5%、75/80 = 93.75%，与项目页的 45.0% 与 93.8% 一致（汇总为本归档按图计算，图 9 的数字由文本抽取得到，行列对应按任务顺序与正文引用的 12/20、18/20、20/20 交叉验证）。
- 与 BC Aggregation、DRWR、RL100 的对比只在 Pick Eggplant 上做（相同数据预算）：BC Aggregation 14/20，DRWR 12/20，离线 RoboFFT 18/20，在线 RoboFFT 20/20，RL100 20/20（第 5.4 节）。其他三项任务没有这类基线对比。

### 6) 作者自述的局限（第 6 节）

- surrogate 只是真实对数似然的替代，尺度随噪声日程、动作维度、策略结构变化，需要温度标定等额外校准。
- 依赖 on-policy PPO，数据效率有限，难以复用历史轨迹，单机器人场景下真机部署困难；离线到在线范式只能部分缓解，且仍需迭代采集与离线重训。

## 与既有项目页 / 仓库归档的差异

- 项目页给出的「RoboFFT-F 约 1.53× 于 ReinFlow、RoboFFT-D 约 38.0× 于 DPPO」与论文第 5.1 节一致，但论文限定为单次训练更新时间（不含 rollout），且 38.0× 仅覆盖四个 state 输入任务。
- 项目页的真机 45.0% → 93.8% 与论文图 9 一致；论文同时给出逐任务次数（见上）。
- 仓库归档所述「VLA 与真机代码不在当前 release」属仓库 README 信息，本次通读论文未再核实。

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-robofft.md`](../../wiki/entities/paper-robofft.md) — 论文、项目页与代码统一实体
- [`wiki/methods/diffusion-policy.md`](../../wiki/methods/diffusion-policy.md) — 扩散策略与前向加噪背景
- [`wiki/methods/ppo.md`](../../wiki/methods/ppo.md) — surrogate ratio 与 PPO clipping
