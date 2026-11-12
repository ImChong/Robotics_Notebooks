# HiRE（arXiv:2609.27068）

> 来源归档（paper）

- **标题：** HiRE: Hindsight Reward Editing for Policy Finetuning
- **作者：** Haoyi Niu*、Zhengtao Han*（UC Berkeley）、Yufeng Ji、Zhongyu Li（香港中文大学）、Koushil Sreenath（UC Berkeley）；* 同等贡献
- **会场：** CoRL 2026（首页脚注：10th Conference on Robot Learning, Austin TX）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.27068>（v1，2026-09-22，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2609.27068>
- **项目页：** <https://hire-project.github.io/>
- **代码：** <https://github.com/HiRE-Project/HiRE>
- **入库日期：** 2026-10-18（2026-11-12 升级为全文通读归档）
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）；Glen Berseth《My CoRL 2026 Reading List》（清单序号 094/101，P2）
- **一句话说明：** 用成功与失败轨迹对照，压低“看似接近目标却会失败”的状态奖励，给策略微调补上控制相关反馈。
- **核对范围：** 通读 arXiv v1 PDF 全文：摘要、引言、相关工作（第 1 到 2 节）、预备知识与方法（第 3 到 4 节，含式 1 到 8）、实验（第 5 节，图 1 到 5）、结论与局限（第 6 节）、参考文献，以及全部附录（A 实现细节含表 1 到 3，B 数学解释，C.1 消融含图 8 到 11）。图表读了正文引用的数值与图 1 曲线形状；项目页与 GitHub 仓库代码未读（仅入库日核查可达），论文附带的项目页定性视频未看。
- **通读：** 已完成

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库与项目页均可达，项目页链接 GitHub 仓库。仓库内容本次未读。

## 全文要点（均为论文自述，章节号为论文章节）

1. **问题（第 1 节）：** 稀疏奖励缺过程反馈，人工设计奖励成本高且有偏，基础视觉（语言）表征的相似度奖励「不具控制感知」（例：机器人已把物体掉落，仍因朝放置区移动而被高奖励）。
2. **方法（第 4 节）：** 理想奖励取成功与失败状态密度的对数比（式 1，借用最大熵逆强化学习的判别器对数几率）；用 von Mises-Fisher 核做核密度估计，把对数密度写成 LogSumExp（式 2 到 4），得势函数 Φ_HiRE(s) = L^{κ+}_{B+}(sim(s,g+)) − λ·L^{κ−}_{B−}(sim(s,g−))，λ = κ−/κ+（式 5）；再用势函数式奖励塑形并入稀疏奖励：R = R_sparse + w(γΦ(s′) − Φ(s))（式 6）。作者称这样不改变最优策略（引 Ng 等 1999）。
3. **缓冲区（第 4.2 节与附录 A.3）：** 正缓冲区 = 专家演示离线部分 + 成功回合的 FIFO 在线部分；负缓冲区 = 失败回合的 FIFO。仿真里只存失败回合的终止帧，在线正样本也只存成功终止帧；仿真环境无失败检测，超时才判失败（A.1）；真机由人工标注成败（A.2）。
4. **主实现（第 5 节与附录 A.3）：** 冻结 DINOv2（dinov2_vits14）编码器，相似度为 patch 平均余弦并在两个相机取平均（式 9）；底座为 DICE-RL（冻结预训练生成式策略，学残差 actor 与 critic 集成）；塑形项除以动作块长度 Ta = 8，并乘 w_dense(t) = 1 − 近期成功率，随成功率上升衰减（式 10）；回放的专家演示只保留稀疏奖励（A.3，引 RLPD）。
5. **仿真结果（第 5.2 节，图 1，三个随机种子均值）：** 基础 BC 策略成功率 Stack_Three 18.0%、Three_Piece_Assembly 6.0%、Threading 1.0%、Tool_Hang 2.0%（第 5.1 节）。Stack_Three：稀疏奖励后期降至约 30%，HiRE 约 50%；Three_Piece_Assembly：稀疏约 70%，HiRE 超过 80%；Tool_Hang：稀疏奖励「完全失败」，HiRE 约 95%；Threading：稀疏奖励最终达到相当水平，HiRE 用明显更少的在线交互达到同样水平（图中 HiRE 末段约 90%，为读图估计）。
6. **基线对比（第 5.2 节）：** RoboMeter-4B、TOPReward（Qwen3-VL-8B-Instruct）「表现不一致、最终成功率显著低于 HiRE」；GCR（自行复现，作者称无官方实现）在两个多阶段任务上达到 30% 成功率约需 HiRE 的 3 倍交互，在 Tool_Hang 与 Threading 上无实质进展。
7. **真机结果（第 5.3 节，图 3，附录 A.2，表 2）：** I2RT YAM 机械臂，两任务基础策略均 20%。Bottle_Pick-and-Place 到 67%（每个检查点评估 30 次，6 个初始位置各 5 次；表 2 中 BC 6/30，在线 60 与 70 回合均 20/30）；Towers_of_Hanoi 到 73.3%（每个检查点评估 15 次），对应 3.67 倍，20 回合时 53.3%。基线只比了稀疏与 RoboMeter：Hanoi 上稀疏峰值 53.3%、RoboMeter 峰值 46.7%，50 回合时分别跌到 30.0% 与 20.0%，作者归因于「奖励被利用或奖励估计有噪声」。示教数量：Bottle 117 条、Hanoi 50 条。
8. **分析与消融（第 5.4 节与附录 C.1）：** 图 2 对比原始 DINO 目标相似度与 HiRE 势函数（仅在 Threading 与 Stack_Three 的代表性轨迹上定性展示）；图 4 critic 损失，稀疏奖励损失过早偏低被作者称为「价值函数坍缩」，RoboMeter 损失不收敛（作者用 probably 措辞）；图 5 t-SNE 显示负缓冲区分布更分散。SigLIP（ViT-B/16）可替换但 DINOv2 在四个任务上一致更好（图 8）；λ ∈ {0.1, 0.3, 0.7, 0.9} 在多阶段任务上不敏感，Threading 对 λ 变小或缓冲区加到 512 更敏感，Tool_Hang 上 λ = 0.9 无法持续提升（图 9）；去掉势函数塑形（w/o PBRS）或改用原始相似度，在 Tool_Hang 上均接近 0（图 10，作者注明各对比组的奖励缩放与回放设置也不同，不是单变量消融）；只用正缓冲区在 Stack_Three 上与完整版接近，在 Tool_Hang 上更慢且终值更低（图 11）。
9. **局限（第 6 节）：** 成功 / 失败信号忽略中间的控制相关信息；可引入人类引导；可设计更精细的缓冲区构造（如处理成功轨迹里的重试与恢复片段）。

## 核对中发现的细节

- 摘要称「至少 3 倍于基础策略」；该倍数只在两个真机任务上有明确数字（67% 对 20%、73.3% 对 20%）。仿真任务基础成功率 1% 到 18%，终值 Stack_Three 约 50%，引言所说「从低于 10% 提升到接近 100%」在正文数值（Tool_Hang 约 95%，Threading 图中约 90%）下大体成立但不适用于 Stack_Three 与 Three_Piece_Assembly。
- 真机每个检查点仅 30 或 15 次评估，论文未给出真机的随机种子数或置信区间（仿真为 3 个种子）。
- 负缓冲区设置在文内有出入：正文称仅含失败回合终止状态（第 4.2 节、附录 A.3 仿真容量 128、仅终止帧）；真机附录 A.2 写负缓冲区 320、取失败回合最后 16 帧，正缓冲区 4096，κ+ = 10、κ− = 1、λ = 0.1。
- 基线集合：仿真 4 个（稀疏、RoboMeter、TOPReward、GCR），真机只有 2 个（稀疏、RoboMeter）；所有主结果都叠在 DICE-RL 上，未在其他 RL 算法上给出实验（论文称兼容 PPO、TD3、SAC 为方法抽象的论断，第 4.2 节）。

## 对 wiki 的映射

- [paper-hire-hindsight-reward-editing](../../wiki/entities/paper-hire-hindsight-reward-editing.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
