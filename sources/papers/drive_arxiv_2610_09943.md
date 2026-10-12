# DRIVE 论文来源归档

## 元数据

- **题目：** Many Ways to Succeed: Diversity-Driven RL Fine-Tuning for VLA Generalization
- **方法：** DRIVE（Diversity-driven RL fIne-tuning for VLA gEneralization）
- **作者：** Haoru Li, Jinmei Liu, Zhiyong Wang, Xiaoming Li, Zhenhong Sun, Daoyi Dong, Chunlin Chen, Zhi Wang
- **单位（HTML 正文）：** 南京大学；哈尔滨工业大学（深圳）；澳大利亚国立大学；悉尼科技大学
- **预印本：** https://arxiv.org/abs/2610.09943 （v1，2026-10-07，cs.RO，交叉 cs.LG）
- **HTML：** https://arxiv.org/html/2610.09943v1
- **代码与项目页：** arXiv 页面未列项目页、GitHub 或 HuggingFace；HTML 末段（约 10K 字符）未读到代码声明，截至 2026-10-12 未核实到公开代码。

## 论文要解决的问题

RL 微调（RFT）能借助闭环经验提升 VLA，但分布外（OOD）条件下表现仍有限。作者分析发现：RFT 在整体上收窄行为，同时让成功轨迹内部更多样、用更少 rollout 就能成功、比监督微调（SFT）覆盖更多潜在可行解。他们据此假设：成功模式覆盖越广，在分布偏移下越可能有替代策略可用。

## 方法与训练信息

1. **RL 基座：** 带 Flow-SDE 采样的 PPO（clip ε=0.2，γ=0.99，GAE λ=0.95）；多样性项以塑形奖励加入 r = r_env + α·r_div。
2. **分组：** 64 组 × 每组 8 次随机 rollout = 每轮 512 条轨迹；同组共享任务与环境条件，只在组内计算相似度。
3. **轨迹特征与时间对齐：** 每个决策步取 VLM 前缀 token 的均值池化特征，末尾追加终止观测特征；用 Global Alignment Kernel（GAK，非 DTW，对所有单调对齐路径求和）对齐，局部相似度为平移后的余弦映射到 [0,1]，并按各自的自相似度归一化，从而对执行快慢不敏感。
4. **多样性度量：** d_i = 1 − 该轨迹与组内其余轨迹的平均归一化相似度；仅在成功 rollout 内做 min-max 归一化并截断到 [0,1]，失败 rollout 的 d̂=0。
5. **内在奖励：** 势函数 Φ(s)=y·d̂(s)（y 为成功指示），r_div=γΦ(s′)−Φ(s)，沿轨迹求和化为终止块上的 γ^L·y·d̂；附录按 Ng et al. 1999 证明势函数塑形不改变最优策略集合。
6. **系数与设置：** α 随基准与骨干在约 0.02–0.10 间取值（π0 上 LIBERO-Plus Spatial/Object 0.10、Goal/Long 0.05，ManiSkill3 0.10；π0.5 多为 0.10），训练中恒定；actor 学习率约 5e-6、critic 约 1e-4，PPO 更新轮数 3–5，全局 batch 2048 或 5120，Flow-SDE 去噪步 3–5、噪声 σ 0.3–0.5，episode 视野 80–480。算力与 GPU 小时在已读部分未报告。
7. **骨干：** π0 与 π0.5，同基准内从同一 SFT checkpoint 起步。

## 论文报告的结果

- **基准与 OOD 划分：** LIBERO-Plus 四套件（IND 池每套件 250 个任务定义，OOD 为同一十个基础任务上的留出扰动）；ManiSkill3 PutOnPlateInScene25Main-v3（IND 320 episode，OOD 13 种设置共三组：视觉语言 6、语义 4、执行 3）；RoboTwin 2.0 的 Click Bell、Press Stapler（IND 1,000 配置抽 350 评测，OOD 500 个留出配置）。
- **基线：** SFT、Vanilla RFT（仅环境奖励）、Higher Noise（σ=0.7）、KL 正则（β=0.5）、Clip-Higher（上界 0.28）。
- **π0（IND/OOD 平均）：** SFT 36.6/29.2；Vanilla RFT 66.6/48.1；DRIVE 69.3/53.4（OOD +5.3）。分项 OOD 如 RoboTwin 62.1→71.4、ManiSkill3 35.5→40.3、LIBERO-Long 46.1→50.8。
- **π0.5：** SFT 66.6/57.9；Vanilla RFT 85.9/71.1；Higher Noise 85.2/70.1；Clip-Higher 85.3/72.6；KL 87.1/71.4；DRIVE 88.2/73.1（OOD +2.0）。论文称 DRIVE 在 π0 与 π0.5 上均在 12 个基准划分中的 11 个有提升；Clip-Higher 的 OOD 均值（72.6）与 DRIVE 差距仅 0.5。
- **分析：** RFT 过程中全部 rollout 的相似度上升，而成功 rollout 间的相似度下降；成功率 π0 41%→75%、π0.5 52%→77%；Pass@K 与 Coverage@K（以 K=256 的 SFT 成功为参照）显示 RFT 以少得多的 rollout 达到相同覆盖。
- **真机（双臂 AgileX PiPER-X，Click Bell/Press Stapler，每条件 20 次）：** Click Bell OOD 平均 65.0→71.7，Press Stapler 63.3→75.0；两任务平均 OOD 64.1%→73.3%（+9.2），Clean 77.5%→82.5%。
- **消融：** 成功条件化与 α=0.1/0.2 比较仅以图给出，正文无数值。

## 复现与解释边界

- 作者列出的局限：组内两两 GAK 增加计算；依赖 VLM 表征能否刻画有意义的行为差异；未来工作含可学习多样性度量、更长视野、更多本体、在线真机微调。
- π0.5 上相对 Vanilla RFT 的提升只有 +2.0 点，且与 Clip-Higher 接近；真机每格仅 20 次试验。
- 无可核实的官方代码，实现细节（算力、总步数）以论文为准且部分未读到。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-drive-diversity-rl-vla.md
- 方法入口：wiki/methods/ppo.md、wiki/methods/vla.md

## 一手来源

- <https://arxiv.org/abs/2610.09943>
- <https://arxiv.org/html/2610.09943v1>
