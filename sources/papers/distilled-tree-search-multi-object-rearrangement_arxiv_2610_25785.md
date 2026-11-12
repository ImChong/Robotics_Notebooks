# Distilled Tree Search Rearrangement（arXiv:2610.25785）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-distilled-tree-search-multi-object-rearrangement.md`）

- **标题：** Long-Horizon Multi-Object Rearrangement via Distilled Tree Search over Continuous Actions
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.25785>（v1，2026-10-19，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2610.25785>
- **作者：** Kallol Saha, Yining Hong, Amber Li, Yejin Choi, Maxim Likhachev, David Held（首页：CMU Robotics Institute 与 Stanford University）
- **arXiv 评论栏：** Accepted at CoRL 2026 (Oral). Project page: <https://distilled-search.github.io>
- **入库日期：** 2026-10-26（通读升级：2026-11-12）
- **一句话说明：** 用 progressive-widening MCTS 在 GPU 并行仿真里自生成长程重排计划，周期性把搜索树蒸馏进点云条件的策略-价值模型，测试时无需搜索与仿真。
- **核对范围：** 下载 arXiv v1 PDF（34 页，pdftotext 全文）并逐段通读正文 §1–§7（含 Fig.1–8、Table 1–2）、附录 A（基线细节）、B（算法 / 超参 / 网络结构 / sim2real 增强 / 预训练数据 / 训练指标，Table 3–5）、C（任务设定）、D（补充定量分析，Table 6–7、Fig.13–15）、F.1–F.4 的推导；附录 E 只看图注与文字说明，树图节点内容仅浏览，参考文献列表仅浏览；Fig.5 柱状图数值是从渲染页图上目视读取，精度有限。未看项目页视频，未运行任何代码，论文未给出代码仓库链接。
- **通读：** 已完成

## 开源状态

- 论文与 arXiv 评论栏只给出项目页 <https://distilled-search.github.io>，未给出代码仓库链接（本次通读未见 GitHub 地址）。

## 核心摘录（均出自原文）

- **问题设定（§3）：** 状态是每个物体的 6D 位姿 s ∈ R^{No×6}；目标检查 G:S→{0,1}；测试时只有带物体分割标签的点云；动作是目标末端位姿，经运动规划（cuRobo）转轨迹；训练时才可用仿真器 f_sim（ManiSkill3）。只有子任务（Pick / Place）数据集 D，没有任何长程任务演示。仅在仿真里定义进度 κ:S→[0,1]，进度奖励 r = κ(s') − κ(s)。
- **MCTS（§4.1）：** 搜索图是以根为起点的树，顶点存仿真状态、奖励、访问数 N、价值 Q。扩展时由策略 π_θ(a|o) 采样连续动作，经运动规划与仿真得到子顶点。progressive widening：仅当 |children(v)| < k·N(v)^α 才扩展（附录 Table 4：k=2.0、α=0.3；附录 A.4 给出 UCB 常数 C=1.5）。
- **价值引导 rollout（§4.2、附录 B.1）：** rollout 与树下降是同一过程，每步对当前顶点批量采样 B 个动作（附录 Table 4：B=32），按 UCB 选子节点；未访问子节点的先验在第一轮是边上的进度奖励，之后用 Q_ϕ(o,a) 初始化。回报 R = κ(s_T) − κ(s_0)，回传并做运行平均。Q_ϕ 预测「该动作导致任务成功的概率」。终止：达成目标、视野耗尽（K=物体数，3–7 个）或无可行动作。
- **GPU 并行（§4.2）：** 多条下降并发，扩展批量化；动作经 GPU 并行运动规划（cuRobo）与 ManiSkill3 并行仿真，新状态批量用价值函数打分。
- **迭代蒸馏（§4.3、式 1）：** 价值目标 y_i 是顶点 v_i 的已访问后代叶子里成功叶子的比例，加权 BCE，权重 w_i = log(1+|leaves(v_i)|)。策略用每一条产生可行新顶点的边训练（不只成功计划），损失沿用 3DFA 的 flow-matching；子任务数据集 D 只用于预训练，蒸馏时不再混入。蒸馏后重置搜索树，用新 π_θ / Q_ϕ 继续（Algorithm 1）。附录 B.2：蒸馏在训练集每完整规划一轮后进行，共 2 轮，之后纯用蒸馏网络。
- **网络（附录 B.3–B.4）：** 策略改自 3D FlowMatch Actor，视觉骨干换成 PointNet++ 风格点云编码器（4096 点 xyz，EdgeConv，k=16 近邻），两个独立训练的 rectified-flow 模型（抓取 + 以抓取为条件的放置）；假设已有目标物体分割掩码。价值函数看全场景点云，动作编成抓取 / 放置两个位姿 token，4 层 8 头注意力，sigmoid 输出成功概率。
- **任务（§5.1、附录 C）：** 仿真 Shelf Packing（κ=装入货架的体积占比）、Object Stacking（κ=塔高/物体数）、Constrained Packing（用于与测试时规划器对比）；真机 Shelf Packing 与 Table Bussing（Franka Panda + Robotiq）。每任务共享一个策略-价值模型，50 个随机化训练实例，15 个留出实例；物体网格来自 ACRONYM。单张 RTX 4090；货架任务搜索生成 263,650 个顶点（其中 11,015 个被扩展），耗时 47.7 小时，每棵树 95 MB（§5.1）。仿真预算：货架 270,000、堆叠 27,000 次仿真器调用（Table 4）。
- **主结果（Table 1，留出集，32 个环境种子平均，95% 置信区间）：** Shelf Packing：DISCO 奖励 90.9±6.7、成功率 80.0±9.5%；Beam Search 6.7%、MPPI 13.3%、RL-PD 20.0%、DISCO w/o Policy 6.7%、DISCO w/o PW 0.0%。Object Stacking：DISCO 94.7±2.0 / 66.7±5.5%；Beam Search 33.3%、MPPI 13.3%、RL-PD 20.0%、w/o Policy 33.3%、w/o PW 26.7%。训练期找到的成功计划数：Shelf Packing DISCO 463.16 对 RL-PD 138.52、w/o PW 133.14。
- **训练集覆盖（§5.2 RQ1）：** DISCO 是唯一解决 100% 货架训练任务与 98% 堆叠训练任务的方法；Beam Search（只按进度奖励）训练集只解 28% 货架任务，留出集 6.7%。
- **Constrained Packing（§5.2 RQ4、Fig.5、附录 A.5、C.3）：** DISCO 直接 rollout 成功率 85.9%，文中称与依赖关系抽象的测试时规划器相当（Fig.5 目视：SPOT 约 84.7%、Points2Plans 约 80.7%、eRD-T 约 23.7%、P-RD 约 5.8%）。测试集为 Points2Plans 发布的 5 个场景 × 100 个随机种子 = 500 次；成功判据是作者改写的（不复现 Points2Plans 的谓词指标），且改用统一的抓取-放置执行接口。
- **真机（Table 2、Fig.6、附录 C.4）：** 每任务 3 次试验。Shelf Packing：3DFA 奖励 33.3、成功 0/3；DISCO 奖励 88.9、成功 2/3。Table Bussing：3DFA 27.8、0/3；DISCO 83.3、1/3。真机基线只有 3DFA（即 DISCO 蒸馏所用的架构）。
- **迭代蒸馏分析（§5.2 RQ5、Fig.7、附录 D.1–D.2）：** 留出货架成功率，策略单独：P0 13.3% → P1 20.0% → P2 46.7%；固定 P1，把 Q1 换成 Q2 由 40.0% 升到 66.7%。每 25 次搜索迭代找到的目标计划数由 3.36 升到 19.96。Table 6：策略训练目标用「所有可行边」得 52.8±38.9% 对「仅成功计划边」32.3±24.9%；预训练数据量 25%/50%/75%/100% D（444/888/1332/1776 条）后奖励 28.4/32.5/41.0/41.9。
- **MCTS 行为（附录 D.3–D.4）：** 根动作 Q 估计随搜索收敛；访问数与 Q 的 Spearman ρ=+0.15（弱正相关，作者称「早期上升后平台」）。
- **论文自述局限（§7）：** 依赖初始策略能提出可行动作；只评估末端执行器抓放，关节空间 / 灵巧控制是未来工作；对仿真保真度敏感，进度度量错位会偏置搜索（进度奖励是 potential-based）；尚未使用基础模型或 VLA。

## 对 wiki 的映射

- [paper-distilled-tree-search-multi-object-rearrangement](../../wiki/entities/paper-distilled-tree-search-multi-object-rearrangement.md)

## 参考来源

- 策展来源：[【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
- Saha, K., Hong, Y., Li, A., Choi, Y., Likhachev, M., Held, D. (2026). *Long-Horizon Multi-Object Rearrangement via Distilled Tree Search over Continuous Actions*. CoRL 2026. <https://arxiv.org/abs/2610.25785>
