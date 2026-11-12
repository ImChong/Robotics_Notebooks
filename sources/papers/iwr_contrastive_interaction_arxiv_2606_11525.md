# Learning Object Manipulation from Scratch via Contrastive Interaction（arXiv:2606.11525）

> 来源归档（ingest）

- **标题：** Learning Object Manipulation from Scratch via Contrastive Interaction
- **短名：** IWR（Interaction-Weighted Resampling）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.11525>（v1 2026-06-10，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.11525>
- **作者：** Tongle Shen、Caleb Chuck、Fan Feng、Biwei Huang（Chuck、Feng、Huang 标 Equal advising）
- **机构：** UC San Diego（Shen、Feng、Huang）、UT Austin（Chuck）（首页署名）
- **会议：** CoRL 2026（项目页标注；PDF 首页未见会议名）
- **项目页：** <https://iwr-arxiv.github.io/>
- **代码：** 论文与项目页未给出仓库地址（截至 2026-10-10 核查）
- **入库日期：** 2026-10-10；2026-11-12 通读后改写
- **博客归档：** [aether_geometry_of_contact.md](../blogs/aether_geometry_of_contact.md)
- **一句话说明：** 把操作动力学写成分段光滑（切换仿射）模型，指出交互引起的模式切换让标准 CRL 的内积能量难以表示，并提出只改正样本未来采样权重的 IWR；仿真平均 +19.8%，真机 UR5e air hockey 5/20 → 12/20。
- **核对范围：** 通读 arXiv v1 PDF 正文第 1-7 节（摘要、引言、相关工作、背景、分析、方法、实验、结论与局限）与附录 A（命题 1 的引理与证明）、B（扩展相关工作仅略读）、C、D（训练设置与全部超参表）、E（环境设置）、F（真机设置）、G、H（Table A12 方差表、Table A13/A14 消融）；图 12 训练曲线与图 A4-A10 的轨迹图为图像，pdftotext 无法读出，只据图注与正文使用，未逐张看图；参考文献表只扫了条目，未逐篇核对。
- **通读：** 已完成

## 开源状态（步骤 2.5，2026-10-10）

- **结论：未列代码**。论文正文与项目页只有方法、分析与结果表，无 GitHub 链接。

## 核心摘录（面向 wiki 编译，均出自论文，括号为章节）

1. **问题与主张（摘要、第 1 节）：** CRL 在运动类任务上成功，但在交互丰富的操作上吃力；作者认为原因之一是物体中心的交互（接触、抓取）引起底层动力学模式变化。
2. **背景（第 3 节）：** 折扣未来占用 ρ^π(g|s,a)；CRL 估计其与回放边缘分布的密度比 E*（式 1），用内积 φ(s,a)ᵀψ(g) 参数化（式 2），InfoNCE 加 LogSumExp 正则训练（式 3）。
3. **分析（第 4 节）：** 沿用 Eysenbach 等 2024 对 CRL 的线性高斯视角（式 4）；操作中目标因子 y 只在交互指示 ω=1 时受动作影响（式 5），表示层写成切换仿射模型 ψ_{t+1}=A_ω ψ_t + ω b_t + ε（式 6）。两个挑战：采样（交互瞬间稀疏且潜在，均匀采样被交互前后的平滑转移占据）与误差传播（命题 1，式 7）。
4. **命题 1 的精确口径（第 4.3 节、附录 A.5）：** 正文写 sup|Ê_k−E_k| ∝ ‖A0^k‖‖e‖ + ½‖A0^k‖²‖e‖²；附录推导的是上界 ≤，且依赖假设 3（终点落在正确均值的局部二次区域 ‖ψ(y_f)−μ_k‖≤1）、单次交互（one-reset）、局部仿射假设 2。
5. **方法（第 5 节，式 8）：** 对锚点 (s,a) 在同轨迹的候选未来 g_k=s_{t+k}，用 d=‖u_p−y_p‖₂（执行器与目标位置距离）算权重 w_k=ε+exp(−‖d−c‖²/(2σ²))，对**正样本未来**重采样；CRL 损失、critic、actor 不变；负样本仍来自回放边缘分布。c 是「宽松」的交互阈值（附录 D.4：至少取真实接触距离的 1.5 倍）。
6. **实验设置（第 6.1 节、附录 D、E）：** Box2D 5 个任务（center、goal、hard、hard-velocity、maze）、Meta-World 4 个任务（push、pick-place、peg-insert-side、sweep-into）、Air Hockey（速度观测；real-transfer 用历史观测加噪声）；基线 PPO、SAC、SAC+HER、SAC+HINT、CRL（正文写 SGCRL）、CRTR（R=8，等价于 c→∞ 的均匀采样）；5 个种子；所有方法 1000 万环境步、256 个训练环境、批大小 64、MLP(512,512)、表示维度 256（Table A2）。
7. **报告口径（附录 D.6）：** 每 20 万步记录 5 个种子的均值，Table 1 报告其中**最好**的分数。
8. **主结果（Table 1、Table A12）：** 平均相对提升 +19.8%；Box2D hard 0.565±0.084（CRTR 0.365±0.069）；Meta-World pick-place 0.570±0.196（CRTR 0.305±0.163）；push 上 IWR 0.730 低于 CRTR 0.750；多个任务提升仅 1.8%-4.8%，且标准差重叠；PPO/SAC 系在 Meta-World 与真机几乎为 0。
9. **真机（第 6.2 节、附录 F）：** UR5e + Robotiq 夹爪握拍，倾斜 air hockey 台，120Hz 顶置相机 blob 检测加单应性，20Hz 控制环；零样本 sim-to-real（只在仿真训练，域随机化：拍密度、冰球阻尼、重力各 ±25%），评测 4×5 共 20 个目标；Table 1：IWR 12/20、CRL 5/20、CRTR 2/20、其余 0/20。作者归因：IWR 更稳定地击球且力度更向上，其他方法多半只击球一次。
10. **消融（附录 H.2，Table A13/A14）：** 仅 6 个任务；c 取 0.5×/1.5×、2σ² 取 0.5×/2× 的多数设置优于主表所选值（如 pick-place 2σ²=0.5× 为 0.844），push 在 c=1.0× 与 2σ²=1.0× 下低于 CRTR；作者称「grid search 结果甚至好于主文所选结果」。
11. **作者自述局限（第 7 节）：** IWR 仍用单峰模型表示多峰操作动力学（只改采样）；CRL 只是一种无监督 RL，表示能力有限；假设因子化状态与启发式交互指示，需与端到端感知结合；CRL 样本效率代价高，真机失败来自 sim-to-real 差距。
12. **t-SNE（附录 G，图 A11）：** SGCRL 的 φ(s,a) 在 Meta-World reach 上轨迹平滑，在 Air Hockey 上于交互处跳变（红圈）。

**未在论文中找到的说法（先前的博客解读里出现过）：** Box2D hard 上「控制 34 ticks vs 90 ticks」、pick-place「6/20 vs 13/20」等定性数字，论文 PDF 中没有出现。

**对 wiki 的映射：** [paper-geometry-of-contact](../../wiki/entities/paper-geometry-of-contact.md)
