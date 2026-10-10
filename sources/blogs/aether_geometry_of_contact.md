# The Geometry of Contact: Learning Object Manipulation from Scratch（Aether AI 博客）

> 来源归档（blog / Aether AI 官方 Field notes #06）

- **标题：** The Geometry of Contact: Learning Object Manipulation from Scratch
- **类型：** blog（论文解读文，配套 arXiv 论文）
- **作者 / 组织：** 页面署名「Paper Shen · Chuck · Feng · Huang」/ Aether AI 博客（aetherlabs.ai）
- **原始链接：** <https://aetherlabs.ai/articles/the-geometry-of-contact.html>
- **博客索引：** <https://aetherlabs.ai/blog.html>（编号 06 · Manipulation；标签 Reinforcement Learning · Manipulation）
- **发表日期：** 2026-07-16
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓取静态 HTML 后抽取正文与结果表
- **一句话说明：** 解释为什么对比强化学习（CRL）在接触丰富的操作上失灵——接触让可达性几何「在接触处折出一道缝」——并提出 **Interaction-Weighted Resampling（IWR）**：只改正样本未来状态的采样分布，在接触附近加权；真实 UR 臂桌上冰球（air hockey）从 CRL 的 **5/20 → 12/20**。

## 开源 / 项目页核查（步骤 2.5，截至 2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 | arXiv:2606.11525（见 [论文归档](../papers/iwr_contrastive_interaction_arxiv_2606_11525.md)） |
| 项目页 | <https://iwr-arxiv.github.io/>（可访问；标注 **CoRL 2026**） |
| 代码 | **未列出**：博客、项目页、arXiv 摘要均无 GitHub 链接 |
| 硬件 / 数据 | 未公开 |

## 核心摘录（归纳，非全文）

### 背景：CRL 的可达性几何

- 目标条件 RL 关心的是折扣未来占用密度 \(\rho^\pi(g\mid s,a)\)；CRL 用状态-动作编码器 \(\phi(s,a)\) 和目标编码器 \(\psi(g)\) 的内积逼近 \(\log\rho^\pi(g\mid s,a)-\log\bar\rho_B(g)\)，用 InfoNCE 训练（同轨迹后续状态为正样本）。
- 作者把它读成「可达性地形」：策略沿能量爬向目标。地形必须忠实于真实动力学。

### 为什么接触会破坏地图

- 运动 / 到达：动作小变化 → 结果小变化，表示平滑；t-SNE 上 \(\phi(s,a)\) 轨迹连续。
- 操作：夹爪接触物体前动作对物体无效，接触后一步之内物体位姿与速度进入可控范围 → 可达性在接触处不连续；air hockey 的 t-SNE 轨迹「长时间漂移 + 击球瞬间跳变」。
- 形式化：把操作建模为 **分解式、分段光滑马尔可夫过程**。局部（高斯插值）分析下：运动时 \(\psi_{t+1}\approx A_0\psi_t\)（线性），接触时 \(\psi_{t+1}\approx A_1\psi_t+b_t\)（仿射，偏置项是模式切换的标志）。
- 误差传播：接触处的局部误差 \(e\) 之后被被动动力学放大，\(\sup|\hat E_k-E_k|\propto\|A_0^k\|\|e\|+\tfrac12\|A_0^k\|^2\|e\|^2\)。「决定一局操作的那一刻，正是平滑模型拟合最差、且误差会传播的那一刻。」

### IWR

- 标准 CRL 沿轨迹均匀采正样本未来；被动帧占多数，接触帧几乎采不到。
- IWR 只改采样分布：\(w_k=\epsilon+\exp(-\|d_{t+k}-c\|_2/2\sigma^2)\)，\(d\) 是粗略接近度信号（如夹爪-物体距离），\(c\) 是接触阈值；\(\epsilon\) 保底保留普通转移，\(\sigma\) 控制聚集带宽。
- 不需要人工接触标签或特权检测器；不加网络、不加奖励整形、不加辅助损失；InfoNCE critic 与 actor 更新不变。

### 结果表（自报，成功率；真机列为 20 次进球数）

| 任务 | PPO | SAC | SAC+HER | SAC+HINT | CRL | CRTR | IWR | 相对最佳对比基线 |
|------|-----|-----|---------|----------|-----|------|-----|------------------|
| Air Hockey（仿真） | 0.617 | 0.145 | 0.398 | 0.422 | 0.695 | 0.727 | **0.742** | +2.1% |
| Air Hockey（real-transfer） | 0.160 | 0.215 | 0.129 | 0.125 | 0.477 | 0.465 | **0.500** | +4.8% |
| Air Hockey（真机） | 0/20 | 0/20 | 0/20 | 0/20 | 5/20 | 2/20 | **12/20** | +140% |
| Box2D（center） | 0.086 | 0.058 | 0.088 | 0.088 | 0.278 | 0.274 | **0.288** | +3.6% |
| Box2D（goal） | 0.089 | 0.046 | 0.086 | 0.064 | 0.450 | 0.558 | **0.709** | +27.1% |
| Box2D（hard） | 0.060 | 0.042 | 0.064 | 0.076 | 0.317 | 0.365 | **0.565** | +54.8% |
| Box2D（hard velocity） | 0.148 | 0.149 | 0.152 | 0.139 | 0.387 | 0.377 | **0.436** | +12.7% |
| Box2D（maze） | 0.033 | 0.012 | 0.031 | 0.035 | 0.217 | 0.206 | **0.223** | +2.8% |
| Meta-World（peg insert） | 0 | 0 | 0 | 0 | 0.430 | 0.367 | **0.438** | +1.9% |
| Meta-World（pick place） | 0 | 0 | 0.004 | 0 | 0.266 | 0.305 | **0.570** | +86.9% |
| Meta-World（push） | 0 | 0 | 0.004 | 0 | 0.699 | **0.750** | 0.730 | —（低于 CRTR） |
| Meta-World（sweep into） | 0 | 0.004 | 0.020 | 0.004 | 0.805 | 0.910 | **0.926** | +1.8% |

- 平均：相对最佳对比基线 **+19.8%**（博客写「across the manipulation suite」，摘要写「in simulation」）。
- 定性：Box2D（hard）CRTR 34 ticks 控制 vs IWR 90 ticks；Meta-World pick-place 示例 CRTR 6/20 vs IWR 13/20。

### 真机

- UR 机械臂持推板 + 顶置相机跟踪冰球；仿真训练后 **零样本** 迁移，相机图像用单应性（homography）对齐到仿真坐标。
- 无模型基线 0/20；CRL 5/20、CRTR 2/20；IWR 12/20（25% → 60%）。作者称是「首个仅凭目标设定与自身交互经验训练的真实 air hockey 机器人」（自报的「首个」）。

## 可信度边界

- 真机 20 次试验、单一任务，没有置信区间；「+140%」是 5→12 的相对提升。
- 博客未写清仿真种子数与训练步数；Meta-World push 一项 IWR 不是最好。
- 代码未公开，无法复核实现细节（如 \(d\) 的具体定义在每个环境中如何取）。

**对 wiki 的映射：** [paper-geometry-of-contact](../../wiki/entities/paper-geometry-of-contact.md)
