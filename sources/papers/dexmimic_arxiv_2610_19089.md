# DexMimic: A Generalist Dexterous Hand-Object Tracking Policy via Value-Guided Reset Allocation（arXiv:2610.19089）

> 来源归档（ingest）

- **标题：** DexMimic: A Generalist Dexterous Hand-Object Tracking Policy via Value-Guided Reset Allocation
- **arXiv：** <https://arxiv.org/abs/2610.19089>（HTML v2：<https://arxiv.org/html/2610.19089v2>），cs.RO / cs.AI；v1 2026-10-13，v2 2026-10-15（仅改作者单位）；13 页含 4 页补充材料
- **代码：** 项目页与论文均未列（2026-10-17 核查）
- **项目页：** <https://zhexiluo.github.io/DexMimic/>（[sources/sites/dexmimic-project.md](../sites/dexmimic-project.md)）
- **作者：** Zhexi Luo, Zhixuan Xu, Rui Yang, Zeyu Xiao, Yiwen Hou, Lin Shao；新加坡国立大学（NUS）与 RoboScience
- **入库日期：** 2026-10-17
- **一句话说明：** 跨大量手-物交互轨迹联合训练一个通用跟踪策略，用价值函数在线调整重置分布，把 rollout 分给难阶段和高期望回报的起始状态。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 无代码、数据集、权重链接；无 "coming soon" 标注。论文 HTML 亦无代码发布声明。
- **结论：** **未列开源链接**（截至 2026-10-17 未发现代码），不能按已开源处理。

## 摘录 1：方法（论文 HTML v2）

- 算法为标准 PPO，actor 与 critic 共享 MLP 主干。
- 观测：手与物体状态历史（偏移 {0,-1,-2,-4,-8}）、未来参考目标（偏移 {0,1,2,4,8}）、交互特征、运动学基础动作、剩余参考步数、128 维物体几何编码。
- 动作：腕平移、腕旋转、手指关节目标的残差，叠加在重定向手参考的运动学基础动作上。
- 奖励：任务无关的跟踪奖励，五组等权：手姿态、物体姿态、速度、参考引导接触、机械功率。
- 阶段难度：参考帧每 15 帧一个 bin，难度向前传播 5 个 bin、衰减 0.8，经 EMA 平滑后用于采样阶段。
- 重置分配：初始化分布分解为阶段项与状态项；每阶段 3 个候选槽，按 critic 分数比例采样，0.1 概率用固定参考锚点；候选存活比例须大于 0.1 才入缓冲，存储分数每次更新衰减 1e-5。

## 摘录 2：结果

- 基准：OakInk-V2 与 HOT3D，共 1,944 任务、140 物体（1,302 单手 + 642 双手轨迹）；零样本 L1（相似轨迹、见过物体）、L2（未见轨迹）、L3（未见物体）；联合训练子集 20/50/200/500 条。
- 50 条双手任务：DexMimic 成功率 88.40%、每轨迹训练 0.12 h、物体旋转误差 0.07 rad、平移误差 3.76 mm；ManipTrans 56.64%、1.86 h；SPIDER 39.68%、0.78 h（时间低 15.5x 与 6.5x）。
- 零样本（20 到 500 条）：L1 8.00% 到 68.50%，L2 10.63% 到 55.89%，L3 2.00% 到 51.65%。
- 微调：在 20 条未见轨迹上达到 80% 训练成功率，比从头训练快 5.3x（项目页：0.24 h 对 1.27 h）。
- 重置消融（500 条）：RSI 66.51% / 0.150 h / FastFail 96.34%；Uniform 82.59% / 18.60 min / 35.65%；DexMimic 84.73% / 5.76 min / 41.70%；最差轨迹成功数 0/50 提升到 38/50；相对 Uniform 时间减 69.0%。
- 其他手（20 任务各 1,000 rollout 成功数）：Wuji 872、Inspire 861、Shadow 832、Allegro 759。失败分析：OakInk 子集 826/1,017 成功（81.219%），主要失败为物体跟踪。
- 真机：7-DoF xArm + 20-DoF Wuji 右手 + RealSense D435i，8 个 3D 打印 OakInk 物体；开环回放仿真成功 rollout，物体位姿用 FoundationPose++，逆运动学用 PyRoki；任务含倒、舀、倾、摇；**论文未报告量化真机结果**。
- 局限：受限于 HOI 数据集规模；对含噪重建的手-物交互的鲁棒性留作后续。

## 建议 wiki 动作

- 新建 `wiki/entities/paper-dexmimic.md`。
