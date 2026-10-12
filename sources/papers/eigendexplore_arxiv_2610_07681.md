# EigenDEXplore 论文来源归档

## 元数据

- **题目：** EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors
- **作者：** Harsh Gupta, Tyler Ga Wei Lum（共同一作）, Changhao Wang, Chuer Pan, C. Karen Liu, Jeannette Bohg, Shuran Song
- **单位：** Stanford University
- **预印本：** https://arxiv.org/abs/2610.07681 （v1，2026-10-06，cs.RO，交叉 cs.AI；15 页、12 图）
- **HTML：** https://arxiv.org/html/2610.07681v1
- **项目页：** https://eigendexplore.github.io/ （见 [eigendexplore-project.md](../sites/eigendexplore-project.md)）
- **代码：** https://github.com/hgupt3/eigendexplore （MIT，见 [eigendexplore.md](../repos/eigendexplore.md)）

## 论文要解决的问题

灵巧操作的有效行为需要多关节协同，而 RL 与采样式轨迹优化通常对每个关节独立加噪，难以发现协调动作。已有工作用人手数据学低维协同动作空间来缩小抓取搜索，但会牺牲通用操作所需的表达力；混合关节与协同空间的方法又增加维度与冗余。论文系统比较了不同操作场景、动作维度、探索策略与人类数据来源，结论是：人类运动先验用来“结构化探索”优于“改动作表示”。

## 方法与训练信息

1. **人类先验：** 取自 Lightwheel 的 EgoSuite-Open100K 第一视角视频中的手部姿态（另比较 GRAB、ARCTIC），双手镜像到统一规范系；用改编自 ManipTrans 的运动学重定向映射到各机器人手（手专属对齐、缩放与指对应）。
2. **PCA：** 对重定向后的关节角做 PCA，保留累计方差达 90%（默认）的前 k 个分量；基矩阵 E∈R^{k×d} 位于归一化到 [−1,1] 的关节坐标（Sharpa 22 关节 k=9，XHand 12 关节 k=6）。
3. **探索噪声：** q⋆ = a_q + σ_q⊙η_q + Eᵀ(σ_e⊙η_e)，η_q~N(0,I_d)，η_e~N(0,I_k)；初始 σ_e,i ∝ √λ_i（与人手姿态散布一致），并重缩放使总初始方差与纯关节空间噪声一致、两项各占一半；等效协方差 Σ = diag(σ_q²) + Eᵀ diag(σ_e²) E。动作空间保持为绝对归一化关节位置目标（维度 d）不变。
4. **RL（PPO）：** 策略输出关节空间均值 μ_q；PPO 同时学习 θ、σ_q、σ_e，E 固定；对数概率与熵按上述 Σ 计算；部署时确定性执行 μ_q，E 与探索参数全部丢弃。
5. **轨迹优化（SPIDER）：** 每轮采样 N 条候选序列（N 未报告），用同一噪声扰动当前目标，仿真后保留代价最低的 10%，按权重更新目标；σ_q、σ_e 固定并遵循 SPIDER 的噪声调度。
6. **对比基线动作空间：** Joint-Space（d）、Eigen-Space（k）、Eigen-Residual（d+k）；附录含 delta 动作变体。
7. **涉及的手：** Sharpa（22 DoF）、Wuji（20）、Allegro（16）、XHand（12）、Schunk（9）、Inspire（6）。
8. 未报告：绝对 σ 值、PPO 超参、学习率、候选数 N。

## 论文报告的结果

- **SimToolReal（Sharpa 手臂工具操作）：** 9B 帧、容差 6 cm/50°、5 个种子，平均连续成功 EigenDEXplore 10.6 对 Joint-Space 7.9。60B 帧并逐步收紧到 1 cm/8° 的课程（每方法取一次中位运行）：EigenDEXplore 14.1、Eigen-Residual 12.0、Joint-Space 4.9，Eigen-Space 早期占优后衰减。
- **DeXtreme（Allegro 立方体重定向）：** 2B 帧 + 自动域随机化、5 种子，12.3 对 8.8；Eigen-Space 被限制在 16 关节中的 10 维子空间，表现差。
- **DexMachina（ARCTIC 双手盒子/笔记本跟踪，约 2B 帧，ADD-AUC）：** 有课程与辅助奖励时与 Joint-Space 接近，Schunk 盒子 0.59→0.84；仅任务奖励时 Allegro 盒子 0.46→0.80，Inspire 几乎无变化。
- **SPIDER（OakInk2 十任务，10 种子，六只手）：** 相对 Joint-Space 的平均重定向代价降幅 Wuji 31%、Sharpa 25%、Allegro 22%、XHand 14%、Schunk 4%、Inspire 0.4%，平均 16%。
- **数据来源对比（SimToolReal 9B 帧）：** EgoSuite 10.6、ARCTIC 9.8、GRAB 9.4、Joint-Space 7.9。
- **保留方差消融（Sharpa，SPIDER）：** 仅 50% 方差的纯本征噪声比 Joint-Space 差（0.126 对 0.089）；与关节噪声叠加时 90% 最优（0.066，约低 25%）。随机正交方向（同特征值）比 Joint-Space 还差，说明增益来自人类派生方向。
- **真机：** Sharpa 手臂在锤击、擦拭、舀取、刷四类工具任务上（FoundationPose 物体轨迹，两个策略均训 60B 帧即原 120B 预算的一半，每个物体-轨迹组合 10 次）平均任务进度 82% 对 33%；Allegro 立方体重定向 40 次试验平均连续成功 8.45→11.90（+41%），最长 32→41。项目页另给出 71% 对 42% 的汇总，与论文 82%/33% 口径不同，未能核对差异来源。

## 复现与解释边界

- 作者列出局限：线性本征基固定，数据里没有的模式或非线性协同仍靠关节噪声；依赖运动学重定向，对形态差异大的手可能迁移差；低自由度手收益小；噪声逐步独立采样，时间相关探索留作未来工作。
- 论文引言声称发布基准代码、即插即用实现与六只手的预训练 PCA 模型；仓库已含对应资产（见 repos 归档），但笔者仅浏览了目录，未运行。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-eigendexplore.md
- 相关：wiki/methods/spider-physics-informed-dexterous-retargeting.md、wiki/entities/paper-dexmachina.md

## 一手来源

- <https://arxiv.org/abs/2610.07681>
- <https://arxiv.org/html/2610.07681v1>
- <https://eigendexplore.github.io/>
- <https://github.com/hgupt3/eigendexplore>
