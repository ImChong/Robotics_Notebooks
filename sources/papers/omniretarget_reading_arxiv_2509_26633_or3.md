# OmniRetarget 论文通读（交互保留重定向视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-hrl-stack-03-omniretarget.md`）

- **标题：** OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction
- **作者：** Lujie Yang*, Xiaoyu Huang*, Zhen Wu*, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi（Amazon FAR / MIT / UC Berkeley / Stanford / CMU）
- **类型：** paper / humanoid motion retargeting / data generation
- **原文：** <https://arxiv.org/abs/2509.26633>（2026-11 通过 arXiv PDF v3，15 Jun 2026 版获取全文）
- **一句话说明：** 用 interaction mesh 的 Laplacian 形变能 + 硬运动学约束逐帧重定向人类演示，并通过重解优化做地形 / 物体 / 位姿增广；下游 RL 仅 5 reward + 4 DR 即可 G1 零样本上真机。
- **核对范围：** 通读摘要、§I–§VI 全文（方法 §III、RL §IV、实验 §V 含 Table II）及附录 A–E（数据源预处理、各基线 PHC/GMR/VideoMimic/IMMA 的数学形式、增广细节、Sequential SOCP 细节、下游 RL 失败直方图）；参考文献列表仅浏览，未逐条查阅被引论文，也未阅读 holosoma 源码本身（另有归档）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§I）：** 既有重定向（PHC、GMR、VideoMimic）多为无约束或软惩罚的关键点匹配，出现脚滑、穿透，且不显式保留人-物-环境交互。
- **Interaction mesh（§III-A）：** 对关键关节 + 物体/环境采样点做 Delaunay 四面体化；物体/环境表面采样比关节更密；Laplacian 坐标 L(p_i)=p_i−Σw_ij p_j，实验用均匀权重 1/|N(i)|；目标为最小化源/目标 mesh 的 Laplacian 差（式 2）。
- **逐帧优化（式 3a–3e）：** 变量为浮动基位姿（四元数+平移）与关节角；目标 = Laplacian 误差 + 时间平滑项 ‖q_t−q_{t−1}‖_Q；约束 = SDF 非穿透、关节界、速度界、stance 脚位置固定（源运动 xy 速度低于阈值判为 stance，正文写 1 cm/s）。
- **求解（§III-A、附录 D）：** 定制 SQP 式 Sequential SOCP，每帧至多 10 次迭代，上一帧最优解热启动，信赖域半径 ε=0.2；用 Drake 的自动微分处理 S³ 上四元数导数。换 embodiment（G1/H1/Booster T1）只需改关键点对应与碰撞模型。
- **增广（§III-B、附录 C）：** 固定源 mesh、改目标物体位姿 / 尺寸 / 地形高度后重解；物体初始位姿扰动随时间指数衰减混入原轨迹（式 14）；mesh 在物体局部系构建以保证物体旋转时 Laplacian 不变（Fig. 8）；增加下半身锚定 nominal 轨迹的代价（式 4–5）防止退化为刚体变换；抬高地形时在地面采样网格点进 mesh。
- **RL（§IV）：** 观测 = 参考关节位/速、骨盆位姿误差、骨盆线/角速度、关节位/速、上一步动作（敏捷动作屏蔽骨盆线位置误差与速度）；5 reward = body tracking、object tracking、action rate、soft joint limit、self-collision（>1 N 二值）；沿用 BeyondMimic 权重不调参；物体偏离 1.0 m 或 45° 时终止；机器人 DR 仅 4 项（躯干 COM、关节默认位 ±0.01 rad、随机推、观测噪声），物体质量 0.1–2 kg / COM ±0.08 m / 惯量 50–150% / 形状 ±10%。箱子搬运共享一个多任务策略，攀台每条参考一个策略。
- **真机（§V-A）：** 4.6 kg 椅子搬运→踏台→跳下翻滚的 30 s 序列；0.9 m（约机身 70%）攀台；斜坡爬行；墙翻约 0.5 s、峰值角速度 15 rad/s、线速度 3.5 m/s，真机 5/5 成功（放宽终止阈值至 0.5 m 并去掉脚关节朝向奖励）。
- **增广效果（§V-A.b）：** 全增广数据训练并评估成功率 79.1%，仅评估 nominal 为 82.2%；仅用 DR 扰动物体/位姿表现差。
- **基准（§V-B，Table II）：** 数据 2.78 h OMOMO 搬箱 + 1 h 自采 MoCap + 4.6 h LAFAN1（共约 8.4 h，将开源）。OMOMO 物体交互：PHC 成功率 71.28%、GMR 50.83%、VideoMimic 3.85%、OmniRetarget 82.20%；地形交互：PHC 52.63%、GMR 78.94%、VideoMimic 51.75%、OmniRetarget 94.73%；LAFAN1 上与 Unitree 官方重定向均 100%，但 OmniRetarget 穿透更小。OmniRetarget 脚滑为 0；因约束线性化偶有小穿透，可由 RL 修复；GMR 物体任务接触保留得分最高（0.99）但源于手部被缩放进物体内部的穿透。下游评测用 39 个挑战性动作，成功判据为训练终止准则。
- **失败分析（附录 E）：** 攀台地形下 PHC/VideoMimic 近半数动作完全失败，GMR 四个、OmniRetarget 仅一个；作者认为剩余失败来自过于简单的 RL（无 curriculum）。
- **局限与展望（§VI）：** 逐帧优化大体有效，未来可联合优化整条轨迹以更稳健处理视频等噪声源，或学习视觉策略。
- **相关工作定位（§II、Table I/III）：** IMMA 同为 interaction mesh，但未开源、多阶段（先 mesh 再 IK）、不考虑关节/速度界与环境/物体交互。

## 对 wiki 的映射

- [`wiki/entities/paper-hrl-stack-03-omniretarget.md`](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2509.26633>
