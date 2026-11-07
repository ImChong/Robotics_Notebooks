# CMU 16-745 课程站与讲义通读（Spring 2025）

> 来源归档（原文通读 · 支撑 `wiki/entities/cmu-optimal-control-curriculum.md`、`wiki/formalizations/lqr.md`、`wiki/concepts/optimal-control.md`）

- **标题：** CMU 16-745 *Optimal Control and Reinforcement Learning*（Zachary Manchester，Spring 2025）官方课程站与讲义
- **类型：** course
- **原文：**
  - 课程站首页、Lectures、Background 页 <https://optimalcontrol.ri.cmu.edu/>（2026-11-07 抓取 HTML 并转文本）
  - 讲义仓 <https://github.com/Optimal-Control-16-745/lecture-notebooks>（`git clone --depth 1`，提交 `e7c23f6`，2025-04-18；README 自述为 *hand-written lecture notes and corresponding Jupyter notebooks*，许可 CC BY-NC-SA 4.0）
- **入库日期：** 2026-11-07
- **一句话说明：** 无需教材的 24 讲研究生课：动力学与稳定性、数值优化、Pontryagin / LQR / DP、凸 MPC、非线性轨迹优化（DDP、直接配点、SQP）、三维旋转、混杂系统、ILC、随机最优控制与 LQG、Kalman 对偶、凸松弛着陆、驾驶博弈、数据驱动 / 行为克隆。
- **核对范围：** 课程站首页（描述、先修、日程、学习目标、评分、项目要求）、Lectures 页（2025 / 2024 / 2023 三张讲次表）、Background 页全文通读；讲义仓 README 与全部 24 讲的 `Lecture N.pdf` 文字层读过（手写讲义经 OCR，噪声大，第 7–10 讲读得最细，其余讲次读到主题、关键定义与算法步骤，公式细节未逐字核对；第 20 讲为 John Zhang 的 33 页嘉宾讲座幻灯片，文字清晰）。读了 `Lecture 8/lqr-riccati.ipynb` 的代码结构（wiki 已有时序图依据）。**未观看任何视频**，未运行 notebook，未读作业与测验，也未能解析 YouTube 播放列表页逐条标题（页面抓取只返回壳）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **课程定位（首页）：** 目标是让机器人"快速、高效、鲁棒"地运动，涵盖非线性动力学、线性系统理论、经典最优控制、数值优化、状态估计、系统辨识与强化学习；先修为强线代、Python / MATLAB / Julia 经验、基础 ODE。周二 / 周四 12:30–1:50，录像上线；无统一教材（Background 页给出线代、线性系统与优化的免费资料，如 Nocedal & Wright、Boyd 凸优化）；无考试，项目占 50%、作业 40%（共 4 次，GitHub 分发、Gradescope 提交）、测验 5%、参与 5%。
- **学习目标（首页）：** 分析动力系统稳定性；设计稳定平衡点与轨迹的 LQR；用离线轨迹优化为非线性系统设计轨迹；用在线凸优化实现 MPC；理解随机性与模型不确定性的影响；模型不可得时直接优化反馈策略。
- **2025 讲次（Lectures 页，20 讲）：** 1 引言与动力学；2 离散化与稳定性；3–5 优化 Pt.1–3；6 正则化、merit 函数与控制史；7 确定性最优控制与 Pontryagin；8 LQR in 3 Ways；9 可控性与动态规划；10 凸 MPC；11 非线性轨迹优化；12 DDP；13 直接轨迹优化；14–16 三维旋转 / 旋转优化 / 四元数 LQR 与四旋翼；17 混杂系统与足式；18 ILC；19 随机最优控制与 LQG；20 How to Walk。2024 表另含 Kalman 滤波与对偶、凸松弛与火箭着陆、足式与 QP 行走、自动驾驶、变分法；2023 表另含鲁棒控制与极小极大、强化学习。
- **讲义仓编号（24 个目录）：** 与站点 2025 表前 18 讲对应；仓内 `Lecture 19`（随机最优控制与 LQG）、`Lecture 20`（How to Walk 嘉宾讲座）、`Lecture 21`（最优估计 / Kalman / 对偶，含 `lqg.ipynb`）、`Lecture 22`（火箭软着陆与凸松弛）、`Lecture 23`（自动驾驶与博弈论 MPC）、`Lecture 24`（数据驱动 / 行为克隆，含 `oscillator.ipynb`、`quadrotor-clone.ipynb`）。站点 2025 表只列到第 20 讲，第 21–24 讲的讲义存在于仓库。
- **LQR 的三种讲法（第 7–9 讲）：** 离散时间问题 min Σ(½xᵀQx + ½uᵀRu) + 终端项，约束 x_{k+1}=A x_k + B u_k。(1) 间接打靶（Pontryagin 离散形式：x_{k+1}=∂_λH，λ_k=∂_xH，u=argmin H；LQR 下 u_k = −R⁻¹Bᵀλ_{k+1}；"前向滚出、反向求协态、线搜索"，课程称此类间接 / 打靶法随算力提高已大多不再使用）；(2) 把整条轨迹堆成一个等式约束 QP（Cz=d，KKT 线性方程组，结构极稀疏，"比打靶好得多"）；(3) 反向 Riccati 递推 P_N=Q_N，K_k=(R+BᵀP_{k+1}B)⁻¹BᵀP_{k+1}A，P_k=Q+AᵀP_{k+1}(A−BK_k)，再正向滚出——与 QP 精确一致；QP 复杂度 O((N(n+m))³)，Riccati 为 O(N(n+m)³)。时不变问题的无限时域极限 P_∞、K_∞ 可用 `dare` 求。反馈策略可处理初值变化与扰动。
- **可控性与 DP（第 9 讲）：** 时不变情形可控性矩阵 C=[B AB … Aⁿ⁻¹B] 满秩（凯莱–哈密顿定理保证 n 步足够）；Bellman 最优性原理：最优轨迹的子轨迹对相应子问题也最优；值函数 V_k(x)；LQR 下 V=½xᵀP x；拉格朗日乘子 λ_k=∇V_k(x_k)（"动力学乘子是 cost-to-go 梯度"，对非线性同样成立）；DP 保证全局最优但受维数灾难限制，近似 DP / 函数逼近是现代 RL 的基础。
- **凸 MPC（第 10 讲）：** 约束破坏 Riccati 解，但可在线解 QP；MPC = "带约束的 LQR"，以 LQR 的 cost-to-go 作终端代价，无约束时对任意时域与 LQR 完全一致；前 H 步显式约束优化使状态接近参考、之后约束不再起作用；V 越准时域可越短。示例为平面四旋翼悬停线性化。凸优化无虚假局部最优，牛顿法 5–10 次迭代，可给出时间上界。
- **非线性轨迹优化（第 11–13 讲）：** DDP 用 cost-to-go 与 action-value 的二阶展开做牛顿步（前馈 + 反馈，可提前停止）；DDP 与 DIRCOL 对比：DDP 总是动力学可行、可用 TVLQR 跟踪、局部收敛快，但状态约束难处理；DIRCOL / SQP 只在收敛时满足动力学、可处理任意约束、可从不可行初值出发；直接配点 Hermite–Simpson 精度同 RK3 却少调用动力学（讲义称可省约 30–50% 计算）；最小时间问题把步长 h 作决策变量并加界。
- **旋转与四元数 LQR（第 14–16 讲）：** 欧拉角有奇异；四元数双覆盖、天然需要在切空间上求导；朴素线性化含四元数状态的系统会得到不可控的线性系统，需用姿态雅可比 E(q) 约化；Wahba 问题例。
- **混杂系统（第 17 讲）：** 两种接触建模 —— 时间步进 / 接触隐式（互补条件，实质是 KKT，不需预设接触模式但优化更难）与事件驱动 / 混杂（守卫函数 + 跳跃映射，需预设模式序列，轨迹优化与 MPC 中常用）。
- **ILC（第 18 讲）：** 面对模型误差的四种选项（参数辨识、学习模型、学习策略、迁移 / 迭代学习）；ILC 可看作用真机 rollout 取代模型预测的 SQP / 高斯–牛顿，每次只更新前馈，反馈增益固定。
- **随机最优控制与 LQG、Kalman（第 19、21 讲）：** LQG 下噪声项为常数，确定性等价（用 E[x] 代入 LQR）与分离原理成立，一般情形不成立；Kalman 滤波是递归线性 MMSE 估计，步骤预测 / 新息 / 增益 / Joseph 形式更新；估计与控制对偶，MMSE 问题等价于一个 LQR。
- **第 20 讲 How to Walk（John Zhang，2025-04-01 嘉宾讲座）：** 内容为步行机器人历史（Honda ASIMO 的倒立工业臂式准静态、CMU / MIT Leg Lab 的浮动基动力学、Boston Dynamics、MIT Cheetah、2025 年 RL）、凸 MPC（单刚体模型、假设：腿质量惯量相对本体可忽略、角速度与俯仰 / 横滚小、足端能较好跟踪参考；数百 Hz）、iLQR 非线性 MPC（不易处理额外约束，接触通过前向仿真隐式满足）、接触隐式方法与 sim2real RL（GPU 大规模并行仿真）。**该讲没有讲 LIP / ZMP。**
- **火箭着陆与凸松弛（第 22 讲）：** 推力范围 T_min ≤ ‖T‖ ≤ T_max 非凸，引入松弛变量 σ ≥ ‖T‖ 凸化；该松弛是紧的（证明依据 Pontryagin 最小原理）。
- **驾驶与博弈（第 23 讲）：** "冻结机器人问题"，Nash 均衡式联合轨迹优化（ALGAMES，增广拉格朗日）。
- **数据驱动控制（第 24 讲）：** 系统辨识坐标不唯一；用输入输出数据矩阵（Hankel）预测并做行为克隆 / 数据驱动 MPC；需要充分激励，闭环 on-policy 数据不够。

## 对 wiki 的映射

- [`wiki/entities/cmu-optimal-control-curriculum.md`](../../wiki/entities/cmu-optimal-control-curriculum.md)
- [`wiki/formalizations/lqr.md`](../../wiki/formalizations/lqr.md)
- [`wiki/concepts/optimal-control.md`](../../wiki/concepts/optimal-control.md)

## 参考来源

- 课程站 <https://optimalcontrol.ri.cmu.edu/>；讲义仓 <https://github.com/Optimal-Control-16-745/lecture-notebooks>；既有 ingest 归档 [`sites/cmu_optimal_control_16_745.md`](../sites/cmu_optimal_control_16_745.md)、[`repos/optimal_control_16_745.md`](../repos/optimal_control_16_745.md)、[`cmu_optimal_control_16_745_2025_youtube.md`](cmu_optimal_control_16_745_2025_youtube.md)
