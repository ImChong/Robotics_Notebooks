# 视频发布清单：从头学人形机器人算法 · 零基础篇

按 [主路线：运动控制 → 物理智能全栈成长路线](../../roadmap/motion-control.md)（站点 [roadmap.html?id=roadmap-motion-control](https://imchong.github.io/Robotics_Notebooks/roadmap.html?id=roadmap-motion-control)）从 L−1 讲到 L5.1 的 PPO，面向完全零基础的初学者；讲完 060 接上姊妹系列《从头学人形机器人算法》（论文精读）的 001 PPO。共 60 集，**每集只讲一个主题**，按发布顺序排好。格式参考 [Robot_Learning_Paper_Notebooks 的论文视频发布清单](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/scripts/paper_video/RELEASE_PLAN.md)。每发一集，把那一行的 `[ ]` 改成 `[x]`。截至 2026-10-11 尚未发布；下一集为 001。

- 🎬 `<ep>`：本目录已有分镜与旁白源稿（`episodes/<ep>`），成片用 `render.mjs` 渲染（见 [README](README.md)）
- 🆕：还没有视频，要新做
- `[x]` 只表示已经发布
- 每行破折号后先写「只讲」什么，再写排序依据：`§L1.2` 等指主路线章节；标〔取舍〕的是拆分或排序判断，路线原文没这么写
- 「素材 cNN」指 [路线讲解长视频](../roadmap-motion-control-video/README.md) 的讲解稿 `chapters/NN-*.txt` 里可复用的幻灯片（引号内为标题原文），`svg/` 为同目录示意图；单集要在此基础上展开，不是直接剪长视频
- 标题格式：「从头学人形机器人算法 · 零基础篇 (0NN)：<本集主题>」

## 每集一个主题

- 一集只回答一个问题；用到的前置概念只用一两句回顾，并说明在第几集讲过
- 片尾只预告下一集要回答的问题，不提前展开
- 每集至少给一个能跑的最小例子，或一道主路线「自测题」并当场给出答案

## 第一季 · 全景与数学打底（L−1、L0）

- [ ] 001 [四步循环：感知 → 规划 → 控制 → 执行](../../roadmap/motion-control.md#l1-序言机器人技术栈全景--怎么读这条路线) 🎬 `ep001` — 只讲四步循环，以及本系列只钻「控制」这一盒；§L−1「30 秒看懂"一台机器人在干嘛"」。素材 c02「任何机器人都在循环跑同一个 4 步」、`svg/loop4.svg`。〔取舍〕§L−1 没有自测题，片尾用自编小题；§L−1 把 SLAM 放在规划盒、§L7.1 放在感知层，片中按 §L7.1：感知负责建图定位，规划只拿地图找路
- [ ] 002 [为什么从人形 / 双足学起](../../wiki/entities/humanoid-robot.md) 🆕 — 只讲人形的四个难点：高自由度、浮动基、接触切换、强不稳定；§L−1「为什么以"人形 / 双足"为主载体」。素材 c02「为什么以人形 / 双足为主载体」
- [ ] 003 [向量与矩阵：机器人里的线性代数](../../wiki/entities/linear-algebra-curriculum.md) 🆕 — 只讲向量、矩阵乘法在描述位置和变换时的含义；§L0 核心问题「线性代数在机器人里到底怎么用」
- [ ] 004 用 Python / NumPy 写第一段矩阵代码 🆕 — 只讲把 003 的运算写成 NumPy 代码并跑通；§L0 前置知识「会写 Python」、学完输出「能用 NumPy 写简单矩阵运算」。〔取舍〕先讲概念再上代码
- [ ] 005 [旋转矩阵：为什么旋转不能直接相加](../../wiki/comparisons/so3-rotation-representations.md) 🆕 — 只讲旋转矩阵的两条性质（正交、行列式为 1），以及逐元素相加 / 插值会跳出 SO(3)；§L0 自测题 1。素材 c03「旋转矩阵为什么不能直接相加 / 插值」、`svg/so3manifold.svg`
- [ ] 006 [欧拉角、四元数、旋转矩阵：三种表示怎么选](../../wiki/comparisons/so3-rotation-representations.md) 🆕 — 只讲三种表示的取舍（万向锁、双重覆盖、冗余）；§L0 自测题 3、推荐读「旋转表示方法对比」。素材 c03「旋转怎么表示：三种方案的取舍」
- [ ] 007 [齐次变换与 SE(3)：位姿 = 朝向 + 位置](../../wiki/formalizations/se3-representation.md) 🆕 — 只讲 (R, p) 怎么拼成 4×4 矩阵，点和方向向量的变换有什么不同；§L0 自测题 2。素材 c03「位姿 = 朝向 R + 位置 p」、`svg/se3.svg`
- [ ] 008 [优化的直觉：变量、目标、约束](../../wiki/entities/numerical-optimization-curriculum.md) 🆕 — 只讲「找让代价最小的变量」这一个框架，为第四季的 LQR / QP / MPC 和第六季的梯度更新铺垫；§L0 核心问题「优化问题的直觉是什么」。素材 c03「优化直觉 + 这一层要做出什么」

## 第二季 · 机器人学骨架（L1）

§L1 原文建议分三步走：L1.1 SE(3) 与刚体变换 → L1.2 FK / IK → L1.3 雅可比。005–007 已讲旋转和齐次变换，这一季从 twist 接着讲。

- [ ] 009 [Twist 与螺旋轴：用一个六维量描述刚体速度](../../wiki/formalizations/spatial-twist-wrench-poe.md) 🆕 — 只讲 twist 和 screw axis；§L1.1、Modern Robotics Ch 3。素材 c04「三步走，外加三个六维量」
- [ ] 010 [矩阵指数：从螺旋运动到位姿](../../wiki/formalizations/lie-group-rigid-body-motions.md) 🆕 — 只讲 exp 怎样把「绕螺旋轴转多少」变成一个 SE(3) 位姿；§L1.1「矩阵指数 / PoE」、§L0 自测题 3
- [ ] 011 [正运动学：用 PoE 从关节角算末端位姿](../../wiki/formalizations/forward-kinematics.md) 🆕 — 只讲指数积公式把每个关节连乘起来，手算一个 2–3 自由度机械臂；§L1.2、推荐做什么「手写 `FKinSpace`」。素材 c04「正运动学：指数积公式 PoE」、`svg/poe.svg`
- [ ] 012 [逆运动学：已知末端反求关节角](../../wiki/formalizations/inverse-kinematics.md) 🆕 — 只讲 IK 为什么会多解或无解（肘上 / 肘下）；§L1.2、自测题 2。素材 c04「逆运动学：解往往不唯一」、`svg/ik_elbow.svg`
- [ ] 013 [雅可比矩阵：关节速度 → 末端速度](../../wiki/formalizations/robot-jacobian.md) 🆕 — 只讲雅可比是速度的线性映射；§L1.3。素材 c04「雅可比：速度的线性映射，力的对偶映射」前半（力的对偶留到 019）
- [ ] 014 [Space Jacobian 与 Body Jacobian](../../wiki/formalizations/robot-jacobian.md) 🆕 — 只讲两种雅可比的坐标系差别和伴随变换换算；§L1.3、自测题 1
- [ ] 015 [用 Pinocchio 加载 URDF，跑出 FK 与雅可比](../../wiki/entities/pinocchio.md) 🆕 — 只讲 [URDF](../../wiki/concepts/urdf-robot-description.md) 描述了什么，以及用 Pinocchio 算出的结果和 011、013 手写结果对齐；§L1 推荐做什么、学完输出「能自己建模一个简单机器人」

## 第三季 · 动力学与刚体建模（L2）

- [ ] 016 [动力学方程：力矩怎么变成加速度](../../roadmap/motion-control.md#l2-动力学与刚体建模) 🆕 — 只讲 $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$ 三项各自的物理意义；§L2.1、自测题 1。素材 c05「一行方程：固定基开链动力学」
- [ ] 017 [RNEA / CRBA / ABA：三个动力学算法各求什么](../../wiki/entities/pinocchio.md) 🆕 — 只讲三个算法的输入输出和计算复杂度；§L2.1、自测题 4。素材 c05「三个递归算法：RNEA / CRBA / ABA」
- [ ] 018 [重力补偿：让机械臂悬停](../../wiki/concepts/gravity-compensation.md) 🆕 — 只讲 $g(q)=\mathrm{RNEA}(q,0,0)$ 的控制用法；§L2 推荐读「Gravity Compensation」。〔取舍〕放在 RNEA 之后，当作第一个动力学应用
- [ ] 019 [力旋量与雅可比转置：末端力 → 关节力矩](../../wiki/formalizations/spatial-twist-wrench-poe.md) 🆕 — 只讲 wrench 和 $\tau = J^\top F$（虚功原理）；§L2 核心问题「wrench、Jacobian transpose、虚功原理…」、Modern Robotics Ch 5。素材 c04「雅可比：速度的线性映射，力的对偶映射」后半
- [ ] 020 [浮动基：人形的身体也是自由变量](../../wiki/concepts/floating-base-dynamics.md) 🆕 — 只讲状态为什么要加上 base 位姿和速度；§L2.2、自测题 2。素材 c05「浮动基：身体本身也是自由变量」
- [ ] 021 [接触与摩擦锥：脚能对地面用多大的力](../../wiki/formalizations/contact-wrench-cone.md) 🆕 — 只讲接触力受摩擦锥约束，以及接触力怎样经 $J_c^\top$ 变成关节力矩；§L2.2、自测题 3。素材 c05「接触：力怎么变成力矩，又受什么限制」、`svg/friction_cone.svg`
- [ ] 022 [关节摩擦：库仑、粘滞与 Stribeck](../../wiki/concepts/joint-friction-models.md) 🆕 — 只讲关节摩擦模型；§L2「Physical AI 视角补充」第 1 条。素材 c05「Physical AI 视角：网络控制的不是理想刚体」
- [ ] 023 [执行器：真实电机有饱和、带宽和延迟](../../wiki/concepts/implicit-explicit-actuator-modeling.md) 🆕 — 只讲执行器的非理想性，以及仿真里怎么建模；§L2「Physical AI 视角补充」第 2 条
- [ ] 024 [状态估计：机器人怎么知道自己的姿态和速度](../../wiki/concepts/state-estimation.md) 🆕 — 只讲 IMU 与编码器融合估计浮动基状态；§L2「Physical AI 视角补充」第 3 条。〔取舍〕022–024 原文是一节里的三条，按一集一主题拆成三集

## 第四季 · 控制基础与最优化（L3）

〔取舍〕§L3 核心问题把 PID / LQR / MPC 并列；这里先讲反馈（025–029），再讲最优化（030–034）。

- [ ] 025 [PID：最经典的反馈控制](../../wiki/methods/pid-control.md) 🆕 — 只讲 P、I、D 三项各管什么；§L3 核心问题、L4.0 方法谱系表 PID 行
- [ ] 026 [轨迹生成：三次 / 五次时间缩放](../../wiki/entities/modern-robotics-book.md) 🆕 — 只讲怎样生成起止平滑的参考轨迹；§L3 推荐做什么「复现 Modern Robotics Ch 9 的三次/五次时间缩放轨迹」
- [ ] 027 [计算力矩控制：用动力学模型抵消非线性](../../wiki/methods/computed-torque-control.md) 🆕 — 只讲 computed torque = 动力学前馈 + PD 反馈；§L3 推荐做什么「加 PD / computed torque tracking」。素材 c06「反馈控制：PD 与计算力矩」
- [ ] 028 [阻抗控制与导纳控制](../../wiki/concepts/impedance-control.md) 🆕 — 只讲「运动进、力出」和「力进、运动出」的区别；§L3 自测题 4，另见[导纳控制](../../wiki/methods/admittance-control.md)。素材 c06「和环境接触：阻抗 vs 导纳」
- [ ] 029 [关节 PD 与 MIT 模式：Policy ≠ 底层控制器](../../roadmap/motion-control.md#policy-vs-low-level-controller) 🆕 — 只讲 $\tau = K_p(q_{des}-q) + K_d(\dot q_{des}-\dot q) + \tau_{ff}$，以及策略出 $q_{des}$、底层 PD 闭环的分工；§L3「Policy ≠ 底层控制器」、Physical AI Core Path 第 2 步。素材 c06「Policy ≠ 底层控制器」「五种控制模式与第一类 sim2real 坑」。〔取舍〕放在阻抗控制之后，因为 MIT 模式的 PD 就是关节级阻抗；第六季 PPO 的动作空间沿用这一集
- [ ] 030 [最优控制：把「控制得好」写成代价函数](../../wiki/concepts/optimal-control.md) 🆕 — 只讲状态、输入、代价、动力学约束组成的最优控制问题（OCP）；§L3 核心问题「最优控制的核心思想是什么」。素材 c10「共同的数学外壳：最优控制问题 OCP」
- [ ] 031 [LQR：倒立摆的最优反馈](../../wiki/formalizations/lqr.md) 🆕 — 只讲 LQR 求出 $u=-Kx$ 并稳住倒立摆；§L3 推荐做什么「用 Python 写一个倒立摆的 LQR 控制器」。素材 c06「最优控制：LQR 与 MPC」前半
- [ ] 032 [二次规划 QP：为什么控制里到处是凸 QP](../../wiki/formalizations/quadratic-programming.md) 🆕 — 只讲 QP 的形式和凸性判断，用 OSQP 跑一个小例子；§L3 推荐做什么「用 qpOASES 或 OSQP 跑一个简单 QP」、自测题 2。素材 c06「QP：为什么控制里到处都是凸二次规划」
- [ ] 033 [MPC：滚动时域的在线优化](../../wiki/methods/model-predictive-control.md) 🆕 — 只讲「预测一段、只执行第一步、再重算」，以及它和 LQR 何时等价；§L3 推荐做什么「理解 MPC 的滚动时域思想」、自测题 1。素材 c06「最优控制：LQR 与 MPC」后半、`svg/mpc_horizon.svg`
- [ ] 034 [任务优先级：HQP 与零空间投影](../../wiki/concepts/hqp.md) 🆕 — 只讲低优先级任务怎样只在高优先级任务的零空间里动；§L3 自测题 3，另见[零空间控制](../../wiki/concepts/null-space-control.md)。素材 c06「HQP：用零空间实现任务优先级」

## 第五季 · 人形运动控制主干（L4）

- [ ] 035 [人形控制的方法链：模型从粗到细，控制从慢到快](../../roadmap/motion-control.md#l4-人形运动控制主干) 🆕 — 只讲 L4.0 的两条轴（模型粒度、控制频率），给 036–044 一张地图；§L4.0。素材 c07「两条轴，把 L1–L3 串成 L4 的方法链」「方法谱系（上）：model-based 从粗到细、从慢到快」
- [ ] 036 [线性倒立摆 LIP：把人形看成一根会走的杆](../../wiki/concepts/lip-zmp.md) 🆕 — 只讲 LIP 的三个假设和线性方程；§L4.1、自测题 1。素材 c08「线性倒立摆：三个假设，一个线性方程」、`svg/lip_pendulum.svg`
- [ ] 037 [ZMP 与支撑多边形：不摔倒的判据](../../wiki/formalizations/zmp-lip.md) 🆕 — 只讲 ZMP 留在支撑多边形内意味着什么；§L4.1、自测题 2。素材 c08「ZMP 与支撑多边形：经典的平衡判据」、`svg/support_polygon.svg`
- [ ] 038 [用 ZMP 参考生成质心轨迹](../../wiki/concepts/lip-zmp.md) 🆕 — 只讲落脚点 → ZMP 参考 → 质心轨迹的步态生成；§L4.1 推荐做什么、推荐读 Kajita 等的 ZMP 预观控制论文。素材 c08「怎么用它生成步态：从 ZMP 参考到质心轨迹」
- [ ] 039 [捕获点 / DCM：已经在倒，下一步踩哪](../../wiki/concepts/capture-point-dcm.md) 🆕 — 只讲 DCM 怎样决定落脚点；§L4.1、自测题 3。素材 c08「DCM / 捕获点：已经在倒，下一步踩哪？」、`svg/dcm.svg`
- [ ] 040 [质心动力学：挥手扭腰也算进平衡](../../wiki/concepts/centroidal-dynamics.md) 🆕 — 只讲 6D 质心动量、质心动量矩阵和「模型降阶」；§L4.2。素材 c09 三页、`svg/centroidal.svg`
- [ ] 041 [轨迹优化：离线算出一整段动作](../../wiki/methods/trajectory-optimization.md) 🆕 — 只讲 TrajOpt 和 MPC 的区别（离线一次 vs 在线滚动）；§L4.3、自测题 2。素材 c10「TrajOpt vs MPC：离线一次，还是在线滚动」
- [ ] 042 [质心 MPC：在线规划质心轨迹与接触力](../../wiki/methods/model-predictive-control.md) 🆕 — 只讲把 040 的质心模型放进 033 的 MPC，以及接触切换怎么处理；§L4.3、自测题 3。素材 c10「凸 MPC vs 非线性 MPC，接触切换怎么处理」「MPC 步态发抖：由外到内的排查顺序」
- [ ] 043 [全身控制 WBC：把任务分配到每个关节](../../wiki/concepts/whole-body-control.md) 🆕 — 只讲 WBC 在分层里的位置：接住 MPC 的参考，每个周期分配关节力矩；§L4.4 场景隐喻与「上一层的局限」。〔取舍〕先讲 WBC 是什么，再讲它的一种实现 TSID
- [ ] 044 [TSID：把全身控制写成一个 QP](../../wiki/formalizations/tsid-formulation.md) 🆕 — 只讲 TSID 的决策变量、等式 / 不等式约束和任务代价；§L4.4、自测题 1–2。素材 c11「TSID：把全身控制写成一个 QP」「冲突时听谁的：优先级与阻抗」「L4 收束：一条完整的方法链」

## 第六季 · 强化学习：从 MDP 到 PPO（L5.0、L5.1）

主线按 §L5.1「RL 主干概念链」：MDP → 价值函数 → 策略梯度 → Actor-Critic → GAE → PPO。

- [ ] 045 [为什么需要强化学习：从「算出来」到「试出来」](../../wiki/comparisons/wbc-vs-rl.md) 🆕 — 只讲传统控制在哪里吃力，以及 RL 是能力扩展层、不是替代；§L5 开头、§L5.1「上一层的局限」。素材 c12「三条心法：RL / IL 是能力扩展层」
- [ ] 046 [策略 π(a|s)：智能体、环境与奖励](../../wiki/concepts/embodied-rl-minimal-closed-loop.md) 🆕 — 只讲岔路口比喻，以及离散动作和连续动作的区别；§L5.0「策略直觉（岔路口比喻）」
- [ ] 047 [MDP 五元组：S、A、R、P、γ](../../wiki/formalizations/mdp.md) 🆕 — 只讲五元组和折扣因子 γ；§L5.0「MDP 与 POMDP」表。素材 c12「MDP 五元组，以及真机上的 POMDP」前半
- [ ] 048 [POMDP：真机看不到全部状态](../../wiki/formalizations/pomdp.md) 🆕 — 只讲部分可观测，以及为什么要用观测历史；§L5.0、§L5.1 推荐读「POMDP — 真机部署前必读」。素材同上后半
- [ ] 049 [具身 RL 最小闭环：在 PyBullet 里手写 S、A、R、P](../../wiki/entities/pybullet.md) 🆕 — 只讲 KUKA 臂定点任务里五元组怎样对应代码和 `stepSimulation`，先不接 RL；§L5.0 最小闭环实验第 1 步，详见[具身 RL 最小闭环](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)
- [ ] 050 [奖励设计：机器人会钻奖励的空子](../../wiki/concepts/reward-design.md) 🆕 — 只讲 reward shaping 和奖励投机（「小跳着前进」）；§L5.1 推荐做什么「理解 reward shaping…」、自测题 3。素材 c12「奖励投机：机器人学会了"小跳着前进"」。〔取舍〕原文这道题在 PPO 之后；这里提前到 049 之后，因为手写 R 时就会遇到
- [ ] 051 [价值函数 V(s) 与 Q(s,a)](../../wiki/formalizations/bellman-equation.md) 🆕 — 只讲 V 和 Q 各自估计什么（长期折扣回报）；§L5.1 概念链第 2 环。素材 c12「RL 主干概念链：从价值到 Actor-Critic」
- [ ] 052 [策略梯度：沿着回报变大的方向改策略](../../wiki/methods/policy-optimization.md) 🆕 — 只讲 $\nabla \log \pi \cdot$ Advantage 的直觉；§L5.1 概念链第 3 环。素材同上
- [ ] 053 [Actor-Critic：一个出动作，一个打分](../../wiki/methods/reinforcement-learning.md) 🆕 — 只讲 actor 和 critic 的分工；§L5.1 概念链第 4 环。素材同上
- [ ] 054 [TD 误差：用下一步的估值修正这一步](../../wiki/formalizations/bellman-equation.md) 🆕 — 只讲 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$；§L5.1「PPO 一次迭代里要真正看懂的 8 个概念」第 3 个。素材 c12「TD 误差与 GAE：偏差和方差之间的旋钮」前半。〔取舍〕概念链没把 TD 误差单列，按「8 个概念」的顺序放在 GAE 前
- [ ] 055 [GAE：在偏差和方差之间调一个旋钮](../../wiki/methods/gae.md) 🆕 — 只讲 λ 加权的 TD 误差怎样得到优势 $\hat A_t$；§L5.1 概念链第 5 环、[GAE 论文（arXiv:1506.02438）](https://arxiv.org/abs/1506.02438)。素材同上后半
- [ ] 056 [信赖域：策略一步不能迈太大](../../wiki/methods/policy-optimization.md) 🆕 — 只讲为什么策略更新幅度要受限（TRPO 的思想）；§L5.0「策略单次迭代不能跳变太大」、§L5.1 缩写表「TRPO：PPO 的前身思想」。〔取舍〕原文没有单列 TRPO，加这一集是为了让 057 的 clip 有来由
- [ ] 057 [PPO 的核心：重要性比与 clip](../../wiki/methods/ppo.md) 🆕 — 只讲 $r_t = \pi_\theta/\pi_{old}$ 和 $\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)$，以及 ε 太大 / 太小的后果；§L5.1 自测题 1、[PPO 论文（arXiv:1707.06347）](https://arxiv.org/abs/1707.06347)。素材 c12「PPO：裁剪让每次更新只走一小步」、`svg/ppo_clip.svg`
- [ ] 058 [PPO 的一次完整迭代](../../wiki/methods/ppo.md) 🆕 — 只讲并行 rollout → GAE → 多 epoch minibatch 更新这一圈，对照 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 代码；§L5.1「8 个概念」整条链。素材 c12「一次 PPO 迭代，以及为什么人形 RL 主流用 PPO」前半
- [ ] 059 [动手：在 CartPole 上训练 PPO](../../wiki/concepts/cartpole.md) 🆕 — 只讲用 Gymnasium `CartPole-v1` 跑通 PPO 并读训练曲线；§L5.0 最小闭环实验第 2 步、§L5.1 推荐做什么
- [ ] 060 [PPO vs SAC：为什么人形 RL 主流用 PPO](../../wiki/comparisons/ppo-vs-sac.md) 🆕 — 只讲 on-policy 和 off-policy 的样本效率差别，以及大规模并行仿真为什么偏向 PPO；§L5.0「PPO vs SAC 具身分工」、§L5.1 自测题 2。素材 c12「一次 PPO 迭代，以及为什么人形 RL 主流用 PPO」后半。系列收尾，片尾预告 §L5.2「把 PPO 用到人形行走」

## 扩展规则

- 本清单止于 §L5.1 的 PPO。之后按主路线顺序（L5.2 人形 RL → L5.3 模仿学习 → L5.4 动作重定向 → L6 Sim2Real …）从 061 起追加，已发视频不改编号。
- 想更快讲到 PPO：§L5.1 的前置知识只写了「L2 + L3」，可以把第五季（L4）挪到第六季之后。〔取舍〕本清单按主路线「三句话先懂这条路线」的顺序：先打通传统控制主干，再接学习方法。
- 主路线对应章节改动后，先改这里的「只讲」和依据，再做视频。
- 本系列讲概念，不逐篇精读论文；看完 060 想读 PPO 原论文，可以接 Robot_Learning_Paper_Notebooks 论文视频系列的 001 PPO。
