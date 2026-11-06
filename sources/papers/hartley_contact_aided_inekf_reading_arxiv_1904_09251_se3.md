# Contact-Aided InEKF 论文通读（足式机器人状态估计视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/state-estimation.md`）

- **标题：** Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation
- **作者：** Ross Hartley, Maani Ghaffari, Ryan M. Eustice, Jessy W. Grizzle（University of Michigan）
- **类型：** paper / state-estimation / legged
- **原文：** <https://arxiv.org/abs/1904.09251>（arXiv v2，2019-11；通过 arXiv PDF 获取全文，pdftotext 提取）
- **一句话说明：** 把「IMU + 接触点 + 前向运动学」的足式状态估计写成矩阵李群 SE_{N+2}(3) 上的右不变 EKF，误差动力学对确定性部分严格 log-linear，线性化不依赖当前估计，从而比四元数 EKF 对差初值收敛更快。
- **核对范围：** 通读 §1–§13 正文（背景、InEKF 理论 §3、SO(3) 动机例 §4、右不变 EKF 推导 §5 含可观性 §5.4、仿真 §6、IMU 偏置 §7、接触增删 §8、Cassie 实验 §9、左不变 / robo-centric §10–§11、额外传感器与 SLAM 关系 §12）；附录 A 离散化、B 李群表达式、C 误差状态换算仅浏览结构与结论，未逐式推导核对。未阅读其引用的 Barrau & Bonnabel 原始 InEKF 论文、Bloesch 2012 原文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题与动机（§1–§2）：** 足式机器人需要位姿与速度；纯视觉方案受光照 / 环境影响，故希望有只用本体感知（IMU、编码器、接触）的低层高频估计器。Bloesch 等的四元数 EKF（QEKF）是代表；本文在其思路上改用不变误差以获得更好收敛与一致性。
- **EKF 的局限（§2.1）：** EKF 对当前估计线性化，至多局部稳定，初值差可发散；不可观状态可能被线性化虚假地视为可观（可用 observability-constrained EKF 缓解）。error-state（间接）EKF 配合 IMU 积分（strapdown）可绕开复杂动力学建模。
- **足式估计脉络（§2.2）：** 运动学航位推算噪声大、会漂；Bloesch 2012 把接触足位置并入状态、接触建模为布朗运动以容许打滑，并证明纯本体感知下绝对位置与偏航不可观；Fallon、Nobili 用 LiDAR 配准消除位置 / 偏航漂移。
- **InEKF 理论（§3）：** 右 / 左不变误差 η^r = X̄X⁻¹、η^l = X⁻¹X̄；若动力学满足 group-affine（式 2），则不变误差动力学与轨迹无关（定理 1），且误差的对数满足线性微分方程并可精确恢复非线性误差（定理 2，log-linear）；观测若写成 Y = X⁻¹b + V（右不变）形式，创新项同样自治。
- **状态与模型（§5.1–5.2）：** 状态 X ∈ SE_{N+2}(3)，含姿态 R、速度 v、位置 p 与 N 个接触点位置 d_i；输入为 IMU 角速度 / 加速度；接触点速度在触地时视为零均值白噪声（容许打滑）。确定性动力学满足 group-affine，A_t 为常数且幂零（3 阶），协方差按 Riccati 方程传播。
- **前向运动学量测（§5.3）：** 编码器 + 正运动学给出足端相对机体位置，具有右不变观测形式；H = [0 0 −I I]，更新用 exp(K Π X̄ Y) X̄，H 与 A 均不依赖状态估计。
- **可观性（§5.4）：** 线性可观矩阵显示绝对位置与绕重力轴的偏航不可观，与 Bloesch 的非线性分析一致，且线性化不会虚增可观秩，因此不需要额外的 observability-constrained 处理。
- **仿真（§6，Cassie 行走仿真）：** 100 次随机初始姿态（±30°）与速度（±1 m/s），RIEKF 收敛明显快于 QEKF；确定性情形 RIEKF 的线性化误差恒为 0，QEKF 随初始误差增大而变差；加噪声后不再精确，但仍更准。协方差经指数映射后为李群上的 concentrated Gaussian，能表达位置分布的弯曲形状，甚至偏航完全未知的圆环分布。
- **IMU 偏置（§7）：** 不存在能同时包含偏置且保持 group-affine 的李群，故偏置并入后理论保证部分丧失，成为「不完美 InEKF」，但实验中仍优于 QEKF。
- **接触增删（§8）：** 断开接触即边缘化（删除对应行列与协方差块）；新接触由前向运动学初始化均值，并用线性映射增广协方差。
- **Cassie 实验（§9）：** IMU 800 Hz、编码器 2000 Hz，腿弹簧形变作二值接触传感器；100 次真机离线重放中 RIEKF 比 QEKF 收敛更快更可靠；动捕实验 60 s、约 15 m，终点误差小于行走距离的 5%；室外约 200 m、7 分 45 秒的长程里程计留在人行道上；搭载 Velodyne VLP-32C 做 LiDAR 局部地图。滤波器可 >2000 Hz 运行。
- **左不变 / robo-centric / 其他传感器（§10–§12）：** 左右不变误差经伴随映射互换；robo-centric 版本直接估计机体系速度；GPS、磁力计、地标等观测可纳入同一框架，接触点与 SLAM 地标形式相同但无数据关联问题，接触速度按白噪声处理。开源 C++ 库 invariant-ekf。
- **未来工作（§13）：** 不变平滑器、在线标定运动学参数、检测步态模式以加约束、视觉-惯性-接触里程计。

## 对 wiki 的映射

- [`wiki/concepts/state-estimation.md`](../../wiki/concepts/state-estimation.md)
