# 线性 / 扩展卡尔曼滤波（KF / EKF）一手资料索引

> 来源归档（ingest）

- **标题：** Kalman Filter (KF) 与 Extended Kalman Filter (EKF) 经典论文、教材与权威教程
- **类型：** paper / textbook / course（合集）
- **入库日期：** 2026-06-01
- **一句话说明：** 汇总 KF 原始论文、连续时间推广、EKF 系统化教材与 MIT 课程讲义，作为 `wiki/formalizations/kalman-filter.md` 与 `wiki/formalizations/ekf.md` 的原始依据。
- **沉淀到 wiki：** 是 → [kalman-filter](../../wiki/formalizations/kalman-filter.md)、[ekf](../../wiki/formalizations/ekf.md)、[state-estimation](../../wiki/concepts/state-estimation.md)

## 为什么值得保留

- 机器人状态估计（IMU、足端、VIO）几乎都以 **KF → EKF → InEKF / 优化** 为脉络；一手资料可避免把 EKF 当成「凭空出现的工程技巧」。
- 本索引只收录 **原始论文、经典教材章节、大学官方讲义 / 教程 PDF**，不收录二手博客转述。

## 核心摘录

### 1) Kalman (1960) — 离散时间线性卡尔曼滤波（KF 源头）

- **来源：** R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, ASME Journal of Basic Engineering, 82(1):35–45, 1960. [UNC 镜像 PDF](https://www.cs.unc.edu/~welch/media/pdf/kalman1960.pdf)
- **要点：**
  - 将「带噪观测下的状态估计」表述为 **线性动力系统 + 线性观测** 下的递推最小方差估计。
  - 给出 **预测–校正（predict–update）** 两步结构：先由模型外推，再用观测残差与 **Kalman 增益** $K_k$ 融合。
  - 假设：过程噪声 $w_k$、观测噪声 $v_k$ 为零均值高斯且白；线性模型 $x_{k+1}=Ax_k+Bu_k+w_k$，$z_k=Cx_k+v_k$。
- **对 wiki 的映射：** [kalman-filter](../../wiki/formalizations/kalman-filter.md)、[ekf](../../wiki/formalizations/ekf.md)（EKF 页内 KF 回顾）、[state-estimation](../../wiki/concepts/state-estimation.md)

### 2) Kalman & Bucy (1961) — 连续时间线性滤波

- **来源：** R. E. Kalman, R. S. Bucy, *New Results in Linear Filtering and Prediction Theory*, ASME Journal of Basic Engineering, 83:95–108, 1961.
- **要点：**
  - 将离散 KF 推广到 **连续时间** 随机微分方程，得到 **Kalman–Bucy 滤波器**（Riccati 微分方程形式）。
  - 与最优控制中的 Riccati 方程共享数学结构，为 **LQR–LQG–Kalman 对偶** 奠基（见 `sources/papers/lqr_ilqr_primary_refs.md`）。
- **对 wiki 的映射：** [kalman-filter](../../wiki/formalizations/kalman-filter.md)、[lqr](../../wiki/formalizations/lqr.md)

### 3) Gelb (ed.) (1974) — *Applied Optimal Estimation*（EKF 工程化经典）

- **来源：** A. Gelb (Editor), *Applied Optimal Estimation*, MIT Press, 1974.（各章作者含 Kalman 学派与 NASA 应用团队）
- **要点：**
  - 系统化 **扩展卡尔曼滤波**：在非线性 $f,h$ 上对当前估计点做一阶 Taylor，用雅可比 $F_k,H_k$ 代入标准 KF 递推。
  - 覆盖 **连续–离散混合**、自适应噪声、实时导航等工程议题；是航天与早期机器人估计的「标准参考书」。
  - EKF 并非 Kalman (1960) 原文题名，但本书及同期 NASA 应用（Stanley F. Schmidt 等）确立了 **「线性化 + KF」** 范式。
- **对 wiki 的映射：** [ekf](../../wiki/formalizations/ekf.md)、[sensor-fusion](../../wiki/concepts/sensor-fusion.md)

### 4) Maybeck (1979) — *Stochastic Models, Estimation, and Control* Vol.1

- **来源：** P. S. Maybeck, *Stochastic Models, Estimation, and Control*, Vol. 1, Academic Press, 1979.
- **要点：**
  - 从 **随机过程、Bayes 估计** 出发推导 KF；对 **非线性估计** 讨论线性化滤波与误差协方差传播。
  - 适合需要 **概率论底座** 的读者，与 Gelb 的工程口径互补。
- **对 wiki 的映射：** [kalman-filter](../../wiki/formalizations/kalman-filter.md)、[ekf](../../wiki/formalizations/ekf.md)

### 5) Simon (2006) — *Optimal State Estimation*

- **来源：** D. Simon, *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*, Wiley, 2006. [作者课程页](https://engineering.uc.edu/~simondj/optimalestimation.html)
- **要点：**
  - 现代教材：KF / EKF / UKF / 粒子滤波 / 滑窗平滑 **并列**；EKF 章节强调 **一致性（consistency）** 与 **Joseph 形式协方差更新** 的数值稳定写法。
  - 机器人读者常用作 Gelb 的更新替代，数学符号与 MATLAB 示例更贴近当代课程。
- **对 wiki 的映射：** [ekf](../../wiki/formalizations/ekf.md)、[kalman-filter-vs-optimization-based-estimation](../../wiki/comparisons/kalman-filter-vs-optimization-based-estimation.md)

### 6) Welch & Bishop (2006) — *An Introduction to the Kalman Filter*（教程 PDF）

- **来源：** G. Welch, G. Bishop, UNC 技术报告 TR 95-041（2006 修订版）. [PDF](https://www.cs.unc.edu/~welch/kalman/kalmanIntro.html) · [配套站点 kalmanfilter.net](https://www.kalmanfilter.net/)
- **要点：**
  - **非期刊一手教程**：从直觉、一维例子到多维 KF 递推；被广泛用作课程讲义与实现入门。
  - 明确区分 **先验 / 后验** 协方差符号，适合实现前快速对齐记号。
- **对 wiki 的映射：** [kalman-filter](../../wiki/formalizations/kalman-filter.md)；课程归档见 [welch_bishop_kalman_filter](../courses/welch_bishop_kalman_filter.md)

### 7) MIT Underactuated Robotics — 状态估计与 KF 章节（课程讲义）

- **来源：** R. Tedrake, *Underactuated Robotics* (MIT 6.832 / 6.821)，[课程站](https://underactuated.csail.mit.edu/) · [Ch.16 Estimation](https://underactuated.csail.mit.edu/estimation.html)
- **要点：**
  - 从 **Bayes 滤波** 推导 KF；讨论 **EKF 在足式 / 操作机器人** 中的应用与局限。
  - 与 **LQR、MPC** 同书互链，适合「控制 + 估计」联合学习路径。
- **对 wiki 的映射：** [state-estimation](../../wiki/concepts/state-estimation.md)、[ekf](../../wiki/formalizations/ekf.md)；课程归档见 [mit_underactuated_kalman_lqr](../courses/mit_underactuated_kalman_lqr.md)

## 推荐继续阅读（外部）

- Hartley et al. (2020) InEKF — 足式机器人（已收录 `sources/papers/state_estimation.md`）
- Bar-Shalom, Li, Kirubarajan, *Estimation with Applications to Tracking and Navigation* — 目标跟踪标准教材

## 当前提炼状态

- [x] KF / EKF 七类一手来源摘录与 wiki 映射
- [x] 新建 `wiki/formalizations/kalman-filter.md` 并回写 EKF / state-estimation 参考来源
- [ ] 后续可补：Joseph 形式、平方根滤波、误差状态 EKF（ESKF）专节
