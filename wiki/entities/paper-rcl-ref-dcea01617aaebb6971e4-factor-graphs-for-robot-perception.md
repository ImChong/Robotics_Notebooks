---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-11-26
venue: "Foundations and Trends in Robotics"
summary: "Frank Dellaert 与 Michael Kaess 2017 年经典专著（FnT Robotics Vol. 6, No. 1-2, pp. 1–139），系统阐述因子图作为机器人感知与 SLAM 统一概率图模型在非线性最小二乘 MAP 推断、稀疏变量消元（Cholesky/QR/COLAMD/嵌套剖分）、Bayes Tree 增量平滑建图（iSAM2）、李群流形（SO(2)/SO(3)/SE(2)/SE(3)/S^2）局部坐标优化与大规模/分布式/非高斯感知中的数学基础与工程实践。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_ref_dcea01617aaebb6971e4_factor-graphs-for-robot-perception.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Factor Graphs for Robot Perception

**Factor Graphs for Robot Perception**（Frank Dellaert & Michael Kaess，*Foundations and Trends in Robotics* 2017，Vol. 6, No. 1-2, pp. 1–139，DOI: `10.1561/2300000043`）收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 158/564** 篇，分组 **Foundational work**。本页基于专著全文（第 1–7 章及附录）完成系统精读与核心公式、算法及实证结论核实。

## 一句话定义

**Factor Graphs for Robot Perception** 系统建立了以**二分因子图**为核心的机器人感知与 SLAM 统一求解框架：将基于观测条件化的后验最大后验（MAP）推断等价转化为稀疏非线性最小二乘问题，并通过变量消元与稀疏矩阵分解、基于 **Bayes Tree** 的增量平滑建图（**iSAM2**）以及李群/流形局部切空间收缩（Retraction）实现大规模实时精确状态估计。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图，因子图最核心的机器人感知问题 |
| MAP | Maximum a Posteriori | 最大后验估计，高斯噪声假设下等价于非线性最小二乘 |
| SAM / $\sqrt{\text{SAM}}$ | Smoothing and Mapping / Square Root SAM | 平滑与建图，基于信息矩阵平方根 $R$ 的全轨迹批量优化 |
| iSAM / iSAM2 | Incremental Smoothing and Mapping | 增量平滑与建图；iSAM2 引入 Bayes Tree 实现流式重线性化与局部更新 |
| AMD / COLAMD | Approximate Minimum Degree / Column AMD | 近似最小度变量消元重排启发式，用于大幅减少 Cholesky/QR 填充（Fill-in） |
| ND / TSAM | Nested Dissection / Tectonic SAM | 嵌套剖分分治排序与子图分解方法，平面图下实现 $O(n^{1.5})$ 复杂度 |
| EKF / SRIF | Extended Kalman Filter / Square-Root Information Filter | 扩展卡尔曼滤波与平方根信息滤波，对应链式因子图固定滞后边际化特例 |
| GLC | Generic Linear Constraints | 通用线性约束，用于长期建图中的节点移除与图稀疏化 |

## 为什么重要

- **统一建模语言**：不同于需要保留显式观测变量的贝叶斯网络，因子图仅保留未知状态变量 $X$ 并将已知观测 $Z$ 吸收进似然因子 $l(X_i; z_i)$，直接揭示优化目标的稀疏结构与条件独立性。
- **打通图模型与稀疏线性代数**：专著严格证明**变量消元算法**与稀疏 **QR / Cholesky 分解**一一等价，并以此引出连通二者的数据结构 **Bayes Tree**，解决了传统增量矩阵分解难以增量重线性化的根本痛点。
- **奠基现代工业级感知库**：文中体系直接对应 GTSAM、iSAM 与 g2o 等核心后端的数学与软件架构，支撑惯性预积分（VIO）、密集三维重建、水下/农业场机器人及大规模多机协同 SLAM。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 158/564 |
| 分组 | Foundational work |
| 出处 | Foundations and Trends in Robotics, Vol. 6, No. 1-2 (2017), pp. 1–139 |
| 作者与机构 | Frank Dellaert（Georgia Institute of Technology）、Michael Kaess（Carnegie Mellon University） |
| DOI / 论文 | DOI: `10.1561/2300000043` · <https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html> |
| 子类 / 象限 | 三维表示与状态估计 · 理论与规划 · 不适用 |

## 核心机制

1. **因子图定义与高斯 MAP 最小二乘转化（Ch. 1–2）**
   - 因子图是二分图 $F = (U, V, E)$，包含因子节点 $\phi_i \in U$ 与变量节点 $x_j \in V$，定义全局未归一化密度分解 $\phi(X) = \prod_i \phi_i(X_i)$（Eq. 1.24）。
   - 对已知观测 $Z$ 条件化后，高斯噪声模型 $\phi_i(X_i) \propto \exp(-\frac{1}{2}\|h_i(X_i) - z_i\|_{\Sigma_i}^2)$ 的 MAP 估计等价于非线性最小二乘（Eq. 2.4）：
     $$X^{\text{MAP}} = \arg\min_X \sum_i \|h_i(X_i) - z_i\|_{\Sigma_i}^2$$
   - 经泰勒线性化 $h_i(X_i^0 + \Delta_i) \approx h_i(X_i^0) + H_i \Delta_i$ 与白化 $A_i = \Sigma_i^{-1/2} H_i$、$b_i = \Sigma_i^{-1/2}(z_i - h_i(X_i^0))$，化为标准线性最小二乘 $\Delta^* = \arg\min_\Delta \|A\Delta - b\|_2^2$（Eq. 2.13）。
   - **批量非线性求解器（Ch. 2.4–2.5）**：直接求解正规方程 $A^\top A \Delta^* = A^\top b$（Cholesky 分解 $\Lambda = A^\top A = R^\top R$）或数值更稳定的 Householder QR 分解 $Q^\top A = \begin{bmatrix} R \\ 0 \end{bmatrix}$、$Q^\top b = \begin{bmatrix} d \\ e \end{bmatrix}$ 并回代 $R\Delta^* = d$；外层迭代采用高斯-牛顿（GN）、Levenberg-Marquardt（LM，阻尼正规方程 $(A^\top A + \lambda \operatorname{diag}(A^\top A))\Delta_{\text{lm}} = A^\top b$，Algorithm 2.1）或 Powell's Dogleg（结合最速下降与 GN 方向并以增益比 $\rho$ 控制信赖域，每次迭代仅需单次矩阵分解）。

2. **稀疏性利用、变量消元与消元排序（Ch. 3–4）**
   - **消元算法（Algorithms 3.1 & 3.2）**：按序逐个消去变量 $x_j$，将其邻接因子相乘构造乘积因子 $\psi(x_j, S_j)$，通过部分 QR 分解（Eq. 3.18–3.20，多波前 Multifrontal QR）拆分为条件密度与分隔符新因子 $\psi(x_j, S_j) = p(x_j | S_j)\tau(S_j)$，最终将因子图化为弦图（Chordal）贝叶斯网络 / 上三角平方根信息矩阵 $R$，随后按逆序回代求 MAP。
   - **填充（Fill-in）与重排策略**：消元顺序不改变 MAP 解但决定分隔符规模 $s_j$ 与非零填充量（求最小填充为 NP-hard）。先消地标再消位姿的 Schur 补技巧会在位姿间产生稠密连接；采用 **AMD / COLAMD**（尤其块级语义 AMD）或基于图划分（METIS）的 **Nested Dissection（Algorithm 4.1）** 可大幅降低计算量（链式图 $O(n)$，平面图依托 $O(\sqrt{n})$ 分隔符达 $O(n^{1.5})$，远优于稠密分解 $O(n^3)$）。

3. **增量平滑建图与 Bayes Tree / iSAM2（Ch. 5）**
   - 线性增量场景可通过 Givens 旋转对 $R$ 逐行清零增量更新（Eq. 5.5–5.7）；协方差形式下边际化直接切片，信息形式下需 Schur 补，而平方根信息形式 $R$ 中**叶子变量可直接丢弃行列实现零计算边际化**（对应固定滞后平滑 SRIS 与卡尔曼滤波 SRIF）。
   - **Bayes Tree（Ch. 5.4）**：将消元得到的弦贝叶斯网络按极大团组织为有向团树，每个节点 $C_k = F_k : S_k$ 存储条件密度 $p(F_k | S_k)$（联合密度 $p(X) = \prod_k p(F_k | S_k)$，Eq. 5.15）。
   - **iSAM2（Algorithm 5.1 & §5.4.3）**：新因子仅影响相关变量到根节点路径上的团，孤立子树（Orphans）保持不变直接重挂；结合受限排序 **CCOLAMD**（将最新变量压向根节点以缩小后续受影响子树）、**部分状态更新**（回代差值低于阈值即截断传播）与**流式重线性化（Fluid Relinearization）**（仅对偏离线性化点超阈值的变量所在团重新提取原因子并局部消元），实现无需周期性全局批处理的精确增量非线性优化。

4. **李群与一般流形上的局部坐标优化（Ch. 6）**
   - 对 $SO(2)/SO(3)/SE(2)/SE(3)$ 等非欧流形，直接向量加法会破坏正交性且维度冗余；通过 hat 算子 $\hat{\cdot}: \mathbb{R}^d \to \mathfrak{g}$ 与指数映射构建流形邻域局部更新 $a \oplus \xi = a \cdot \exp(\hat{\xi})$（Eq. 6.33，其中 $SO(3)$ 具有闭式 Rodrigues 公式 Eq. 6.8），在切空间局部坐标 $\xi \in \mathbb{R}^d$ 上求解线性化雅可比后再映射回流形。
   - 对非群结构的一般流形 $M$（如 3D 单位方向球面 $S^2$）定义光滑**收缩（Retraction）** $R_a(\xi)$（例如对 $S^2$ 用 QR 构造切平面基 $B_p$ 并归一化 $R_p(\xi) = \frac{p + B_p \xi}{\|p + B_p \xi\|}$，Eq. 6.38），或在 $SE(3)$ 上采用满足一阶相容性的高效近似收缩 $R_T(\xi)$（Eq. 6.39）。

### 方法流程图

```mermaid
flowchart LR
    OBS["多传感器观测与先验 (里程计/视觉/IMU/GPS)"] --> FG["构造二分因子图 F = (U, V, E)"]
    FG --> MANI["流形局部坐标线性化 (李群 exp / Retraction + 白化 A, b)"]
    MANI --> ORDER["稀疏消元排序 (Block COLAMD / CCOLAMD / Nested Dissection)"]
    ORDER --> ELIM["部分 QR 变量消元 -> 弦贝叶斯网络 / 上三角矩阵 R"]
    ELIM --> BT["构建/局部编辑 Bayes Tree (仅重算受影响路径至根节点)"]
    BT --> UPD["部分回代更新状态 X ⊕ ξ 与流式重线性化检测"]
    UPD -- "偏离超阈值或新测量到达" --> MANI
    UPD -- "收敛" --> MAP["输出全局 MAP 轨迹与地图估计"]
```

## 评测与指标

本专著以理论推导与代表性基准/实车系统案例交叉验证核心机制（非单一基准刷榜论文），正文直接给出的关键量化对比与工程指标如下：

| 实验/案例场景 | 对比维度 | 量化指标与原文结论 |
|---|---|---|
| 消元复杂度玩具 SLAM（5 变量，Eq. 4.4–4.5） | 多波前稀疏 QR vs 稠密 QR | 浮点运算量从 **5,072 flops** 降至 **1,156 flops**（仅 22.8%） |
| 2D 模拟 SLAM（95 位姿 + 24 地标，333 未知量 / 1,126 标量观测，Fig. 4.2） | 位姿优先顺序 vs COLAMD 排序 | $R$ 非零元从 **9,399** 降至 **4,168**（减半以上，解完全一致） |
| 更大规模 SLAM 稀疏对比（Fig. 4.4–4.7） | 朴素 Schur 补 vs 标量 AMD vs 块级 AMD | $R$ 非零元由约 **280 万** 降至约 **25 万**（标量 AMD），再借助因子图块语义进一步降至约 **13 万**（**20× 压缩**） |
| 平面图嵌套剖分（$n = 25,000$ 节点块世界，Fig. 4.9） | 分隔符与子图规模 | 首层分隔符仅 $|C| = 108$（$\sqrt{n} = 158.1$），渐近复杂度从 $O(n^3)$ 降至 **$O(n^{1.5})$** |
| 长期建图稀疏化（GLC，27 次户外建图 / 15 个月，Fig. 7.7） | 节点与因子规模压缩 | 节点数与因子数均实现 **>10× 缩减**并保持统计一致性 |
| 多模态非高斯 SLAM（mmiSAM，Fig. 7.6） | 模糊数据关联处理 | 在含上千变量且理论模态数超过 **$2^{400}$** 的歧义场景下仍具可行性 |
| DARPA ASPN 3.3 km 城市多传感器导航（Fig. 7.10） | 1 步短时平滑 vs 并发平滑建图（CSM） | 3D 位置 RMS 误差从 **19.19 m** 降至 **5.85 m** |

## 与其他工作对比

- **对比 EKF / 滤波派方法**：经典 EKF 在每步边际化旧状态并永久固定线性化点，应用于本质非线性的 SLAM 时会产生**不可逆的线性化误差与一致性退化**；因子图平滑（SAM / iSAM2）保留历史关联或仅按阈值流式重线性化，在 KITTI 视觉惯性导航（Fig. 7.1）中几乎达到全量批量优化精度。
- **对比早期矩阵更新派 iSAM1 与直接库（CHOLMOD / SuiteSparseQR）**：iSAM1 基于 Givens 旋转维护扁平矩阵 $R$，必须周期性停顿做全局批重排与重线性化；iSAM2 通过 **Bayes Tree** 将图模型与矩阵分解融合，天然支持局部重线性化与局部回代更新。
- **对比开源非线性后端（GTSAM / g2o / Ceres）**：专著以 GTSAM 为主要思想载体（`g2o` 使用“超图”概念），同样强调鲁棒核函数（Huber / Pseudo-Huber）与块稀疏线性求解器在实车异常值抑制中的必要性；同分组工作可参见 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)。

## 常见误区或局限

1. **误将“消元顺序”等同于改变最优解**：在无舍入误差下，任意变量消元顺序都给出完全相同的 MAP 估计与边缘分布；排序（COLAMD / ND）改变的是**图拓扑、填充量（Fill-in）与分解速度**，而非目标函数最优值。
2. **在固定滞后平滑中“直接删掉旧位姿”而非边际化**：Ch. 5.3.2 明确指出，直接丢弃因子图中的最早位姿会彻底损失历史观测信息；正确做法是先通过消元得到条件密度，再作为叶子节点安全边际化（或转为先验因子保留信息）。
3. **误以为标准 Gauss-Newton / Dogleg 适用欠约束系统**：Ch. 2.5.4 指出 GN 与 Powell's Dogleg 要求雅可比 $A$ 列满秩（$A^\top A$ 可逆）；若出现单目尺度/深度退化等欠约束情况，必须切换为带阻尼的 Levenberg-Marquardt 或补充先验因子。
4. **纯最小二乘对数据关联外点敏感**：高斯假设下的二次代价会被错误闭环严重拉偏；工程中需结合鲁棒代价函数（Pseudo-Huber 等）、离散开关变量（Switchable Constraints / Max-Mixture）或非高斯 Bayes Tree（mmiSAM）。

## 结论

**《Factor Graphs for Robot Perception》构建了从二分因子图建模、稀疏变量消元与排序、Bayes Tree 增量平滑（iSAM2）到李群流形局部坐标优化的完整理论与算法闭环，是现代机器人状态估计与 SLAM 后端工程的标准基石。**

- **设计感知后端时优先显式构建块级因子图**：利用位姿/地标块结构执行 Block-AMD 或 CCOLAMD 排序，相比标量级矩阵处理或朴素 Schur 补可显著减少非零元填充（典型案例达 20×）。
- **在线实时建图采用 Bayes Tree 增量更新而非周期性全量批重算**：借助局部受影响路径重消元与流式重线性化，兼顾全局一致性与实时性，避免 EKF 式不可逆线性化漂移。
- **长期运行系统必须叠加图稀疏化与抗外点机制**：单纯增量平滑随时间仍会无界增长且受闭环外点威胁，生产部署应结合固定滞后平滑（SRIS）、GLC 图稀疏化或鲁棒核函数（Pseudo-Huber）。
- **涉及 3D 姿态/旋转状态时严禁直接欧氏向量加减**：统一通过李代数切空间坐标 $\xi$ 与指数映射或一致收缩 $R_a(\xi)$ 构造局部雅可比与状态更新，彻底规避欧拉角奇异与正交约束破坏。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_ref_dcea01617aaebb6971e4_factor-graphs-for-robot-perception.md`](../../sources/papers/rcl_awesome_wam_ref_dcea01617aaebb6971e4_factor-graphs-for-robot-perception.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文（CMU 主页 / FnT Robotics 2017）](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.html)
