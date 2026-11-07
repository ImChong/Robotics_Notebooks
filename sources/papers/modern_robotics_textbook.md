# modern_robotics_textbook

> 来源归档（ingest）

- **标题：** Modern Robotics: Mechanics, Planning, and Control
- **类型：** book / textbook
- **作者：** Kevin M. Lynch (Northwestern University) & Frank C. Park (Seoul National University)
- **出版：** Cambridge University Press, 2017
- **ISBN：** 9781107156302
- **官方资源：** https://hades.mech.northwestern.edu/index.php/Modern_Robotics
- **PDF 镜像：** https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
- **配套 Coursera 专项课程：** Modern Robotics 6 门课
- **入库日期：** 2026-04-30（首次）；2026-11-07（原文通读复核）
- **核对版本：** 官方 PDF 镜像（Lynch & Park, 2017-05-03 preprint）
- **核对范围：** 下载官方 PDF 并转文本，读完：目录与前言结构；Ch2 摘要（Grübler 公式、完整/非完整约束）；Ch3 §3.2 旋转矩阵 / 角速度 / 指数坐标 / 矩阵对数、§3.3–3.5 齐次变换·旋量·指数坐标·汇总表；Ch4 §4.1 PoE（空间式）、§4.2 URDF、摘要与附注；Ch5 §5.1 空间/体雅可比及解析雅可比（Example 5.5）、§5.2 静力学 τ=JᵀF、§5.3 奇异分析（Case I–V）、§5.4 可操作度椭球与三种标量度量；Ch6 §6.1 解析 IK（PUMA / Stanford）、§6.2 牛顿–拉夫森数值 IK、§6.3 逆速度运动学（质量矩阵加权伪逆）、摘要与附注；Ch8 §8.7 约束动力学与摘要（Lagrange / Newton–Euler 算法、M(θ)、任务空间 Λ(θ)）；Ch12 §12.2 库仑摩擦与摩擦锥、wrench cone、力封闭与摘要；附录 B（欧拉角 / 单位四元数）、附录 C（Denavit–Hartenberg、PoE 与 DH 关系）。**未读：** Ch7 闭链、Ch9 轨迹、Ch10 规划、Ch11 控制、Ch13 轮式机器人、Ch8 其余小节（拉格朗日推导细节、电机 / 齿轮 / 摩擦）、Ch12 其余小节（接触运动学、形封闭、操作）、各章习题与配套软件代码。
- **通读：** 已完成
- **沉淀到 wiki：** 是 → [`wiki/entities/modern-robotics-book.md`](../../wiki/entities/modern-robotics-book.md)

## 一句话说明

经典的现代机器人学教材，独特之处是**全程使用李群（Lie group）/ 螺旋理论（screw theory）**作为统一数学语言来描述刚体运动、运动学、动力学与控制，覆盖从配置空间到全身控制、抓取、移动机器人的完整体系。

## 为什么值得保留

1. **唯一系统教完螺旋理论的本科级教材**：传统教材（Craig、Spong、Siciliano）多用 D-H 参数；本书用 PoE（Product of Exponentials）公式，更几何、更简洁。
2. **配套 Coursera 专项课程 + Python/MATLAB/Mathematica 库**：可以代码验证。
3. **覆盖广度**：13 章覆盖配置空间 → 运动学 → 动力学 → 轨迹 → 规划 → 控制 → 抓取 → 移动机器人，是 RL/LLM 时代之前传统机器人学的「最大公约数」教材。
4. **官方 PDF 永久免费**：作者直接提供，无需购买。

## 章节目录

| 章节 | 标题 | 主要主题 |
|------|------|---------|
| Ch 1 | Preview | 全书路线图 |
| Ch 2 | Configuration Space | 自由度、C-space、拓扑、约束 |
| Ch 3 | **Rigid-Body Motions** | SO(3)/SE(3)、旋转矩阵、齐次变换、**twist / wrench / 指数坐标** |
| Ch 4 | **Forward Kinematics** | **PoE 公式（Product of Exponentials）**、空间形式 vs 物体形式 |
| Ch 5 | **Velocity Kinematics and Statics** | 雅可比矩阵（空间/物体）、奇异性、可操作度椭球 |
| Ch 6 | Inverse Kinematics | 解析解、数值解（牛顿-拉夫森 §6.2.1–6.2.2；**Example 6.1** 平面 2R 三步收敛） |
| Ch 7 | Kinematics of Closed Chains | 并联机构、Stewart 平台、Grübler 公式 |
| Ch 8 | **Dynamics of Open Chains** | Lagrange / Newton-Euler、惯性矩阵 M(q)、动力学方程 |
| Ch 9 | Trajectory Generation | 多项式时间律、最短路径、最优时间轨迹 |
| Ch 10 | Motion Planning | C-space 障碍、网格、采样规划（PRM/RRT）、虚拟势场 |
| Ch 11 | **Robot Control** | 关节空间 PD/PID、计算力矩控制、任务空间控制、力控、阻抗控制 |
| Ch 12 | Grasping and Manipulation | 接触模型、力封闭、形封闭、操作平面 |
| Ch 13 | Wheeled Mobile Robots | 全向 / 非完整、里程计、控制 |

加粗章节是与本知识库现有 wiki 页面**直接对应**的核心章节。

## 关键术语对应

教材主要术语已在对应 wiki 实体页 [`wiki/entities/modern-robotics-book.md`](../../wiki/entities/modern-robotics-book.md) 的"章节地图"中给出全部映射。本 source 文件不再单独列出具体 wiki 链接，避免双重维护。

## 与其他教材的关系

| 维度 | Modern Robotics (Lynch-Park) | Craig | Siciliano | Featherstone |
|------|------------------------------|-------|-----------|--------------|
| 数学语言 | **李群 / 螺旋理论** | D-H 参数 | 兼有 | 空间向量代数 |
| 本科可读 | ✅ | ✅ | △ | ❌（研究级） |
| 涵盖动力学 | 中等 | 浅 | 深 | 极深（递推算法） |
| 覆盖控制 | 中等 | 中等 | 深 | ❌ |
| 涵盖规划/抓取 | ✅ | ❌ | △ | ❌ |
| 配套代码 | ✅ Python/MATLAB | ❌ | △ | ❌ |
| 在线视频 | ✅ Coursera | ❌ | ❌ | ❌ |

## Ch 6 数值 IK 与 Newton–Raphson（§6.2，一手算例）

- **标量 NR：** $\theta_{k+1}=\theta_k-(\partial g/\partial\theta)^{-1}g(\theta_k)$（式 6.2.1 段）
- **向量 IK：** $\Delta\theta=J^\dagger(x_d-f(\theta))$；SE(3) 用 $V_b=\log(T_{bs}^{-1}T_{sd})$ + $J_b$
- **Example 6.1：** 2R 平面臂，链长 1 m，目标 $(30^\circ,90^\circ)$，初值 $(0^\circ,30^\circ)$，**3 次迭代**收敛
- **wiki 映射：** [`newtons-method.md`](../../wiki/methods/newtons-method.md)、[`inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)、[`newton_raphson_method_primary_refs.md`](newton_raphson_method_primary_refs.md)

## 2026-11-07 通读复核补充（出自 PDF 原文）

- **twist / wrench 排序：** 本书 twist 写 $\mathcal V=(\omega,v)$（角速度在前）；wrench 写 $\mathcal F=(m,f)$（力矩在前）。空间惯量 $\mathcal G_b=\mathrm{diag}(\mathcal I_b,\,mI)$（质心系），换系 $\mathcal G_a=[\mathrm{Ad}_{T_{ba}}]^\top\mathcal G_b[\mathrm{Ad}_{T_{ba}}]$（§8 摘要）。
- **DH 约定（附录 C）：** 本书 DH 用 $T_{i-1,i}=\mathrm{Rot}(\hat x,\alpha_{i-1})\mathrm{Trans}(\hat x,a_{i-1})\mathrm{Trans}(\hat z,d_i)\mathrm{Rot}(\hat z,\phi_i)$（帧 $\{i\}$ 绑在关节 $i$ 的 $\hat z$ 轴上，参数下标 $i-1$ 对应前一连杆），并明说另有以 $\hat x$ 为关节轴的其他约定；书中给出 $T_{i-1,i}=M_ie^{[A_i]\theta_i}$（式 C.5–C.7），即 PoE 可由 DH 导出。教材同时指出 PoE 无需连杆帧、统一处理转动 / 移动关节，DH 的唯一优势是参数最少（4 个）。
- **解的个数：** 一般 6R 开链 IK 最多 16 组解（Lee & Liang；Raghavan & Roth，§6.7）；带偏置的 PUMA 型臂位置逆解有 4 组（左 / 右肩 × 肘上 / 肘下，§6.1.1），腕部角由 ZYX 欧拉角解出。
- **数值 IK：** $\Delta\theta=J^\dagger(\theta^i)\,e$，SE(3) 目标用体旋量 $\mathcal V_b=\log(T_{sb}^{-1}T_{sd})$ 作 $e$；奇异时 $J^{-1}$ 不存在才改用 Moore–Penrose 伪逆。书中**没有**阻尼最小二乘（DLS）内容。
- **可操作度（§5.4）：** $A=JJ^\top$；$\mu_1=\sqrt{\lambda_{\max}/\lambda_{\min}}$，$\mu_2=\lambda_{\max}/\lambda_{\min}$（条件数），$\mu_3=\sqrt{\det A}$（体积，越大越好）；线速度与角速度因量纲不同应分开画两个椭球；力椭球与可操作度椭球主轴重合、半轴互为倒数。
- **奇异性（§5.3）：** 空间 / 体雅可比秩相同，奇异性与固定系 / 末端系选择无关；常见情形：两转动轴共线、三平行共面轴、四轴共点（如腕心在肩轴正上方）、四轴共面、六轴交于一线。
- **静力学（§5.2）：** $\tau=J^\top(\theta)F$；$n<6$ 时有 $\mathrm{Null}(J^\top)$ 方向机器人不能主动施力但可被动抵抗外力。
- **单位四元数在附录 B.3**（不是 Ch 3 正文）：$q=(\cos\frac\theta2,\hat\omega\sin\frac\theta2)$，$q$ 与 $-q$ 对应同一 $R$；指数坐标的对数在 $\sin\theta\to0$ 数值敏感，三参数表示必有奇异（§3.2.3.3 脚注）。
- **摩擦锥（§12.2）：** $\sqrt{f_x^2+f_y^2}\le\mu f_z$，$f_z\ge0$，半角 $\alpha=\tan^{-1}\mu$；内接棱锥低估摩擦力、外接高估；接触力转为 wrench $F=([p]f,f)$ 后摩擦锥成为 wrench cone，多接触取正张成。
- **接触动力学范围（§12.5 附注）：** 书中明说基于线性 / 非线性互补问题（Stewart、Pang & Trinkle、Trinkle 等）的摩擦接触刚体仿真「not covered in this chapter」；Ch 12 只含接触运动学、库仑摩擦、形封闭 / 力封闭与操作。
- **约束动力学（§8.7）：** $\tau=M\ddot\theta+h+A^\top\lambda$，$A\dot\theta=0$；投影 $P=I-A^\top(AM^{-1}A^\top)^{-1}AM^{-1}$。本书动力学针对**固定基座开链**，无浮动基章节。

## 当前提炼状态

- [x] 章节目录与教材定位
- [x] 与本项目已有 wiki 页面的术语对照
- [x] Ch 6.2 Newton–Raphson + Example 6.1 算例摘录
- [ ] 后续可专题深挖：PoE Forward Kinematics 形式化、Wheeled Mobile Robots 单页（当前 wiki 未覆盖）、传统采样规划（RRT/PRM）单页
