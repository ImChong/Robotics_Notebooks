# MuJoCo 官方文档通读（Overview / Computation 全章 + Modeling 选读）

> 来源归档（原文通读 · 支撑 `wiki/entities/mujoco.md`）

- **标题：** MuJoCo Documentation — Overview、Computation、Modeling（含 Tips and tricks）
- **作者：** Google DeepMind / MuJoCo 开发团队（Computation 章引 Todorov 2014 *Analytically-invertible dynamics with contacts and constraints*）
- **类型：** 官方文档 / physics-engine
- **原文：** <https://mujoco.readthedocs.io/en/stable/overview.html>、<https://mujoco.readthedocs.io/en/stable/computation/index.html>、<https://mujoco.readthedocs.io/en/stable/modeling.html>（2026-11 抓取 stable 版 HTML）
- **一句话说明：** MuJoCo 在广义坐标下做刚体动力学，把摩擦接触写成凸优化（软约束、可解析求逆），并提供可重配置的计算管线与 MJCF 建模语言。
- **核对范围：** Overview 章通读（特性、模型实例、示例、模型元素前半）；Computation 章通读（动力学框架、执行器、积分器、约束类型、凸约束求解 / 求解器 / 岛、solref-solimp 参数化、碰撞检测、休眠、管线 26 步、可微性、参考文献）；Modeling 章只读了 MJCF 机制导言、默认类机制、solver parameters、contact 参数起始部分与 Tips and tricks（性能 / 防滑 / 回差 / 恢复系数 / 约束精度），Actuators、Sensors、Delays、Composite、Deformable、URDF 扩展、MoCap、内存、Assets 等参考型小节仅浏览标题。未取得 Todorov, Erez, Tassa (IROS 2012) 论文全文（IEEE 付费墙，作者主页直链失效），页面对该论文的论述仅依赖文档自述与本库既有页面。
- **通读：** 未完成

## 核心摘录（均出自文档）

- **定位（Overview）：** MuJoCo = Multi-Joint dynamics with Contact；2021-10 由 DeepMind 收购并免费，2022-05 开源；C/C++ 库，模型用 MJCF（也可载入 URDF），运行时零内存分配，mjModel（常量）与 mjData（动态量）分离，便于多线程采样与有限差分。
- **关键特性：** 广义坐标 + 基于优化的接触（文档称 MuJoCo 首创此结合）；接触为凸优化而非 NP-hard 的 LCP/NCP；默认 Newton 求解器（二次收敛），另有 CG 与 PGS；腱（tendon）几何、通用执行器模型（含激活态，使动力学为三阶）；可重配置管线（mj_step、flag、回调）；约束岛与休眠；插件；GPU 后端 MJX（JAX）与 MuJoCo Warp。
- **接触模型（Computation）：** 去掉 LCP 的严格互补约束，换成代价项；允许软接触与互补性违反，文档认为软接触下这反而更物理；经验上典型机器人模型 10 次 PGS 扫描已与全局最优几乎无差别；连续时间定义，避开 Painleve 悖论；逆动力学唯一且可解析求解（问题对角化，比正向更易）。
- **动力学框架：** M v̇ = τ + Jᵀf − c；c 用 RNE、M 用 CRB、稀疏 LᵀDL 分解；约束力由广义 Gauss 最小约束原理给出的原问题 / 对偶问题决定，软度由对角正则 R 与参考加速度 a_ref（弹簧阻尼）控制。
- **积分器：** Euler（半隐式，含隐式关节阻尼）、implicit、implicitfast（推荐，成本近 Euler 但更稳）、RK4（适合近保守系统）、discrete（文档标注 2026-09 新增，仍在变动）。时间步长是最重要的单个参数。
- **求解器：** Newton（精确 Hessian + 增量 Cholesky）、CG、PGS（椭圆锥用自适应 QCQP 更新）；NoSlip 为抑制滑移的后处理，无单一优化目标，偶尔致不稳；热启动取「上一步力」与「零力」中代价较低者。
- **约束参数：** impedance d∈(0,1)（solimp）与 solref（timeconst, dampratio；timeconst 应不小于 2 倍步长，refsafe 强制）；负值为直接 (−刚度, −阻尼) 格式，可做完全弹性碰撞；摩擦维度与 friction loss 无位置残差，仅阻尼；d0=0 才使接触起始光滑可微。
- **碰撞：** 广相 sweep-and-prune + 中相 AABB BVH；仅凸 geom 碰撞，网格取凸包，非凸物体需分解成凸 geom 并并入同一 body；默认 nativeccd（GJK/EPA）与 multiccd；contype/conaffinity 位掩码过滤。
- **确定性与可微：** 管线是确定性的，但复现仅保证同版本同架构；解析导数仅覆盖光滑动力学，约束 / 接触导数靠 mjd_transitionFD、mjd_inverseFD 有限差分。
- **Tips（Modeling）：** 调速顺序：加大步长 → 选 implicitfast → dense/sparse Jacobian → 求解器与迭代数 → 碰撞（多线程、网格换基元、减面）→ 椭圆 / 金字塔锥；防滑：检查力是否在摩擦锥内、condim 4/6、椭圆锥加大 impratio、必要时 noslip_iterations 1-3；高频抖动：减步长 / 加 armature / 用隐式积分器；约束尺度异常时开 fwdinv 诊断、diagexact 修正。
- **管线语义陷阱：** 管线完全命令式；改 qpos 后须手动调 mj_forward 等才能让笛卡尔量一致；mj_step 后 mjData 里的量对应上一状态。

## 对 wiki 的映射

- [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)
