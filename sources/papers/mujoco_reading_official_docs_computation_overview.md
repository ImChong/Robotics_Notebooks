# MuJoCo 官方文档通读（Overview / Computation / Modeling 三章）

> 来源归档（原文通读 · 支撑 `wiki/entities/mujoco.md`）

- **标题：** MuJoCo Documentation — Overview、Computation、Modeling（含 Tips and tricks）
- **作者：** Google DeepMind / MuJoCo 开发团队（Computation 章引 Todorov 2014 *Analytically-invertible dynamics with contacts and constraints*）
- **类型：** 官方文档 / physics-engine
- **原文：** <https://mujoco.readthedocs.io/en/stable/overview.html>、<https://mujoco.readthedocs.io/en/stable/computation/index.html>、<https://mujoco.readthedocs.io/en/stable/modeling.html>（2026-11 抓取 stable 版 HTML）
- **一句话说明：** MuJoCo 在广义坐标下做刚体动力学，把摩擦接触写成凸优化（软约束、可解析求逆），并提供可重配置的计算管线与 MJCF 建模语言。
- **核对范围：** Overview 章通读（特性、模型实例、示例、模型元素前半）；Computation 章通读（动力学框架、执行器、积分器、约束类型、凸约束求解 / 求解器 / 岛、solref-solimp 参数化、碰撞检测、休眠、管线 26 步、可微性、参考文献）；Modeling 章（2026-11 抓取 stable 版 modeling.html 全文，转纯文本后逐节读完）通读全部小节：Introduction、Procedural Modeling、MJCF 机制（运动学树、默认类、坐标系、朝向）、Solver parameters（impedance / reference / friction）、Contact parameters 与 override、User parameters、Solver settings、Actuators（group disable、shortcuts、force limits、length range、stateful / activation limits、muscles）、Sensors、Delays、Cameras、Composite、Deformable、Including files、Naming、URDF 扩展、MoCap、Memory allocation、Assets by reference、Tips and tricks 全部五节。局限：Modeling 章中以链接形式指向的其他章节（XML Reference 全部元素 / 属性表、API Reference、Model Editing / mjSpec、MJX、MJZ / OpenUSD 等）未逐条读，仅就各章被 Modeling 引用的语义作摘录；muscles 小节的 FLV 曲线图与 MATLAB 脚本、各小节图示 / 示例模型视频未查看。未取得 Todorov, Erez, Tassa (IROS 2012) 论文全文（IEEE 付费墙，作者主页直链失效），页面对该论文的论述仅依赖文档自述与本库既有页面。
- **通读：** 已完成

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

## Modeling 章补充笔记（2026-11 通读 stable 版 modeling.html）

- **MJCF 机制：** 运动学树由嵌套 body 构成（区别于 URDF 的 link + joint）；joint 的作用是在父子间「创造自由度」，body 内无 joint 即与父体焊死，一个 body 可含多个 joint（如两个滑动 + 一个转动建平面运动）。默认类类似 CSS：子类继承父类，元素创建时从当前激活类的「哑元素」拷贝属性；激活类优先级 = 元素自身 class > 最近祖先 body 的 childclass > 顶层类；已定义的属性不能再回到「未定义」态。朝向可用 quat（首选，无转换）、axisangle、euler（顺序由 compiler/eulerseq 全模型统一）、xyaxes、zaxis，内部一律转单位四元数。
- **Solver parameters：** 约束动力学近似为 a + d·(b v + k r) = (1−d)·a_unc；impedance d 被夹在 [0.0001, 0.9999]，solimp 默认 (0.9, 0.95, 0.001, 0.5, 2) 描述 d 随违反量 r 的 sigmoid 变化；solref 默认 (0.02, 1)，正值为 (timeconst, dampratio)，负值为直接 (−刚度, −阻尼)，直接格式可令阻尼为 0 得到完全弹性碰撞（文档示例 solref="-1000 0"）；恒定 d 下静止穿透深度公式见文档式 (2)(3) 后。摩擦维度与 friction loss 的 r≡0：阻抗恒为 d0，solref 的 dampratio 被忽略，动力学是一阶指数衰减（无弹簧）。想要完全可微的接触/限位应设 d0=0。
- **接触参数混合规则：** 两 geom 参数不同时按 priority 与 solmix 合成——condim 取高优先级者，同级取较大；friction 取高优先级者，同级逐元素取大；margin / gap 直接相加（不看 priority）；solref/solimp 同优先级时按 solmix 加权平均，但任一 solref 为非正（直接格式）时改为逐元素取小。各向异性摩擦只能在显式 pair 中给，因为单个 geom 无从知道接触切平面朝向。Contact override（flag override + o_margin / o_solref / o_solimp）只作用于接触，可做优化中的 continuation（早期让接触「远距离起作用」）。
- **Solver settings：** Jacobian auto 在自由度 ≤60 时取 dense，否则 sparse；Newton 通常约 5 次迭代、极少超 20，宜配 1e-10 级容差；大模型 + 椭圆锥 + 大量滑动接触时 Hessian 更新多、或排序致高 fill-in，改用 CG；PGS 适合自由度多于约束、可容忍不精确解的场景，且顺序更新会破坏对称性，Newton/CG 并行更新保持对称；NoSlip 是接在主求解器后的 PGS 后处理，只更新摩擦维度并忽略正则，文档明言它「不再求解良定义的优化问题」，复杂多接触下见过不稳定。
- **Actuators：** shortcut（motor、position、velocity、intvelocity、orientation、pid、damper、cylinder、muscle、adhesion、dcmotor）都只是解析时展开成同一种 general 执行器，不是独立模型元素；shortcut 参数只从同 shortcut 或增益类型匹配的 general 默认继承。actuatorgroupdisable / mjOption.disableactuator 可运行时按 group 关停执行器而不改 nu。力限幅有三层：ctrlrange（控制量）、forcerange（执行器输出）、joint/tendon 的 actuatorfrcrange（变换后合力，多执行器共用一关节时应在此限幅，对应传感器 jointactuatorfrc，而 actuatorfrc 报限幅前的值）。actuator_lengthrange 是近似量，可显式给、拷贝关节/腱限位或编译期用「关闭接触/重力等后的阻尼仿真」自动求，肌肉常见的收敛错误多因没写关节限位。intvelocity = 积分器 + 位置伺服，活动态即位置设定点，需 actrange 限幅防止积分越过关节限位（actrange 恒用弧度）；文档指出真实机器人中带速度语义的执行器多为这种形式，纯速度反馈往往不稳。
- **Muscles：** 肌腱总长 = 肌肉长 L_M + 恒长腱 L_T（腱不可伸长），由 lengthrange 与 range（默认缩放范围 0.75–1.05）解出 L_0 与 L_T；力 = −FLV(L,V,act)·F_0，F_0 未给时为 scale/acc0；激活是一阶滤波，timeconst 默认 (0.01, 0.04)，有效时间常数按 Millard 2013 随 act 变化，tausmooth>0 时用 sigmoid 平滑切换。与 OpenSim 对比：激活动力学一致；不建模腱弹性、羽状角，包裹物体仅球和无限圆柱且两个包裹体之间须隔一个固定 site。文档提醒：工作范围对行为的影响大于 FLV 曲线形状，不宜过度相信模型参数。
- **Sensors / Delays：** 传感器输出只写入 sensordata，不影响仿真。通用属性 noise（仅存储标准差，不影响仿真）、cutoff、nsample（历史环形缓冲）、interp（zoh / linear / cubic）、delay、interval = (period, phase)。delay 与 interval 须 nsample>0；读在写之前，故最小有效延迟为一个时间步；delay > timestep·nsample 会非因果外推且不报错；zoh 下延迟对小变化不连续，linear/cubic 可令延迟连续 / 可微。执行器同样支持 nsample/delay/interp。
- **Cameras：** 相机看向自身 −Z，+X 右、+Y 上；默认透视，projection=orthographic 时 fovy 为长度；可用 focal、principal 表达轴向不等焦距与主点偏移，此时须同时给 sensorsize 与 resolution。
- **Composite / Deformable：** composite 现仅保留 cable 类型（单一运动学树、球关节串联，因而严格不可伸长，可用大步长，依赖 mujoco.elasticity.cable 插件，含扭转 / 弯曲刚度，基于 Bishop 无扭转标架）；其余类型弃用，重复物体改用 replicate，软体改用 flex。MuJoCo 3.0 起的 flex 是由无质量可伸缩元素（1D 胶囊 / 2D 三角形 / 3D 四面体）连接 body 顶点的真实可变形体，接触力按权重分配给至多 8 个 body；形变保持有两种选择：边方程式软等式约束（集总刚度，允许大步长）或分片线性有限元式连续体（可分别设剪切与体积刚度，基于 Saint Venant-Kirchhoff 超弹性）；flexcomp 提供 grid / mesh 等自动生成，另有 trilinear（8 节点）与 quadratic（27 节点）背景网格参数化以减少自由度；全顶点属同一 body 的刚性 flex 可当作非凸网格碰撞使用。
- **Include / Naming：** include 把被包含文件顶层元素的子元素替换进主文件，允许重复的 MJCF 段（如多个 worldbody 自动合并）；全局设置（如 compiler/angle）取最后一次定义，混用度/弧度模型会出错；同类元素名须唯一，同一文件重复 include 是解析错误。名字仅在被引用（如腱的 site）、可视化标注或 mj_name2id / mj_id2name 时才需要，文档建议其余情况不命名，可读性用 XML 注释实现。
- **URDF：** MuJoCo 可载入 URDF，并识别顶层 robot 下的自定义 mujoco 元素（含 compiler、option、size）；strippath、angle、fusestatic、discardvisual 的默认值与 MJCF 不同；URDF 不做 schema 校验，拼错的属性会被静默忽略；建议转换为 MJCF 后用 include 补充 MuJoCo 专有信息。
- **MoCap：** mocap body 是世界的静态子节点（mocap="true"），通过每步写 mjData.mocap_pos / mocap_quat 驱动；对物理引擎它们是固定的，接触缺少相对速度分量，接触力偏小且可能不稳；更好的做法是再加一个普通 body，用 weld 等式约束连到 mocap body，由 solref/solimp 软硬度决定与接触约束的取舍。mocap body 及其无自由度后代自成焊接组，mocap 与静态几何或彼此之间不产生接触，启用休眠时 mocap 视为清醒。
- **Memory / Assets：** mjData 在 mj_makeData 时预分配，分 buffer（定长）与 arena（动态，接触和约束从头部、栈从尾部分配），由 size/memory 一个数控制；arena 不足时：接触超限则不再加新接触（警告）；约束超限则本步关闭约束求解器（警告）；Newton 构造 Hessian 内存不足触发 mjWARN_CNSTRFULL 并回退到 CG；栈不足为硬错误。可用 mjData.maxuse_arena 看峰值，文档设计目标约 50% 利用率；CG 最省内存、PGS 最费。Assets by reference：纹理可只存路径不存像素（compiler passthrough / texture embed），tex_adr==-1 表示按引用，mj_loadTexture 读取，mj_embedAssets 可转回内嵌；需要变换的纹理仍内嵌。
- **Tips 补充：** 回差（backlash）用两个同轴 hinge 建模——J1 带 armature 受驱动，J2 小范围限位，armature 缺失则关节空间惯量矩阵奇异；也可改用 J2=0 的软等式约束并让阻抗随违反量增大，过渡更柔和。恢复系数见上文直接 solref 格式。约束精度：约束阻抗用的对角近似在编译期按 qpos0 一次计算，在各向异性惯量、复杂运动链或远离 qpos0 时可能失准，表现为 badqacc、穿透过大、滑移，用 fwdinv 诊断、diagexact 在运行时精确计算。静止「慢滑移」：椭圆锥下摩擦维度纯阻尼无位置残差，持续切向载荷会有非零稳态滑移速度，文档称这是预期行为，只能用 impratio 与 noslip_iterations 抑制，不保证精确粘着。

## 对 wiki 的映射

- [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)
