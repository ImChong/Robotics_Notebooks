# 机器人建模与数字孪生（CAD → 仿真 → 数字孪生）

**首屏导读**：

- **为谁**：想把机器人从机械 CAD 装配体一路做成物理可信的仿真资产、跨引擎一致模型与可联调数字孪生的人。
- **路径**：Stage 0–5 全链主干（建模分层 → CAD 准备 → 描述格式管线 → 动力学/闭链/接触 → 仿真三层与场景 → 标定与 V&V）+ Stage 6（SIL/HIL 在环闭环；产线多机 OpenUSD 孪生按项目选读）。
- **纵深路线**：[运动控制主路线](motion-control.md) 与 [Sim2Real](depth-sim2real.md) / [Real2Sim](depth-real2sim.md) / [整机硬件设计](depth-humanoid-hardware-design.md) 共用的正向工程建模与孪生底座。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>建模目标与孪生分层<br/><em>三类物理题 · Model/Shadow/Twin</em>"]
  S1["<b>Stage 1</b><br/>面向仿真的 CAD 准备<br/><em>B-rep/坐标零位 · 简化与质量补全</em>"]
  S2["<b>Stage 2</b><br/>描述格式与资产管线<br/><em>URDF / MJCF / SDF / OpenUSD</em>"]
  S3["<b>Stage 3</b><br/>动力学、闭链与接触<br/><em>惯量/armature · 并联/执行器/摩擦</em>"]
  S4["<b>Stage 4</b><br/>仿真三层、传感与拼装<br/><em>物理/平台/学习 · SimReady 场景</em>"]
  S5["<b>Stage 5</b><br/>模型验证、标定与 V&V<br/><em>五层惯量核查 · SysID · Sim2Sim</em>"]
  S6["<b>Stage 6</b><br/>在环验证与数字孪生闭环<br/><em>SIL → PIL → HIL · 双向同步</em>"]
  S6B["<b>选读分支</b><br/>产线/多机 OpenUSD 孪生<br/><em>Omniverse · 多机与场景协同</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5
  S5 -->|控制/RL/Sim2Real 上游| S6
  S6 -. "按项目选读" .-> S6B
  S5 -. "参数回写 / 结构重模" .-> S3
  S6 -. "实机漂移再标定" .-> S5

  classDef stage fill:#142a3a,stroke:#16a085,stroke-width:2px,color:#fff
  classDef opt fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,stroke-dasharray: 4 3,color:#e2e8f0
  class S0,S1,S2,S3,S4,S5,S6 stage
  class S6B opt
```

## 这条路径怎么用

- 目标读者是**负责机器人仿真建模、资产管线或数字孪生联调的工程师与研究者**：不论你是从机械结构工程师手里接过 STEP 装配体，还是接手开源机器人的 URDF/MJCF 准备上线训练、做 MPC/WBC 或接 ROS 2 在环联调，都能从 Stage 0 顺着把“为什么仿真表现与真机对不上”从模型根部修透
- **Stage 0–5 是通用主干**：走完即可交付一套“坐标系与零位冻结、惯量物理一致、闭链与接触可信、跨引擎 Sim2Sim 对齐、经过真机标定回写”的可复用机器人仿真资产；**Stage 6** 把资产推进到 SIL → PIL → HIL 在环闭环与真机双向同步，其中「产线/多机 OpenUSD 孪生」可按项目需求选读，单机算法研究不强制展开
- 每个阶段包含前置知识、核心问题、推荐读什么、推荐做什么与学完输出什么；工具不绑定单一商业 CAD 或单个仿真器，核心概念可在 FreeCAD/SolidWorks 与 MuJoCo/Isaac Sim/Gazebo 之间直接迁移

**和其他纵深路线的明确边界：**
- **vs [整机硬件设计纵深](depth-humanoid-hardware-design.md)**：整机硬件交付的是**可制造、可上电的物理样机**（结构 FEA/模态、配电、EMC、通信总线），只在其 Stage 1/6 简述“导出 URDF/交付数字副本”；本路线直接消费其 CAD 装配体与实测参数，把“数字副本如何建模、验证、标定并演进为数字孪生”完整展开。传统工程图、加工公差与强度疲劳计算留给整机硬件纵深，不重复堆砌。
- **vs [Real2Sim 纵深](depth-real2sim.md)**：二者输入方向镜像——Real2Sim 是**逆向/感知驱动**（相机视频、点云、3DGS → 外部真实场景、交互物体、episode 孪生）；本路线是**正向/工程驱动**（CAD 装配体与硬件设计 → 机器人本体与受控工装 → 实时双向孪生）。二者在 **Stage 4 的 OpenUSD / SimReady 场景拼装** 处汇合（标定机器人本体 + Real2Sim 真实环境 = 完整仿真世界）。
- **vs [Sim2Real 纵深](depth-sim2real.md)**：Sim2Real 假设已有可用仿真模型，重点解决**策略分布迁移**（域随机化、特权蒸馏、RMA、视觉迁移、真机残差微调）；本路线解决**模型资产与孪生系统本身可不可信**，本路线 Stage 5 的标定资产与 V&V 报告正是 Sim2Real Stage 1 的上游真值。
- **vs [传统模型控制](depth-classical-control.md) / [主路线](motion-control.md)**：主路线 L1–L2 与传统控制教的是**拿模型之后怎么算正逆动力学、MPC 与 WBC**；本路线位于主路线 L1/L2 前后，讲清**模型从哪里来、为什么直接导出的文件不能直接用、以及如何校验**。
- **vs [具身测评纵深](depth-embodied-eval.md)**：具身测评判断**策略或具身大模型的能力边界与基准外推性**；本路线只对**仿真底座与数字孪生体本身的物理保真度（V&V）**负责。

---

## Stage 0 建模目标与数字孪生分层：先钉死“拿模型干什么”

**建模不是画得越细越好，数字孪生也不是 3D 展示大屏；在导入第一份 CAD 前，先钉死物理问题类型、保真度预算和孪生集成等级。**

### 前置知识
- 会看基本的三维坐标、质量与刚体运动概念
- 了解机器人从设计到仿真训练/真机部署的大致流程（主路线 L0 或硬件常识即可）

### 核心问题
- **你要回答哪一类物理问题？** 载荷与寿命验证（[Adams](../wiki/entities/adams.md) / MBD+FEA）、控制与策略高速采样（MuJoCo / Isaac Lab）、还是感知与系统集成（Isaac Sim / Gazebo）——选错工具会陷入“用 RL 引擎校零件疲劳”或“用工业 CAE 跑百万步强化学习”的死胡同
- **四层物理保真度投资排序**：几何/URDF → 刚体动力学 → 接触/摩擦 → 执行器，误差会被上层逐级放大，其中**几何/URDF 标定成本最低且收益最高**
- **Digital Model → Digital Shadow → Digital Twin 三级分界**：离线模型 vs 单向遥测回放/标定 vs 双向实时闭环，决定你需要准备到哪一层实时接口与版本管理
- **正向 CAD-to-Sim 与逆向 Real2Sim 如何对接**：分清自己建的是“图纸已知的机器人本体”，还是“感知扫描的未知外部世界”

### 推荐读什么
- [Adams vs MuJoCo vs Isaac Sim：按三类物理问题选型](../wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md) — 明确 MBD 载荷寿命、控制策略采样、感知系统集成三类工具边界
- [仿真物理保真度链路选型指南](../wiki/queries/simulation-physics-fidelity.md) 与 [物理保真度 ↔ sim2real gap](../wiki/concepts/physics-fidelity-sim2real-gap.md) — 四层保真度因果与投入优先级
- [机器人数字孪生（Digital Twin）](../wiki/concepts/digital-twin.md) — 概念主页：NASA 范式、Model / Shadow / Twin 分级与五层构成
- [Adams（ADAMS 多体动力学谱系）](../wiki/entities/adams.md) — 理解工业多体仿真与现代机器人物理引擎的同源与分工

### 推荐做什么
- 写一页「项目建模任务书」：明确目标用途（MPC/WBC、RL 训练、ROS 2 SIL、产线监控）、需要集成的孪生等级（Model / Shadow / Twin），以及允许删繁就简的部件清单
- 给自己的机器人列出四层保真度预算表，标出哪些层必须实测标定、哪些层可以暂用近似

### 学完输出什么
- 一份**建模目标与孪生分层定义表**，避免一开始就在紧固件网格面数或渲染贴图上浪费算力
- 能用一句话分清 Digital Model、Digital Shadow、Digital Twin 与 Real2Sim 的工程边界

---

## Stage 1 面向仿真的 CAD 准备：从制造图纸到可计算几何

**制造用的 CAD 追求可加工、不漏一颗螺钉；仿真用的模型追求坐标系统一、零位冻结、拓扑清晰和质量完整——直接导出 STEP 往往把几百万面片和负号陷阱一起带进引擎。**

### 前置知识
- Stage 0 内容
- 会打开至少一款 CAD 工具（如 [FreeCAD](../wiki/entities/freecad.md)、SolidWorks、Onshape）或理解装配体概念

### 核心问题
- **B-rep 实体与离散网格的本质差异**：STEP 存的是精确解析/NURBS 边界表示（适合加工和特征修改），仿真引擎几乎都消费离散三角网格（STL/OBJ/DAE）或基本体近似；曲面细分（tessellation）公差如何取舍
- **坐标系与三零位冻结**：遵循 REP-103 右手系（x 前、y 左、z 上，SI 米/千克/弧度），冻结 **CAD 装配零位、运动学算法零位、电机编码器零点** 三张对照表；对称镜像左右臂时为什么绝对不能机械地把 `axis` 一起取反
- **面向仿真的装配体清洗（De-featuring）**：移除螺纹、倒角、小螺钉、logo 凹凸等无关紧固件；合并同一刚性连杆下无相对运动的子零件（rigid sub-assembly）
- **补全 CAD 默认缺失的物理项**：线束、胶水、电池内部结构、外包壳常未建入结构 CAD 或密度写错，造成总重偏差与质心漂移
- **参数化脚本 CAD 与 AI 草稿能力边界**：用 Python/CadQuery 参数化建模与 Text-to-CAD 生成支架/夹具草稿时，如何校验 B-rep 合法性与物理尺寸

### 推荐读什么
- [Parametric CAD Workflow · 参数化 CAD 建模流程](../wiki/methods/parametric-cad-workflow.md) — 约束参数、特征再生与 STEP/STL 导出复核规范
- [FreeCAD](../wiki/entities/freecad.md) 与 [NVIDIA USD BRep / SMLib](../wiki/entities/nvidia-usd-brep-smlib.md) — 开源参数化 CAD 与开源 B-rep 内核到网格细分转换机制
- [文字生成 CAD（Text-to-CAD）](../wiki/concepts/text-to-cad.md) · [CAD Skills](../wiki/entities/cad-skills.md) — 参数化脚本生成与自然语言建草稿的适用范围
- [3D-as-Code、网格与机器人仿真描述](../wiki/comparisons/3d-as-code-to-robot-simulation.md) — 程序化建模、网格与物理资产三层边界
- [前沿模型 × 3D / CAD / 机器人（MIT CDFG Survey）](../wiki/overview/frontier-models-3d-cad-robotics-survey.md) — AI 辅助建 CAD 草稿与仿真资产的现实成熟度证据边界

### 推荐做什么
- 选一台开源机械臂或单腿 CAD 装配体，完成一次面向仿真的清洗：隐藏紧固件、按运动连杆合并子装配、按 REP-103 对齐 `base_link` 与各关节轴向，写出「CAD–算法–编码器零位对照表」
- 对每个连杆导出质量属性，对比 CAD 密度积分总重与 BOM 理论/实测总重，列出漏件清单（线束、驱动板、电池等）

### 学完输出什么
- 一套**清洗并按刚体分组的装配体导出包**（STEP 源 + 分连杆视觉网格 + 坐标系与零位冻结表）
- 一份**漏件质量与惯量核对清单**，确保下游描述文件不从错的基准起步

---

## Stage 2 描述格式与资产管线：URDF / MJCF / SDF / OpenUSD

**“格式能打开”不等于“语义没丢”。URDF、MJCF、SDFormat 与 OpenUSD 对闭链、角度单位、默认轴向、惯量符号和传感器的规定各不相同，工程上要建立“单真值源 + 可回归导出链”。**

### 前置知识
- Stage 1 内容（已冻结坐标系、零位与分连杆几何）
- 基础 XML/YAML/Python 阅读能力

### 核心问题
- **四大描述格式怎么选、怎么分工**：[URDF](../wiki/concepts/urdf-robot-description.md) 做 ROS/Pinocchio 运动树通用入口，MJCF 做 MuJoCo 接触/执行器/闭链原生体，SDFormat 做 Gazebo 世界与原生传感器描述，OpenUSD（`UsdPhysics`）做跨工具协作与大规模仿真/数字孪生容器
- **从 STEP 到 URDF/Xacro 的工程管线**：使用 [step2urdf](../wiki/entities/step2urdf.md)、[URDF-Studio](../wiki/entities/urdf-studio.md)、[StackForce](../wiki/entities/stackforce.md) 或 CAD 导出插件时，如何按模块拆 Xacro、分离 `visual` 与 `collision`、做凸分解（CoACD/VHACD）或基本体包络代理
- **OpenUSD 资产组织与协作**：Layer、Reference、Payload、Variant（LIVERPS）与 Instancing 如何让多工程师在不互相覆盖的前提下并行维护视觉、碰撞与传感器层
- **跨格式转换的五大高频失真点**：`UsdPhysics` 角度用**度**而 URDF/MJCF/SDF 用**弧度**；URDF 默认关节轴 `(1,0,0)` vs SDF `(0,0,1)`；URDF 省略 `<inertial>` 变零质量 vs OpenUSD 按水密度估算；URDF 惯量积负号约定；MuJoCo 对非凸 mesh 默认静默取凸包
- **现成开源机器人模型目录**：何时自己从头建，何时从公开模型库（[robot_descriptions.py](../wiki/entities/robot-descriptions-py.md) 等）起步并二次核验

### 推荐读什么
- [URDF vs MJCF vs SDFormat vs OpenUSD：机器人描述格式横向对比](../wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md) — **本阶段核心选型页**（逐条对齐一手官方规范）
- [URDF（统一机器人描述格式）](../wiki/concepts/urdf-robot-description.md) 与 [机器人模型描述与资产接入](../wiki/methods/robot-model-description-workflow.md) — `<link>`/`<joint>`/`<inertial>`/`visual`/`collision` 核心结构
- [轮式双臂人形 URDF 建模与工具链](../wiki/queries/wheeled-dual-arm-humanoid-urdf-workflow.md) — 从 CAD 零位、xacro 分层到跨仿真器转换的七步实操细节（含惯性 `rpy` 与 `mimic` 陷阱）
- [step2urdf](../wiki/entities/step2urdf.md) · [URDF-Studio](../wiki/entities/urdf-studio.md) · [StackForce](../wiki/entities/stackforce.md) — 浏览器端与工作站级 CAD→URDF/MJCF/USD 转换工具链
- [NVIDIA Learn OpenUSD](../wiki/entities/nvidia-learn-openusd.md) — OpenUSD Stage / LIVERPS / Asset Structure / Validation 官方自学路径
- [机器人描述目录选型](../wiki/comparisons/robot-description-catalogs.md) · [URDD（Beyond URDF）](../wiki/entities/paper-urdd-universal-robot-description-directory.md) — 开源模型目录与标准化预处理目录

### 推荐做什么
- 拿 Stage 1 的装配体（或一台轮式双臂/双足开源结构），走通“STEP → Xacro/URDF → `check_urdf` → MJCF / OpenUSD 导出”完整流水线，并编写自动校验脚本核对总质量、关节轴向、限位单位与左右镜像一致性
- 故意将一个凹面夹爪直接作为 MuJoCo 非凸 mesh 导入并开启碰撞框可视化，亲眼确认“静默凸包化”为什么会挡住抓手内腔，再用凸分解或复合基本体修复

### 学完输出什么
- 一份**可版本控制的“单真值源 + 多格式导出”机器人资产包**（URDF/Xacro + MJCF/SDF/USD 衍生产物）
- 一张**本机器人的跨格式转换回归检查单**（覆盖单位、零位、关节顺序、碰撞近似）

---

## Stage 3 动力学、闭链、执行器与接触建模：把“能动”变成“物理真实”

**模型在 RViz 里能转不代表它能在动力学求解器里站稳；这一层决定重力矩、冲击能量、并联机构运动耦合和力控响应是否符合物理定律。**

### 前置知识
- Stage 2 内容
- 了解基本的刚体转动惯量张量、PD 控制与摩擦概念（可对照主路线 L2）

### 核心问题
- **刚体 10 参数的物理一致性与惯量积陷阱**：为什么单位阵占位、任意放大主惯量或直接照抄正号 CAD 非对角惯量会让模型发飘或穿地；如何对齐主轴使惯量积为零并检验三角不等式（$J_1 + J_2 \ge J_3$）
- **连杆惯量 vs 转子折算惯量（`armature`）严格分账**：高减速比关节中电机转子折算惯量 $N^2 I_r$ 往往比连杆本身还大，为何必须写进执行器 `armature` 而非直接加进 URDF 连杆 `<mass>`
- **超越单树：并联/闭链机构怎么建**：人形推杆踝、四连杆膝、行星滚柱丝杠或指尖连杆在 URDF 中不能直接建真闭环；如何在 MJCF（`<equality>` + 空间腱）、SDFormat（闭环图 + `must_be_loop_joint`）与 OpenUSD（`excludeFromArticulation`）中正确表达姿态相关的非线性传动比与限位耦合
- **执行器与关节摩擦建模**：Implicit vs Explicit 执行器差异；粘滞阻尼、库仑摩擦、Stribeck 效应与减速器背隙（backlash）何时必须进入模型
- **接触动力学与可微仿真**：硬接触 LCP vs 凸优化软接触（MuJoCo `solref`/`solimp`）vs 可微仿真/软体耦合（FEM/MPM/XPBD）对抓握、滑移和足地冲击的影响

### 推荐读什么
- [连杆与转子惯量（Robot Link and Rotor Inertia）](../wiki/concepts/robot-link-and-rotor-inertia.md) 与 [Armature 建模](../wiki/concepts/armature-modeling.md) — 连杆 10 参数与转子折算惯量的严格边界
- [人形机器人并联关节解算](../wiki/concepts/humanoid-parallel-joint-kinematics.md) 与 [Closing the Loop：G1 推杆踝闭链 MuJoCo 建模](../wiki/entities/g1-pushrod-ankle-closed-loop-modeling.md) — 真机联合限位、空间腱/实体杆闭链与非恒定传动比的实战样本
- [Implicit / Explicit 执行器建模](../wiki/concepts/implicit-explicit-actuator-modeling.md) 与 [关节摩擦模型](../wiki/concepts/joint-friction-models.md) — 执行器延迟、力矩饱和与摩擦谱系
- [Contact Dynamics](../wiki/concepts/contact-dynamics.md) 与 [Differentiable Simulation](../wiki/concepts/differentiable-simulation.md) — 接触求解机制、滑移成因与可微物理路线

### 推荐做什么
- 在 MuJoCo 中开启等效惯量盒（equivalent inertia box）与接触点可视化，检查自己机器人的每个 link 惯量盒是否包裹在真实几何内部，移除所有占位单位阵
- 对一个并联踝或四连杆机构，对比“开环树近似 + `mimic`”与“MJCF 实体杆/腱闭环约束”在大偏转姿态下的末端力矩与关节限位边界差异

### 学完输出什么
- 一份**物理自洽的多体动力学模型**（通过等效惯量盒目检、闭链约束验证与 `armature` 分账）
- 能定量解释“把 `armature` 塞进 URDF 连杆质量会怎样污染重力项 $g(q)$”以及“为什么刚体模型不配软接触调参会在落地时弹跳或缓慢滑移”

---

## Stage 4 仿真三层分工、传感器建模与场景拼装

**拿到标配的机器人本体模型后，还需要选对物理引擎与仿真平台、挂齐传感器与外参，并在共享场景里把机器人物件与环境（含 Real2Sim 资产）拼起来。**

### 前置知识
- Stage 3 内容
- 了解相机内/外参、IMU、深度图或 ROS 2 传感器消息概念

### 核心问题
- **物理引擎 / 仿真平台 / 学习框架三层分工**：不要把“渲染慢”归咎于物理求解器，也不要把“接口不同步”算到任务奖励头上；按任务选型 [MuJoCo](../wiki/entities/mujoco.md)、[Isaac Sim](../wiki/entities/isaac-sim.md)、[Gazebo Sim](../wiki/entities/gazebo-sim.md)、[Genesis](../wiki/entities/genesis-sim.md)、[Newton](../wiki/entities/newton-physics.md) 或 [Drake](../wiki/entities/drake.md)
- **传感器接口、坐标与噪声延迟建模**：区分 REP-103 机身系与相机 `optical frame`（z 前 x 右 y 下）；模拟相机畸变、深度噪声/孔洞、IMU 零偏随机游走、足底力/六维力传感器量程与滤波延迟
- **场景拼装与资产库复用（与 Real2Sim 汇合点）**：使用 [NVIDIA SimReady Central](../wiki/entities/nvidia-simready-central.md)、[光轮 SimReady](../wiki/entities/lightwheel-simready.md) 或 [Real2Sim 纵深](depth-real2sim.md) 重建的真实场景资产，与正向建模的机器人本体在 OpenUSD/MJCF 中合装成可重置任务环境
- **零后端浏览器轻量孪生可视化**：何时使用 [robot-viewer](../wiki/entities/robot-viewer.md) 或 [mjswan](../wiki/entities/mjswan.md)（MuJoCo WASM + ONNX）做跨团队评审、在线交互施力与模型巡检

### 推荐读什么
- [机器人仿真三层分工（Physics / Platform / Learning）](../wiki/concepts/robot-simulation-three-layers.md) — 厘清求解器、仿真平台、学习框架三层边界
- [Locomotion RL 仿真器选型指南](../wiki/queries/simulator-selection-guide.md) — 主流开源/工业仿真引擎横向选型与生态坐标
- [机器人仿真环境集成](../wiki/methods/robot-simulation-integration.md) · [机器人传感器接口与集成](../wiki/methods/robot-sensor-integration.md) · [机器人仿真场景生成](../wiki/methods/robot-simulation-scene-generation.md) — 控制插件挂载、REP-103/光学帧对齐与场景重置检查
- [NVIDIA Omniverse](../wiki/entities/nvidia-omniverse.md) · [OmniGraph](../wiki/entities/omnigraph.md) · [NVIDIA SimReady Central](../wiki/entities/nvidia-simready-central.md) · [光轮 SimReady](../wiki/entities/lightwheel-simready.md) — OpenUSD 底座、可视化逻辑图与物理校验资产库
- [Pinocchio](../wiki/entities/pinocchio.md) · [robot-viewer](../wiki/entities/robot-viewer.md) · [mjswan](../wiki/entities/mjswan.md) — 运动学动力学解算对比及浏览器轻量化模型预览/互动栈

### 推荐做什么
- 用同一个机器人模型搭建“机器 + 操作台/楼梯场景 + RGB-D + IMU + 接触传感器”，对比其在 MuJoCo 与 Isaac Sim（或 Gazebo）下的传感器坐标系、步长与实时因子（RTF）
- 用 `robot-viewer` 或 `mjswan` 搭建一个浏览器可打开的模型检查页，允许团队成员交互拖动关节与施加外力

### 学完输出什么
- 一套**带完整传感器配置与可重置场景的机器人仿真环境**
- 一份**跨平台传感器坐标/时间戳对账记录**（确认相机光学系、IMU 轴向与力传感方向零反号）

---

## Stage 5 模型验证、标定与 V&V：让仿真跟真机“物理对账”

**没有经过真机验证的模型只是假说。把正向仿真与反向动力学标定分开，按由廉价到昂贵的五层阶梯校准参数，并做跨引擎回归。**

### 前置知识
- Stage 4 内容
- 有真机传感器日志（称重、悬空静止/激励运动电流、关节位置/速度）或至少能跑多引擎对比脚本

### 核心问题
- **正向仿真 vs 反向标定的本质区别**：仿真已知参数推响应，标定已知响应反推参数；**参数性误差（质量/质心/摩擦）能靠标定消除，结构性误差（柔性/间隙/接触模型错配）靠调参只会越调越偏**
- **URDF 惯量对照真机的五层抽检阶梯**：约定核对 → 书桌物理一致性（正定+三角不等式）→ 台秤称重与悬挂测 CoM → 悬空静止力矩对齐 $g(q)$ → 激励轨迹动力学回归（最小参数集 + LMI 物理一致性约束）
- **整机闭环与单关节悬空标定的差异**：单关节台架测不到整机装配偏差、线束拉扯与寿命漂移，如何结合 [PRIME](../wiki/entities/prime-system-id.md) 与 [MuJoCo SysID](../wiki/entities/mujoco-sysid.md) 联合估计与足底接触校准
- **Sim2Sim 跨引擎交叉回归**：在同一条开环/闭环轨迹下，对比 Pinocchio、MuJoCo、Isaac Sim 的质心轨迹、接触力与力矩残差，把“导入导出错配”提前扼杀在仿真内部

### 推荐读什么
- [动力学仿真与动力学标定：区别与四个接口](../wiki/concepts/robot-dynamics-simulation-vs-calibration.md) — 厘清参数性误差与结构性误差及四大耦合接口
- [URDF 连杆惯量对照真机检查](../wiki/queries/urdf-link-inertia-real-robot-check.md) — **本阶段核心实操清单**（0–4 五层抽检与失败改法）
- [System Identification](../wiki/concepts/system-identification.md) · [MuJoCo SysID](../wiki/entities/mujoco-sysid.md) — 有界最小二乘、可辨识性分析与参数回写 MjSpec/URDF
- [人形整机闭环惯量标定](../wiki/concepts/humanoid-closed-loop-inertia-calibration.md) · [PRIME](../wiki/entities/prime-system-id.md) — 绑定机身序列号的运动学/惯量/IMU/足底力闭环标定架构

### 推荐做什么
- 按五层抽检清单对自己或开源的机器人描述做一遍审计：用 Pinocchio `computeTotalMass` / `centerOfMass` / `computeGeneralizedGravity` 核对总质量、静力矩与质心位置
- 跑通一次“采集真机激励轨迹 → 辨识摩擦/转子惯量/连杆质心 → 写回模型 → 留出轨迹复放力矩残差对比”闭环，交付带版本号的模型资产

### 学完输出什么
- 一份**带版本号、绑定机身序列号的已标定仿真模型 + 参数不确定度区间表**
- 一份**模型 V&V 验证报告**（含 Sim2Sim 一致性指标与真机静/动力学残差对比曲线）

---

## Stage 6 在环验证与数字孪生系统闭环（产线/多机孪生按项目选读）

**当标定过的物理模型接入真实软件中间件与计算硬件，并能持续接收真机遥测状态做在线同步与寿命/安全推演时，Digital Model 才真正长成可工程依赖的 Digital Twin。**

### 前置知识
- Stage 5 内容（已交付通过 V&V 的标定仿真模型）
- 了解 ROS 2 节点通信、实时调度与嵌入式计算平台约束

### 核心问题
- **SIL → PIL → HIL 在环验证阶梯**：
  - **SIL（软件在环）**：把真实感知/规划/WBC/RL 软件栈通过 ROS 2 Bridge 接到仿真模型，验证接口契约、话题 QoS 与逻辑回归；
  - **PIL（处理器在环）**：保留未改动生产固件，把 CAN/EtherCAT/I2C 总线调度与通信抖动纳入仿真闭环；
  - **HIL（硬件在环）**：将真实计算板（如 Jetson）或伺服驱动接进实时仿真 plant，在全机落地前暴露算力、内存与总线瓶颈。
- **Digital Shadow → Digital Twin 实时闭环搭建**：
  - **Shadow**：将真机高频状态日志或在线 ROS 2 topic 实时镜像到仿真场景中回放与可视化诊断（定位碰撞、过流或姿态发散瞬间）；
  - **Twin**：在双向实时同步下利用仿真做前向安全性校验、预测控制或远程操作保护，并在磨损/换载荷后触发增量再标定。
- **选读分支 · 产线与多机 OpenUSD 数字孪生**：在工厂/仓储场景中，利用 OpenUSD Reference/Variant/Instancing 与 Omniverse 组合多台异构机器人、传送带、AGV 与 [SimReady](../wiki/entities/nvidia-simready-central.md) 工装，完成全工位节拍、干涉与感知传感器布局验证。

### 推荐读什么
- [Software-in-the-Loop（SIL，软件在环）](../wiki/concepts/software-in-the-loop.md) — 纯仿真软件栈接口回归与最小闭环设计
- [处理器在环 Sim2Real（PITL/PIL）](../wiki/concepts/processor-in-the-loop-sim2real.md) — 把固件调度与 CAN/I2C 延迟语义拉进闭环
- [Hardware-in-the-Loop（HIL，硬件在环）](../wiki/concepts/hardware-in-the-loop.md) — Jetson/驱动硬件接入实时仿真的软硬件联调边界
- [机器人数字孪生（Digital Twin）](../wiki/concepts/digital-twin.md) — 从 Shadow 回放到双向 Twin 与终身再标定的安全联锁要求
- [仿真评测基础设施](../wiki/concepts/simulation-evaluation-infrastructure.md) · [NVIDIA Omniverse](../wiki/entities/nvidia-omniverse.md) —（产线/多机选读）规模化环境编排与可复现仿真基建
- 下游出口：[Sim2Real 纵深](depth-sim2real.md) · [Real2Sim 纵深](depth-real2sim.md) · [RL 运动控制纵深](depth-rl-locomotion.md) · [具身测评纵深](depth-embodied-eval.md)

### 推荐做什么
- 搭建一条最小 **ROS 2 SIL + Digital Shadow** 闭环：在 Isaac Sim / MuJoCo 中跑通你的控制节点，再录一段真实（或另一引擎生成的）状态流回放驱动孪生体，对比两者同一任务下的关节力矩与接触状态差异
- **（选读）** 用 OpenUSD Sublayer/Reference 把已标定的机器人资产装配到一个工位场景（如桌面装配或仓储货架），为同一个工作站切换两种末端夹爪 Variant 并重新验证可达性与碰撞

### 学完输出什么
- 一套**可复用的 SIL / HIL 测试台架与遥测回放（Digital Shadow）管线**
- 能针对具体项目做出“单机数字孪生够用，还是需要产线级 OpenUSD 多机孪生”的清晰工程决策，并无缝向控制/RL/Sim2Real 团队交付底座

---

## 快速入口汇总

| 阶段 | 核心问题 | 知识页入口 |
|---|---|---|
| Stage 0 | 建模目标、物理题分类与孪生分级 | [机器人数字孪生](../wiki/concepts/digital-twin.md) · [三类物理问题选型](../wiki/comparisons/adams-vs-mujoco-vs-isaac-sim-three-problem-types.md) |
| Stage 1 | 面向仿真的 CAD 清洗、零位与质量补全 | [参数化 CAD 建模流程](../wiki/methods/parametric-cad-workflow.md) · [文字生成 CAD](../wiki/concepts/text-to-cad.md) |
| Stage 2 | 描述格式选型与跨格式转换流水线 | [URDF vs MJCF vs SDF vs OpenUSD](../wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md) · [轮式双臂 URDF 工作流](../wiki/queries/wheeled-dual-arm-humanoid-urdf-workflow.md) |
| Stage 3 | 惯量物理一致性、转子 `armature` 与闭链 | [连杆与转子惯量](../wiki/concepts/robot-link-and-rotor-inertia.md) · [G1 推杆踝闭链建模](../wiki/entities/g1-pushrod-ankle-closed-loop-modeling.md) |
| Stage 4 | 仿真三层、传感器外参与场景拼装 | [机器人仿真三层分工](../wiki/concepts/robot-simulation-three-layers.md) · [仿真器选型指南](../wiki/queries/simulator-selection-guide.md) |
| Stage 5 | 模型验证、真机惯量抽检与系统辨识 | [仿真与标定区别](../wiki/concepts/robot-dynamics-simulation-vs-calibration.md) · [URDF 惯量真机检查](../wiki/queries/urdf-link-inertia-real-robot-check.md) |
| Stage 6 | SIL → PIL → HIL 与双向孪生（含多机选读） | [SIL](../wiki/concepts/software-in-the-loop.md) · [HIL](../wiki/concepts/hardware-in-the-loop.md) · [NVIDIA Omniverse](../wiki/entities/nvidia-omniverse.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制 → 物理智能全栈成长路线](motion-control.md)（本路线是 L1/L2 建模基础、L6 资产对齐与系统集成的工程展开）
- 姊妹与上下游纵深路线（完整目录见 [路线总览](motion-control.md#depth-optional-index)）：
  - [人形整机硬件设计](depth-humanoid-hardware-design.md) — 上游物理样机设计与 CAD 源头，向本路线交付整机装配与硬件指标
  - [Real2Sim](depth-real2sim.md) — 逆向感知重建路线（视频/点云/3DGS → 外部真实场景/物体），与本路线在 OpenUSD / SimReady 场景拼装处汇合
  - [Sim2Real](depth-sim2real.md) — 下游策略迁移路线，消费本路线 Stage 5 交付的标定模型与执行器/延迟参数
  - [传统模型控制](depth-classical-control.md) — 消费本路线标定的质心、惯量与运动学树求解 MPC/WBC
  - [具身模型测评](depth-embodied-eval.md) — 在本路线提供的可信仿真与孪生底座上评测策略与大模型

## 参考来源

- Orlandea (1973), *Node-Analogous, Sparsity-Oriented Methods for Simulation of Mechanical Dynamic Systems*（[ADAMS 一手学术资料索引](../sources/papers/adams_orlandea_primary_refs.md)）— 本方向的起点里程碑（1973）
- [Glaessgen & Stargel (2012) NASA/USAF Digital Twin 原文归档](../sources/papers/glaessgen_stargel_2012_nasa_digital_twin.md) — Digital Twin 权威定义与六维框架
- [ROS URDF XML 官方规范](../sources/sites/ros_urdf_xml_spec.md) · [MuJoCo 官方文档通读](../sources/papers/mujoco_reading_official_docs_computation_overview.md) · [SDFormat 官方规范](../sources/sites/sdformat_official_spec.md) · [OpenUSD `UsdPhysics` 官方规范](../sources/sites/openusd_usdphysics_spec.md)
- [URDF 连杆惯量对照真机检查一手资料索引](../sources/papers/urdf_link_inertia_real_robot_check.md)
- [NVIDIA Learn OpenUSD 课纲归档](../sources/courses/nvidia_learn_openusd.md) · [Isaac Sim SIL 教程](../sources/sites/nvidia-isaac-sim-sil-tutorial.md) · [Isaac Sim HIL 教程](../sources/sites/nvidia-isaac-sim-hil-tutorial.md)
