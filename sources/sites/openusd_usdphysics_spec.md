# OpenUSD `UsdPhysics` 官方规范通读（`PixarAnimationStudios/OpenUSD` `release`）

> 来源归档（ingest · 一手规范通读）

- **标题：** `UsdPhysics` — USD Physics Schema Overview & Schema Definition
- **来源：** Pixar Animation Studios / OpenUSD（与 NVIDIA / Apple 等联合制定）
- **原文：**
  - [`pxr/usd/usdPhysics/overview.dox`](https://github.com/PixarAnimationStudios/OpenUSD/blob/release/pxr/usd/usdPhysics/overview.dox)
  - [`pxr/usd/usdPhysics/schema.usda`](https://github.com/PixarAnimationStudios/OpenUSD/blob/release/pxr/usd/usdPhysics/schema.usda)
  - [`pxr/usd/usdPhysics/tokens.h`](https://github.com/PixarAnimationStudios/OpenUSD/blob/release/pxr/usd/usdPhysics/tokens.h)
- **入库日期：** 2026-11-28
- **核对范围：** 完整通读上述三个官方源码文件的全部内容（`overview.dox` 684 行、`schema.usda` 886 行、`tokens.h` 412 行）。范围限于 OpenUSD 核心刚体物理 Schema（`UsdPhysics`）；NVIDIA Isaac Sim 专属 `PhysxSchema` 扩展与渲染材质 `UsdShade` 细节未逐文件展开。
- **通读：** 已完成
- **一句话说明：** OpenUSD 官方以非破坏性 API Schema 挂载在 3D 场景图之上的核心刚体与铰接体物理规范，用统一 Stage 支撑多软件协作与大规模仿真/数字孪生场景合成。

## 一手规范核对要点

1. **API Schema 非破坏性叠加设计（`overview.dox` §USD Implementation）：**
   - 为避免膨胀 USD 节点数量并保持与既有图形管线兼容，刚体、碰撞、质量、材质、限位与驱动均实现为 **API Schema**（附加在现有 `UsdGeomXformable` / `UsdGeomGprim` / `UsdShadeMaterial` 上，可通过 `RemoveAPI()` 无损撤销）；仅 `PhysicsScene`、`PhysicsCollisionGroup` 与 `PhysicsJoint` 系列为独立 IsA Schema。
2. **单位制度与常见跨格式陷阱（`overview.dox` §Units & `schema.usda`）：**
   - 距离由 Stage 元数据 `metersPerUnit` 指定（注意机器人常用米制而 DCC/CAD 默认常为厘米/毫米）；质量由 `kilogramsPerUnit` 指定；时间使用秒（`TimeCodesPerSecond`）；**角度与角速度统一采用度（`degrees` / `degrees/second`）**——这与 URDF/MJCF/SDF 默认弧度制截然不同，转换时极易出错。
   - 引用（reference）不同长度单位资产时可通过 Xform 缩放修正，但**质量单位切换不能靠简单缩放修正**，需通过覆盖 opinion 显式换算。
3. **质量与惯量解析优先级（`PhysicsMassAPI` & `overview.dox` §Body Mass Properties）：**
   - **显式质量优于密度与体积估算**；且与常见直觉相反，**父级刚体的显式 `mass` 会覆盖子级冲突设置**（避免扣减出现负质量）；但子级碰撞体的 `density` 会覆盖父级密度，未指定时默认水密度 `1000.0 kg/m³`。
   - 惯量拆分为质心 `centerOfMass`、主轴对角惯量 `diagonalInertia` 与主轴四元数 `principalAxes`（注意无非对角惯量积字段，消除了 URDF 负号惯量积争议）。
4. **碰撞体与网格近似（`PhysicsCollisionAPI` & `PhysicsMeshCollisionAPI`）：**
   - 碰撞可附加在基本图元（Sphere/Cylinder/Cube/Cone/Capsule/Plane，其中 Plane 仅可作静态或 kinematic 无限半空间）或 `UsdGeomMesh`。
   - `PhysicsMeshCollisionAPI.approximation` 定义 6 个标准档位：`none`、`convexDecomposition`、`convexHull`、`boundingSphere`、`boundingCube`、`meshSimplification`；也支持同级放一个 `purpose="guide"` 的定制碰撞代理网。
   - 碰撞过滤支持粗粒度 `PhysicsCollisionGroup`（带 `mergeGroup` 跨资产合并与 `invertFilteredGroups`）以及高优先级的点对点 `PhysicsFilteredPairsAPI`。
5. **关节、驱动与闭链处理（`PhysicsJoint` / `ArticulationRootAPI`）：**
   - 关节为对称双参考系设计（`body0`/`localPos0`/`localRot0` 与 `body1`/`localPos1`/`localRot1`），支持 `PhysicsFixedJoint`、`PhysicsRevoluteJoint`、`PhysicsPrismaticJoint`、`PhysicsSphericalJoint`、`PhysicsDistanceJoint` 及通用 D6 关节；原生支持 `breakForce`/`breakTorque`。
   - `PhysicsDriveAPI`（`multipleApply` 按自由度挂载）提供隐式弹簧阻尼驱动模型：$\text{force/acc} = \text{stiffness}(p_{\text{target}}-p) + \text{damping}(v_{\text{target}}-v)$，区分 `force` 与 `acceleration` 两种模式。
   - **减缩坐标树与闭链切换**：在根体或父节点挂 `PhysicsArticulationRootAPI` 即可提示引擎使用减缩坐标求解；遇到机械闭链时，可通过在闭环关节上设置 `physics:excludeFromArticulation = true` 显式将其转为最大坐标约束，从而在标准图结构下无歧义切开闭链。
6. **边界注记：**
   - `UsdPhysics` 核心规范聚焦刚体、碰撞与关节驱动，**不包含相机/LiDAR/IMU 传感器模型、电机电枢（armature）专用字段或软体 FEM/MPM 规范**；这些能力在仿真中需由平台扩展（如 Isaac Sim 的 `PhysxSchema` / sensor schema）或特定引擎补充。

## 对 wiki 与路线的映射

- [`wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md`](../../wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md) — OpenUSD 核心物理规范依据
- [`wiki/entities/nvidia-learn-openusd.md`](../../wiki/entities/nvidia-learn-openusd.md) — OpenUSD 教学路径互补
- [`roadmap/depth-robot-modeling-digital-twin.md`](../../roadmap/depth-robot-modeling-digital-twin.md) — Stage 2 格式选型、Stage 3 闭链与 Stage 4 场景组合依据
