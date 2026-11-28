# SDFormat (SDF) 官方规范通读（`gazebosim/sdformat` `sdf15` · `sdf/1.11`）

> 来源归档（ingest · 一手规范通读）

- **标题：** SDFormat (Simulation Description Format) Specification 1.11 / `libsdformat15`
- **来源：** Open Source Robotics Foundation (OSRF) / Gazebo
- **原文：**
  - [gazebosim/sdformat README (`sdf15`)](https://github.com/gazebosim/sdformat/blob/sdf15/README.md)
  - `sdf/1.11/` 规范文件组：[`root.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/root.sdf)、[`world.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/world.sdf)、[`model.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/model.sdf)、[`link.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/link.sdf)、[`joint.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/joint.sdf)、[`inertial.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/inertial.sdf)、[`frame.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/frame.sdf)、[`pose.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/pose.sdf)、[`geometry.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/geometry.sdf)、[`collision.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/collision.sdf)、[`visual.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/visual.sdf)、[`surface.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/surface.sdf)、[`mimic.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/mimic.sdf)、[`sensor.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/sensor.sdf)、[`physics.sdf`](https://github.com/gazebosim/sdformat/blob/sdf15/sdf/1.11/physics.sdf)
- **入库日期：** 2026-11-28
- **核对范围：** 抓取并完整通读 `sdf15` 分支 `README.md` 与上述 15 个 `sdf/1.11/*.sdf` 官方 XML Schema 定义文件的全部元素、默认属性、单位和描述文本。未运行 `libsdformat` C++ 解析测试套件，也未逐一展开子传感器参数文件（如 `camera.sdf`、`imu.sdf` 等）。
- **通读：** 已完成
- **一句话说明：** Gazebo 原生的系统级机器人与世界仿真 XML 规范，把世界物理、多模型嵌套、显式参考系、非树运动闭环与 20+ 种传感器统一在一个格式中。

## 一手规范核对要点

1. **世界与模型并重（`root.sdf` / `world.sdf` / `model.sdf`）：**
   - `<sdf>` 顶层即可封装 `<world>`（含重力、磁场、风场、气氛、GUI、灯光、道路、状态快照、多个 `<physics>` 配置块与 `<include merge>`）或 `<model>`；支持 `<model>` 内嵌套子 `<model>` 与 `<include placement_frame>`，适合整套工作站与机器人联合建模。
2. **显式坐标系 `<frame>` 与 `<pose>`（`frame.sdf` / `pose.sdf`）：**
   - 通过 `@attached_to`（随谁刚性运动）与 `@relative_to`（初值数值在哪个系下表达，支持 `::` 跨嵌套作用域）解耦运动学附着与坐标表达；`<pose>` 支持 `euler_rpy`（可选 `degrees="true"`）与 `quat_xyzw` 四元数两种旋转格式。
3. **闭链与丰富关节类型（`joint.sdf` / `mimic.sdf`）：**
   - 支持 `continuous`、`revolute`、`gearbox`、`revolute2`、`prismatic`、`ball`、`screw`（新增 `screw_thread_pitch` 取代旧 `thread_pitch`）、`universal`、`fixed`。
   - **默认 `<axis><xyz>` 为 `0 0 1`**（与 URDF 默认 `1 0 0` 不同），且可通过 `@expressed_in` 在任意命名帧中表达轴向。
   - **原生支持闭环机构**：关节可在 `<world>` 或 `<model>` 任意两帧之间建立，并通过 `<simbody><must_be_loop_joint>` 显式指定闭环切割点；`<mimic>` 还增加 `reference` 参数（与 URDF 仅 `multiplier`/`offset` 不同）。
4. **惯量、几何与接触表面（`inertial.sdf` / `geometry.sdf` / `surface.sdf`）：**
   - 保留与 URDF 一致的负号惯量积约定（`-m x y`），同时支持 `inertial@auto="true"` 基于碰撞几何体和密度自动估算惯量、以及水下机器人的 21 元对称 `fluid_added_mass` 矩阵。
   - `<geometry>` 支持 `box`、`capsule`、`cone`、`cylinder`、`ellipsoid`、`heightmap`、`image`、`mesh`、`plane`、`polyline`、`sphere`。
   - `<surface>` 提供弹性恢复（`bounce`）、ODE/Bullet 摩擦与扭转摩擦（`torsional`）、比特掩码碰撞过滤（`collide_bitmask` / `category_bitmask`）、杨氏模量/泊松比及 DART 软接触参数。
5. **原生传感器与多物理后端（`sensor.sdf` / `physics.sdf`）：**
   - `<sensor>` 原生内置 `camera`、`depth_camera`、`rgbd_camera`、`segmentation_camera`、`boundingbox_camera`、`thermal_camera`、`lidar`/`gpu_lidar`、`imu`、`force_torque`、`contact`、`magnetometer`、`altimeter`、`navsat`、`sonar` 等。
   - `<physics>` 支持声明 `ode`、`bullet`、`simbody`、`dart` 引擎及专属调参块（`max_step_size`、`real_time_factor`、CFM/ERP 等）。

## 对 wiki 与路线的映射

- [`wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md`](../../wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md) — 四大格式对比核心规范依据
- [`roadmap/depth-robot-modeling-digital-twin.md`](../../roadmap/depth-robot-modeling-digital-twin.md) — Stage 2 描述格式选型与 Stage 4 传感器建模依据
