# ROS URDF XML 官方规范（robot / link / joint / transmission / sensor / gazebo）

> 来源归档（ingest · 一手规范通读）

- **标题：** Unified Robot Description Format (URDF) XML Specification — ROS Wiki
- **来源：** Open Robotics / ROS Wiki
- **原文：**
  - [urdf/XML](https://wiki.ros.org/urdf/XML)
  - [urdf/XML/robot](https://wiki.ros.org/urdf/XML/robot)
  - [urdf/XML/link](https://wiki.ros.org/urdf/XML/link)
  - [urdf/XML/joint](https://wiki.ros.org/urdf/XML/joint)
  - [urdf/XML/Transmission](https://wiki.ros.org/urdf/XML/Transmission)
  - [urdf/XML/sensor](https://wiki.ros.org/urdf/XML/sensor)
  - [urdf/XML/Gazebo](https://wiki.ros.org/urdf/XML/Gazebo)
- **入库日期：** 2026-11-28
- **核对范围：** 通过 ROS Wiki `?action=raw` 抓取并完整通读上述 7 个官方 XML 规范页面的全部原始条文（元素、必填/可选属性、默认值、单位与官方警示注记）。未逐一核查外部历史链接（如废弃的 PR2 `pr2_controller_manager` 细则）。
- **通读：** 已完成
- **一句话说明：** ROS 生态描述单个机器人运动学树、刚体惯量与视碰几何的基础 XML 规范，同时明确记录了其对负号惯量积的约定、以及闭链/传感/仿真的原生态边界。

## 一手规范核对要点

1. **`<robot>` 根元素（`urdf/XML/robot`）：**
   - 顶层子元素仅包含 `<link>`、`<joint>`、`<transmission>`（PR2 起步扩展，连接执行器与关节）、`<gazebo>`（Gazebo 仿真扩展）。主文件必填 `name`。
   - **原生不包含 `<world>`、环境模型或标准物理引擎配置块**。
2. **`<link>` 与惯量陷阱（`urdf/XML/link`）：**
   - `<inertial>` 可选，但**未指定时默认零质量、零惯量**。
   - `<origin xyz rpy>` 定义质心系 $C$ 相对 link 系 $L$ 的位姿（RPY 固定轴弧度）；**$C$ 不必对齐主惯量轴**。
   - `<inertia>` 采用 **负号惯量积约定**（$I_{xy}=-\int xy\,\rho\,dV$）；官方建议将质心系对齐主惯量轴使惯量积为零，避免 CAD 符号相反。
   - `<visual>` 与 `<collision>` 可各自包含多个几何体，支持 `<box>`、`<cylinder>`（对称轴固定沿 **Z** 轴）、`<sphere>`、`<mesh>`（`package://...`，官方推荐碰撞网格单 link 少于 1000 面或优先基本体）。支持自定义 XML 元素（如 `<collision_checking>`）由外部解析器读取。
3. **`<joint>` 与树形约束（`urdf/XML/joint`）：**
   - 仅支持 **6 种类型**：`revolute`、`continuous`、`prismatic`、`fixed`、`floating`（6-DoF）、`planar`。
   - `<origin>` 定义子 link 相对父 link 的变换，关节位于子 link 原点；`<axis>` 默认 `(1, 0, 0)`（注意与 SDFormat 默认 `(0, 0, 1)` 不同！）。
   - `<dynamics>` 仅含标量 `damping` 与 `friction`；`<limit>` 对 `revolute`/`prismatic` 必填 `effort` 与 `velocity`。
   - `<mimic>`（ROS Groovy 引入）仅表达线性跟随公式 `value = multiplier * other_joint_value + offset`，**并非一般机械闭链动力学约束**；每个 `<joint>` 连接单一 `<parent>` 与 `<child>` 形成树形拓扑。
4. **`<transmission>` 与 `<sensor>`（`urdf/XML/Transmission` & `sensor`）：**
   - `<transmission>` 描述电机执行器与关节之间的功率守恒耦合（含 `<mechanicalReduction>` 与 `<hardwareInterface>`），原文注明当前主要由 `ros_control` 使用。
   - `<sensor>` 页面顶部官方明确给出警示：虽在 URDF DOM 中有实现，但**在应用中从未真正落地（项目已被放弃）**；实际仿真传感器需借助 SDFormat / `<gazebo>` 标签或平台专有 schema。

## 对 wiki 与路线的映射

- [`wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md`](../../wiki/comparisons/urdf-vs-mjcf-vs-sdf-vs-openusd.md) — 四大格式对比主据
- [`wiki/concepts/urdf-robot-description.md`](../../wiki/concepts/urdf-robot-description.md) — 概念页补充规范交叉
- [`roadmap/depth-robot-modeling-digital-twin.md`](../../roadmap/depth-robot-modeling-digital-twin.md) — Stage 2 描述格式规范依据
