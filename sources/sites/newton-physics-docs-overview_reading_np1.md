# Newton Physics 官方文档 Overview 通读（stable 1.7.0 首页）

> 来源归档（原文通读 · 支撑 `wiki/entities/newton-physics.md`）

- **标题：** Newton Physics Documentation — Overview
- **作者：** The Newton Developers（Disney Research、Google DeepMind、NVIDIA 发起，Linux Foundation 维护）
- **类型：** site / official-docs / physics-engine
- **原文：** <https://newton-physics.github.io/newton/stable/guide/overview.html>（2026-11 经代理获取全文 HTML，页面标注 Newton Physics 1.7.0）
- **一句话说明：** 官方总览页：定位、关键特性、核心对象（ModelBuilder/Model/State/Contacts/Control/Solver/Sensors/Importer/Viewer）与标准仿真循环。
- **核对范围：** 通读该 Overview 页正文全文（Key Features、Core Concepts 含流程图、Simulation Loop、Quick Links）。该页是索引性质的短页；其下属的 Installation、Concepts（Worlds/Articulations/Collisions 等）、Solvers 各页、API Reference、FAQ 及 GitHub 源码未在此次通读，相关事实依赖 wiki 页面既有内容与其他 sources 归档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** GPU 加速、可扩展、可微的物理仿真引擎，面向机器人学、研究与高级仿真；构建于 NVIDIA Warp 之上并集成 MuJoCo Warp，提供现代 Python API。
- **关键特性：** GPU 加速；多求解器（XPBD、VBD、MuJoCo、Featherstone、SemiImplicit、Kamino、ImplicitMPM、Style3D）；模块化可扩展；支持可微仿真；可从 URDF、MJCF、USD 等导入；由 Disney Research、Google DeepMind、NVIDIA 维护。
- **核心对象：** ModelBuilder 为建模入口；Model 封装物体、关节、形状与物理参数；State 为随时间变化的位置速度；Contacts 存放 `CollisionPipeline.collide` 产生的接触集；Control 为关节目标与力等输入；Solver 负责积分、接触与约束；Sensors 由 State/Contacts/site/shape 计算观测；Importer 为 `add_urdf()/add_mjcf()/add_usd()`；另有可选 solver observables（加速度、接触力等诊断量，与状态分开存放）。
- **仿真循环：** 构建/导入 → finalize 为 Model → 创建 Sensors 与 CollisionPipeline，分配 State/Control/Contacts → `collide` → Solver 步进 → 更新传感器/渲染/导出。
- **导航：** 文档含 Worlds、Articulations、Deformable Objects、Collisions and Contacts、Coupled Solvers、Actuators、Solvers（MuJoCo/Kamino/VBD）、API 参考、FAQ、Migration Guide、Isaac Lab 等章节（仅见目录，未读）。

## 对 wiki 的映射

- [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

## 参考来源

- 官方文档：<https://newton-physics.github.io/newton/stable/guide/overview.html>
