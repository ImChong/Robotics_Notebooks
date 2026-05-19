# Newton Physics 官方文档：Overview

> 来源归档

- **标题：** Newton Physics Documentation — Overview
- **类型：** site（官方文档）
- **来源：** newton-physics 文档站（stable）
- **链接：** https://newton-physics.github.io/newton/stable/guide/overview.html
- **入库日期：** 2026-05-19
- **一句话说明：** Newton 架构总览：Warp + MuJoCo Warp、多求解器、ModelBuilder 仿真循环与 URDF/MJCF/USD 导入的官方权威说明。
- **沉淀到 wiki：** 是 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 特性列表（文档原文摘要）

- GPU 加速（NVIDIA Warp）
- 多求解器实现：**XPBD、VBD、MuJoCo、Featherstone、SemiImplicit**
- 模块化：可扩展新求解器与组件
- **可微仿真**（ML 与优化）
- 导入/导出：**URDF、MJCF、USD** 等
- 开源维护方：Disney Research、Google DeepMind、NVIDIA

## 核心对象与仿真循环

### 架构对象

| 类型 | 职责 |
|------|------|
| **ModelBuilder** | 从基元或导入资产构建模型（`add_urdf` / `add_mjcf` / `add_usd`） |
| **Model** | 刚体、关节、形状与物理参数的世界描述 |
| **State** | 位姿、速度等动态状态；可选 extended attributes（如刚体加速度） |
| **Contacts** | `Model.collide` 产出的接触集；可含接触力等扩展量 |
| **Control** | 关节目标、力矩等控制输入 |
| **Solver** | 积分、接触与约束；多后端可选 |
| **Sensors** | 从 State / Contacts / site / shape 计算观测 |
| **Viewer** | 实时或离线可视化 |

### 标准步进顺序

1. `ModelBuilder` 构建或导入 → `finalize` 为 `Model`
2. 分配 `State`、`Control`、`Contacts`；按需创建 Sensors
3. `Model.collide` 填充当前接触集
4. `Solver.step`（state + control + contacts）
5. 更新传感器、渲染或导出

## 对 wiki 的映射

- 仿真循环 Mermaid 与 API 名词表 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)「流程总览」
- 与 MuJoCo Warp / MJCF 对齐理解 → [`wiki/entities/mjlab.md`](../../wiki/entities/mjlab.md)、[`wiki/entities/mujoco-mjx.md`](../../wiki/entities/mujoco-mjx.md)
