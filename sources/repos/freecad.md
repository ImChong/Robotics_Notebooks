# FreeCAD（官方源码仓库）

- **标题**：FreeCAD
- **类型**：repo
- **来源**：FreeCAD 组织（GitHub）
- **链接**：<https://github.com/FreeCAD/FreeCAD>
- **克隆**：`https://github.com/FreeCAD/FreeCAD.git` / `git@github.com:FreeCAD/FreeCAD.git`
- **官网**：<https://www.freecad.org>
- **入库日期**：2026-07-03
- **一句话说明：** FreeCAD 官方 **开源参数化 3D CAD** 源码仓库（C++/Python），以 **OpenCASCADE** 为几何内核，覆盖机械设计、装配、FEM、CAM 与 **Robot 工作台**；整体以 **LGPL-2.1** 许可发布，是机器人硬件链路中 **CAD→STEP/网格→URDF** 的常见零许可成本入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/freecad.md`](../../wiki/entities/freecad.md)

## 仓库概况（2026-07-03 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`FreeCAD/FreeCAD`） |
| 默认分支 | `main` |
| 主要语言 | C++ |
| Stars / Forks | ~31.9k / ~5.7k |
| 描述 | Official source code of FreeCAD, a free and opensource multiplatform 3D parametric modeler. |
| 许可 | LGPL-2.1 |
| Topics | `cad`, `opencascade`, `3d-printing`, `fem`, `cam`, `bim`, `engineering` 等 |

## README 摘要

> FreeCAD is an open-source parametric 3D modeler made primarily to design real-life objects of any size. Parametric modeling allows you to easily modify your design by going back into your model history to change its parameters.

**核心技术与文档入口（README 链接）：**

- 用户 Wiki：<https://wiki.freecad.org>
- 开发者手册：<https://freecad.github.io/DevelopersHandbook/>
- 工作台列表：<https://wiki.freecad.org/Workbenches>
- Python 脚本：<https://wiki.freecad.org/Power_users_hub>
- 论坛：<https://forum.freecad.org>
- 预编译包：<https://github.com/FreeCAD/FreeCAD/releases/latest>

**底层技术栈（README）：**

- **OpenCASCADE** — B-rep 几何内核
- **Coin3D** — Open Inventor 兼容 3D 场景表示
- **Python** — 脚本与宏 API
- **Qt** — 图形界面

## 与机器人研究/工程的关联点

- **机械 CAD 上游**：支架、夹具、连杆、外壳等 **参数化零件** 设计；导出 **STEP / STL / OBJ** 供 [step2urdf](../../wiki/entities/step2urdf.md)、[URDF-Studio](../../wiki/entities/urdf-studio.md) 或厂商 URDF 管线消费。
- **惯量与结构**：**Part / Assembly** 工作台可导出实体几何；**FEM** 工作台支持有限元分析（与 WBC/MPC 中的结构刚度认知互补，但不替代动力学标定）。
- **Robot 工作台与社区插件**：内置 **Robot** 工作台支持连杆-关节建模与运动学预览；社区 **CROSS**、**RobotCAD**、**RobotCreator** 等插件可从 CAD 生成 **URDF/xacro** 与 ROS2 描述包（ROS 文档亦将 FreeCAD 列为 URDF 导出工具之一）。
- **与 DCC 分工**：相对 [Blender](../../wiki/entities/blender.md) 的 **动画/渲染/网格** 强项，FreeCAD 强调 **制造级 B-rep、约束草图与工程图**——机器人栈中常作 **硬件真值 CAD**，而非动捕后处理。
- **OpenCASCADE 同族**：与浏览器端 [step2urdf](../../wiki/entities/step2urdf.md)（OpenCascade.js）共享 **STEP/B-rep** 语义，便于 **同一装配体** 在桌面 CAD 精修后导出 STEP 再转 URDF。

## 对 wiki 的映射

- 升格页面：[wiki/entities/freecad.md](../../wiki/entities/freecad.md)
- 交叉引用：[wiki/entities/blender.md](../../wiki/entities/blender.md)、[wiki/entities/step2urdf.md](../../wiki/entities/step2urdf.md)、[wiki/concepts/urdf-robot-description.md](../../wiki/concepts/urdf-robot-description.md)、[wiki/concepts/text-to-cad.md](../../wiki/concepts/text-to-cad.md)、[wiki/queries/wbc-implementation-guide.md](../../wiki/queries/wbc-implementation-guide.md)

## 参考链接

- 源码仓库：<https://github.com/FreeCAD/FreeCAD>
- 官网：<https://www.freecad.org>
- Wiki：<https://wiki.freecad.org>
- ROS 2 URDF 导出工具列表（含 FreeCAD 生态）：<https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/Exporting-an-URDF-File.html>
