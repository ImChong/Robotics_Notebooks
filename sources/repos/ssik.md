# ssik

> 来源归档

- **标题：** ssik
- **类型：** repo
- **机构：** 华盛顿大学 Personal Robotics Lab（UW PRL）
- **链接：** https://github.com/personalrobotics/ssik
- **文档站：** https://personalrobotics.github.io/ssik/
- **PyPI：** https://pypi.org/project/ssik/
- **Stars：** ~125（2026-07）
- **入库日期：** 2026-07-21
- **复核日期：** 2026-11-19
- **通读：** 已完成
- **核对范围：** `README.md`、`docs/`（`teleop.md`、`architecture.md` 等）、快速上手与覆盖率文档、`src/ssik/` 核心模块（`cli.py`、`manipulator.py`、`_kinbody.py`、`tracker.py`、`postprocess.py`、`refinement/__init__.py`、`core/dispatcher.py`、`core/codegen.py`）
- **许可证：** BSD-3-Clause
- **代码：** **已开源** — GitHub 仓库 + PyPI 轮子（`pip install ssik`）；`ssik build` 需 `ssik[urdf]` 从 URDF 生成 per-arm artifact；内置 `_ssik_native` C++ 后端并支持导出零 Python 依赖的纯头文件 `ssik_cpp`
- **一句话说明：** 6R/7R 旋转关节机械臂的解析逆运动学库：覆盖 27 家厂商 72 款预置机械臂，支持四级子问题分解调度、C++ 原生后端微秒级求解、`self_motion` 自运动流形图表、`Tracker` 在线流式 IK 与非 Pieper / 非 SRS 构型。
- **沉淀到 wiki：** 是 → [`wiki/entities/ssik.md`](../../wiki/entities/ssik.md)

---

## 核心定位

- **问题：** 数值 IK（DLS、TracIK、MINK 等）依赖种子、通常只收敛到单一解；EAIK / IK-Geo 等解析库对 **非 Pieper 6R** 与 **7R 冗余臂** 常直接拒绝。
- **做法：** 子问题分解（Raghavan–Roth、Manocha–Canny、Husty–Pfurner 等）+ 统一 dispatch；**构建期** 把每臂预处理烘焙进单文件 `.py` artifact，**运行时** 无 URDF 解析、无 sympy。
- **预置臂：** 轮子内置 19+ 款（UR 系、Franka Panda/FR3、iiwa14、xArm6/7、JACO2、PiPER、Rizon、OpenArm、R1 Pro 等）；非标准工具链或夹具需 `ssik build`。
- **API：** `solve(T_target)` → `list[Solution]`，每项含 `q`、`fk_residual`、`refinement_used`；支持 `q_seed` / `seed_tolerance` 做轨迹连续 IK。
- **运行时序：** 见 [wiki/entities/ssik.md § 源码运行时序图](../../wiki/entities/ssik.md#源码运行时序图)（`prebuilt/*_ik.py` → `ssik.solvers.*` → `ssik.postprocess` → 可选 `ssik.refinement`）。
- **类结构：** 见 [wiki/entities/ssik.md § 源码类图](../../wiki/entities/ssik.md#源码类图)（`Manipulator` / `ArmArtifact` / `KinBody` / `Solution` / `DispatchPlan`）。

---

## 对 wiki 的映射

- 实体页：[ssik](../../wiki/entities/ssik.md)
- 任务交叉：[Manipulation](../../wiki/tasks/manipulation.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)
- 相邻栈：[MoveIt 2](../../wiki/entities/moveit2.md)、[cuRobo](../../wiki/entities/curobo.md)、[Pinocchio 快速上手](../../wiki/queries/pinocchio-quick-start.md)

---

## 代码深读补充（2026-11-19）

- **覆盖规模扩展：** 预置机械臂扩展至 **27 家厂商共 72 款模型**，覆盖通用 6R（含非 Pieper 偏移腕部）与 7R（含 `seven_r.spherical_shoulder` 与 `seven_r.srs`）。
- **四级调度器与代码生成（`core/dispatcher.py` / `core/codegen.py`）：**
  - 按几何结构族分派四级求解策略，并对奇异折痕附近引入 `_polished` 精修层。
  - 构建期生成独立单文件求解器，支持直接启用随包内置的 C++ 原生后端 **`_ssik_native`**（默认启用，较纯 Python 快 2–100×：6R 约 10–460 µs，7R SRS/球形肩约 120–360 µs，joint-lock 7R 约 1.1–12 ms），并可通过 `cli.py`（`classify` / `build` / `add-arm`）导出零 Python 依赖的**纯头文件 C++（`ssik_cpp`，如 `ssik::iiwa14_ik`）**。
- **核心 API 与流式遥操作（`manipulator.py` / `_kinbody.py` / `tracker.py` / `postprocess.py` / `refinement/__init__.py`）：**
  - 支持 `[−2π, 2π]` 全绕数（winding）枚举、冗余臂 **`self_motion` 自运动流形图表（Chart）**，以及 `Diagnostic` 与 `TolerancePolicy` 诊断体系。
  - `ssik.Tracker`（`TrackerStep`、`TrackStatus` = `OK` / `LIMITED` / `HELD` / `JUMPED`、`Redundancy`）专为 VR/SpaceMouse/mocap 流式遥操作设计：在 7R 图表臂上固定冗余坐标并沿流形连续追踪，避免解分支跳变（elbow flip），支持速率限幅与 `set_redundancy()` 在不移动末端前提下滑动肘部零空间。
