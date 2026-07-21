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
- **许可证：** BSD-3-Clause
- **代码：** **已开源** — GitHub 仓库 + PyPI 轮子（`pip install ssik`）；`ssik build` 需 `ssik[urdf]` 从 URDF 生成 per-arm artifact
- **一句话说明：** 6R/7R 旋转关节机械臂的解析逆运动学库：按臂生成单文件 artifact，返回全部 IK 分支且默认 FK 残差低于典型机器人重复精度；覆盖非 Pieper 6R 与非 SRS 7R 几何。
- **沉淀到 wiki：** 是 → [`wiki/entities/ssik.md`](../wiki/entities/ssik.md)

---

## 核心定位

- **问题：** 数值 IK（DLS、TracIK、MINK 等）依赖种子、通常只收敛到单一解；EAIK / IK-Geo 等解析库对 **非 Pieper 6R** 与 **7R 冗余臂** 常直接拒绝。
- **做法：** 子问题分解（Raghavan–Roth、Manocha–Canny、Husty–Pfurner 等）+ 统一 dispatch；**构建期** 把每臂预处理烘焙进单文件 `.py` artifact，**运行时** 无 URDF 解析、无 sympy。
- **预置臂：** 轮子内置 19+ 款（UR 系、Franka Panda/FR3、iiwa14、xArm6/7、JACO2、PiPER、Rizon、OpenArm、R1 Pro 等）；非标准工具链或夹具需 `ssik build`。
- **API：** `solve(T_target)` → `list[Solution]`，每项含 `q`、`fk_residual`、`refinement_used`；支持 `q_seed` / `seed_tolerance` 做轨迹连续 IK。
- **运行时序：** 见 [wiki/entities/ssik.md § 源码运行时序图](../wiki/entities/ssik.md#源码运行时序图)（`prebuilt/*_ik.py` → `ssik.solvers.*` → `ssik.postprocess` → 可选 `ssik.refinement`）。

---

## 对 wiki 的映射

- 实体页：[ssik](../../wiki/entities/ssik.md)
- 任务交叉：[Manipulation](../../wiki/tasks/manipulation.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)
- 相邻栈：[MoveIt 2](../../wiki/entities/moveit2.md)、[cuRobo](../../wiki/entities/curobo.md)、[Pinocchio 快速上手](../../wiki/queries/pinocchio-quick-start.md)
