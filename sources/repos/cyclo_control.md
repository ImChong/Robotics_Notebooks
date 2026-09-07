# cyclo_control（ROBOTIS Cyclo 运动控制）

> 来源归档

- **标题：** cyclo_control
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/cyclo_control
- **机构：** ROBOTIS（乐百机器人）
- **许可：** Apache-2.0
- **Stars：** ~40（2026-09-07）
- **入库日期：** 2026-09-07
- **一句话说明：** ROBOTIS Physical AI 真机运动控制栈：Pinocchio 运动学 + OSQP 优化 + ROS 2 节点，覆盖 AI Worker（FFW）、OMX/OMY 臂与重定向工具；对接 cyclo_lab / cyclo_intelligence 导出的策略。
- **沉淀到 wiki：** [cyclo-control](../../wiki/entities/cyclo-control.md)

---

## 开源状态（2026-09-07 项目页核查）

| 组件 | 状态 |
|------|------|
| GitHub 仓库 | **已开源**（Apache-2.0） |
| 依赖 | ROS 2 **Jazzy**、`numpy<2`、vcs/rosdep |
| 上游致谢 | dyros_robot_controller、dex-retargeting、pinocchio、osqp-eigen |

---

## 仓库结构（README）

| 包 | 角色 |
|----|------|
| `cyclo_motion_controller_core/` | 运动学、QP 优化、控制器实现、Python 重定向 |
| `cyclo_motion_controller_ros/` | ROS 2 节点、launch、AI Worker / OMX / OMY 配置 |
| `cyclo_motion_controller_ros_py/` | 重定向脚本与测试 |
| `cyclo_motion_controller_models/` | URDF/SRDF 与 RViz 可视化 |
| `osqp_eigen_vendor/` | vendored osqp-eigen |

控制器族：`ffw_*`（AI Worker）、`omx_*`、`omy_*`（movel/movej + 可选 interactive marker）。

---

## 核心摘录

### 1) Cyclo 栈中的「执行层」

- [cyclo_lab](./cyclo_lab.md) 训练策略 → [cyclo_intelligence](./cyclo_intelligence.md) BT+VLA 编排 → **本仓** 将轨迹/速度指令落到真机关节与底盘。
- 与 [cyclo_mjlab](./cyclo_mjlab.md) K1 mjlab 路径的 ONNX 部署说明互补：mjlab 导出后仍须对接 bringup + 本类控制节点。

**对 wiki 的映射：** [cyclo-control](../../wiki/entities/cyclo-control.md)、[robotis](../../wiki/entities/robotis.md)

### 2) 构建与可视化

```bash
# ROS 2 Jazzy workspace；README 要求 numpy<2
ros2 launch cyclo_motion_controller_ros omy_controller.launch.py start_interactive_marker:=true
ros2 launch cyclo_motion_controller_models view_ffw_sg2_follower.launch.py
```

**对 wiki 的映射：** [cyclo-control](../../wiki/entities/cyclo-control.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)

### 3) 重定向与优化栈

- 核心控制器衍生自 **dyros_robot_controller**（SNU）；重定向模块参考 **dex-retargeting**。
- 优化层：**osqp-eigen** + Pinocchio 运动学。

**对 wiki 的映射：** [motion-retargeting](../../wiki/concepts/motion-retargeting.md)

---

## 对 wiki 的映射

- **wiki/entities/cyclo-control.md** — 独立详情节点（本 ingest 新建）
- **wiki/entities/robotis.md** — Cyclo 地图补链
- **sources/repos/robotis-git.md** — 组织表已列本仓，回链实体页

## 当前提炼状态

- [x] 项目页与 README 结构摘录
- [x] 开源核查（Apache-2.0）
- [x] wiki 实体页
